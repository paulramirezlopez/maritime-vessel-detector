#!/usr/bin/env python3
"""Utilities for lifting tiled maritime annotations back to parent images."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_utils import (
    SHIP_LABEL,
    CVATBox,
    normalize_whitespace,
    parse_cvat_xml,
    polygon_area,
    read_image_size,
    rotated_box_to_points,
    source_group_key,
    truncate_rotated_box_to_fit,
)
from preprocess.obb_utils import polygon_to_rotated_box
from preprocess.xview_preprocess import MARITIME_IDS

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DOTA_TILE_RE = re.compile(r"^(?P<parent>.+?)_(?P<x>\d+)_(?P<y>\d+)$")
XVIEW_TILE_RE = re.compile(r"^(?P<parent>.+?)_row(?P<row>\d+)_col(?P<col>\d+)$")
EDGE_TOUCH_EPS = 2.5
ANGLE_THRESHOLD_DEG = 10.0
PAIRWISE_IOU_THRESHOLD = 0.12
PAIRWISE_GAP_RATIO = 0.18
CLEAR_WINNER_SCORE_GAP = 15.0


@dataclass(frozen=True)
class RecoverySourceConfig:
    dataset: str
    split: str
    cleaned_xml: Path
    membership_root: Path
    raw_label_root: Path | None = None
    raw_geojson_path: Path | None = None
    tile_kind: str = "dota"
    xview_stride: int = 824
    excluded_parent_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ParentImage:
    parent_id: str
    image_name: str
    image_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class CandidateAnnotation:
    dataset: str
    split: str
    parent_id: str
    parent_image_name: str
    parent_image_path: Path
    source_kind: str
    source_xml: str
    source_image: str
    tile_name: str
    tile_x: int
    tile_y: int
    tile_width: int
    tile_height: int
    parent_width: int
    parent_height: int
    points: tuple[tuple[float, float], ...]
    box: tuple[float, float, float, float, float]
    edge_touch: bool
    border_margin: float
    source_split: str = ""

    @property
    def source_priority(self) -> int:
        if self.source_kind in {"second_pass", "smart_retile"}:
            return 2000
        if self.source_kind in {"merged"}:
            return 1500
        if self.source_kind in {"cleaned", "assembled", "first_pass"}:
            return 1000
        return 100

    @property
    def angle(self) -> float:
        return self.box[4] % 180.0

    @property
    def score(self) -> float:
        return float(
            self.source_priority
            + min(self.border_margin, 100.0)
            + (10.0 if not self.edge_touch else 0.0)
        )

    @property
    def area(self) -> float:
        return abs((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def discover_membership_images(membership_root: Path) -> dict[str, ParentImage]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None

    parent_map: dict[str, ParentImage] = {}
    for image_path in sorted(
        path for path in membership_root.rglob("*") if path.is_file() and is_image_file(path)
    ):
        parent_id = image_path.stem
        if parent_id in parent_map and parent_map[parent_id].image_path != image_path:
            raise ValueError(
                f"Duplicate parent image stem detected in membership root: {parent_id}"
            )
        width, height = read_image_size(image_path)
        parent_map[parent_id] = ParentImage(
            parent_id=parent_id,
            image_name=image_path.name,
            image_path=image_path,
            width=width,
            height=height,
        )
    return parent_map


def parse_dota_tile_origin(image_name: str) -> tuple[str, int, int] | None:
    match = DOTA_TILE_RE.match(Path(image_name).stem)
    if not match:
        return None
    return match.group("parent"), int(match.group("x")), int(match.group("y"))


def parse_xview_tile_origin(image_name: str, *, stride: int = 824) -> tuple[str, int, int] | None:
    match = XVIEW_TILE_RE.match(Path(image_name).stem)
    if not match:
        return None
    return (
        match.group("parent"),
        int(match.group("col")) * stride,
        int(match.group("row")) * stride,
    )


def _tile_margin(points: Iterable[tuple[float, float]], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    margin = float("inf")
    for x, y in points:
        margin = min(margin, x, y, max_x - x, max_y - y)
    return float(max(margin, 0.0))


def _touches_edge(points: Iterable[tuple[float, float]], width: int, height: int) -> bool:
    return _tile_margin(points, width, height) <= EDGE_TOUCH_EPS


def _project_points(
    points: Iterable[tuple[float, float]], angle_deg: float
) -> tuple[float, float, float, float]:
    theta = math.radians(angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -math.sin(theta), math.cos(theta)
    us: list[float] = []
    vs: list[float] = []
    for x, y in points:
        us.append(x * ux + y * uy)
        vs.append(x * vx + y * vy)
    return min(us), max(us), min(vs), max(vs)


def _rect_iou_and_gaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float, float, float]:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b

    inter_u = max(0.0, min(a1, b1) - max(a0, b0))
    inter_v = max(0.0, min(a3, b3) - max(a2, b2))
    union_u = max(a1, b1) - min(a0, b0)
    union_v = max(a3, b3) - min(a2, b2)
    iou = 0.0
    if union_u > 0.0 and union_v > 0.0:
        iou = (inter_u * inter_v) / (union_u * union_v)

    gap_u = max(0.0, max(a0, b0) - min(a1, b1))
    gap_v = max(0.0, max(a2, b2) - min(a3, b3))
    span_u = max(a1 - a0, b1 - b0)
    span_v = max(a3 - a2, b3 - b2)
    return iou, gap_u, gap_v, inter_u, inter_v, max(span_u, span_v)


def _candidate_compatible(a: CandidateAnnotation, b: CandidateAnnotation) -> bool:
    angle_diff = abs(a.angle - b.angle)
    angle_diff = min(angle_diff, 180.0 - angle_diff)
    if angle_diff > ANGLE_THRESHOLD_DEG:
        return False

    rect_a = _project_points(a.points, a.angle)
    rect_b = _project_points(b.points, a.angle)
    iou, gap_u, gap_v, inter_u, inter_v, max_span = _rect_iou_and_gaps(rect_a, rect_b)
    if iou >= PAIRWISE_IOU_THRESHOLD:
        return True

    span_u_a = rect_a[1] - rect_a[0]
    span_v_a = rect_a[3] - rect_a[2]
    span_u_b = rect_b[1] - rect_b[0]
    span_v_b = rect_b[3] - rect_b[2]
    gap_u_thresh = max(4.0, PAIRWISE_GAP_RATIO * max(span_u_a, span_u_b))
    gap_v_thresh = max(4.0, PAIRWISE_GAP_RATIO * max(span_v_a, span_v_b))

    if (a.edge_touch or b.edge_touch) and gap_u <= gap_u_thresh and gap_v <= gap_v_thresh:
        return True

    if inter_u > 0.0 and inter_v > 0.0:
        return True

    return False


def _cluster_candidates(candidates: list[CandidateAnnotation]) -> list[list[CandidateAnnotation]]:
    if not candidates:
        return []

    parent = list(range(len(candidates)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if _candidate_compatible(candidates[i], candidates[j]):
                union(i, j)

    grouped: dict[int, list[CandidateAnnotation]] = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        grouped[find(idx)].append(candidate)

    clusters = list(grouped.values())
    clusters.sort(
        key=lambda cluster: (
            cluster[0].parent_id,
            min(candidate.tile_y for candidate in cluster),
            min(candidate.tile_x for candidate in cluster),
            len(cluster),
        )
    )
    return clusters


def _candidate_to_annotation(candidate: CandidateAnnotation, *, merged: bool, cluster_size: int) -> dict[str, object]:
    xtl, ytl, xbr, ybr, rotation = candidate.box
    return {
        "label": SHIP_LABEL,
        "xtl": round(xtl, 2),
        "ytl": round(ytl, 2),
        "xbr": round(xbr, 2),
        "ybr": round(ybr, 2),
        "rotation": round(rotation, 3),
        "points": [[round(x, 2), round(y, 2)] for x, y in candidate.points],
        "source_kind": candidate.source_kind,
        "source_split": candidate.source_split,
        "source_xml": candidate.source_xml,
        "source_image": candidate.source_image,
        "source_tile": candidate.tile_name,
        "tile_x": candidate.tile_x,
        "tile_y": candidate.tile_y,
        "tile_width": candidate.tile_width,
        "tile_height": candidate.tile_height,
        "cluster_size": cluster_size,
        "merged": merged,
        "score": round(candidate.score, 3),
    }


def _best_candidate(cluster: list[CandidateAnnotation]) -> CandidateAnnotation:
    return max(
        cluster,
        key=lambda candidate: (
            candidate.score,
            candidate.parent_width * candidate.parent_height,
            -candidate.tile_y,
            -candidate.tile_x,
        ),
    )


def _merge_cluster(cluster: list[CandidateAnnotation]) -> CandidateAnnotation | None:
    if len(cluster) < 2:
        return None

    reference = _best_candidate(cluster)
    if len(cluster) > 3:
        return None

    for candidate in cluster:
        angle_diff = abs(reference.angle - candidate.angle)
        angle_diff = min(angle_diff, 180.0 - angle_diff)
        if angle_diff > ANGLE_THRESHOLD_DEG:
            return None

    ref_angle = reference.angle
    projected_points: list[tuple[float, float]] = []
    for candidate in cluster:
        for x, y in candidate.points:
            theta = math.radians(ref_angle)
            ux, uy = math.cos(theta), math.sin(theta)
            vx, vy = -math.sin(theta), math.cos(theta)
            projected_points.append((x * ux + y * uy, x * vx + y * vy))

    if not projected_points:
        return None

    theta = math.radians(ref_angle)
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -math.sin(theta), math.cos(theta)

    us = [u for u, _ in projected_points]
    vs = [v for _, v in projected_points]
    u0, u1 = min(us), max(us)
    v0, v1 = min(vs), max(vs)
    if u1 <= u0 or v1 <= v0:
        return None

    merged_points = [
        (u0 * ux + v0 * vx, u0 * uy + v0 * vy),
        (u1 * ux + v0 * vx, u1 * uy + v0 * vy),
        (u1 * ux + v1 * vx, u1 * uy + v1 * vy),
        (u0 * ux + v1 * vx, u0 * uy + v1 * vy),
    ]

    if not all(
        -EDGE_TOUCH_EPS <= x <= reference.parent_width - 1 + EDGE_TOUCH_EPS
        and -EDGE_TOUCH_EPS <= y <= reference.parent_height - 1 + EDGE_TOUCH_EPS
        for x, y in merged_points
    ):
        return None

    box = polygon_to_rotated_box(merged_points, clockwise=False)
    if box is None:
        return None

    xtl, ytl, xbr, ybr, rotation = box
    fitted_points = truncate_rotated_box_to_fit(
        CVATBox(
            label=SHIP_LABEL,
            xtl=xtl,
            ytl=ytl,
            xbr=xbr,
            ybr=ybr,
            rotation=rotation,
        ),
        reference.parent_width,
        reference.parent_height,
        clockwise=True,
    )
    if fitted_points is None:
        return None
    fitted_box = polygon_to_rotated_box(fitted_points, clockwise=False)
    if fitted_box is None:
        return None
    xtl, ytl, xbr, ybr, rotation = fitted_box
    return CandidateAnnotation(
        dataset=reference.dataset,
        split=reference.split,
        parent_id=reference.parent_id,
        parent_image_name=reference.parent_image_name,
        parent_image_path=reference.parent_image_path,
        source_kind="merged",
        source_xml=";".join(sorted({candidate.source_xml for candidate in cluster})),
        source_image=reference.source_image,
        tile_name="|".join(sorted({candidate.tile_name for candidate in cluster})),
        tile_x=min(candidate.tile_x for candidate in cluster),
        tile_y=min(candidate.tile_y for candidate in cluster),
        tile_width=reference.parent_width,
        tile_height=reference.parent_height,
        parent_width=reference.parent_width,
        parent_height=reference.parent_height,
            points=tuple(fitted_points),
            box=(xtl, ytl, xbr, ybr, rotation),
            edge_touch=any(candidate.edge_touch for candidate in cluster),
            border_margin=min(candidate.border_margin for candidate in cluster),
            source_split=";".join(
                sorted({candidate.source_split for candidate in cluster if candidate.source_split})
            ),
        )


def _candidate_from_points(
    *,
    dataset: str,
    split: str,
    parent: ParentImage,
    source_kind: str,
    source_xml: str,
    source_image: str,
    tile_name: str,
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
    local_points: list[tuple[float, float]],
    parent_points: list[tuple[float, float]],
    source_split: str = "",
) -> CandidateAnnotation | None:
    if len(parent_points) != 4:
        return None
    if polygon_area(parent_points) <= 0.0:
        return None

    box = polygon_to_rotated_box(parent_points, clockwise=False)
    if box is None:
        return None

    xtl, ytl, xbr, ybr, rotation = box
    # First-pass edge tiles can contain rotated rectangles whose corners extend
    # beyond the parent image. Keep a true rectangle while fitting it in-bounds.
    fitted_points = truncate_rotated_box_to_fit(
        CVATBox(
            label=SHIP_LABEL,
            xtl=xtl,
            ytl=ytl,
            xbr=xbr,
            ybr=ybr,
            rotation=rotation,
        ),
        parent.width,
        parent.height,
        clockwise=True,
    )
    if fitted_points is None:
        return None
    fitted_box = polygon_to_rotated_box(fitted_points, clockwise=False)
    if fitted_box is None:
        return None
    xtl, ytl, xbr, ybr, rotation = fitted_box
    return CandidateAnnotation(
        dataset=dataset,
        split=split,
        parent_id=parent.parent_id,
        parent_image_name=parent.image_name,
        parent_image_path=parent.image_path,
        source_kind=source_kind,
        source_split=source_split,
        source_xml=source_xml,
        source_image=source_image,
        tile_name=tile_name,
        tile_x=tile_x,
        tile_y=tile_y,
        tile_width=tile_width,
        tile_height=tile_height,
        parent_width=parent.width,
        parent_height=parent.height,
        points=tuple(fitted_points),
        box=(xtl, ytl, xbr, ybr, rotation),
        edge_touch=_touches_edge(local_points, tile_width, tile_height),
        border_margin=_tile_margin(local_points, tile_width, tile_height),
    )


def load_cleaned_candidates(
    *,
    dataset: str,
    split: str,
    cleaned_xml: Path,
    membership: dict[str, ParentImage],
    tile_kind: str,
    xview_stride: int = 824,
) -> tuple[dict[str, list[CandidateAnnotation]], dict[str, object]]:
    candidate_map: dict[str, list[CandidateAnnotation]] = defaultdict(list)
    stats: dict[str, object] = {
        "cleaned_images_seen": 0,
        "cleaned_parent_ids": set(),
        "stale_cleaned_parent_ids": set(),
        "missing_parent_ids": set(),
    }

    if not cleaned_xml.exists():
        return candidate_map, stats

    images = parse_cvat_xml(cleaned_xml, allowed_labels=(SHIP_LABEL,))
    stats["cleaned_images_seen"] = len(images)

    for image in images:
        parent_id = source_group_key(image.name)
        if parent_id not in membership:
            stats["stale_cleaned_parent_ids"].add(parent_id)
            continue

        if tile_kind == "dota":
            parsed = parse_dota_tile_origin(image.name)
        else:
            parsed = parse_xview_tile_origin(image.name, stride=xview_stride)
        if parsed is None:
            continue
        _, tile_x, tile_y = parsed
        parent = membership[parent_id]

        for box in image.boxes:
            local_box = CVATBox(
                label=normalize_whitespace(box.label),
                xtl=box.xtl,
                ytl=box.ytl,
                xbr=box.xbr,
                ybr=box.ybr,
                rotation=box.rotation,
            )
            local_points = rotated_box_to_points(local_box, clockwise=True)
            parent_points = [(x + tile_x, y + tile_y) for x, y in local_points]
            candidate = _candidate_from_points(
                dataset=dataset,
                split=split,
                parent=parent,
                source_kind="cleaned",
                source_split=split,
                source_xml=str(cleaned_xml),
                source_image=str(parent.image_path),
                tile_name=image.name,
                tile_x=tile_x,
                tile_y=tile_y,
                tile_width=image.width,
                tile_height=image.height,
                local_points=local_points,
                parent_points=parent_points,
            )
            if candidate is not None:
                candidate_map[parent_id].append(candidate)
                stats["cleaned_parent_ids"].add(parent_id)

    return candidate_map, stats


def load_dota_raw_candidates(
    *,
    dataset: str,
    split: str,
    parent_ids: Iterable[str],
    membership: dict[str, ParentImage],
    raw_label_root: Path,
) -> tuple[dict[str, list[CandidateAnnotation]], list[str]]:
    candidate_map: dict[str, list[CandidateAnnotation]] = defaultdict(list)
    missing_raw_labels: list[str] = []

    for parent_id in sorted(parent_ids):
        parent = membership.get(parent_id)
        if parent is None:
            continue
        label_path = raw_label_root / f"{parent_id}.txt"
        if not label_path.exists():
            missing_raw_labels.append(parent_id)
            continue

        with label_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                label = normalize_whitespace(parts[8])
                if label != SHIP_LABEL:
                    continue
                try:
                    points = [
                        (float(parts[0]), float(parts[1])),
                        (float(parts[2]), float(parts[3])),
                        (float(parts[4]), float(parts[5])),
                        (float(parts[6]), float(parts[7])),
                    ]
                except ValueError:
                    continue
                candidate = _candidate_from_points(
                dataset=dataset,
                split=split,
                parent=parent,
                source_kind="raw_fallback",
                source_split=split,
                source_xml=str(label_path),
                source_image=str(parent.image_path),
                tile_name=parent.image_name,
                tile_x=0,
                    tile_y=0,
                    tile_width=parent.width,
                    tile_height=parent.height,
                    local_points=points,
                    parent_points=points,
                )
                if candidate is not None:
                    candidate_map[parent_id].append(candidate)

    return candidate_map, missing_raw_labels


def load_xview_raw_candidates(
    *,
    dataset: str,
    split: str,
    parent_ids: Iterable[str],
    membership: dict[str, ParentImage],
    raw_geojson_path: Path,
) -> tuple[dict[str, list[CandidateAnnotation]], list[str]]:
    candidate_map: dict[str, list[CandidateAnnotation]] = defaultdict(list)
    missing_geojson_parents: list[str] = []

    if not raw_geojson_path.exists():
        return candidate_map, list(parent_ids)

    data = json.loads(raw_geojson_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        try:
            type_id = int(props.get("type_id", -1))
        except (TypeError, ValueError):
            continue
        if type_id not in MARITIME_IDS:
            continue
        parent_id = Path(str(props.get("image_id", ""))).stem
        if not parent_id:
            continue
        bounds = str(props.get("bounds_imcoords", "")).strip()
        if not bounds:
            continue
        parts = [part.strip() for part in bounds.split(",") if part.strip()]
        if len(parts) != 4:
            continue
        try:
            xmin, ymin, xmax, ymax = (float(value) for value in parts)
        except ValueError:
            continue
        grouped[parent_id].append((xmin, ymin, xmax, ymax))

    for parent_id in sorted(parent_ids):
        parent = membership.get(parent_id)
        if parent is None:
            continue
        if parent_id not in grouped:
            missing_geojson_parents.append(parent_id)
            continue
        for xmin, ymin, xmax, ymax in grouped[parent_id]:
            local_points = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
            ]
            candidate = _candidate_from_points(
                dataset=dataset,
                split=split,
                parent=parent,
                source_kind="raw_fallback",
                source_split=split,
                source_xml=str(raw_geojson_path),
                source_image=str(parent.image_path),
                tile_name=parent.image_name,
                tile_x=0,
                tile_y=0,
                tile_width=parent.width,
                tile_height=parent.height,
                local_points=local_points,
                parent_points=local_points,
            )
            if candidate is not None:
                candidate_map[parent_id].append(candidate)

    return candidate_map, missing_geojson_parents


def recover_parent_split(
    config: RecoverySourceConfig,
    *,
    parent_id_filter: set[str] | None = None,
) -> dict[str, object]:
    membership = discover_membership_images(config.membership_root)
    membership_ids = set(membership) - set(config.excluded_parent_ids)
    if parent_id_filter is not None:
        membership_ids &= parent_id_filter

    cleaned_candidates, cleaned_stats = load_cleaned_candidates(
        dataset=config.dataset,
        split=config.split,
        cleaned_xml=config.cleaned_xml,
        membership=membership,
        tile_kind=config.tile_kind,
        xview_stride=config.xview_stride,
    )

    stale_cleaned_parent_ids = sorted(cleaned_stats["stale_cleaned_parent_ids"])
    cleaned_parent_ids = set(cleaned_stats["cleaned_parent_ids"])
    missing_cleaned_parent_ids = sorted(membership_ids - cleaned_parent_ids)

    fallback_candidates: dict[str, list[CandidateAnnotation]] = defaultdict(list)
    missing_raw_ids: list[str] = []
    if config.raw_label_root is not None and missing_cleaned_parent_ids:
        raw_map, raw_missing = load_dota_raw_candidates(
            dataset=config.dataset,
            split=config.split,
            parent_ids=missing_cleaned_parent_ids,
            membership=membership,
            raw_label_root=config.raw_label_root,
        )
        fallback_candidates.update(raw_map)
        missing_raw_ids.extend(raw_missing)

    if config.raw_geojson_path is not None and missing_cleaned_parent_ids:
        raw_map, raw_missing = load_xview_raw_candidates(
            dataset=config.dataset,
            split=config.split,
            parent_ids=missing_cleaned_parent_ids,
            membership=membership,
            raw_geojson_path=config.raw_geojson_path,
        )
        for parent_id, candidates in raw_map.items():
            fallback_candidates[parent_id].extend(candidates)
        missing_raw_ids.extend(raw_missing)

    candidate_map: dict[str, list[CandidateAnnotation]] = defaultdict(list)
    for parent_id in membership_ids:
        candidate_map[parent_id].extend(cleaned_candidates.get(parent_id, ()))
        candidate_map[parent_id].extend(fallback_candidates.get(parent_id, ()))

    parent_records: list[dict[str, object]] = []
    review_entries: list[dict[str, object]] = []
    total_candidates = 0
    total_clusters = 0
    merged_clusters = 0
    ambiguous_clusters = 0
    raw_fallback_parents = 0
    empty_parents = 0

    for parent_id in sorted(membership_ids):
        parent = membership[parent_id]
        candidates = candidate_map.get(parent_id, [])
        total_candidates += len(candidates)
        annotations: list[dict[str, object]] = []
        source_xmls = sorted({candidate.source_xml for candidate in candidates})
        source_kind = "empty"
        raw_fallback_used = any(candidate.source_kind == "raw_fallback" for candidate in candidates)
        if raw_fallback_used:
            raw_fallback_parents += 1

        if candidates:
            clusters = _cluster_candidates(candidates)
            total_clusters += len(clusters)
            for cluster_index, cluster in enumerate(clusters):
                merged_candidate = _merge_cluster(cluster)
                if merged_candidate is not None:
                    selected = merged_candidate
                    annotations.append(
                        _candidate_to_annotation(
                            selected, merged=True, cluster_size=len(cluster)
                        )
                    )
                    merged_clusters += 1
                    source_kind = "merged" if source_kind == "empty" else source_kind
                    continue

                best = _best_candidate(cluster)
                annotations.append(
                    _candidate_to_annotation(best, merged=False, cluster_size=len(cluster))
                )
                if len(cluster) > 1:
                    sorted_cluster = sorted(cluster, key=lambda candidate: candidate.score, reverse=True)
                    score_gap = sorted_cluster[0].score - sorted_cluster[1].score
                    if score_gap < CLEAR_WINNER_SCORE_GAP:
                        ambiguous_clusters += 1
                        review_entries.append(
                            {
                                "dataset": config.dataset,
                                "split": config.split,
                                "parent_id": parent_id,
                                "cluster_index": cluster_index,
                                "candidate_count": len(cluster),
                                "action": "needs_manual_review",
                                "score_gap": round(score_gap, 3),
                                "candidates": [
                                    {
                                        "source_kind": candidate.source_kind,
                                        "source_xml": candidate.source_xml,
                                        "source_image": candidate.source_image,
                                        "source_tile": candidate.tile_name,
                                        "score": round(candidate.score, 3),
                                        "edge_touch": candidate.edge_touch,
                                        "border_margin": round(candidate.border_margin, 3),
                                        "box": [
                                            round(candidate.box[0], 2),
                                            round(candidate.box[1], 2),
                                            round(candidate.box[2], 2),
                                            round(candidate.box[3], 2),
                                            round(candidate.box[4], 3),
                                        ],
                                    }
                                    for candidate in sorted_cluster
                                ],
                            }
                        )

                source_kind = "cleaned" if any(
                    candidate.source_kind == "cleaned" for candidate in cluster
                ) else "raw_fallback"
        else:
            empty_parents += 1

        if not annotations:
            source_kind = "empty"
        elif any(annotation["merged"] for annotation in annotations):
            source_kind = "merged"
        elif any(annotation["source_kind"] == "cleaned" for annotation in annotations):
            source_kind = "cleaned"
        else:
            source_kind = "raw_fallback"

        parent_records.append(
            {
                "dataset": config.dataset,
                "split": config.split,
                "parent_id": parent_id,
                "image_name": parent.image_name,
                "image_path": str(parent.image_path),
                "width": parent.width,
                "height": parent.height,
                "source_kind": source_kind,
                "raw_fallback_used": raw_fallback_used,
                "annotation_count": len(annotations),
                "merged_count": sum(1 for annotation in annotations if annotation["merged"]),
                "review_cluster_count": sum(
                    1 for entry in review_entries if entry["parent_id"] == parent_id
                ),
                "source_xmls": source_xmls,
                "annotations": annotations,
            }
        )

    summary = {
        "dataset": config.dataset,
        "split": config.split,
        "membership_parent_count": len(membership_ids),
        "cleaned_parent_count": len(cleaned_parent_ids & membership_ids),
        "stale_cleaned_parent_count": len(stale_cleaned_parent_ids),
        "stale_cleaned_parent_ids": stale_cleaned_parent_ids,
        "missing_current_parent_count": len(missing_cleaned_parent_ids),
        "missing_current_parent_ids": missing_cleaned_parent_ids,
        "raw_fallback_parents": raw_fallback_parents,
        "raw_fallback_missing_sources": sorted(set(missing_raw_ids)),
        "lifted_candidate_count": total_candidates,
        "cluster_count": total_clusters,
        "merged_cluster_count": merged_clusters,
        "ambiguous_cluster_count": ambiguous_clusters,
        "empty_parent_count": empty_parents,
        "output_parent_count": len(parent_records),
        "review_entry_count": len(review_entries),
        "excluded_parent_ids": sorted(config.excluded_parent_ids),
    }

    return {
        "dataset": config.dataset,
        "split": config.split,
        "summary": summary,
        "parents": parent_records,
        "review_entries": review_entries,
    }


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cvat_xml(path: Path, recovered: dict[str, object]) -> None:
    root = ET.Element("annotations")
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = SHIP_LABEL
    ET.SubElement(label, "attributes")

    image_id = 0
    for parent in recovered["parents"]:
        image_elem = ET.SubElement(
            root,
            "image",
            id=str(image_id),
            name=str(parent["image_name"]),
            width=str(parent["width"]),
            height=str(parent["height"]),
        )
        image_id += 1
        for annotation in parent["annotations"]:
            ET.SubElement(
                image_elem,
                "box",
                label=SHIP_LABEL,
                xtl=f"{annotation['xtl']:.2f}",
                ytl=f"{annotation['ytl']:.2f}",
                xbr=f"{annotation['xbr']:.2f}",
                ybr=f"{annotation['ybr']:.2f}",
                rotation=f"{annotation['rotation']:.3f}",
                occluded="0",
                z_order="0",
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def format_summary(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        f"Dataset: {report['dataset']}",
        f"Split: {report['split']}",
        f"Membership parents: {summary['membership_parent_count']}",
        f"Cleaned parents in membership: {summary['cleaned_parent_count']}",
        f"Stale cleaned parents excluded: {summary['stale_cleaned_parent_count']}",
        f"Current-membership parents missing from cleaned XML: {summary['missing_current_parent_count']}",
        f"Raw fallback parents used: {summary['raw_fallback_parents']}",
        f"Lifted candidate annotations: {summary['lifted_candidate_count']}",
        f"Clusters: {summary['cluster_count']}",
        f"Merged clusters: {summary['merged_cluster_count']}",
        f"Ambiguous clusters: {summary['ambiguous_cluster_count']}",
        f"Empty parents: {summary['empty_parent_count']}",
        f"Output parents: {summary['output_parent_count']}",
        f"Review entries: {summary['review_entry_count']}",
    ]
    if summary["stale_cleaned_parent_ids"]:
        lines.append(
            "Stale cleaned parent IDs: " + ", ".join(summary["stale_cleaned_parent_ids"][:12])
        )
    if summary["missing_current_parent_ids"]:
        lines.append(
            "Missing current parent IDs: "
            + ", ".join(summary["missing_current_parent_ids"][:12])
        )
    return "\n".join(lines) + "\n"
