#!/usr/bin/env python3
"""Read-only structural audit for the current grid-tile recovery dataset.

The audit deliberately distinguishes parent-coordinate second-pass OBBs from
the tile-local, threshold-filtered labels used by the active v9 training view.
It writes evidence and proposed fixes only; it never changes datasets or labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union
from shapely.strtree import STRtree

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from dataset_utils import CVATBox
from roi_smart_retile_utils import project_box_to_tile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TILE_ROOT = REPO_ROOT / "data/tiled/grid_current_with_recovery"
DEFAULT_PARENT_LABEL_ROOT = REPO_ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery"
DEFAULT_TRAINING_ROOT = REPO_ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3"
DEFAULT_PARENT_IMAGE_ROOT = REPO_ROOT / "data/parent_images/active"
DEFAULT_ROI_ROOT = REPO_ROOT / "data/metadata/roi_grids"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports/tile_coverage_recovery_audit"
EPSILON = 1e-6


@dataclass(frozen=True)
class Tile:
    key: str
    row: dict[str, str]
    width: int
    height: int
    x: int
    y: int

    @property
    def parent_key(self) -> str:
        return self.row["group_key"]

    @property
    def source(self) -> str:
        return self.row["dataset"].lower()

    @property
    def output_split(self) -> str:
        return self.row["split"]

    @property
    def source_split(self) -> str:
        return self.row["source_split"]

    @property
    def tile_source(self) -> str:
        return self.row.get("tile_source") or "unknown"

    @property
    def rect(self) -> Polygon:
        return shapely_box(self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class TileLabel:
    tile_key: str
    line_number: int
    class_id: int
    normalized: tuple[float, ...]
    points: tuple[tuple[float, float], ...]
    polygon: Polygon
    raw_line: str


@dataclass(frozen=True)
class ParentOBB:
    instance_id: str
    group_key: str
    dataset: str
    source_split: str
    parent_name: str
    points: tuple[tuple[float, float], ...]
    box: CVATBox
    width: int
    height: int
    source_kind: str

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.points)


@dataclass
class ParentLabelIndex:
    """Tile-local labels lifted into parent space with a per-tile spatial index."""

    records: list[tuple[TileLabel, Polygon]]
    tree: STRtree | None
    local_records: list[TileLabel]
    local_tree: STRtree | None

    def query(self, polygon: Polygon) -> Iterable[tuple[TileLabel, Polygon]]:
        if self.tree is None:
            return ()
        return (self.records[int(index)] for index in self.tree.query(polygon))

    def query_local(self, polygon: Polygon) -> Iterable[TileLabel]:
        if self.local_tree is None:
            return ()
        return (self.local_records[int(index)] for index in self.local_tree.query(polygon))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    parser.add_argument(
        "--label-root",
        "--parent-label-root",
        dest="parent_label_root",
        type=Path,
        default=DEFAULT_PARENT_LABEL_ROOT,
        help="Second-pass parent OBB directory or its instances.jsonl file.",
    )
    parser.add_argument("--training-dataset-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--tile-metadata", type=Path, default=None)
    parser.add_argument("--roi-metadata-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--parent-image-root", type=Path, default=DEFAULT_PARENT_IMAGE_ROOT)
    parser.add_argument(
        "--suppression-ledger",
        type=Path,
        default=None,
        help="Optional edge-fragment suppression ledger; listed tile/instance pairs are documented rather than high-severity missing labels.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--dataset", choices=("dota", "xview", "hrsc"), default=None)
    parser.add_argument("--split", choices=("train", "val"), default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing audit output root.")
    parser.add_argument("--preview-count", type=int, default=100)
    parser.add_argument("--edge-margin-px", type=float, default=3.0)
    parser.add_argument("--min-valid-visible-fraction", type=float, default=0.25)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.85)
    parser.add_argument("--containment-threshold", type=float, default=0.95)
    parser.add_argument("--centroid-distance-threshold", type=float, default=8.0)
    parser.add_argument("--small-object-threshold-px", type=float, default=12.0)
    parser.add_argument("--run-model-fn-audit", action="store_true")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.20)
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def parse_csv(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.replace("[", "").replace("]", "").split(",") if part.strip().isdigit()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(title: str, summary: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    lines = [f"# {title}", "", "## Summary", ""]
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        lines.append(f"- `{key}`: {value}")
    lines += ["", f"## Findings ({len(issues)})", ""]
    if issues:
        fields = ["severity", "issue_type", "dataset", "parent_image", "tile_image", "details"]
        lines += ["| Severity | Issue | Dataset | Parent | Tile | Details |", "|---|---|---|---|---|---|"]
        for issue in issues[:100]:
            lines.append(
                "| {severity} | {issue_type} | {dataset} | {parent_image} | {tile_image} | {details} |".format(
                    severity=issue.get("severity", ""), issue_type=issue.get("issue_type", ""),
                    dataset=issue.get("dataset", ""), parent_image=issue.get("parent_image", ""),
                    tile_image=issue.get("tile_image", ""), details=str(issue.get("details", "")).replace("|", "/"),
                )
            )
        if len(issues) > 100:
            lines.append(f"\nOnly the first 100 findings are rendered here; see the JSON report for all {len(issues)}.")
    else:
        lines.append("No findings.")
    return "\n".join(lines) + "\n"


def write_module(output_root: Path, stem: str, title: str, summary: dict[str, Any], issues: list[dict[str, Any]], records: list[dict[str, Any]] | None = None) -> None:
    payload: dict[str, Any] = {"summary": summary, "issues": issues}
    if records is not None:
        payload["records"] = records
    write_json(output_root / f"{stem}.json", payload)
    (output_root / f"{stem}.md").write_text(markdown_summary(title, summary, issues), encoding="utf-8")


def resolve_parent_obb_path(path: Path) -> Path:
    path = repo_path(path)
    return path / "instances.jsonl" if path.is_dir() else path


def load_tile_rows(path: Path, dataset: str | None, split: str | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tile_path", "label_path", "dataset", "source_split", "split", "group_key", "parent_id", "tile_x", "tile_y", "tile_width", "tile_height"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Invalid tile metadata: {path}")
    return [row for row in rows if (dataset is None or row["dataset"] == dataset) and (split is None or row["split"] == split)]


def load_training_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["image_name"]: row for row in rows}


def make_tiles(rows: Iterable[dict[str, str]]) -> tuple[dict[str, Tile], dict[str, list[Tile]]]:
    by_key: dict[str, Tile] = {}
    by_parent: dict[str, list[Tile]] = defaultdict(list)
    for row in rows:
        key = Path(row["tile_path"]).name
        tile = Tile(key=key, row=row, width=parse_int(row["tile_width"]), height=parse_int(row["tile_height"]), x=parse_int(row["tile_x"]), y=parse_int(row["tile_y"]))
        by_key[key] = tile
        by_parent[tile.parent_key].append(tile)
    return by_key, by_parent


def parent_image_path(tile: Tile, parent_root: Path) -> Path:
    source = tile.source
    split = tile.source_split
    source_name = Path(tile.row.get("source_image", "")).name
    name = (
        tile.row.get("parent_name")
        or tile.row.get("source_image_name")
        or source_name
        or f"{tile.row['parent_id']}.png"
    )
    return parent_root / source / split / name


def parse_yolo_obb_line(line: str, tile: Tile, line_number: int) -> tuple[TileLabel | None, list[str]]:
    parts = line.split()
    errors: list[str] = []
    if len(parts) != 9:
        return None, ["wrong_field_count"]
    try:
        class_id = int(parts[0])
        values = tuple(float(value) for value in parts[1:])
    except ValueError:
        return None, ["non_numeric"]
    if class_id != 0:
        errors.append("invalid_class_id")
    if not all(math.isfinite(value) for value in values):
        errors.append("non_finite")
    if any(value < -EPSILON or value > 1.0 + EPSILON for value in values):
        errors.append("normalized_out_of_bounds")
    points = tuple((values[index] * tile.width, values[index + 1] * tile.height) for index in range(0, 8, 2))
    polygon = Polygon(points)
    if len(set(points)) != 4:
        errors.append("non_distinct_corners")
    if not polygon.is_valid:
        errors.append("self_intersection_or_invalid_polygon")
    if polygon.area <= EPSILON:
        errors.append("zero_area")
    return TileLabel(tile.key, line_number, class_id, values, points, polygon, line.rstrip()), errors


def load_tile_labels(training_root: Path, tiles: dict[str, Tile]) -> tuple[dict[str, list[TileLabel]], list[dict[str, Any]], set[str]]:
    index = load_training_rows(training_root / "dataset_index.csv")
    records: dict[str, list[TileLabel]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    referenced_labels: set[str] = set()
    for key, tile in tiles.items():
        row = index.get(key)
        if row is None:
            issues.append({"severity": "high", "issue_type": "tile_missing_from_training_index", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "details": "Tile map row has no v9 dataset-index row."})
            continue
        label = training_root / row["label_path"]
        referenced_labels.add(str(label.resolve(strict=False)))
        if not label.is_file():
            issues.append({"severity": "high", "issue_type": "missing_training_label", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "details": str(label)})
            continue
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parsed, errors = parse_yolo_obb_line(line, tile, line_number)
            if parsed is not None:
                records[key].append(parsed)
            for error in errors:
                issues.append({"severity": "high" if error in {"wrong_field_count", "non_numeric", "non_finite", "zero_area", "self_intersection_or_invalid_polygon"} else "medium", "issue_type": error, "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "line_number": line_number, "details": line})
    return records, issues, referenced_labels


def load_removed_labels(training_root: Path, tiles: dict[str, Tile]) -> dict[str, list[TileLabel]]:
    path = training_root / "removed_annotations.csv"
    result: dict[str, list[TileLabel]] = defaultdict(list)
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tile = tiles.get(row.get("tile_image", ""))
            if tile is None or not row.get("label_line"):
                continue
            parsed, _ = parse_yolo_obb_line(row["label_line"], tile, 0)
            if parsed is not None:
                result[tile.key].append(parsed)
    return result


def load_parent_obbs(path: Path, dataset: str | None, selected_split: str | None) -> list[ParentOBB]:
    records: list[ParentOBB] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            group_key = str(row["parent_group_key"])
            source, source_split, _ = group_key.split(":", 2)
            if dataset and source != dataset:
                continue
            # DOTA source split and output split are identical; xView/HRSC train
            # can be present in either output split after parent-level assignment.
            if selected_split == "val" and source in {"dota"} and source_split != "val":
                continue
            if selected_split == "train" and source == "dota" and source_split != "train":
                continue
            values = [float(value) for value in row["obb_cxcywhr"]]
            points = tuple((float(x), float(y)) for x, y in row["obb_vertices_xy"])
            if len(points) != 4 or values[2] <= 0 or values[3] <= 0:
                continue
            records.append(ParentOBB(
                instance_id=str(row["instance_id"]), group_key=group_key, dataset=source,
                source_split=source_split, parent_name=str(row["parent_image_filename"]), points=points,
                box=CVATBox(label="ship", xtl=values[0] - values[2] / 2, ytl=values[1] - values[3] / 2, xbr=values[0] + values[2] / 2, ybr=values[1] + values[3] / 2, rotation=values[4]),
                width=parse_int(row.get("image_width")), height=parse_int(row.get("image_height")), source_kind=str(row.get("obb_source", "unknown")),
            ))
    return records


def load_suppression_ledger(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"Suppression ledger must contain a list of entries: {path}")
    return {
        (str(entry["parent_annotation_id"]), str(entry["suppressed_tile"]))
        for entry in entries
        if isinstance(entry, dict) and entry.get("parent_annotation_id") and entry.get("suppressed_tile")
    }


def polygon_iou(first: Polygon, second: Polygon) -> float:
    if first.is_empty or second.is_empty:
        return 0.0
    union = first.union(second).area
    return first.intersection(second).area / union if union > EPSILON else 0.0


def intersection_over_smaller(first: Polygon, second: Polygon) -> float:
    smaller = min(first.area, second.area)
    return first.intersection(second).area / smaller if smaller > EPSILON else 0.0


def centroid_distance(first: Polygon, second: Polygon) -> float:
    return first.centroid.distance(second.centroid)


def labels_match(expected: Polygon, candidates: Iterable[TileLabel], centroid_threshold: float) -> tuple[TileLabel | None, dict[str, float]]:
    best: TileLabel | None = None
    best_metrics: dict[str, float] = {}
    for candidate in candidates:
        if candidate.polygon.area <= EPSILON:
            continue
        iou = polygon_iou(expected, candidate.polygon)
        ios = intersection_over_smaller(expected, candidate.polygon)
        distance = centroid_distance(expected, candidate.polygon)
        ratio = candidate.polygon.area / expected.area if expected.area > EPSILON else math.inf
        matched = (iou >= 0.5 or ios >= 0.75) and distance <= centroid_threshold and 0.4 <= ratio <= 2.5
        if matched and (best is None or iou > best_metrics["iou"]):
            best = candidate
            best_metrics = {"iou": round(iou, 6), "intersection_over_smaller": round(ios, 6), "centroid_distance": round(distance, 4), "area_ratio": round(ratio, 6)}
    return best, best_metrics


def parent_labels_match(expected: Polygon, candidates: Iterable[tuple[TileLabel, Polygon]] | ParentLabelIndex, centroid_threshold: float) -> tuple[TileLabel | None, dict[str, float]]:
    """Match a parent OBB against tile-local labels lifted back to parent space."""
    best: TileLabel | None = None
    best_metrics: dict[str, float] = {}
    expected_bounds = expected.bounds
    candidate_records = candidates.query(expected) if isinstance(candidates, ParentLabelIndex) else candidates
    for candidate, polygon in candidate_records:
        bounds = polygon.bounds
        if bounds[2] < expected_bounds[0] or expected_bounds[2] < bounds[0] or bounds[3] < expected_bounds[1] or expected_bounds[3] < bounds[1]:
            continue
        if polygon.area <= EPSILON:
            continue
        iou = polygon_iou(expected, polygon)
        ios = intersection_over_smaller(expected, polygon)
        distance = centroid_distance(expected, polygon)
        ratio = polygon.area / expected.area if expected.area > EPSILON else math.inf
        matched = (iou >= 0.5 or ios >= 0.75) and distance <= centroid_threshold and 0.4 <= ratio <= 2.5
        if matched and (best is None or iou > best_metrics["iou"]):
            best = candidate
            best_metrics = {"iou": round(iou, 6), "intersection_over_smaller": round(ios, 6), "centroid_distance": round(distance, 4), "area_ratio": round(ratio, 6)}
    return best, best_metrics


def labels_in_parent_coordinates(tiles: dict[str, Tile], labels: dict[str, list[TileLabel]]) -> dict[str, ParentLabelIndex]:
    lifted_records: dict[str, list[tuple[TileLabel, Polygon]]] = defaultdict(list)
    for key, records in labels.items():
        tile = tiles[key]
        lifted_records[key] = [(record, Polygon([(x + tile.x, y + tile.y) for x, y in record.points])) for record in records]
    return {
        key: ParentLabelIndex(
            records,
            STRtree([polygon for _, polygon in records]) if records else None,
            [record for record, _ in records],
            STRtree([Polygon(record.points) for record, _ in records]) if records else None,
        )
        for key, records in lifted_records.items()
    }


def fitted_projection_match(obb: ParentOBB, tile: Tile, candidates: Iterable[TileLabel] | ParentLabelIndex, centroid_threshold: float) -> tuple[TileLabel | None, dict[str, float]]:
    """Match against the exact rectangle-preserving projection used by assembly."""
    projection = project_box_to_tile(
        obb.box,
        tile_origin=(tile.x, tile.y),
        tile_width=tile.width,
        tile_height=tile.height,
        clockwise=True,
        min_retained_fraction=0.0,
    )
    if projection is None:
        return None, {}
    projected_polygon = Polygon(projection.points)
    local_candidates = candidates.query_local(projected_polygon) if isinstance(candidates, ParentLabelIndex) else candidates
    return labels_match(projected_polygon, local_candidates, centroid_threshold)


def grid_rectangles(width: int, height: int, grids: Iterable[int]) -> list[Polygon]:
    # Keyboard layout: 7 8 9 / 4 5 6 / 1 2 3.
    positions = {7: (0, 0), 8: (1, 0), 9: (2, 0), 4: (0, 1), 5: (1, 1), 6: (2, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2)}
    xs = [round(width * index / 3) for index in range(4)]
    ys = [round(height * index / 3) for index in range(4)]
    return [shapely_box(xs[col], ys[row], xs[col + 1], ys[row + 1]) for grid in grids if grid in positions for col, row in [positions[grid]]]


def load_roi_maps(root: Path) -> dict[tuple[str, str], dict[str, list[int]]]:
    paths = {
        ("dota", "train"): root / "dota_train_maritime_grids.json",
        ("dota", "val"): root / "dota_val_maritime_grids.json",
        ("xview", "train"): root / "xview_train_grids.json",
    }
    maps: dict[tuple[str, str], dict[str, list[int]]] = {}
    for key, path in paths.items():
        if not path.is_file():
            maps[key] = {}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.values() if isinstance(payload, dict) else payload
        mapping: dict[str, list[int]] = {}
        for item in items:
            if isinstance(item, dict) and item.get("image"):
                mapping[Path(str(item["image"])).name] = sorted({int(value) for value in item.get("grids", [])})
        maps[key] = mapping
    return maps


def inventory(tile_root: Path, training_root: Path, tiles: dict[str, Tile], labels: dict[str, list[TileLabel]], referenced_labels: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    Image.MAX_IMAGE_PIXELS = None
    issues: list[dict[str, Any]] = []
    dimensions: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_tile_counts: Counter[str] = Counter()
    annotation_counts: list[int] = []
    for tile in tiles.values():
        source_counts[tile.source] += 1
        split_counts[tile.output_split] += 1
        source_tile_counts[tile.tile_source] += 1
        annotation_counts.append(len(labels.get(tile.key, [])))
        image_path = tile_root / tile.row["tile_path"]
        if not image_path.is_file():
            issues.append({"severity": "high", "issue_type": "missing_tile_image", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": str(image_path)})
            continue
        try:
            with Image.open(image_path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                if image.width != tile.width or image.height != tile.height:
                    issues.append({"severity": "high", "issue_type": "tile_dimension_metadata_mismatch", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": f"metadata={tile.width}x{tile.height}, file={image.width}x{image.height}"})
        except Exception as error:  # Pillow raises several image-specific errors.
            issues.append({"severity": "high", "issue_type": "unreadable_tile_image", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": str(error)})
    # A filtered audit should only compare label files belonging to the selected
    # tile-map slice. Otherwise a DOTA-only run would call xView and HRSC labels
    # orphaned even though they are intentionally outside that invocation.
    training_index = load_training_rows(training_root / "dataset_index.csv")
    all_label_files = {
        str((training_root / row["label_path"]).resolve(strict=False))
        for key, row in training_index.items()
        if key in tiles and row.get("label_path")
    }
    orphan_labels = sorted(all_label_files - referenced_labels)
    for label in orphan_labels:
        issues.append({"severity": "medium", "issue_type": "orphan_training_label", "dataset": "", "parent_image": "", "tile_image": Path(label).name, "details": label})
    duplicate_names = [key for key, count in Counter(tile.key for tile in tiles.values()).items() if count > 1]
    for key in duplicate_names:
        issues.append({"severity": "high", "issue_type": "duplicate_tile_filename", "dataset": "", "parent_image": "", "tile_image": key, "details": "Duplicate tile filename in tile map."})
    positive = sum(count > 0 for count in annotation_counts)
    return {
        "tile_images": len(tiles), "training_label_files_referenced": len(referenced_labels), "images_with_labels": positive,
        "negative_images": len(tiles) - positive, "positive_negative_ratio": round(positive / max(1, len(tiles) - positive), 6),
        "orphan_labels": len(orphan_labels), "duplicate_image_stems": len(duplicate_names), "image_hashes": "not_computed_to_avoid hashing all source images",
        "source_counts": dict(sorted(source_counts.items())), "split_counts": dict(sorted(split_counts.items())),
        "tile_source_counts": dict(sorted(source_tile_counts.items())), "dimension_distribution": dict(sorted(dimensions.items())),
        "annotation_count": {"total": sum(annotation_counts), "min": min(annotation_counts, default=0), "max": max(annotation_counts, default=0), "median": sorted(annotation_counts)[len(annotation_counts) // 2] if annotation_counts else 0},
    }, issues


def validate_metadata(tiles: dict[str, Tile], parent_root: Path, parent_sizes: dict[str, tuple[int, int]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[int, int]]]:
    Image.MAX_IMAGE_PIXELS = None
    issues: list[dict[str, Any]] = []
    resolved_sizes: dict[str, tuple[int, int]] = dict(parent_sizes)
    seen: dict[tuple[str, int, int, int, int], str] = {}
    for tile in tiles.values():
        if tile.width <= 0 or tile.height <= 0 or tile.x < 0 or tile.y < 0:
            issues.append({"severity": "high", "issue_type": "impossible_tile_offset_or_size", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": f"{tile.x},{tile.y},{tile.width},{tile.height}"})
        identity = (tile.parent_key, tile.x, tile.y, tile.width, tile.height)
        if identity in seen:
            issues.append({"severity": "high", "issue_type": "duplicate_tile_record", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": f"Duplicates {seen[identity]}"})
        seen[identity] = tile.key
        parent_path = parent_image_path(tile, parent_root)
        if not parent_path.is_file():
            issues.append({"severity": "medium", "issue_type": "unresolved_parent_image_reference", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": str(parent_path)})
            continue
        if tile.parent_key not in resolved_sizes:
            issues.append({"severity": "low", "issue_type": "parent_dimensions_unavailable", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": "No parent OBB record provides image dimensions."})
            continue
        width, height = resolved_sizes[tile.parent_key]
        if tile.x + tile.width > width or tile.y + tile.height > height:
            issues.append({"severity": "high", "issue_type": "tile_bounds_outside_parent", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": f"tile=({tile.x},{tile.y},{tile.x + tile.width},{tile.y + tile.height}) parent={width}x{height}"})
    return {"tile_records": len(tiles), "resolved_parent_images": len(resolved_sizes), "issue_count": len(issues)}, issues, resolved_sizes


def label_geometry(labels: dict[str, list[TileLabel]], tiles: dict[str, Tile], duplicate_iou: float, edge_margin: float, small_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    issues: list[dict[str, Any]] = []
    dense: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tiny = boundary = total = 0
    for key, records in labels.items():
        tile = tiles[key]
        for record in records:
            total += 1
            min_x, min_y, max_x, max_y = record.polygon.bounds
            sides = sorted([math.dist(record.points[index], record.points[(index + 1) % 4]) for index in range(4)])
            touches = min_x <= edge_margin or min_y <= edge_margin or max_x >= tile.width - edge_margin or max_y >= tile.height - edge_margin
            if touches:
                boundary += 1
                issues.append({"severity": "low", "issue_type": "boundary_touching_obb", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "line_number": record.line_number, "details": f"bounds={tuple(round(value, 2) for value in record.polygon.bounds)}"})
            if sides and sides[0] < small_threshold:
                tiny += 1
                issues.append({"severity": "low", "issue_type": "tiny_obb", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "line_number": record.line_number, "details": f"minimum_side={sides[0]:.3f}"})
        if len(records) >= 2:
            if len(records) > 80:
                dense[key].append({"severity": "low", "issue_type": "dense_scene_pairwise_review_required", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "details": f"{len(records)} labels; exhaustive duplicate pairing intentionally skipped."})
                continue
            # Spatial hashing limits comparisons to nearby labels. A dense marina
            # may contain hundreds of ships, where all-pairs comparison is both
            # slow and uninformative for duplicate detection.
            bucket_size = 64
            buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
            bounds = [record.polygon.bounds for record in records]
            for index, (min_x, min_y, max_x, max_y) in enumerate(bounds):
                for grid_x in range(int(min_x // bucket_size), int(max_x // bucket_size) + 1):
                    for grid_y in range(int(min_y // bucket_size), int(max_y // bucket_size) + 1):
                        buckets[(grid_x, grid_y)].append(index)
            seen_pairs: set[tuple[int, int]] = set()
            for bucket in buckets.values():
                for offset, first_index in enumerate(bucket):
                    for second_index in bucket[offset + 1:]:
                        pair = (min(first_index, second_index), max(first_index, second_index))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        first, second = records[pair[0]], records[pair[1]]
                        first_bounds, second_bounds = bounds[pair[0]], bounds[pair[1]]
                        overlap = not (first_bounds[2] < second_bounds[0] or second_bounds[2] < first_bounds[0] or first_bounds[3] < second_bounds[1] or second_bounds[3] < first_bounds[1])
                        distance = centroid_distance(first.polygon, second.polygon)
                        if not overlap and distance > 8.0:
                            continue
                        iou = polygon_iou(first.polygon, second.polygon) if overlap else 0.0
                        if iou >= duplicate_iou:
                            issue = {"severity": "medium", "issue_type": "likely_duplicate_tile_label", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "details": f"lines={first.line_number},{second.line_number}; iou={iou:.4f}"}
                            issues.append(issue); dense[key].append(issue)
                        elif len(records) >= 10 and distance <= 8.0 and iou < 0.3:
                            issue = {"severity": "low", "issue_type": "nearby_distinct_dense_scene_labels", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": key, "details": f"lines={first.line_number},{second.line_number}; iou={iou:.4f}"}
                            dense[key].append(issue)
    return {"parsed_labels": total, "tiny_labels": tiny, "boundary_touching_labels": boundary, "issue_count": len(issues)}, issues, dense


def parent_coverage(
    parent_obbs: list[ParentOBB],
    tiles_by_parent: dict[str, list[Tile]],
    labels: dict[str, ParentLabelIndex],
    removed: dict[str, ParentLabelIndex],
    policy: dict[str, Any],
    args: argparse.Namespace,
    suppressions: set[tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    ship_overlap = float(policy.get("ship_overlap_threshold", 0.5))
    for obb in parent_obbs:
        parent_tiles = tiles_by_parent.get(obb.group_key, [])
        parent_polygon = obb.polygon
        parent_bounds = parent_polygon.bounds
        parent_area = parent_polygon.area
        expected: list[tuple[Tile, float]] = []
        edge_fragments = 0
        for tile in parent_tiles:
            if parent_bounds[2] < tile.x or parent_bounds[0] > tile.x + tile.width or parent_bounds[3] < tile.y or parent_bounds[1] > tile.y + tile.height:
                continue
            visible_fraction = parent_polygon.intersection(tile.rect).area / parent_area if parent_area > EPSILON else 0.0
            if visible_fraction < ship_overlap:
                continue
            if visible_fraction < args.min_valid_visible_fraction:
                edge_fragments += 1
                continue
            expected.append((tile, visible_fraction))
        matched: list[dict[str, Any]] = []
        filtered: list[str] = []
        missing: list[str] = []
        suppressed: list[str] = []
        for tile, retained in expected:
            match, metrics = parent_labels_match(parent_polygon, labels.get(tile.key, []), args.centroid_distance_threshold)
            if match is None:
                label_index = labels.get(tile.key)
                match, metrics = fitted_projection_match(obb, tile, label_index if label_index else [], args.centroid_distance_threshold)
            if match is not None:
                matched.append({"tile": tile.key, "tile_source": tile.tile_source, "retained_fraction": round(retained, 6), "label_line": match.line_number, **metrics})
                continue
            removed_match, _ = parent_labels_match(parent_polygon, removed.get(tile.key, []), args.centroid_distance_threshold)
            if removed_match is None:
                removed_index = removed.get(tile.key)
                removed_match, _ = fitted_projection_match(obb, tile, removed_index if removed_index else [], args.centroid_distance_threshold)
            if removed_match is not None:
                filtered.append(tile.key)
            elif (obb.instance_id, tile.key) in suppressions:
                suppressed.append(tile.key)
            else:
                missing.append(tile.key)
        if not parent_tiles:
            status = "uncovered_by_any_tile"; severity = "high"; detail = "No current tile maps to this parent."
        elif not expected:
            status = "covered_but_clipped_below_threshold" if edge_fragments else "uncovered_by_any_tile"
            severity = "medium"; detail = f"candidate_fragments={edge_fragments}"
        elif missing:
            status = "covered_but_missing_label"; severity = "high"; detail = f"missing_tiles={','.join(missing)}"
        elif not matched and filtered and not suppressed:
            status = "intentionally_filtered_training_label"; severity = "low"; detail = f"v9_filter_tiles={','.join(filtered)}"
        elif len(matched) + len(filtered) + len(suppressed) != len(expected):
            status = "ambiguous_match"; severity = "medium"; detail = "Expected projections were not all matched."
        else:
            sources = {item["tile_source"] for item in matched}
            if suppressed:
                status = "covered_with_intentional_edge_suppression"
            else:
                status = "covered_by_recovery_tile" if sources == {"annotation_recovery"} else "covered_by_roi_tile" if sources <= {"smart_grid", "strict_grid"} else "covered_and_labeled"
            severity = "none"; detail = ""
        row = {"instance_id": obb.instance_id, "dataset": obb.dataset, "source_split": obb.source_split, "parent_image": obb.parent_name, "parent_group_key": obb.group_key, "source_kind": obb.source_kind, "status": status, "expected_tile_count": len(expected), "matched_tile_count": len(matched), "filtered_tile_count": len(filtered), "suppressed_tile_count": len(suppressed), "edge_fragment_count": edge_fragments, "matched_tiles": matched, "missing_tiles": missing, "filtered_tiles": filtered, "suppressed_tiles": suppressed, "details": detail}
        records.append(row); status_counts[status] += 1
        if severity != "none":
            issues.append({"severity": severity, "issue_type": status, "dataset": obb.dataset, "parent_image": obb.parent_name, "tile_image": missing[0] if missing else (expected[0][0].key if expected else ""), "annotation_id": obb.instance_id, "details": detail, "coverage_record": row})
    return {"parent_obbs": len(parent_obbs), "status_counts": dict(sorted(status_counts.items())), "issue_count": len(issues), "projection_policy": {"ship_overlap_threshold": ship_overlap, "minimum_valid_visible_fraction": args.min_valid_visible_fraction, "full_containment_threshold": args.containment_threshold}}, issues, records


def roi_risk(parent_obbs: list[ParentOBB], tiles_by_parent: dict[str, list[Tile]], coverage_records: list[dict[str, Any]], parent_sizes: dict[str, tuple[int, int]], roi_maps: dict[tuple[str, str], dict[str, list[int]]], policy: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage = {row["instance_id"]: row for row in coverage_records}
    issues: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    strict = policy.get("strict_parent_overrides", {})
    for obb in parent_obbs:
        if obb.dataset not in {"dota", "xview"} or obb.group_key not in parent_sizes:
            continue
        grids = roi_maps.get((obb.dataset, obb.source_split), {}).get(obb.parent_name)
        if not grids:
            continue
        regions = grid_rectangles(*parent_sizes[obb.group_key], grids)
        selected = unary_union(regions)
        row = coverage.get(obb.instance_id, {})
        inside = selected.intersects(obb.polygon) and selected.intersection(obb.polygon).area > EPSILON
        center_inside = selected.contains(obb.polygon.centroid)
        if not center_inside and row.get("status") in {"uncovered_by_any_tile", "covered_but_clipped_below_threshold"}:
            parent_id = obb.group_key.split(":")[-1]
            intentional = parent_id in strict
            issue_type = "intentional_strict_roi_exclusion" if intentional else "possible_roi_grid_exclusion"
            issues.append({"severity": "low" if intentional else "medium", "issue_type": issue_type, "dataset": obb.dataset, "parent_image": obb.parent_name, "tile_image": "", "annotation_id": obb.instance_id, "details": f"selected_grids={grids}; parent_status={row.get('status', 'unknown')}"})
            counts[issue_type] += 1
        elif inside and not center_inside:
            counts["roi_boundary_object"] += 1
    for parent_tiles in tiles_by_parent.values():
        for tile in parent_tiles:
            if tile.source not in {"dota", "xview"} or tile.tile_source == "annotation_recovery":
                continue
            grids = roi_maps.get((tile.source, tile.source_split), {}).get(tile.row.get("parent_name") or tile.row.get("source_image_name", ""))
            if not grids or tile.parent_key not in parent_sizes:
                continue
            roi = unary_union(grid_rectangles(*parent_sizes[tile.parent_key], grids))
            coverage_fraction = tile.rect.intersection(roi).area / tile.rect.area if tile.rect.area else 0.0
            if coverage_fraction < float(policy.get("roi_min_overlap", 0.25)):
                issues.append({"severity": "medium", "issue_type": "tile_below_roi_threshold", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": f"roi_overlap={coverage_fraction:.6f}"})
                counts["tile_below_roi_threshold"] += 1
    return {"roi_mapped_parent_obbs": sum(1 for obb in parent_obbs if obb.dataset in {"dota", "xview"}), "risk_counts": dict(sorted(counts.items())), "issue_count": len(issues)}, issues


def recovery_effectiveness(
    tiles: dict[str, Tile],
    labels: dict[str, list[TileLabel]],
    metadata_path: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def covered_ids(row: dict[str, Any]) -> list[str]:
        values = row.get("covered_instance_ids", [])
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                values = []
        return [str(value) for value in values]

    issues: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    if metadata_path.is_file():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); metadata[row["image_name"]] = row
    plans: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    if plan_path.is_file():
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            xyxy = row.get("tile_xyxy_parent", [])
            if len(xyxy) == 4:
                x1, y1, x2, y2 = (int(value) for value in xyxy)
                plans[(row.get("parent_group_key", ""), x1, y1, x2 - x1, y2 - y1)] = row
    recovery = [tile for tile in tiles.values() if tile.tile_source == "annotation_recovery"]
    labeled = [tile for tile in recovery if labels.get(tile.key)]
    covered_instances: dict[str, list[Tile]] = defaultdict(list)
    planned_recovery_tiles = 0
    for tile in recovery:
        meta = metadata.get(tile.key)
        if meta is None:
            issues.append({"severity": "high", "issue_type": "recovery_tile_missing_metadata", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": "No recovery_tile_metadata entry."})
            continue
        for instance_id in covered_ids(meta):
            covered_instances[instance_id].append(tile)
        plan_key = (tile.parent_key, tile.x, tile.y, tile.width, tile.height)
        if plan_key not in plans:
            issues.append({"severity": "medium", "issue_type": "recovery_tile_missing_plan", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": "No recovery_tile_plan window matches this recovery tile."})
        else:
            planned_recovery_tiles += 1
        if not labels.get(tile.key):
            issues.append({"severity": "medium", "issue_type": "zero_label_recovery_tile", "dataset": tile.source, "parent_image": tile.row.get("parent_name", ""), "tile_image": tile.key, "details": f"reason={meta.get('recovery_reason', '')}"})
    duplicate_instance_coverage = 0
    for instance_id, instance_tiles in covered_instances.items():
        if len(instance_tiles) > 1:
            duplicate_instance_coverage += 1
            issues.append({"severity": "medium", "issue_type": "recovery_instance_covered_by_multiple_windows", "dataset": instance_tiles[0].source, "parent_image": instance_tiles[0].row.get("parent_name", ""), "tile_image": instance_tiles[0].key, "annotation_id": instance_id, "details": f"tiles={','.join(tile.key for tile in instance_tiles)}"})
    redundant_windows = 0
    recovery_by_parent: dict[str, list[Tile]] = defaultdict(list)
    for tile in recovery:
        recovery_by_parent[tile.parent_key].append(tile)
    for parent_tiles in recovery_by_parent.values():
        for index, first in enumerate(parent_tiles):
            first_ids = set(covered_ids(metadata.get(first.key, {})))
            for second in parent_tiles[index + 1:]:
                overlap = first.rect.intersection(second.rect).area
                union = first.rect.union(second.rect).area
                iou = overlap / union if union else 0.0
                second_ids = set(covered_ids(metadata.get(second.key, {})))
                if iou >= 0.70 and (first_ids <= second_ids or second_ids <= first_ids):
                    redundant_windows += 1
                    issues.append({"severity": "low", "issue_type": "redundant_recovery_window", "dataset": first.source, "parent_image": first.row.get("parent_name", ""), "tile_image": first.key, "details": f"other={second.key}; iou={iou:.4f}"})
    return {
        "recovery_tiles": len(recovery),
        "recovery_tiles_with_labels": len(labeled),
        "zero_label_recovery_tiles": len(recovery) - len(labeled),
        "metadata_records": len(metadata),
        "plan_records": len(plans),
        "planned_recovery_tiles": planned_recovery_tiles,
        "recovery_source_instances": len(covered_instances),
        "duplicate_instance_coverage": duplicate_instance_coverage,
        "redundant_recovery_windows": redundant_windows,
        "issue_count": len(issues),
    }, issues


def preview_parent_image(issue: dict[str, Any], tiles: dict[str, Tile], parent_obbs: dict[str, ParentOBB], parent_root: Path, roi_maps: dict[tuple[str, str], dict[str, list[int]]], destination: Path) -> bool:
    annotation_id = issue.get("annotation_id")
    obb = parent_obbs.get(annotation_id) if annotation_id else None
    tile = tiles.get(issue.get("tile_image", ""))
    if obb is None and tile is None:
        return False
    if tile is None:
        candidates = [item for item in tiles.values() if item.parent_key == obb.group_key]
        tile = candidates[0] if candidates else None
    if tile is None:
        return False
    image_path = parent_image_path(tile, parent_root)
    if not image_path.is_file():
        return False
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = min(1.0, 1400 / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    grids = roi_maps.get((tile.source, tile.source_split), {}).get(tile.row.get("parent_name") or tile.row.get("source_image_name", ""), [])
    for region in grid_rectangles(round(image.width / scale), round(image.height / scale), grids):
        coords = [(round(x * scale), round(y * scale)) for x, y in region.exterior.coords]
        draw.polygon(coords, fill=(48, 160, 76, 35))
    for index in (1, 2):
        x = round(image.width * index / 3); y = round(image.height * index / 3)
        draw.line((x, 0, x, image.height), fill=(180, 180, 180, 220), width=2)
        draw.line((0, y, image.width, y), fill=(180, 180, 180, 220), width=2)
    for candidate in (item for item in tiles.values() if item.parent_key == tile.parent_key):
        color = (255, 153, 0, 220) if candidate.tile_source == "annotation_recovery" else (52, 152, 219, 150)
        draw.rectangle((round(candidate.x * scale), round(candidate.y * scale), round((candidate.x + candidate.width) * scale), round((candidate.y + candidate.height) * scale)), outline=color, width=2)
    if obb is not None:
        points = [(round(x * scale), round(y * scale)) for x, y in obb.points]
        draw.line(points + [points[0]], fill=(235, 56, 56, 255), width=4)
    draw.rectangle((0, 0, image.width, 40), fill=(0, 0, 0, 185))
    draw.text((8, 10), f"{issue.get('issue_type', '')}: {tile.row.get('parent_name', '')}", fill="white", font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return True


def write_previews(output_root: Path, issues: list[dict[str, Any]], tiles: dict[str, Tile], parent_obbs: list[ParentOBB], parent_root: Path, roi_maps: dict[tuple[str, str], dict[str, list[int]]], preview_count: int) -> dict[str, str]:
    priority = {"high": 0, "medium": 1, "low": 2}
    def preview_bucket(issue: dict[str, Any]) -> str:
        issue_type = issue.get("issue_type", "")
        if issue_type == "covered_but_missing_label":
            return "missing_label"
        if issue_type in {"uncovered_by_any_tile", "covered_but_clipped_below_threshold"}:
            return "uncovered_or_clipped"
        if issue_type == "ambiguous_match":
            return "edge_or_ambiguous"
        if "roi" in issue_type:
            return "roi"
        if "recovery" in issue_type:
            return "recovery"
        if "dense" in issue_type or "duplicate" in issue_type:
            return "dense_scene"
        if issue_type in {"tiny_obb", "boundary_touching_obb", "invalid_class_id", "non_finite", "zero_area", "self_intersection_or_invalid_polygon"}:
            return "label_geometry"
        return "metadata_or_inventory"

    # Preserve severity ranking within each audit category, then select in a
    # round-robin sequence. This makes the static QC bundle useful for triage
    # instead of filling it entirely with one plentiful issue type.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        buckets[preview_bucket(issue)].append(issue)
    ordered_names = ["missing_label", "uncovered_or_clipped", "edge_or_ambiguous", "roi", "recovery", "dense_scene", "label_geometry", "metadata_or_inventory"]
    for values in buckets.values():
        values.sort(key=lambda row: (priority.get(row.get("severity", "low"), 9), row.get("issue_type", ""), row.get("parent_image", "")))
    selected: list[dict[str, Any]] = []
    while len(selected) < preview_count:
        added = False
        for name in ordered_names:
            if buckets[name] and len(selected) < preview_count:
                selected.append(buckets[name].pop(0))
                added = True
        if not added:
            break
    by_id = {obb.instance_id: obb for obb in parent_obbs}
    paths: dict[str, str] = {}
    for index, issue in enumerate(selected, 1):
        category = issue.get("issue_type", "other")
        name = f"{index:03d}_{issue.get('dataset', 'unknown')}__{Path(issue.get('parent_image', 'unknown')).stem}.png"
        destination = output_root / "qc_previews" / category / name
        if preview_parent_image(issue, tiles, by_id, parent_root, roi_maps, destination):
            paths[f"{category}:{index}"] = str(destination.relative_to(output_root))
            issue["preview_path"] = str(destination.relative_to(output_root))
    # A compact, inspectable contact sheet of every generated preview.
    thumbs = [output_root / value for value in paths.values()]
    if thumbs:
        cards: list[Image.Image] = []
        for path in thumbs:
            with Image.open(path) as image:
                image.thumbnail((360, 260)); cards.append(image.copy())
        page_width, page_height, columns = 4 * 360, 6 * 280, 4
        for page, start in enumerate(range(0, len(cards), 24), 1):
            sheet = Image.new("RGB", (page_width, page_height), "white")
            for offset, image in enumerate(cards[start:start + 24]):
                x = (offset % columns) * 360; y = (offset // columns) * 280
                sheet.paste(image, (x, y))
            target = output_root / "qc_previews" / "summary_contact_sheets" / f"page_{page:02d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True); sheet.save(target, quality=88)
    return paths


def main() -> int:
    args = parse_args()
    tile_root = repo_path(args.tile_root)
    parent_obb_path = resolve_parent_obb_path(args.parent_label_root)
    training_root = repo_path(args.training_dataset_root)
    parent_root = repo_path(args.parent_image_root)
    suppression_ledger = repo_path(args.suppression_ledger) if args.suppression_ledger else None
    roi_root = repo_path(args.roi_metadata_root)
    metadata_path = repo_path(args.tile_metadata) if args.tile_metadata else tile_root / "tile_parent_map.csv"
    output_root = repo_path(args.output_root)
    required = [tile_root / "dataset_index.csv", parent_obb_path, training_root / "dataset_index.csv", metadata_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audit inputs: " + ", ".join(missing))
    tile_rows = load_tile_rows(metadata_path, args.dataset, args.split)
    tiles, tiles_by_parent = make_tiles(tile_rows)
    parent_obbs = load_parent_obbs(parent_obb_path, args.dataset, args.split)
    suppressions = load_suppression_ledger(suppression_ledger)
    policy = json.loads((tile_root / "grid_tile_policy.json").read_text(encoding="utf-8"))
    roi_maps = load_roi_maps(roi_root)
    dry = {"tile_records": len(tiles), "parent_obbs": len(parent_obbs), "tile_metadata": str(metadata_path), "parent_obb_source": str(parent_obb_path), "training_dataset": str(training_root), "suppression_ledger": str(suppression_ledger) if suppression_ledger else None, "suppression_entries": len(suppressions), "model_false_negative_audit": "deferred"}
    if args.dry_run:
        print(json.dumps({"dry_run": dry}, indent=2, sort_keys=True)); return 0
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Audit output exists: {output_root}; use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    started = time.monotonic()
    def stage(message: str) -> None:
        print(f"[audit +{time.monotonic() - started:.1f}s] {message}", flush=True)

    stage("loading tile-local labels")
    labels, label_parse_issues, referenced_labels = load_tile_labels(training_root, tiles)
    removed = load_removed_labels(training_root, tiles)
    lifted_labels = labels_in_parent_coordinates(tiles, labels)
    lifted_removed = labels_in_parent_coordinates(tiles, removed)
    source_parent_sizes = {obb.group_key: (obb.width, obb.height) for obb in parent_obbs if obb.width > 0 and obb.height > 0}
    stage("inventory and tile metadata validation")
    inv_summary, inv_issues = inventory(tile_root, training_root, tiles, labels, referenced_labels)
    meta_summary, meta_issues, parent_sizes = validate_metadata(tiles, parent_root, source_parent_sizes)
    stage("label geometry validation")
    geo_summary, geo_issues, dense = label_geometry(labels, tiles, args.duplicate_iou_threshold, args.edge_margin_px, args.small_object_threshold_px)
    stage("parent-to-tile coverage")
    coverage_summary, coverage_issues, coverage_records = parent_coverage(parent_obbs, tiles_by_parent, lifted_labels, lifted_removed, policy, args, suppressions)
    stage("ROI and recovery checks")
    roi_summary, roi_issues = roi_risk(parent_obbs, tiles_by_parent, coverage_records, parent_sizes, roi_maps, policy, args)
    recovery_summary, recovery_issues = recovery_effectiveness(
        tiles,
        labels,
        tile_root / "recovery_tile_metadata.jsonl",
        tile_root / "recovery_tile_plan.jsonl",
    )
    dense_issues = [issue for values in dense.values() for issue in values]
    write_module(output_root, "inventory", "Tile And Training-Label Inventory", inv_summary, inv_issues)
    write_module(output_root, "tile_metadata_validation", "Tile Metadata Validation", meta_summary, meta_issues)
    write_module(output_root, "label_geometry_validation", "Tile-Local OBB Geometry Validation", geo_summary, label_parse_issues + geo_issues)
    write_module(output_root, "parent_annotation_coverage", "Parent Annotation Coverage", coverage_summary, coverage_issues, coverage_records)
    write_module(output_root, "roi_grid_exclusion_risk", "ROI Grid Exclusion Risk", roi_summary, roi_issues)
    write_module(output_root, "recovery_tile_effectiveness", "Annotation Recovery Tile Effectiveness", recovery_summary, recovery_issues)
    write_module(output_root, "duplicate_dense_scene_risk", "Duplicate And Dense-Scene Risk", {"issue_count": len(dense_issues), "known_limitation": "Absent pre-suppression candidates prevent confirmation of removed adjacent ships."}, dense_issues)
    all_issues = inv_issues + meta_issues + label_parse_issues + geo_issues + coverage_issues + roi_issues + recovery_issues + dense_issues
    stage("rendering ranked previews")
    paths = write_previews(output_root, all_issues, tiles, parent_obbs, parent_root, roi_maps, args.preview_count)
    severity_counts = Counter(issue.get("severity", "low") for issue in all_issues)
    confirmed_high = [issue for issue in all_issues if issue.get("severity") == "high"]
    readiness = "fix_before_more_training" if confirmed_high else "safe_with_minor_cautions" if all_issues else "safe_to_continue"
    summary = {"inputs": dry, "thresholds": {"edge_margin_px": args.edge_margin_px, "min_valid_visible_fraction": args.min_valid_visible_fraction, "duplicate_iou_threshold": args.duplicate_iou_threshold, "containment_threshold": args.containment_threshold, "centroid_distance_threshold": args.centroid_distance_threshold, "small_object_threshold_px": args.small_object_threshold_px}, "inventory": inv_summary, "coverage": coverage_summary, "roi": roi_summary, "recovery": recovery_summary, "severity_counts": dict(sorted(severity_counts.items())), "model_false_negative_audit": {"status": "deferred", "requested": args.run_model_fn_audit, "reason": "Model inference is intentionally out of scope for the initial structural audit."}, "readiness": readiness, "preview_count": len(paths)}
    write_json(output_root / "audit_summary.json", summary)
    queue = sorted([issue for issue in all_issues if issue.get("severity") in {"high", "medium"}], key=lambda issue: ({"high": 0, "medium": 1}.get(issue.get("severity"), 2), issue.get("issue_type", "")))
    queue_rows = [{"priority": index, "issue_type": issue.get("issue_type", ""), "dataset": issue.get("dataset", ""), "split": "", "parent_image": issue.get("parent_image", ""), "tile_image": issue.get("tile_image", ""), "annotation_id": issue.get("annotation_id", ""), "severity": issue.get("severity", ""), "metrics": issue.get("details", ""), "preview_path": issue.get("preview_path", ""), "recommended_action": "inspect_and_plan_fix"} for index, issue in enumerate(queue, 1)]
    fields = ["priority", "issue_type", "dataset", "split", "parent_image", "tile_image", "annotation_id", "severity", "metrics", "preview_path", "recommended_action"]
    write_csv(output_root / "human_review_queue.csv", queue_rows, fields)
    (output_root / "human_review_queue.md").write_text(markdown_summary("Human Review Queue", {"queue_entries": len(queue_rows)}, queue_rows), encoding="utf-8")
    fix_plan = [{"issue_type": issue_type, "count": count, "action": "inspect evidence; do not mutate dataset automatically"} for issue_type, count in Counter(item["issue_type"] for item in queue).items()]
    write_json(output_root / "proposed_fixes.json", fix_plan)
    (output_root / "proposed_fixes.md").write_text("# Proposed Fixes\n\n" + "\n".join(f"- `{row['issue_type']}` ({row['count']}): {row['action']}" for row in fix_plan) + "\n", encoding="utf-8")
    top = ["# Tile Coverage And Recovery Audit", "", f"- Readiness: **{readiness}**", f"- Tile images audited: {len(tiles)}", f"- Parent OBBs audited: {len(parent_obbs)}", f"- Training labels audited: {sum(len(items) for items in labels.values())}", f"- High-severity findings: {severity_counts['high']}", f"- Medium-severity findings: {severity_counts['medium']}", f"- Recovery tiles: {recovery_summary['recovery_tiles']} ({recovery_summary['recovery_tiles_with_labels']} with labels)", f"- Model false-negative discovery: deferred", "", "See module reports, `human_review_queue.csv`, `proposed_fixes.md`, and `qc_previews/`."]
    (output_root / "tile_coverage_recovery_audit.md").write_text("\n".join(top) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "readiness": readiness, "high_severity": severity_counts["high"], "medium_severity": severity_counts["medium"]}, indent=2))
    return 2 if args.strict and confirmed_high else 0


if __name__ == "__main__":
    raise SystemExit(main())
