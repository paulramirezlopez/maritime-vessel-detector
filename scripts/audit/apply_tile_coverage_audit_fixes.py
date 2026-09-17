#!/usr/bin/env python3
"""Create a versioned, audit-repaired successor of the grid-recovery v9 view.

This utility is intentionally additive: it never alters the source tiled tree,
the parent-annotation source, or the existing v9 training dataset.  Unchanged
images are linked into the new tiled dataset; only new recovery windows are
materialized as PNG crops.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

if __package__:
    from .audit_tile_coverage_recovery import ParentOBB, load_parent_obbs
else:
    from audit.audit_tile_coverage_recovery import ParentOBB, load_parent_obbs
from roi_smart_retile_utils import project_box_to_tile, projection_to_yolo_line


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_TILE_ROOT = REPO_ROOT / "data/tiled/grid_current_with_recovery"
SOURCE_PARENT_ROOT = REPO_ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery"
SOURCE_TRAINING_ROOT = REPO_ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3"
SOURCE_AUDIT_ROOT = REPO_ROOT / "reports/tile_coverage_recovery_audit"
ACTIVE_PARENT_ROOT = REPO_ROOT / "data/parent_images/active"
QA_MANIFEST = REPO_ROOT / "data/metadata/qa/parent_image_quality_manifest.csv"

DEFAULT_TILE_ROOT = REPO_ROOT / "data/tiled/grid_current_with_recovery_audit_fixed_v1"
DEFAULT_PARENT_ROOT = REPO_ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery_audit_fixed_v1"
DEFAULT_TRAINING_ROOT = REPO_ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3_audit_fixed_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports/tile_coverage_recovery_audit_fix_v1"

EXCLUDED_PARENT_IDS = {"P1277", "P1591", "P1927", "P2662", "P2769"}
EXCLUSION_REASON = "intentionally_removed_low_gsd_or_low_visual_resolvability"
TILE_SIZE = 1024
SHIP_OVERLAP_THRESHOLD = 0.5
MIN_RETAINED_FRACTION = 0.25
FULL_RETAINED_FRACTION = 0.95
DUPLICATE_LABEL_IOU = 0.85


@dataclass
class TileRow:
    row: dict[str, str]

    @property
    def image_name(self) -> str:
        return self.row["image_name"]

    @property
    def parent_id(self) -> str:
        return self.row["parent_id"]

    @property
    def group_key(self) -> str:
        return self.row["group_key"]

    @property
    def split(self) -> str:
        return self.row["split"]

    @property
    def dataset(self) -> str:
        return self.row["dataset"].lower()

    @property
    def source_split(self) -> str:
        return self.row["source_split"]

    @property
    def x(self) -> int:
        return int(float(self.row["tile_x"]))

    @property
    def y(self) -> int:
        return int(float(self.row["tile_y"]))

    @property
    def width(self) -> int:
        return int(float(self.row["tile_width"]))

    @property
    def height(self) -> int:
        return int(float(self.row["tile_height"]))

    @property
    def source_kind(self) -> str:
        return self.row.get("tile_source") or "smart_grid"

    @property
    def window_key(self) -> tuple[str, int, int, int, int]:
        return (self.group_key, self.x, self.y, self.width, self.height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tile-root", type=Path, default=SOURCE_TILE_ROOT)
    parser.add_argument("--source-parent-root", type=Path, default=SOURCE_PARENT_ROOT)
    parser.add_argument("--source-training-root", type=Path, default=SOURCE_TRAINING_ROOT)
    parser.add_argument("--source-audit-root", type=Path, default=SOURCE_AUDIT_ROOT)
    parser.add_argument("--active-parent-root", type=Path, default=ACTIVE_PARENT_ROOT)
    parser.add_argument("--qa-manifest", type=Path, default=QA_MANIFEST)
    parser.add_argument("--output-tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    parser.add_argument("--output-parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--replace", action="store_true", help="Replace only pre-existing patch output roots.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and classify changes without writing outputs.")
    parser.add_argument("--skip-audit", action="store_true", help="Do not invoke the post-patch audit.")
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def relative_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_id_from_group(group_key: str) -> str:
    return group_key.rsplit(":", 1)[-1]


def active_parent_path(tile: TileRow, root: Path) -> Path:
    name = tile.row.get("parent_name") or tile.row.get("source_image_name") or f"{tile.parent_id}.png"
    return root / tile.dataset / tile.source_split / name


def tile_image_relative_path(tile: TileRow) -> Path:
    """Accept both the historical ``image_path`` and newer ``tile_path`` schemas."""
    value = tile.row.get("tile_path") or tile.row.get("image_path")
    if value:
        return Path(value)
    return Path("images") / tile.split / tile.image_name


def parse_parent_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, ParentOBB]]:
    raw_rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_rows.append(json.loads(line))
    # Do not filter the canonical source here. The patcher needs the complete
    # parent-space inventory before applying its explicit exclusion ledger.
    obbs = {item.instance_id: item for item in load_parent_obbs(path, None, None)}
    return raw_rows, obbs


def load_qa_tags(path: Path) -> dict[str, str]:
    rows, _ = read_csv(path)
    return {
        row["parent_image_id"].strip(): row["qa_tag"].strip().lower()
        for row in rows
        if row.get("parent_image_id") and row.get("qa_tag", "").strip().lower() in {"small", "low", "bad"}
    }


def load_issues(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("issues", []))


def parse_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def tile_polygon_from_line(line: str, tile: TileRow) -> Polygon | None:
    parts = line.split()
    if len(parts) != 9:
        return None
    try:
        points = [(float(parts[index]) * tile.width, float(parts[index + 1]) * tile.height) for index in range(1, 9, 2)]
    except ValueError:
        return None
    polygon = Polygon(points)
    return polygon if polygon.is_valid and polygon.area > 1e-6 else None


def polygon_iou(first: Polygon, second: Polygon) -> float:
    intersection = first.intersection(second).area
    if intersection <= 0.0:
        return 0.0
    union = first.union(second).area
    return intersection / union if union else 0.0


def canonical_tile_sort_key(tile: TileRow) -> tuple[int, str]:
    priority = {"smart_grid": 0, "strict_grid": 1, "annotation_recovery": 2, "full_parent_small_image": 3}
    return (priority.get(tile.source_kind, 9), tile.image_name)


def choose_canonical_tile(rows: list[TileRow]) -> TileRow:
    splits = {tile.split for tile in rows}
    if len(splits) != 1:
        raise ValueError(f"Duplicate tile window crosses splits: {[tile.image_name for tile in rows]}")
    return min(rows, key=canonical_tile_sort_key)


def deduplicate_lines(lines: list[str], tile: TileRow, threshold: float = DUPLICATE_LABEL_IOU) -> tuple[list[str], list[dict[str, Any]]]:
    # The normal v9 view has already been through label QA. Outside the three
    # audit-identified high-IoU pairs we only need deterministic exact-line
    # removal while folding duplicate tile windows; an all-pairs Shapely scan
    # turns dense marina labels into unnecessary O(n^2) work.
    if threshold >= 0.999999:
        kept: list[str] = []
        seen: set[str] = set()
        removed: list[dict[str, Any]] = []
        for line in lines:
            normalized = " ".join(line.split())
            if normalized in seen:
                removed.append({"reason": "exact_duplicate_label", "removed_line": line})
                continue
            seen.add(normalized)
            kept.append(line)
        return kept, removed
    kept: list[str] = []
    kept_polygons: list[Polygon] = []
    removed: list[dict[str, Any]] = []
    for line in lines:
        polygon = tile_polygon_from_line(line, tile)
        if polygon is None:
            removed.append({"reason": "invalid_label_geometry", "removed_line": line})
            continue
        duplicate_index = next((index for index, previous in enumerate(kept_polygons) if polygon_iou(polygon, previous) >= threshold), None)
        if duplicate_index is None:
            kept.append(line)
            kept_polygons.append(polygon)
            continue
        # Prefer the larger valid polygon; it is less likely to be a degraded edge fit.
        previous = kept_polygons[duplicate_index]
        if polygon.area > previous.area:
            removed.append({"reason": "high_iou_duplicate_replaced", "removed_line": kept[duplicate_index], "retained_line": line, "iou": polygon_iou(polygon, previous)})
            kept[duplicate_index] = line
            kept_polygons[duplicate_index] = polygon
        else:
            removed.append({"reason": "high_iou_duplicate", "removed_line": line, "retained_line": kept[duplicate_index], "iou": polygon_iou(polygon, previous)})
    return kept, removed


def projection_for(obb: ParentOBB, tile: TileRow):
    return project_box_to_tile(
        obb.box,
        tile_origin=(tile.x, tile.y),
        tile_width=tile.width,
        tile_height=tile.height,
        clockwise=True,
        min_retained_fraction=0.0,
    )


def v9_filter_reason(line: str, tile: TileRow, qa_tag: str) -> str | None:
    polygon = tile_polygon_from_line(line, tile)
    if polygon is None:
        return "invalid_projection"
    coordinates = list(polygon.exterior.coords)[:-1]
    sides = [math.dist(coordinates[index], coordinates[(index + 1) % 4]) for index in range(4)]
    short_side = min(sides)
    area = polygon.area
    thresholds: list[tuple[float, float, str]] = []
    if tile.dataset == "xview":
        thresholds.append((36.0, 5.0, "xview_aggressive"))
    if qa_tag == "small":
        thresholds.append((36.0, 5.0, "aggressive"))
    elif qa_tag == "low":
        thresholds.append((64.0, 6.0, "low_aggressive"))
    if not thresholds:
        return None
    area_threshold = max(item[0] for item in thresholds)
    side_threshold = max(item[1] for item in thresholds)
    if area < area_threshold or short_side < side_threshold:
        parts = []
        if area < area_threshold:
            parts.append("area")
        if short_side < side_threshold:
            parts.append("short_side")
        return f"threshold_{'_and_'.join(parts)}:{'+'.join(item[2] for item in thresholds)}"
    return None


def candidate_origins(obb: ParentOBB, parent_width: int, parent_height: int, tile_width: int, tile_height: int) -> list[tuple[int, int]]:
    points = obb.polygon.exterior.coords[:-1]
    min_x, min_y, max_x, max_y = obb.polygon.bounds
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    seeds = [
        (center_x - tile_width / 2.0, center_y - tile_height / 2.0),
        (min_x, min_y), (max_x - tile_width, min_y),
        (min_x, max_y - tile_height), (max_x - tile_width, max_y - tile_height),
    ]
    max_x = max(parent_width - tile_width, 0)
    max_y = max(parent_height - tile_height, 0)
    origins = {
        (min(max(round(x), 0), max_x), min(max(round(y), 0), max_y))
        for x, y in seeds
    }
    return sorted(origins)


def choose_recovery_origin(obb: ParentOBB, parent_width: int, parent_height: int) -> tuple[int, int, int, int] | None:
    tile_width = min(TILE_SIZE, parent_width)
    tile_height = min(TILE_SIZE, parent_height)
    best: tuple[float, int, int] | None = None
    for origin_x, origin_y in candidate_origins(obb, parent_width, parent_height, tile_width, tile_height):
        candidate = TileRow({"image_name": "candidate", "parent_id": "", "group_key": obb.group_key, "split": "", "dataset": obb.dataset, "source_split": obb.source_split, "tile_x": str(origin_x), "tile_y": str(origin_y), "tile_width": str(tile_width), "tile_height": str(tile_height)})
        projection = projection_for(obb, candidate)
        score = projection.retained_fraction if projection is not None else -1.0
        # Highest retained fraction wins. The top-left tie-break makes this
        # deterministic without generating a search fan-out.
        candidate_score = (score, -origin_y, -origin_x)
        if best is None or candidate_score > best:
            best = candidate_score
    if best is None or best[0] < MIN_RETAINED_FRACTION:
        return None
    return -best[2], -best[1], tile_width, tile_height


def image_name_for_recovery(tile: TileRow) -> str:
    prefix = "dota" if tile.dataset == "dota" else tile.dataset
    parent = Path(tile.row.get("parent_name") or tile.parent_id).stem
    return f"{prefix}__{parent}_{tile.x:04d}_{tile.y:04d}_recovery_audit_v1.png"


def source_tile_row_for_recovery(obb: ParentOBB, parent_tile: TileRow, x: int, y: int, width: int, height: int) -> TileRow:
    row = dict(parent_tile.row)
    row.update({
        "image_name": "",
        "tile_path": "",
        "label_path": "",
        "tile_x": str(x), "tile_y": str(y), "tile_width": str(width), "tile_height": str(height),
        "roi_mode": "annotation_recovery", "selected_grids": "", "tile_source": "annotation_recovery",
        "recovery_reason": "audit_coverage_repair", "covered_instance_ids": json.dumps([obb.instance_id]),
        "covered_obb_count": "1", "tile_xyxy_parent": json.dumps([x, y, x + width, y + height]),
        "outside_manual_grid": "True", "source_grid_cells": "[]", "coverage_stats": "",
        "empty": "False", "second_pass_annotation_count": "0", "label_source": "second_pass_parent_obbs_audit_fixed_v1",
    })
    tile = TileRow(row)
    row["image_name"] = image_name_for_recovery(tile)
    row["tile_path"] = str(Path("images") / tile.split / row["image_name"])
    row["label_path"] = str(Path("labels") / tile.split / f"{Path(row['image_name']).stem}.txt")
    row["image_path"] = row["tile_path"]
    return TileRow(row)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Tile Coverage Audit Fix v1", "", "## Summary", ""]
    for key, value in summary["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Policy", "", "- Source roots remain untouched.", "- The five approved low-GSD parents are excluded only in this versioned successor.", "- New labels use rectangle-preserving parent-to-tile projection and v9 aggressive threshold filtering.", "- Ambiguous edge fragments are preserved as suppression decisions when a clean tile already exists."])
    return "\n".join(lines) + "\n"


def write_before_after_summary(before_root: Path, after_root: Path, destination: Path) -> None:
    """Write a compact, reviewable delta after the post-patch audit finishes."""
    before = json.loads((before_root / "audit_summary.json").read_text(encoding="utf-8"))
    after = json.loads((after_root / "audit_summary.json").read_text(encoding="utf-8"))
    statuses = sorted(set(before["coverage"]["status_counts"]) | set(after["coverage"]["status_counts"]))
    payload = {
        "before_audit": str(before_root),
        "after_audit": str(after_root),
        "tiles": {"before": before["inventory"]["tile_images"], "after": after["inventory"]["tile_images"]},
        "labels": {"before": before["inventory"]["annotation_count"]["total"], "after": after["inventory"]["annotation_count"]["total"]},
        "coverage_status_counts": {
            status: {"before": before["coverage"]["status_counts"].get(status, 0), "after": after["coverage"]["status_counts"].get(status, 0)}
            for status in statuses
        },
        "recovery": {
            "before": before["recovery"],
            "after": after["recovery"],
        },
        "severity": {"before": before["severity_counts"], "after": after["severity_counts"]},
    }
    write_json(destination / "before_after_summary.json", payload)
    lines = ["# Tile Coverage Audit Fix v1: Before / After", "", "| Measure | Before | After |", "| --- | ---: | ---: |"]
    lines.extend([
        f"| Tiles | {payload['tiles']['before']} | {payload['tiles']['after']} |",
        f"| Tile-local labels | {payload['labels']['before']} | {payload['labels']['after']} |",
        f"| Zero-label recovery tiles | {payload['recovery']['before']['zero_label_recovery_tiles']} | {payload['recovery']['after']['zero_label_recovery_tiles']} |",
    ])
    for status, values in payload["coverage_status_counts"].items():
        lines.append(f"| `{status}` | {values['before']} | {values['after']} |")
    (destination / "before_after_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repair_previews(
    destination: Path,
    decisions: list[dict[str, Any]],
    parent_obbs: dict[str, ParentOBB],
    before_tiles: dict[str, list[TileRow]],
    after_tiles: dict[str, list[TileRow]],
    active_parent_root: Path,
) -> list[dict[str, str]]:
    """Render compact parent-space before/after evidence for actual repairs."""
    selected: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if decision.get("action") != "added_projection":
            continue
        obb = parent_obbs.get(str(decision["instance_id"]))
        if obb is not None:
            selected.setdefault(obb.group_key, decision)
    previews: list[dict[str, str]] = []
    Image.MAX_IMAGE_PIXELS = None
    for group_key, decision in sorted(selected.items())[:50]:
        obb = parent_obbs[str(decision["instance_id"])]
        rows = after_tiles.get(group_key, [])
        if not rows:
            continue
        sample = rows[0]
        path = active_parent_path(sample, active_parent_root)
        if not path.is_file():
            continue
        with Image.open(path) as source:
            source = source.convert("RGB")
            scale = min(1.0, 1200.0 / max(source.size))
            size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
            source = source.resize(size)

        def panel(rows_to_draw: list[TileRow], patched: bool) -> Image.Image:
            canvas = source.copy()
            draw = ImageDraw.Draw(canvas)
            for tile in rows_to_draw:
                color = "#2369c8"
                if patched and tile.image_name == decision.get("tile_image"):
                    color = "#20a35c"
                draw.rectangle((tile.x * scale, tile.y * scale, (tile.x + tile.width) * scale, (tile.y + tile.height) * scale), outline=color, width=3)
            draw.line([(x * scale, y * scale) for x, y in obb.points] + [(obb.points[0][0] * scale, obb.points[0][1] * scale)], fill="#e53935", width=3)
            draw.rectangle((0, 0, min(canvas.width, 520), 28), fill="white")
            draw.text((6, 6), "patched" if patched else "before", fill="black")
            return canvas

        left = panel(before_tiles.get(group_key, []), False)
        right = panel(after_tiles.get(group_key, []), True)
        combined = Image.new("RGB", (left.width + right.width, left.height), "white")
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        filename = f"{group_key.replace(':', '__')}__{decision['instance_id'].replace(':', '_')}.jpg"
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        combined.save(target, quality=88)
        previews.append({"parent_group_key": group_key, "instance_id": str(decision["instance_id"]), "tile_image": str(decision.get("tile_image", "")), "preview_path": str(target)})
    return previews


def main() -> int:
    args = parse_args()
    source_tile_root = repo_path(args.source_tile_root)
    source_parent_root = repo_path(args.source_parent_root)
    source_training_root = repo_path(args.source_training_root)
    source_audit_root = repo_path(args.source_audit_root)
    active_parent_root = repo_path(args.active_parent_root)
    qa_manifest = repo_path(args.qa_manifest)
    output_tile_root = repo_path(args.output_tile_root)
    output_parent_root = repo_path(args.output_parent_root)
    output_training_root = repo_path(args.output_training_root)
    report_root = repo_path(args.report_root)
    parent_jsonl = source_parent_root / "instances.jsonl"

    required = [
        source_tile_root / "dataset_index.csv", source_tile_root / "tile_parent_map.csv",
        source_training_root / "dataset_index.csv", source_audit_root / "parent_annotation_coverage.json",
        source_audit_root / "tile_metadata_validation.json", parent_jsonl, qa_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing patch inputs: " + ", ".join(missing))
    destinations = [output_tile_root, output_parent_root, output_training_root, report_root]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.replace and not args.dry_run:
        raise FileExistsError("Patch destinations already exist: " + ", ".join(str(path) for path in existing))

    tile_rows_raw, tile_fields = read_csv(source_tile_root / "dataset_index.csv")
    training_rows_raw, training_fields = read_csv(source_training_root / "dataset_index.csv")
    tile_map_rows, tile_map_fields = read_csv(source_tile_root / "tile_parent_map.csv")
    tile_rows = {row["image_name"]: TileRow(row) for row in tile_rows_raw}
    training_rows = {row["image_name"]: dict(row) for row in training_rows_raw}
    qa_tags = load_qa_tags(qa_manifest)
    raw_parent_rows, parent_obbs = parse_parent_records(parent_jsonl)
    parent_obbs_by_group: dict[str, list[ParentOBB]] = defaultdict(list)
    for obb in parent_obbs.values():
        parent_obbs_by_group[obb.group_key].append(obb)

    coverage_issues = load_issues(source_audit_root / "parent_annotation_coverage.json")
    roi_issues = load_issues(source_audit_root / "roi_grid_exclusion_risk.json")
    metadata_issues = load_issues(source_audit_root / "tile_metadata_validation.json")
    geometry_issues = load_issues(source_audit_root / "label_geometry_validation.json")
    recovery_issues = load_issues(source_audit_root / "recovery_tile_effectiveness.json")

    duplicate_window_names = {issue["tile_image"] for issue in metadata_issues if issue["issue_type"] == "duplicate_tile_record"}
    duplicate_window_names.update(
        issue["details"].replace("Duplicates ", "")
        for issue in metadata_issues if issue["issue_type"] == "duplicate_tile_record" and issue.get("details", "").startswith("Duplicates ")
    )
    duplicate_label_tiles = {issue["tile_image"] for issue in geometry_issues if issue["issue_type"] == "likely_duplicate_tile_label"}
    zero_recovery_tiles = {issue["tile_image"] for issue in recovery_issues if issue["issue_type"] == "zero_label_recovery_tile"}

    coverage_by_instance = {issue["annotation_id"]: issue for issue in coverage_issues if issue.get("annotation_id")}
    repair_instances = {
        issue["annotation_id"]
        for issue in coverage_issues
        if issue["issue_type"] in {"covered_but_missing_label", "uncovered_by_any_tile"} and issue.get("annotation_id")
    }
    repair_instances.update(issue["annotation_id"] for issue in roi_issues if issue["issue_type"] == "possible_roi_grid_exclusion" and issue.get("annotation_id"))

    rows_by_window: dict[tuple[str, int, int, int, int], list[TileRow]] = defaultdict(list)
    for tile in tile_rows.values():
        if tile.parent_id not in EXCLUDED_PARENT_IDS:
            rows_by_window[tile.window_key].append(tile)
    canonical_by_name: dict[str, TileRow] = {}
    removed_duplicate_names: set[str] = set()
    duplicate_ledger: list[dict[str, Any]] = []
    for window, candidates in rows_by_window.items():
        canonical = choose_canonical_tile(candidates)
        canonical_by_name[canonical.image_name] = canonical
        for candidate in candidates:
            if candidate.image_name != canonical.image_name:
                removed_duplicate_names.add(candidate.image_name)
                duplicate_ledger.append({"action": "removed_duplicate_tile_window", "canonical_tile": canonical.image_name, "removed_tile": candidate.image_name, "window": window})

    excluded_tile_names = {tile.image_name for tile in tile_rows.values() if tile.parent_id in EXCLUDED_PARENT_IDS}
    kept_tiles: dict[str, TileRow] = dict(canonical_by_name)
    parent_tiles: dict[str, list[TileRow]] = defaultdict(list)
    for tile in kept_tiles.values():
        parent_tiles[tile.group_key].append(tile)
    # Keep this immutable snapshot for the repair evidence previews.
    source_parent_tiles = {key: list(value) for key, value in parent_tiles.items()}

    base_lines: dict[str, list[str]] = {}
    for name, tile in kept_tiles.items():
        source_rows = [training_rows[item.image_name] for item in rows_by_window[tile.window_key] if item.image_name in training_rows and item.parent_id not in EXCLUDED_PARENT_IDS]
        lines: list[str] = []
        for row in source_rows:
            path = source_training_root / row["label_path"]
            if path.is_file():
                lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        # Only the audited high-IoU locations are de-duplicated. Window merges
        # still remove exact duplicates to avoid duplicate rows becoming labels.
        threshold = DUPLICATE_LABEL_IOU if name in duplicate_label_tiles else 0.999999
        deduped, removals = deduplicate_lines(lines, tile, threshold)
        base_lines[name] = deduped
        for removal in removals:
            duplicate_ledger.append({"action": "removed_duplicate_label", "tile_image": name, **removal})
    labels_before_projection_repairs = sum(len(lines) for lines in base_lines.values())

    decisions: list[dict[str, Any]] = []
    new_recovery_tiles: list[TileRow] = []
    forced_recovery_instances: set[str] = set()
    unusable_zero_recovery_names: set[str] = set()
    parent_sizes: dict[str, tuple[int, int]] = {}
    Image.MAX_IMAGE_PIXELS = None
    for tile in kept_tiles.values():
        parent_path = active_parent_path(tile, active_parent_root)
        if parent_path.is_file() and tile.group_key not in parent_sizes:
            with Image.open(parent_path) as image:
                parent_sizes[tile.group_key] = image.size

    def add_projection(instance_id: str, tile: TileRow, reason: str) -> bool:
        obb = parent_obbs.get(instance_id)
        if obb is None:
            decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "skipped", "reason": "parent_obb_not_found"})
            return False
        if tile.parent_id in EXCLUDED_PARENT_IDS:
            decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "skipped", "reason": "intentionally_excluded_parent"})
            return False
        projection = projection_for(obb, tile)
        if projection is None or projection.overlap_ratio < SHIP_OVERLAP_THRESHOLD or projection.retained_fraction < MIN_RETAINED_FRACTION:
            decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "skipped", "reason": "below_projection_policy", "overlap": getattr(projection, "overlap_ratio", 0.0), "retained": getattr(projection, "retained_fraction", 0.0)})
            return False
        issue = coverage_by_instance.get(instance_id, {})
        matched = issue.get("coverage_record", {}).get("matched_tiles", [])
        if projection.retained_fraction < FULL_RETAINED_FRACTION and any(float(match.get("retained_fraction", 0.0)) >= FULL_RETAINED_FRACTION for match in matched):
            decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "skipped", "reason": "clean_representation_already_exists", "retained": projection.retained_fraction})
            return False
        line = projection_to_yolo_line(0, projection, tile.width, tile.height)
        filter_reason = v9_filter_reason(line, tile, qa_tags.get(tile.parent_id, "unflagged"))
        if filter_reason:
            decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "skipped", "reason": filter_reason, "retained": projection.retained_fraction})
            return False
        polygon = tile_polygon_from_line(line, tile)
        if polygon is None:
            decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "skipped", "reason": "invalid_projected_geometry"})
            return False
        existing = [tile_polygon_from_line(item, tile) for item in base_lines.setdefault(tile.image_name, [])]
        if any(candidate is not None and polygon_iou(polygon, candidate) >= DUPLICATE_LABEL_IOU for candidate in existing):
            decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "skipped", "reason": "already_represented_geometrically", "retained": projection.retained_fraction})
            return False
        base_lines[tile.image_name].append(line)
        decisions.append({"instance_id": instance_id, "tile_image": tile.image_name, "action": "added_projection", "reason": reason, "overlap": projection.overlap_ratio, "retained": projection.retained_fraction, "label_line": line})
        return True

    # Repair missing projections in existing windows first, including P6769.
    for instance_id, issue in sorted(coverage_by_instance.items()):
        if issue["issue_type"] != "covered_but_missing_label":
            continue
        for name in issue.get("coverage_record", {}).get("missing_tiles", []):
            tile = kept_tiles.get(name)
            if tile is not None:
                add_projection(instance_id, tile, "covered_but_missing_label")
    for name in sorted(zero_recovery_tiles):
        tile = kept_tiles.get(name)
        if tile is None:
            continue
        metadata = next((row for row in load_metadata(source_tile_root / "recovery_tile_metadata.jsonl") if row.get("image_name") == name), {})
        for instance_id in parse_json_field(metadata.get("covered_instance_ids"), []):
            instance_id = str(instance_id)
            if not add_projection(instance_id, tile, "zero_label_recovery_tile"):
                # The original P6769-style window can be spatially invalid for
                # its declared instance. Retire it and replace it below rather
                # than retaining a known zero-label recovery tile.
                forced_recovery_instances.add(instance_id)
                unusable_zero_recovery_names.add(name)
    for name in sorted(unusable_zero_recovery_names):
        tile = kept_tiles.pop(name, None)
        if tile is None:
            continue
        parent_tiles[tile.group_key] = [item for item in parent_tiles[tile.group_key] if item.image_name != name]
        base_lines.pop(name, None)
        duplicate_ledger.append({"action": "removed_invalid_zero_label_recovery_tile", "removed_tile": name})

    # Add a recovery crop for genuinely uncovered/ROI-excluded OBBs only.
    created_windows: set[tuple[str, int, int, int, int]] = set()
    for instance_id in sorted(repair_instances):
        obb = parent_obbs.get(instance_id)
        if obb is None or parent_id_from_group(obb.group_key) in EXCLUDED_PARENT_IDS:
            continue
        issue = coverage_by_instance.get(instance_id, {})
        if issue.get("issue_type") == "covered_but_missing_label" and instance_id not in forced_recovery_instances:
            continue
        existing = parent_tiles.get(obb.group_key, [])
        if any(add_projection(instance_id, tile, "existing_recovery_or_roi_tile") for tile in existing if tile.source_kind == "annotation_recovery"):
            continue
        parent_size = parent_sizes.get(obb.group_key)
        template = next((tile for tile in existing), None)
        if parent_size is None or template is None:
            decisions.append({"instance_id": instance_id, "action": "skipped", "reason": "active_parent_or_template_tile_unavailable"})
            continue
        origin = choose_recovery_origin(obb, parent_size[0], parent_size[1])
        if origin is None:
            decisions.append({"instance_id": instance_id, "action": "skipped", "reason": "cannot_create_valid_recovery_window"})
            continue
        x, y, width, height = origin
        window = (obb.group_key, x, y, width, height)
        tile = next((item for item in parent_tiles[obb.group_key] if item.window_key == window), None)
        created_tile = False
        if tile is None:
            tile = source_tile_row_for_recovery(obb, template, x, y, width, height)
            if tile.window_key not in created_windows:
                created_windows.add(tile.window_key)
                kept_tiles[tile.image_name] = tile
                parent_tiles[obb.group_key].append(tile)
                base_lines[tile.image_name] = []
                new_recovery_tiles.append(tile)
                created_tile = True
        if created_tile:
            # A recovery window is a normal training tile once created. Label
            # every valid parent OBB it contains, not just the trigger OBB,
            # otherwise dense parents acquire synthetic missing-label findings.
            for candidate in parent_obbs_by_group[obb.group_key]:
                min_x, min_y, max_x, max_y = candidate.polygon.bounds
                if max_x < tile.x or min_x > tile.x + tile.width or max_y < tile.y or min_y > tile.y + tile.height:
                    continue
                add_projection(candidate.instance_id, tile, "new_recovery_tile_context")
        else:
            add_projection(instance_id, tile, "new_recovery_tile")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "excluded_tile_count": len(excluded_tile_names), "duplicate_tiles_removed": len(removed_duplicate_names), "planned_new_recovery_tiles": len(new_recovery_tiles), "projection_decisions": Counter(item["action"] for item in decisions)}, indent=2, default=dict))
        return 0

    for path in destinations:
        if path.exists():
            shutil.rmtree(path)
    output_tile_root.mkdir(parents=True)
    output_parent_root.mkdir(parents=True)
    output_training_root.mkdir(parents=True)
    report_root.mkdir(parents=True)

    # Versioned parent source: preserve all non-excluded records exactly.
    kept_parent_rows = [row for row in raw_parent_rows if parent_id_from_group(str(row["parent_group_key"])) not in EXCLUDED_PARENT_IDS]
    write_jsonl(output_parent_root / "instances.jsonl", kept_parent_rows)
    exclusion_rows = [{"parent_id": parent_id, "reason": EXCLUSION_REASON} for parent_id in sorted(EXCLUDED_PARENT_IDS)]
    write_csv(output_parent_root / "exclusion_ledger.csv", exclusion_rows, ["parent_id", "reason"])
    write_json(output_parent_root / "source_manifest.json", {"source_instances_jsonl": str(parent_jsonl), "source_sha256": sha256(parent_jsonl), "excluded_parents": exclusion_rows, "records_kept": len(kept_parent_rows)})

    output_rows: list[dict[str, Any]] = []
    for tile in sorted(kept_tiles.values(), key=lambda item: (item.split, item.image_name)):
        image_destination = output_tile_root / "images" / tile.split / tile.image_name
        label_destination = output_tile_root / "labels" / tile.split / f"{Path(tile.image_name).stem}.txt"
        if tile in new_recovery_tiles:
            parent_path = active_parent_path(tile, active_parent_root)
            with Image.open(parent_path) as image:
                image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height)).save(image_destination)
        else:
            source_image = source_tile_root / tile_image_relative_path(tile)
            relative_link(source_image, image_destination)
        lines, removals = deduplicate_lines(base_lines.get(tile.image_name, []), tile, DUPLICATE_LABEL_IOU if tile.image_name in duplicate_label_tiles else 0.999999)
        for removal in removals:
            duplicate_ledger.append({"action": "removed_duplicate_label", "tile_image": tile.image_name, **removal})
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        row = dict(tile.row)
        row.update({
            "image_name": tile.image_name,
            "tile_path": str(Path("images") / tile.split / tile.image_name),
            "label_path": str(Path("labels") / tile.split / f"{Path(tile.image_name).stem}.txt"),
            "image_path": str(Path("images") / tile.split / tile.image_name),
            "second_pass_annotation_count": str(len(lines)),
            "empty": str(not lines),
            "label_source": "v9_audit_fixed_parent_projection",
        })
        output_rows.append(row)

    output_fields = list(tile_fields)
    for extra in ["image_name", "source_image_name", "source_xml", "image_path", "second_pass_annotation_count", "empty", "label_source", "source_dataset_split", "assignment_origin"]:
        if extra not in output_fields:
            output_fields.append(extra)
    write_csv(output_tile_root / "dataset_index.csv", output_rows, output_fields)
    map_rows = [{field: row.get(field, "") for field in tile_map_fields} for row in output_rows]
    write_csv(output_tile_root / "tile_parent_map.csv", map_rows, tile_map_fields)
    for split in ("train", "val"):
        subset = [row for row in output_rows if row["split"] == split]
        write_csv(output_tile_root / "manifests" / f"{split}.csv", subset, output_fields)
        (output_tile_root / "manifests" / f"{split}.txt").write_text("\n".join(str((output_tile_root / row["image_path"]).absolute()) for row in subset) + "\n", encoding="utf-8")
    policy = json.loads((source_tile_root / "grid_tile_policy.json").read_text(encoding="utf-8"))
    policy.update({"patched_from": str(source_tile_root), "patch_version": "audit_fixed_v1", "excluded_parent_ids": sorted(EXCLUDED_PARENT_IDS)})
    write_json(output_tile_root / "grid_tile_policy.json", policy)

    metadata_by_name = {row.get("image_name"): row for row in load_metadata(source_tile_root / "recovery_tile_metadata.jsonl")}
    recovery_metadata: list[dict[str, Any]] = []
    recovery_plan: list[dict[str, Any]] = []
    kept_window_keys = {tile.window_key for tile in kept_tiles.values()}
    for tile in kept_tiles.values():
        if tile.source_kind not in {"annotation_recovery", "full_parent_small_image"}:
            continue
        if tile.image_name in metadata_by_name:
            recovery_metadata.append(metadata_by_name[tile.image_name])
        elif tile in new_recovery_tiles:
            ids = [item["instance_id"] for item in decisions if item.get("tile_image") == tile.image_name and item["action"] == "added_projection"]
            recovery_metadata.append({"image_name": tile.image_name, "image_path": tile.row["tile_path"], "label_path": tile.row["label_path"], "dataset": tile.dataset, "source_split": tile.source_split, "split": tile.split, "group_key": tile.group_key, "parent_id": tile.parent_id, "parent_image_filename": tile.row.get("parent_name", ""), "tile_source": "annotation_recovery", "recovery_reason": "audit_coverage_repair", "covered_instance_ids": ids, "covered_obb_count": len(ids), "tile_xyxy_parent": [tile.x, tile.y, tile.x + tile.width, tile.y + tile.height], "outside_manual_grid": True})
    for row in load_metadata(source_tile_root / "recovery_tile_plan.jsonl"):
        xyxy = row.get("tile_xyxy_parent", [])
        if len(xyxy) == 4:
            key = (row.get("parent_group_key", ""), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]), int(xyxy[3]) - int(xyxy[1]))
            if key in kept_window_keys:
                recovery_plan.append(row)
    for row in recovery_metadata:
        if row.get("image_name") and row["image_name"] not in metadata_by_name:
            recovery_plan.append({"parent_group_key": row["group_key"], "parent_image_filename": row.get("parent_image_filename", ""), "tile_source": "annotation_recovery", "recovery_reason": "audit_coverage_repair", "covered_instance_ids": row["covered_instance_ids"], "tile_xyxy_parent": row["tile_xyxy_parent"]})
    write_jsonl(output_tile_root / "recovery_tile_metadata.jsonl", recovery_metadata)
    write_jsonl(output_tile_root / "recovery_tile_plan.jsonl", recovery_plan)

    # The training successor owns labels and symlinks all images from patched tiles.
    training_rows_out: list[dict[str, Any]] = []
    for row in output_rows:
        image_name = row["image_name"]
        split = row["split"]
        source_image = output_tile_root / row["image_path"]
        destination_image = output_training_root / "images" / split / image_name
        relative_link(source_image, destination_image)
        destination_label = output_training_root / "labels" / split / f"{Path(image_name).stem}.txt"
        source_label = output_tile_root / row["label_path"]
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_label.write_text(source_label.read_text(encoding="utf-8"), encoding="utf-8")
        training_row = dict(row)
        training_row.update({"tile_path": str(Path("images") / split / image_name), "image_path": str(Path("images") / split / image_name), "label_path": str(Path("labels") / split / destination_label.name)})
        training_rows_out.append(training_row)
    training_fields_out = list(training_fields)
    for field in output_fields:
        if field not in training_fields_out:
            training_fields_out.append(field)
    write_csv(output_training_root / "dataset_index.csv", training_rows_out, training_fields_out)
    for split in ("train", "val"):
        subset = [row for row in training_rows_out if row["split"] == split]
        write_csv(output_training_root / "manifests" / f"{split}.csv", subset, training_fields_out)
        (output_training_root / "manifests" / f"{split}.txt").write_text("\n".join(str((output_training_root / row["image_path"]).absolute()) for row in subset) + "\n", encoding="utf-8")
    yaml = f"path: {output_training_root}\ntrain: manifests/train.txt\nval: manifests/val.txt\nnames:\n  0: ship\n"
    (output_training_root / "data.yaml").write_text(yaml, encoding="utf-8")
    (output_training_root / "train.yaml").write_text(yaml, encoding="utf-8")

    # Preserve the source policy audit rows where their tile still exists, and
    # separately account for labels introduced or filtered by this patch.
    retained_tile_names = {row["image_name"] for row in output_rows}
    for name in ("removed_annotations.csv", "kept_annotations.csv"):
        source_path = source_training_root / name
        if not source_path.is_file():
            continue
        rows, fields = read_csv(source_path)
        filtered = [
            row for row in rows
            if row.get("tile_image") in retained_tile_names
            and row.get("parent_image_id") not in EXCLUDED_PARENT_IDS
        ]
        write_csv(output_training_root / name, filtered, fields)
    added_rows = [item for item in decisions if item.get("action") == "added_projection"]
    skipped_rows = [item for item in decisions if item.get("action") == "skipped"]
    write_csv(
        output_training_root / "audit_added_annotations.csv",
        added_rows,
        sorted({key for item in added_rows for key in item}),
    )
    write_csv(
        output_training_root / "audit_skipped_projections.csv",
        skipped_rows,
        sorted({key for item in skipped_rows for key in item}),
    )
    write_json(
        output_training_root / "filter_report.json",
        {
            "policy_source": str(source_training_root / "filter_report.csv"),
            "policy": "v9_seed3_aggressive_keep_bad",
            "new_projection_additions": len(added_rows),
            "new_projection_filter_removals": sum(str(item.get("reason", "")).startswith("threshold_") for item in skipped_rows),
            "new_projection_other_skips": sum(not str(item.get("reason", "")).startswith("threshold_") for item in skipped_rows),
        },
    )

    counts = {
        "source_tiles": len(tile_rows), "patched_tiles": len(output_rows), "excluded_parent_tiles_removed": len(excluded_tile_names),
        "duplicate_tile_rows_removed": len(removed_duplicate_names), "new_recovery_tiles": len(new_recovery_tiles),
        "parent_obb_records_source": len(raw_parent_rows), "parent_obb_records_patched": len(kept_parent_rows),
        "parent_obb_records_excluded": len(raw_parent_rows) - len(kept_parent_rows),
        "source_training_labels": sum(
            len((source_training_root / row["label_path"]).read_text(encoding="utf-8").splitlines())
            for row in training_rows_raw
            if (source_training_root / row["label_path"]).is_file()
        ),
        "labels_after_duplicate_window_merge": labels_before_projection_repairs,
        "training_labels_after": sum(len(base_lines.get(row["image_name"], [])) for row in output_rows),
        "projection_additions": sum(item["action"] == "added_projection" for item in decisions),
        "projection_skips": sum(item["action"] == "skipped" for item in decisions),
        "new_projection_v9_filter_removals": sum(str(item.get("reason", "")).startswith("threshold_") for item in decisions if item["action"] == "skipped"),
    }
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "counts": counts, "sources": {"tile_root": str(source_tile_root), "parent_root": str(source_parent_root), "training_root": str(source_training_root), "audit_root": str(source_audit_root)}, "outputs": {"tile_root": str(output_tile_root), "parent_root": str(output_parent_root), "training_root": str(output_training_root)}, "excluded_parents": sorted(EXCLUDED_PARENT_IDS), "v9_policy": {"seed": 3, "small": [36.0, 5.0], "low": [64.0, 6.0], "xview": [36.0, 5.0], "keep_bad": True}}
    write_json(report_root / "patch_summary.json", summary)
    write_csv(report_root / "patch_ledger.csv", decisions + duplicate_ledger, sorted({key for item in decisions + duplicate_ledger for key in item}))
    write_csv(report_root / "excluded_parents.csv", exclusion_rows, ["parent_id", "reason"])
    (report_root / "patch_report.md").write_text(report_markdown(summary), encoding="utf-8")
    write_json(output_training_root / "dataset_manifest.json", summary)
    write_json(output_training_root / "dataset_summary.json", summary)
    (output_training_root / "README.md").write_text("Versioned v9 seed-3 aggressive-filtering successor produced by the tile-coverage audit patch. Images link to the patched tiled root; labels are independent.\n", encoding="utf-8")

    previews = write_repair_previews(
        report_root / "repair_previews",
        decisions,
        parent_obbs,
        source_parent_tiles,
        parent_tiles,
        active_parent_root,
    )
    write_csv(
        report_root / "repair_preview_manifest.csv",
        previews,
        ["parent_group_key", "instance_id", "tile_image", "preview_path"],
    )

    if not args.skip_audit:
        if __package__:
            from .audit_tile_coverage_recovery import main as audit_main
        else:
            from audit.audit_tile_coverage_recovery import main as audit_main
        audit_output = report_root / "post_patch_audit"
        old_argv = sys.argv[:]
        try:
            sys.argv = ["audit_tile_coverage_recovery.py", "--tile-root", str(output_tile_root), "--parent-label-root", str(output_parent_root), "--training-dataset-root", str(output_training_root), "--parent-image-root", str(active_parent_root), "--output-root", str(audit_output), "--preview-count", "50"]
            audit_main()
        finally:
            sys.argv = old_argv
        write_before_after_summary(source_audit_root, audit_output, report_root)
    print(json.dumps({"output_tile_root": str(output_tile_root), "output_training_root": str(output_training_root), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
