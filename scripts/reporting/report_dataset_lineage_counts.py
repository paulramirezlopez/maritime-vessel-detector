#!/usr/bin/env python3
"""Create publication-ready counts across the maritime dataset lineage.

The report deliberately keeps unique parent-coordinate annotations separate
from tile-local label occurrences, because overlap means one parent ship may
appear in more than one tile label file.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
MARITIME_IDS = frozenset({40, 41, 42, 44, 45, 47, 49, 50, 51, 52})
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/dataset_lineage_counts"


@dataclass(frozen=True)
class CountRow:
    stage: str
    dataset: str
    source_split: str
    selection: str
    parent_image_count: int | None = None
    annotated_parent_image_count: int | None = None
    tile_image_count: int | None = None
    annotated_tile_image_count: int | None = None
    parent_annotation_count: int | None = None
    tile_annotation_count: int | None = None
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-expected-checks", action="store_true", help="Write observed counts without failing expected-total checks.")
    return parser.parse_args()


def count_dota_ship_labels(labels_dir: Path, image_names: Iterable[str] | None = None) -> tuple[int, int]:
    """Return (ship polygons, positive parents) from DOTA text labels."""
    paths = (
        sorted(labels_dir.glob("*.txt"))
        if image_names is None
        else [labels_dir / f"{Path(name).stem}.txt" for name in image_names]
    )
    annotation_count = 0
    positive_parent_count = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing DOTA label: {path}")
        ships = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 9 and parts[8] == "ship":
                ships += 1
        annotation_count += ships
        positive_parent_count += bool(ships)
    return annotation_count, positive_parent_count


def count_hrsc_objects(labels_dir: Path) -> tuple[int, int]:
    """Return (raw <object> count, positive images) from HRSC XML labels."""
    annotation_count = 0
    positive_parent_count = 0
    for path in sorted(labels_dir.glob("*.xml")):
        object_count = sum(1 for element in ET.parse(path).getroot().iter() if element.tag == "object")
        annotation_count += object_count
        positive_parent_count += bool(object_count)
    return annotation_count, positive_parent_count


def xview_maritime_counts(geojson_path: Path) -> dict[str, int]:
    """Return raw maritime HBB counts keyed by xView parent filename."""
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = defaultdict(int)
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        try:
            type_id = int(properties.get("type_id", -1))
        except (TypeError, ValueError):
            continue
        image_name = Path(str(properties.get("image_id", ""))).name
        if type_id in MARITIME_IDS and image_name:
            counts[image_name] += 1
    return dict(counts)


def count_cvat_shapes(xml_path: Path) -> tuple[int, int]:
    """Return (image entries, annotation shapes) for a CVAT-for-images XML."""
    root = ET.parse(xml_path).getroot()
    images = root.findall("image")
    return len(images), sum(len(image.findall("box")) + len(image.findall("polygon")) + len(image.findall("mask")) for image in images)


def load_triage(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_tiled_dataset(dataset_root: Path) -> tuple[list[CountRow], dict[str, int]]:
    """Validate and count the active tile view from its index and label files."""
    index_path = dataset_root / "dataset_index.csv"
    with index_path.open(encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    required = {"dataset", "split", "parent_id", "tile_path", "label_path"}
    if not index_rows or not required.issubset(index_rows[0]):
        raise ValueError(f"Unexpected tile index schema: {index_path}")

    stats: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"tiles": 0, "nonempty_tiles": 0, "labels": 0, "parents": set()}
    )
    for row in index_rows:
        tile_path = dataset_root / row["tile_path"]
        label_path = dataset_root / row["label_path"]
        if not tile_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing tile or label referenced by {index_path}: {tile_path}, {label_path}")
        label_lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(len(parts) != 9 for parts in label_lines):
            raise ValueError(f"Invalid YOLO OBB field count in {label_path}")
        record = stats[(row["dataset"], row["split"])]
        record["tiles"] = int(record["tiles"]) + 1
        record["labels"] = int(record["labels"]) + len(label_lines)
        record["nonempty_tiles"] = int(record["nonempty_tiles"]) + bool(label_lines)
        record["parents"].add(row["parent_id"])

    result: list[CountRow] = []
    totals = {"tile_images": 0, "annotated_tile_images": 0, "tile_annotations": 0, "represented_parents": 0}
    for (dataset, split), record in sorted(stats.items()):
        parents = record["parents"]
        result.append(
            CountRow(
                stage="current_final_tiled",
                dataset=dataset,
                source_split=split,
                selection="active_v9_audit_fixed_v2",
                parent_image_count=len(parents),
                tile_image_count=int(record["tiles"]),
                annotated_tile_image_count=int(record["nonempty_tiles"]),
                tile_annotation_count=int(record["labels"]),
                notes="Tile annotation count includes valid repeated views in overlapping tiles.",
            )
        )
        totals["tile_images"] += int(record["tiles"])
        totals["annotated_tile_images"] += int(record["nonempty_tiles"])
        totals["tile_annotations"] += int(record["labels"])
        totals["represented_parents"] += len(parents)
    return result, totals


def count_parent_instances(instances_path: Path) -> tuple[list[CountRow], dict[str, int]]:
    stats: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {"annotations": 0, "parents": set()})
    for line_number, line in enumerate(instances_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("coordinate_system") != "parent_image_pixels":
            raise ValueError(f"Non-parent coordinate system at {instances_path}:{line_number}")
        group_key = str(record.get("parent_group_key", ""))
        parts = group_key.split(":")
        if len(parts) < 3:
            raise ValueError(f"Invalid parent group key at {instances_path}:{line_number}: {group_key!r}")
        bucket = stats[(parts[0], parts[1])]
        bucket["annotations"] = int(bucket["annotations"]) + 1
        bucket["parents"].add(group_key)

    rows: list[CountRow] = []
    totals = {"annotated_parents": 0, "parent_annotations": 0}
    for (dataset, split), bucket in sorted(stats.items()):
        parent_count = len(bucket["parents"])
        annotation_count = int(bucket["annotations"])
        rows.append(
            CountRow(
                stage="second_pass_parent",
                dataset=dataset,
                source_split=split,
                selection="current_audit_fixed_v2",
                parent_image_count=parent_count,
                annotated_parent_image_count=parent_count,
                parent_annotation_count=annotation_count,
                notes="Only parents with at least one second-pass OBB appear in instances.jsonl.",
            )
        )
        totals["annotated_parents"] += parent_count
        totals["parent_annotations"] += annotation_count
    return rows, totals


def add_total(rows: list[CountRow], *, stage: str, dataset: str, selection: str, members: list[CountRow], notes: str) -> None:
    def total(field: str) -> int | None:
        values = [getattr(row, field) for row in members if getattr(row, field) is not None]
        return sum(values) if values else None

    rows.append(
        CountRow(
            stage=stage,
            dataset=dataset,
            source_split="all",
            selection=selection,
            parent_image_count=total("parent_image_count"),
            annotated_parent_image_count=total("annotated_parent_image_count"),
            tile_image_count=total("tile_image_count"),
            annotated_tile_image_count=total("annotated_tile_image_count"),
            parent_annotation_count=total("parent_annotation_count"),
            tile_annotation_count=total("tile_annotation_count"),
            notes=notes,
        )
    )


def assert_expected(rows: list[CountRow]) -> list[dict[str, object]]:
    """Check the current intended lineage values and return an auditable ledger."""
    expected = {
        ("raw_source", "dota", "all", "all_labeled_source"): {"parent_image_count": 2423, "parent_annotation_count": 54353},
        ("raw_source", "hrsc", "all", "all_labeled_source"): {"parent_image_count": 1680, "parent_annotation_count": 7655},
        ("raw_source", "xview", "all", "all_labeled_source"): {"parent_image_count": 846, "parent_annotation_count": 5141},
        ("manual_triage", "dota", "all", "maritime_context"): {"parent_image_count": 517, "parent_annotation_count": 52998},
        ("manual_triage", "dota", "all", "low_maritime_context"): {"parent_image_count": 203, "parent_annotation_count": 545},
        ("manual_triage", "xview", "all", "maritime_context"): {"parent_image_count": 150, "parent_annotation_count": 4716},
        ("manual_triage", "xview", "all", "low_maritime_context"): {"parent_image_count": 32, "parent_annotation_count": 34},
        ("first_pass_tiled", "combined", "all", "naive_tiles"): {"tile_image_count": 12596, "tile_annotation_count": 116471},
        ("first_pass_parent", "dota_xview", "all", "pure_first_pass_recovery"): {"parent_image_count": 667, "parent_annotation_count": 45039},
        ("second_pass_parent", "combined", "all", "current_audit_fixed_v2"): {"parent_image_count": 2260, "parent_annotation_count": 61845},
        ("current_final_tiled", "combined", "all", "active_v9_audit_fixed_v2"): {"tile_image_count": 5351, "annotated_tile_image_count": 4959, "tile_annotation_count": 98768},
    }
    indexed = {(row.stage, row.dataset, row.source_split, row.selection): row for row in rows}
    checks = []
    for key, fields in expected.items():
        row = indexed.get(key)
        actual = {field: getattr(row, field) if row else None for field in fields}
        passed = row is not None and all(actual[field] == value for field, value in fields.items())
        checks.append({"key": list(key), "expected": fields, "actual": actual, "passed": passed})
    return checks


def markdown_table(rows: list[CountRow]) -> str:
    header = "| Dataset | Split | Selection | Parent images | Annotated parents | Tile images | Annotated tiles | Parent annotations | Tile annotations |\n"
    rule = "|---|---|---|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for row in rows:
        values = [
            row.dataset,
            row.source_split,
            row.selection.replace("_", " "),
            row.parent_image_count,
            row.annotated_parent_image_count,
            row.tile_image_count,
            row.annotated_tile_image_count,
            row.parent_annotation_count,
            row.tile_annotation_count,
        ]
        body.append("| " + " | ".join("" if value is None else f"{value:,}" if isinstance(value, int) else str(value) for value in values) + " |\n")
    return header + rule + "".join(body)


def require_audit_fixed_symlink(path: Path, target_name: str) -> Path:
    if not path.is_symlink() or path.resolve().name != target_name:
        raise RuntimeError(f"Active root must resolve to {target_name}: {path} -> {path.resolve()}")
    return path.resolve()


def build_report() -> tuple[list[CountRow], dict[str, object]]:
    raw_dota_train = REPO_ROOT / "data/raw/DOTA/train"
    raw_dota_val = REPO_ROOT / "data/raw/DOTA/val"
    raw_hrsc = REPO_ROOT / "data/raw/HRSC"
    raw_xview = REPO_ROOT / "data/raw/xView"
    triage_root = REPO_ROOT / "data/metadata/manual_triage"
    first_root = REPO_ROOT / "data/annotations/first_pass_naive_tiles"
    first_recovery = REPO_ROOT / "data/annotations/recovered_parent/archive/first_pass_naive_before_generalized_offsets"
    active_parent_obb_root = REPO_ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery"
    active_grid_root = REPO_ROOT / "data/tiled/grid_current_with_recovery"
    active_tiles = REPO_ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3"
    resolved_parent_obb_root = require_audit_fixed_symlink(active_parent_obb_root, "obbs_complete_with_recovery_audit_fixed_v2")
    resolved_grid_root = require_audit_fixed_symlink(active_grid_root, "grid_current_with_recovery_audit_fixed_v2")
    resolved_active_tiles = require_audit_fixed_symlink(active_tiles, "v9_threshold_aggressive_keep_bad_seed3_audit_fixed_v2")
    parent_instances = active_parent_obb_root / "instances.jsonl"

    xview_counts = xview_maritime_counts(raw_xview / "xView_train.geojson")
    rows: list[CountRow] = []

    raw_rows: list[CountRow] = []
    for dataset, split, image_dir, label_dir in [
        ("dota", "train", raw_dota_train / "images", raw_dota_train / "labels"),
        ("dota", "val", raw_dota_val / "images", raw_dota_val / "labels"),
    ]:
        annotations, positive = count_dota_ship_labels(label_dir)
        raw_rows.append(CountRow("raw_source", dataset, split, "all_labeled_source", len(list(image_dir.glob("*.png"))), positive, parent_annotation_count=annotations, notes="Raw DOTA `ship` polygons only."))
    hrsc_annotations, hrsc_positive = count_hrsc_objects(raw_hrsc / "labels")
    raw_rows.append(CountRow("raw_source", "hrsc", "train", "all_labeled_source", len(list((raw_hrsc / "images").glob("*"))), hrsc_positive, parent_annotation_count=hrsc_annotations, notes="Raw HRSC `<object>` records."))
    raw_rows.append(CountRow("raw_source", "xview", "train", "all_labeled_source", len(list((raw_xview / "train").glob("*.tif"))), sum(bool(count) for count in xview_counts.values()), parent_annotation_count=sum(xview_counts.values()), notes="Raw xView maritime type IDs only."))
    rows.extend(raw_rows)
    add_total(rows, stage="raw_source", dataset="dota", selection="all_labeled_source", members=raw_rows[:2], notes="DOTA train and val labeled source pools.")
    for dataset in ("hrsc", "xview"):
        members = [row for row in raw_rows if row.dataset == dataset]
        add_total(rows, stage="raw_source", dataset=dataset, selection="all_labeled_source", members=members, notes="Labeled source pool total.")

    triage_rows: list[CountRow] = []
    for key, dataset, split, labels_dir in [
        ("dota_train", "dota", "train", raw_dota_train / "labels"),
        ("dota_val", "dota", "val", raw_dota_val / "labels"),
        ("xview_train", "xview", "train", None),
    ]:
        triage = load_triage(triage_root / f"{key}.csv")
        for selection in ("maritime_context", "low_maritime_context"):
            names = [row["image_name"] for row in triage if row["selection"] == selection]
            if labels_dir is not None:
                annotations, positive = count_dota_ship_labels(labels_dir, names)
            else:
                annotations = sum(xview_counts.get(name, 0) for name in names)
                positive = sum(bool(xview_counts.get(name, 0)) for name in names)
            triage_rows.append(CountRow("manual_triage", dataset, split, selection, len(names), positive, parent_annotation_count=annotations, notes="Manual context membership; annotations remain raw-source ships."))
    rows.extend(triage_rows)
    for dataset in ("dota", "xview"):
        for selection in ("maritime_context", "low_maritime_context"):
            members = [row for row in triage_rows if row.dataset == dataset and row.selection == selection]
            add_total(rows, stage="manual_triage", dataset=dataset, selection=selection, members=members, notes="Combined manual-triage total across source splits.")

    first_tiled_rows: list[CountRow] = []
    for dataset, split, relative in [
        ("dota", "train", "dota/train_annotations.xml"),
        ("dota", "val", "dota/val_annotations.xml"),
        ("hrsc", "train", "hrsc/hrsc_annotations.xml"),
        ("xview", "train", "xview/xview_annotations.xml"),
    ]:
        tile_images, labels = count_cvat_shapes(first_root / relative)
        first_tiled_rows.append(CountRow("first_pass_tiled", dataset, split, "naive_tiles", tile_image_count=tile_images, tile_annotation_count=labels, notes="Combined first-pass CVAT XML only; HRSC job exports are intentionally excluded."))
    rows.extend(first_tiled_rows)
    add_total(rows, stage="first_pass_tiled", dataset="combined", selection="naive_tiles", members=first_tiled_rows, notes="Tile labels include overlap duplication from naive tiling.")

    first_parent_rows: list[CountRow] = []
    for dataset, split, relative in [
        ("dota", "train", "dota/train/parent_annotations.xml"),
        ("dota", "val", "dota/val/parent_annotations.xml"),
        ("xview", "train", "xview/train/parent_annotations.xml"),
    ]:
        parents, annotations = count_cvat_shapes(first_recovery / relative)
        first_parent_rows.append(CountRow("first_pass_parent", dataset, split, "pure_first_pass_recovery", parents, parents, parent_annotation_count=annotations, notes="Archived pure first-pass parent recovery before later overrides."))
    rows.extend(first_parent_rows)
    add_total(rows, stage="first_pass_parent", dataset="dota_xview", selection="pure_first_pass_recovery", members=first_parent_rows, notes="DOTA and xView only; HRSC is direct-parent imagery.")
    hrsc_first = next(row for row in first_tiled_rows if row.dataset == "hrsc")
    first_parent_rows.append(CountRow("first_pass_parent", "hrsc", "train", "direct_parent", hrsc_first.tile_image_count, hrsc_first.tile_image_count, parent_annotation_count=hrsc_first.tile_annotation_count, notes="HRSC first pass used full parent images rather than tiles."))
    rows.append(first_parent_rows[-1])
    add_total(rows, stage="first_pass_parent", dataset="combined", selection="parent_coordinate_or_direct", members=first_parent_rows, notes="Recovered DOTA/xView parents plus direct-parent HRSC.")

    second_parent_rows, second_parent_totals = count_parent_instances(parent_instances)
    rows.extend(second_parent_rows)
    add_total(rows, stage="second_pass_parent", dataset="combined", selection="current_audit_fixed_v2", members=second_parent_rows, notes="Current parent-coordinate OBB source, audit-fixed v2.")

    current_rows, current_totals = count_tiled_dataset(active_tiles)
    rows.extend(current_rows)
    add_total(rows, stage="current_final_tiled", dataset="combined", selection="active_v9_audit_fixed_v2", members=current_rows, notes="Current training view; label occurrences can exceed unique parent OBBs.")

    metadata = {
        "definitions": {
            "parent_image_count": "Source parent-image membership for raw/triage rows; annotated parent groups for parent-level annotation rows; parent IDs represented by tiles for tiled rows.",
            "parent_annotation_count": "Unique parent-coordinate annotation instances, except raw-source rows which count native source ship records.",
            "tile_image_count": "Image records in the relevant tiled dataset or CVAT naive-tile XML.",
            "tile_annotation_count": "YOLO/CVAT tile-local label occurrences. A parent annotation may occur in multiple overlapping tiles.",
        },
        "footnotes": [
            f"Unlabeled source test images are excluded from raw annotation totals: DOTA={len(list((raw_dota_train.parent / 'test/images').glob('*.png')))}, xView={len(list((raw_xview / 'test').glob('*.tif')))}.",
            "The current recovered_parent/current tree is not used for pure first-pass counts because it contains later xView second-pass overrides.",
        ],
        "source_paths": {
            "raw_dota": str(raw_dota_train.parent),
            "raw_hrsc": str(raw_hrsc),
            "raw_xview": str(raw_xview),
            "manual_triage": str(triage_root),
            "first_pass_naive_tiles": str(first_root),
            "first_pass_parent_recovery": str(first_recovery),
            "second_pass_parent_instances": str(parent_instances),
            "active_parent_obb_root": str(active_parent_obb_root),
            "active_parent_obb_root_resolved": str(resolved_parent_obb_root),
            "active_grid_root": str(active_grid_root),
            "active_grid_root_resolved": str(resolved_grid_root),
            "active_final_tiles": str(active_tiles),
            "active_final_tiles_resolved": str(resolved_active_tiles),
        },
        "current_totals": {**second_parent_totals, **current_totals},
    }
    return rows, metadata


def write_outputs(rows: list[CountRow], metadata: dict[str, object], output_dir: Path, checks: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rows": [asdict(row) for row in rows], "metadata": metadata, "expected_total_checks": checks}
    (output_dir / "dataset_lineage_counts.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "dataset_lineage_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    groups = [
        ("Raw Labeled Source Pools", "raw_source"),
        ("Manual Maritime-Context Triage", "manual_triage"),
        ("First-Pass Naive Tiles", "first_pass_tiled"),
        ("First-Pass Parent View", "first_pass_parent"),
        ("Second-Pass Parent OBBs", "second_pass_parent"),
        ("Current Final Tiled Training View", "current_final_tiled"),
    ]
    markdown = ["# Dataset Lineage Counts\n", "## Definitions\n"]
    markdown.extend(f"- **{field}**: {description}\n" for field, description in metadata["definitions"].items())
    for title, stage in groups:
        markdown.extend([f"\n## {title}\n", markdown_table([row for row in rows if row.stage == stage])])
    markdown.append("\n## Validation\n")
    markdown.extend(f"- {'PASS' if check['passed'] else 'FAIL'}: `{':'.join(check['key'])}`\n" for check in checks)
    markdown.append("\n## Notes\n")
    markdown.extend(f"- {note}\n" for note in metadata["footnotes"])
    (output_dir / "dataset_lineage_counts.md").write_text("".join(markdown), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows, metadata = build_report()
    checks = assert_expected(rows)
    write_outputs(rows, metadata, args.output_dir, checks)
    failures = [check for check in checks if not check["passed"]]
    print(json.dumps({"output_dir": str(args.output_dir), "row_count": len(rows), "expected_total_checks_passed": len(checks) - len(failures), "expected_total_check_failures": len(failures)}, indent=2))
    if failures and not args.no_expected_checks:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
