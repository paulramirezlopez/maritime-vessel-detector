#!/usr/bin/env python3
"""Resolve audited overlap fragments into a versioned grid-recovery v2 view."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box as shapely_box

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

if __package__:
    from .apply_tile_coverage_audit_fixes import (  # noqa: E402
        TILE_SIZE, TileRow, active_parent_path, candidate_origins, choose_recovery_origin,
        load_qa_tags, projection_for, relative_link, source_tile_row_for_recovery,
        tile_polygon_from_line, v9_filter_reason,
    )
    from .audit_tile_coverage_recovery import ParentOBB, load_parent_obbs  # noqa: E402
else:
    from audit.apply_tile_coverage_audit_fixes import (  # noqa: E402
        TILE_SIZE, TileRow, active_parent_path, candidate_origins, choose_recovery_origin,
        load_qa_tags, projection_for, relative_link, source_tile_row_for_recovery,
        tile_polygon_from_line, v9_filter_reason,
    )
    from audit.audit_tile_coverage_recovery import ParentOBB, load_parent_obbs  # noqa: E402
from roi_smart_retile_utils import projection_to_yolo_line  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V1_TILE = ROOT / "data/tiled/grid_current_with_recovery_audit_fixed_v1"
V1_PARENT = ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery_audit_fixed_v1"
V1_TRAINING = ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3_audit_fixed_v1"
V1_AUDIT = ROOT / "reports/tile_coverage_recovery_audit_fix_v1/post_patch_audit"
ACTIVE_PARENTS = ROOT / "data/parent_images/active"
QA_MANIFEST = ROOT / "data/metadata/qa/parent_image_quality_manifest.csv"
OUT_TILE = ROOT / "data/tiled/grid_current_with_recovery_audit_fixed_v2"
OUT_PARENT = ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery_audit_fixed_v2"
OUT_TRAINING = ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3_audit_fixed_v2"
OUT_REPORT = ROOT / "reports/tile_coverage_recovery_residual_edge_v2"

# These source OBBs are valid ships but exceed what the fixed 1024px,
# rectangle-preserving projection can represent. They remain in parent-space
# truth and are intentionally absent from the current tile-local training view.
VALID_BUT_UNREPRESENTABLE_LARGE_OBJECTS = {
    "dota_train_4:P2641.png:mask:28d262bdb00d6b62a80a",
    "dota_val_2:P2789.png:rotated_box:b3da1cb564a5cdd5c486",
}


@dataclass
class Case:
    issue_id: str
    parent_image: str
    parent_annotation_id: str
    flagged_tile: str
    flagged_tile_split: str
    flagged_tile_bounds_parent: list[int]
    flagged_visible_fraction: float
    flagged_intersection_area: float
    flagged_projection_valid: bool
    best_existing_tile: str
    best_existing_tile_bounds_parent: list[int]
    best_existing_visible_fraction: float
    best_existing_matching_label_id: str
    best_existing_match_iou: float
    best_existing_match_containment: float
    num_existing_matching_tiles: int
    num_candidate_tiles_intersecting_parent_obb: int
    decision: str
    action_taken: str
    reason: str
    preview_path: str = ""


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("report", "patch", "document"), default="report")
    parser.add_argument("--tile-root", type=Path, default=V1_TILE)
    parser.add_argument("--label-root", type=Path, default=V1_PARENT)
    parser.add_argument("--training-root", type=Path, default=V1_TRAINING)
    parser.add_argument("--post-patch-audit-root", type=Path, default=V1_AUDIT)
    parser.add_argument("--active-parent-root", type=Path, default=ACTIVE_PARENTS)
    parser.add_argument("--qa-manifest", type=Path, default=QA_MANIFEST)
    parser.add_argument("--output-tile-root", type=Path, default=OUT_TILE)
    parser.add_argument("--output-label-root", type=Path, default=OUT_PARENT)
    parser.add_argument("--output-training-root", type=Path, default=OUT_TRAINING)
    parser.add_argument("--output-report-root", type=Path, default=OUT_REPORT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--write-before-after",
        action="store_true",
        help="Write a compact v1-to-v2 audit delta from completed audit reports and exit.",
    )
    parser.add_argument("--suppression-visible-threshold", type=float, default=0.70)
    parser.add_argument("--best-visible-threshold", type=float, default=0.90)
    parser.add_argument("--suppression-preview-count", type=int, default=50)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames or []


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Write a short human-readable companion for the machine ledger."""
    lines = [
        "# Residual Edge-Fragment Resolution v2",
        "",
        f"Mode: `{summary['mode']}`",
        f"Atomic missing-tile cases analyzed: `{summary['atomic_cases']}`",
        f"Recovery targets: `{summary['recovery_targets']}`",
    ]
    if "labels_added" in summary:
        lines.extend([
            f"Projected labels added: `{summary['labels_added']}`",
            f"New physical recovery tiles: `{summary['recovery_tiles_created']}`",
        ])
    lines.extend(["", "## Decisions", ""])
    for decision, count in sorted(summary["decision_counts"].items()):
        lines.append(f"- `{decision}`: {count}")
    lines.extend([
        "",
        "The suppression ledger documents intentionally omitted secondary tile views, "
        "v9 policy filters, fixed-tile large-object exclusions, and views replaced by a deterministic recovery tile. "
        "It is consumed by the follow-up coverage audit.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_exception_queues(report_root: Path, cases: list[Case]) -> None:
    manual = [case for case in cases if case.decision == "unresolved_needs_manual_review"]
    excluded = [case for case in cases if case.decision == "valid_but_unrepresentable_large_object"]
    fields = [
        "parent_image", "parent_annotation_id", "flagged_tile", "flagged_visible_fraction",
        "reason", "preview_path",
    ]
    write_csv(report_root / "manual_review_queue.csv", [asdict(case) for case in manual], fields)
    lines = ["# Residual Edge-Fragment Manual Review", ""]
    if not manual:
        lines.append("No unresolved residual edge-fragment cases remain.")
    else:
        lines.append("These OBBs cannot be represented safely in a 1024px recovery crop using the required rectangle-preserving projection policy.")
        lines.append("")
        lines.append("| Parent | Source OBB | Flagged tile | Reason | Preview |")
        lines.append("| --- | --- | --- | --- | --- |")
        for case in manual:
            lines.append(f"| {case.parent_image} | `{case.parent_annotation_id}` | `{case.flagged_tile}` | `{case.reason}` | `{case.preview_path}` |")
    (report_root / "manual_review_queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    exclusion_rows = [asdict(case) for case in excluded]
    write_csv(report_root / "valid_but_unrepresentable_large_object_ledger.csv", exclusion_rows, fields)
    write_json(report_root / "valid_but_unrepresentable_large_object_ledger.json", {
        "classification": "valid_but_unrepresentable_large_object",
        "policy": "intentional exclusion from the fixed 1024px tile-local OBB training view; parent-space source annotations remain unchanged",
        "entries": exclusion_rows,
    })
    exclusion_lines = ["# Valid But Unrepresentable Large Objects", ""]
    if not excluded:
        exclusion_lines.append("No fixed-tile large-object exclusions are documented.")
    else:
        exclusion_lines.extend([
            "These are valid parent-space ships, retained in parent annotations. They are intentionally excluded only from the current 1024px tile-local training view because rectangle-preserving projection cannot produce a safe OBB.",
            "",
            "| Parent | Source OBB | Tile view | Preview |",
            "| --- | --- | --- | --- |",
        ])
        for case in excluded:
            exclusion_lines.append(f"| {case.parent_image} | `{case.parent_annotation_id}` | `{case.flagged_tile}` | `{case.preview_path}` |")
    (report_root / "valid_but_unrepresentable_large_object_ledger.md").write_text("\n".join(exclusion_lines) + "\n", encoding="utf-8")


def document_large_object_exclusions(report_root: Path) -> None:
    """Reclassify explicit fixed-tile exclusions without rebuilding dataset files."""
    json_path = report_root / "residual_edge_cases.json"
    csv_path = report_root / "residual_edge_cases.csv"
    ledger_path = report_root / "edge_fragment_suppression_ledger.json"
    cases = json.loads(json_path.read_text(encoding="utf-8"))
    changed = 0
    for case in cases:
        if case.get("parent_annotation_id") in VALID_BUT_UNREPRESENTABLE_LARGE_OBJECTS:
            case["decision"] = "valid_but_unrepresentable_large_object"
            case["action_taken"] = "intentional_fixed_tile_exclusion"
            case["reason"] = "valid_but_unrepresentable_large_object"
            old_preview = report_root / str(case.get("preview_path", ""))
            if old_preview.is_file():
                target = report_root / "qc_previews" / "valid_but_unrepresentable_large_object" / old_preview.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target != old_preview:
                    shutil.copy2(old_preview, target)
                    old_preview.unlink()
                case["preview_path"] = str(target.relative_to(report_root))
            changed += 1
    if changed != len(VALID_BUT_UNREPRESENTABLE_LARGE_OBJECTS):
        raise ValueError(f"Expected {len(VALID_BUT_UNREPRESENTABLE_LARGE_OBJECTS)} large-object cases, updated {changed}.")
    fields = list(cases[0]) if cases else list(Case.__annotations__)
    write_json(json_path, cases)
    write_csv(csv_path, cases, fields)

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    for entry in entries:
        if entry.get("parent_annotation_id") in VALID_BUT_UNREPRESENTABLE_LARGE_OBJECTS:
            entry["decision"] = "valid_but_unrepresentable_large_object"
            entry["reason"] = "valid_but_unrepresentable_large_object"
            entry["intentional_exclusion_scope"] = "fixed_1024_tile_training_only"
    write_json(ledger_path, payload)

    typed_cases = [Case(**case) for case in cases]
    write_exception_queues(report_root, typed_cases)
    summary_path = report_root / "residual_edge_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision_counts"] = dict(Counter(case.decision for case in typed_cases))
    summary["documented_large_object_exclusions"] = len(VALID_BUT_UNREPRESENTABLE_LARGE_OBJECTS)
    write_json(summary_path, summary)
    write_summary_markdown(report_root / "residual_edge_summary.md", summary)


def row_tile(row: dict[str, str]) -> TileRow:
    return TileRow(row)


def line_iou(line: str, tile: TileRow, projection_line: str) -> float:
    first, second = tile_polygon_from_line(line, tile), tile_polygon_from_line(projection_line, tile)
    if first is None or second is None:
        return 0.0
    union = first.union(second).area
    return first.intersection(second).area / union if union else 0.0


def output_paths(args_: argparse.Namespace) -> list[Path]:
    return [args_.output_tile_root, args_.output_label_root, args_.output_training_root, args_.output_report_root]


def write_before_after_summary(before_audit: Path, after_audit: Path, report_root: Path) -> None:
    before = json.loads((before_audit / "audit_summary.json").read_text(encoding="utf-8"))
    after = json.loads((after_audit / "audit_summary.json").read_text(encoding="utf-8"))
    statuses = sorted(set(before["coverage"]["status_counts"]) | set(after["coverage"]["status_counts"]))
    payload = {
        "before_audit": str(before_audit),
        "after_audit": str(after_audit),
        "tiles": {"before": before["inventory"]["tile_images"], "after": after["inventory"]["tile_images"]},
        "tile_local_labels": {"before": before["inventory"]["annotation_count"]["total"], "after": after["inventory"]["annotation_count"]["total"]},
        "coverage_status_counts": {
            status: {"before": before["coverage"]["status_counts"].get(status, 0), "after": after["coverage"]["status_counts"].get(status, 0)}
            for status in statuses
        },
        "severity_counts": {"before": before["severity_counts"], "after": after["severity_counts"]},
        "readiness": {"before": before.get("readiness"), "after": after.get("readiness")},
    }
    write_json(report_root / "before_after_counts.json", payload)
    lines = ["# Residual Edge-Fragment Resolution v2: Before / After", "", "| Measure | v1 audit | v2 audit |", "| --- | ---: | ---: |"]
    lines.extend([
        f"| Tiles | {payload['tiles']['before']} | {payload['tiles']['after']} |",
        f"| Tile-local labels | {payload['tile_local_labels']['before']} | {payload['tile_local_labels']['after']} |",
    ])
    for status, values in payload["coverage_status_counts"].items():
        lines.append(f"| `{status}` | {values['before']} | {values['after']} |")
    lines.extend(["", f"Readiness: `{payload['readiness']['before']}` -> `{payload['readiness']['after']}`.", ""])
    (report_root / "before_after_counts.md").write_text("\n".join(lines), encoding="utf-8")


def fallback_template(obb: ParentOBB, fields: list[str]) -> TileRow:
    row = {field: "" for field in fields}
    parent_id = obb.group_key.rsplit(":", 1)[-1]
    output_split = "train"  # These parents have no pre-existing assignment; retain source-train provenance.
    row.update({"dataset": obb.dataset, "source_split": obb.source_split, "split": output_split, "group_key": obb.group_key, "parent_id": parent_id, "parent_name": obb.parent_name, "source_image_name": obb.parent_name, "image_name": "", "tile_x": "0", "tile_y": "0", "tile_width": str(min(TILE_SIZE, obb.width)), "tile_height": str(min(TILE_SIZE, obb.height)), "tile_source": "annotation_recovery", "roi_mode": "annotation_recovery"})
    return TileRow(row)


def render_previews(cases: list[Case], obbs: dict[str, ParentOBB], tiles: dict[str, TileRow], parent_root: Path, report_root: Path, suppression_limit: int) -> None:
    for decision in (
        "intentional_edge_fragment_suppression",
        "valid_intentional_filter",
        "audit_false_positive",
        "repair_add_projected_label",
        "repair_add_or_adjust_recovery_tile",
        "unresolved_needs_manual_review",
    ):
        (report_root / "qc_previews" / decision).mkdir(parents=True, exist_ok=True)
    selected = [case for case in cases if case.decision not in {"intentional_edge_fragment_suppression"}]
    selected += [case for case in cases if case.decision == "intentional_edge_fragment_suppression"][:suppression_limit]
    Image.MAX_IMAGE_PIXELS = None
    for case in selected:
        tile = tiles.get(case.flagged_tile)
        obb = obbs.get(case.parent_annotation_id)
        if tile is None or obb is None:
            continue
        path = active_parent_path(tile, parent_root)
        if not path.is_file():
            continue
        with Image.open(path) as image:
            image = image.convert("RGB")
            scale = min(1.0, 1200 / max(image.size))
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        draw = ImageDraw.Draw(image)
        draw.rectangle((tile.x * scale, tile.y * scale, (tile.x + tile.width) * scale, (tile.y + tile.height) * scale), outline="#e53935", width=3)
        if case.best_existing_tile in tiles:
            best = tiles[case.best_existing_tile]
            draw.rectangle((best.x * scale, best.y * scale, (best.x + best.width) * scale, (best.y + best.height) * scale), outline="#20a35c", width=3)
        points = [(x * scale, y * scale) for x, y in obb.points]
        draw.line(points + [points[0]], fill="#ffd600", width=3)
        draw.rectangle((0, 0, min(image.width, 1000), 42), fill="white")
        draw.text((6, 6), f"{case.decision}  visible={case.flagged_visible_fraction:.3f}  best={case.best_existing_visible_fraction:.3f}", fill="black")
        target = report_root / "qc_previews" / case.decision / f"{case.issue_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True); image.save(target, quality=88)
        case.preview_path = str(target.relative_to(report_root))
    rendered = [report_root / case.preview_path for case in cases if case.preview_path]
    for page, start in enumerate(range(0, len(rendered), 24), 1):
        sheet = Image.new("RGB", (1440, 1560), "white")
        for offset, path in enumerate(rendered[start:start + 24]):
            with Image.open(path) as preview:
                preview.thumbnail((360, 260))
                sheet.paste(preview, ((offset % 4) * 360, (offset // 4) * 260))
        target = report_root / "qc_previews" / "summary_contact_sheets" / f"page_{page:02d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True); sheet.save(target, quality=88)


def main() -> int:
    a = args()
    if a.mode == "document":
        document_large_object_exclusions(a.output_report_root)
        return 0
    if a.write_before_after:
        write_before_after_summary(a.post_patch_audit_root, a.output_report_root / "post_v2_audit", a.output_report_root)
        return 0
    required = [a.tile_root / "dataset_index.csv", a.training_root / "dataset_index.csv", a.label_root / "instances.jsonl", a.post_patch_audit_root / "parent_annotation_coverage.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Missing residual resolver input: " + ", ".join(missing))
    existing = [path for path in output_paths(a) if path.exists()]
    if existing and not a.replace: raise FileExistsError("Output exists; use --replace: " + ", ".join(map(str, existing)))

    index_rows, tile_fields = read_csv(a.tile_root / "dataset_index.csv")
    training_rows, training_fields = read_csv(a.training_root / "dataset_index.csv")
    tiles = {row["image_name"]: row_tile(row) for row in index_rows}
    obbs = {item.instance_id: item for item in load_parent_obbs(a.label_root / "instances.jsonl", None, None)}
    by_group: dict[str, list[ParentOBB]] = defaultdict(list)
    for obb in obbs.values(): by_group[obb.group_key].append(obb)
    qa_tags = load_qa_tags(a.qa_manifest)
    coverage = json.loads((a.post_patch_audit_root / "parent_annotation_coverage.json").read_text(encoding="utf-8"))["issues"]
    residual = [item for item in coverage if item["issue_type"] == "covered_but_missing_label"]
    base_labels = {name: [line for line in (a.training_root / row["label_path"]).read_text(encoding="utf-8").splitlines() if line.strip()] for name, row in {row["image_name"]: row for row in training_rows}.items()}
    cases: list[Case] = []
    additions: list[dict[str, Any]] = []
    suppressions: list[dict[str, Any]] = []
    recovery_targets: set[str] = set()

    # Group cases by parent instance so only its best new usable view is added.
    grouped: dict[str, list[tuple[dict[str, Any], str, float, Any]]] = defaultdict(list)
    for issue in residual:
        record, obb = issue["coverage_record"], obbs.get(issue["annotation_id"])
        if obb is None: continue
        best = max(record["matched_tiles"], key=lambda item: item["retained_fraction"], default=None)
        for name in record["missing_tiles"]:
            tile = tiles[name]; intersection = obb.polygon.intersection(shapely_box(tile.x, tile.y, tile.x + tile.width, tile.y + tile.height)).area
            grouped[obb.instance_id].append((issue, name, intersection / obb.polygon.area if obb.polygon.area else 0.0, best))
    for instance_id, items in grouped.items():
        obb = obbs[instance_id]
        best_existing = max((float(item[3]["retained_fraction"]) for item in items if item[3]), default=0.0)
        best_entry = max((item[3] for item in items if item[3]), key=lambda item: item["retained_fraction"], default=None)
        valid_candidates: list[tuple[dict[str, Any], str, float, str]] = []
        for issue, name, visible, _ in items:
            tile = tiles[name]; projection = projection_for(obb, tile)
            line = projection_to_yolo_line(0, projection, tile.width, tile.height) if projection else ""
            valid = projection is not None and tile_polygon_from_line(line, tile) is not None
            if best_existing >= a.best_visible_threshold:
                decision, reason = "intentional_edge_fragment_suppression", "better_matching_label_at_or_above_0.90"
            elif not valid:
                decision, reason = "repair_add_or_adjust_recovery_tile", "projection_invalid"
            else:
                filter_reason = v9_filter_reason(line, tile, qa_tags.get(tile.parent_id, "unflagged"))
                if filter_reason:
                    decision, reason = "valid_intentional_filter", filter_reason
                # A relaxed comparison is only safe when it still represents
                # the same OBB. A 0.30 overlap can be a neighboring ship in a
                # dense marina, so use the duplicate-equivalence threshold.
                elif any(line_iou(existing, tile, line) >= 0.85 for existing in base_labels.get(name, [])):
                    decision, reason = "audit_false_positive", "relaxed_tile_local_overlap_match"
                else:
                    decision, reason = "repair_add_projected_label", "best_available_valid_projection"
                    valid_candidates.append((issue, name, visible, line))
            best_tile = best_entry["tile"] if best_entry else ""
            best_tile_row = tiles.get(best_tile)
            case = Case(f"{instance_id.replace(':', '_')}__{Path(name).stem}", obb.parent_name, instance_id, name, tile.split, [tile.x, tile.y, tile.x + tile.width, tile.y + tile.height], round(visible, 6), round(obb.polygon.intersection(shapely_box(tile.x, tile.y, tile.x + tile.width, tile.y + tile.height)).area, 4), valid, best_tile, [best_tile_row.x, best_tile_row.y, best_tile_row.x + best_tile_row.width, best_tile_row.y + best_tile_row.height] if best_tile_row else [], best_existing, str(best_entry.get("label_line", "")) if best_entry else "", float(best_entry.get("iou", 0.0)) if best_entry else 0.0, float(best_entry.get("intersection_over_smaller", 0.0)) if best_entry else 0.0, len([item for item in items if item[3]]), len(items), decision, "", reason)
            cases.append(case)
        if best_existing < a.best_visible_threshold and valid_candidates:
            chosen = max(valid_candidates, key=lambda item: (item[2], item[1]))
            for case in [item for item in cases if item.parent_annotation_id == instance_id and item.decision == "repair_add_projected_label"]:
                if case.flagged_tile == chosen[1]:
                    case.action_taken = "add_projected_label"; additions.append({"instance_id": instance_id, "tile_image": chosen[1], "label_line": chosen[3]})
                else:
                    case.decision = "intentional_edge_fragment_suppression"; case.action_taken = "ledger_suppression"; case.reason = "lower_ranked_overlap_after_best_new_projection"
        elif best_existing < a.best_visible_threshold and any(case.decision == "repair_add_or_adjust_recovery_tile" for case in cases if case.parent_annotation_id == instance_id):
            recovery_targets.add(instance_id)
        for case in [item for item in cases if item.parent_annotation_id == instance_id and item.decision == "intentional_edge_fragment_suppression"]:
            case.action_taken = "ledger_suppression"
            suppressions.append({"parent_image": case.parent_image, "parent_annotation_id": case.parent_annotation_id, "suppressed_tile": case.flagged_tile, "reason": case.reason, "flagged_visible_fraction": case.flagged_visible_fraction, "best_existing_tile": case.best_existing_tile, "best_existing_visible_fraction": case.best_existing_visible_fraction, "date_generated": datetime.now(timezone.utc).isoformat(), "script_version": "v2"})
        for case in [item for item in cases if item.parent_annotation_id == instance_id and not item.action_taken]:
            case.action_taken = "preserve_filter" if case.decision == "valid_intentional_filter" else "no_change" if case.decision == "audit_false_positive" else "create_recovery_tile" if case.decision == "repair_add_or_adjust_recovery_tile" else case.action_taken
            if case.decision == "valid_intentional_filter":
                suppressions.append({"parent_image": case.parent_image, "parent_annotation_id": case.parent_annotation_id, "suppressed_tile": case.flagged_tile, "reason": case.reason, "flagged_visible_fraction": case.flagged_visible_fraction, "best_existing_tile": case.best_existing_tile, "best_existing_visible_fraction": case.best_existing_visible_fraction, "date_generated": datetime.now(timezone.utc).isoformat(), "script_version": "v2", "decision": "valid_intentional_filter"})

    # The seven true no-tile parent OBBs join the deterministic recovery path.
    recovery_targets.update(item["annotation_id"] for item in coverage if item["issue_type"] == "uncovered_by_any_tile")
    a.output_report_root.mkdir(parents=True, exist_ok=True)
    render_previews(cases, obbs, tiles, a.active_parent_root, a.output_report_root, a.suppression_preview_count)
    fields = list(asdict(cases[0]).keys()) if cases else list(Case.__annotations__)
    write_csv(a.output_report_root / "residual_edge_cases.csv", [asdict(case) for case in cases], fields)
    write_json(a.output_report_root / "residual_edge_cases.json", [asdict(case) for case in cases])
    write_json(a.output_report_root / "edge_fragment_suppression_ledger.json", {"version": "v2", "entries": suppressions})
    write_json(a.output_report_root / "planned_repairs.json", {"label_additions": additions, "recovery_targets": sorted(recovery_targets)})

    if a.mode == "report":
        summary = {"mode": "report", "decision_counts": dict(Counter(case.decision for case in cases)), "atomic_cases": len(cases), "recovery_targets": len(recovery_targets)}
        write_json(a.output_report_root / "residual_edge_summary.json", summary)
        write_summary_markdown(a.output_report_root / "residual_edge_summary.md", summary)
        return 0

    for path in output_paths(a)[:3]:
        if path.exists(): shutil.rmtree(path)
    shutil.copytree(a.label_root, a.output_label_root)
    labels = {name: list(lines) for name, lines in base_labels.items()}
    for item in additions: labels[item["tile_image"]].append(item["label_line"])
    output_rows = [dict(row) for row in index_rows]
    rows_by_group: dict[str, list[TileRow]] = defaultdict(list)
    for tile in tiles.values(): rows_by_group[tile.group_key].append(tile)
    created: list[TileRow] = []
    Image.MAX_IMAGE_PIXELS = None
    for instance_id in sorted(recovery_targets):
        obb = obbs[instance_id]; candidates = rows_by_group.get(obb.group_key, [])
        template = candidates[0] if candidates else fallback_template(obb, tile_fields)
        parent_path = active_parent_path(template, a.active_parent_root)
        if not parent_path.is_file():
            failure_reason = "active_parent_image_unavailable"
        else:
            with Image.open(parent_path) as image: parent_size = image.size
            origin = choose_recovery_origin(obb, *parent_size)
            failure_reason = "rectangle_preserving_projection_unavailable_for_1024_crop" if origin is None else ""
        if failure_reason:
            for case in [item for item in cases if item.parent_annotation_id == instance_id]:
                is_large_object_exclusion = (
                    instance_id in VALID_BUT_UNREPRESENTABLE_LARGE_OBJECTS
                    and failure_reason == "rectangle_preserving_projection_unavailable_for_1024_crop"
                )
                case.decision = "valid_but_unrepresentable_large_object" if is_large_object_exclusion else "unresolved_needs_manual_review"
                case.action_taken = "intentional_fixed_tile_exclusion" if is_large_object_exclusion else "manual_review_required"
                case.reason = "valid_but_unrepresentable_large_object" if is_large_object_exclusion else failure_reason
                suppressions.append({
                    "parent_image": case.parent_image,
                    "parent_annotation_id": instance_id,
                    "suppressed_tile": case.flagged_tile,
                    "reason": case.reason,
                    "flagged_visible_fraction": case.flagged_visible_fraction,
                    "best_existing_tile": case.best_existing_tile,
                    "best_existing_visible_fraction": case.best_existing_visible_fraction,
                    "date_generated": datetime.now(timezone.utc).isoformat(),
                    "script_version": "v2",
                    "decision": case.decision,
                    "intentional_exclusion_scope": "fixed_1024_tile_training_only" if is_large_object_exclusion else "",
                })
            continue
        x, y, width, height = origin
        existing = next((tile for tile in candidates if (tile.x, tile.y, tile.width, tile.height) == (x, y, width, height)), None)
        if existing is not None:
            target = existing
        else:
            target = source_tile_row_for_recovery(obb, template, x, y, width, height)
            row = dict(target.row)
            prefix = "dota" if target.dataset == "dota" else target.dataset
            row["image_name"] = f"{prefix}__{Path(obb.parent_name).stem}_{x:04d}_{y:04d}_recovery_edge_v2.png"
            row["image_path"] = str(Path("images") / target.split / row["image_name"])
            row["tile_path"] = row["image_path"]
            row["label_path"] = str(Path("labels") / target.split / f"{Path(row['image_name']).stem}.txt")
            target = TileRow(row)
            rows_by_group[obb.group_key].append(target); created.append(target); labels[target.image_name] = []
            with Image.open(parent_path) as image:
                destination = a.output_tile_root / "images" / target.split / target.image_name
                destination.parent.mkdir(parents=True, exist_ok=True); image.crop((x, y, x + width, y + height)).save(destination)
            output_rows.append(dict(target.row))
        for candidate in by_group[obb.group_key]:
            projection = projection_for(candidate, target)
            if projection is None: continue
            line = projection_to_yolo_line(0, projection, target.width, target.height)
            filter_reason = v9_filter_reason(line, target, qa_tags.get(target.parent_id, "unflagged"))
            if tile_polygon_from_line(line, target) is None or filter_reason:
                if filter_reason:
                    visible = candidate.polygon.intersection(shapely_box(target.x, target.y, target.x + target.width, target.y + target.height)).area / candidate.polygon.area
                    if visible >= 0.5:
                        suppressions.append({"parent_image": candidate.parent_name, "parent_annotation_id": candidate.instance_id, "suppressed_tile": target.image_name, "reason": filter_reason, "flagged_visible_fraction": round(visible, 6), "best_existing_tile": "", "best_existing_visible_fraction": 0.0, "date_generated": datetime.now(timezone.utc).isoformat(), "script_version": "v2", "decision": "valid_intentional_filter"})
                continue
            if not any(line_iou(old, target, line) >= 0.85 for old in labels[target.image_name]): labels[target.image_name].append(line)
        # A recovery tile supersedes the original unresolved window for the
        # triggering annotation; future audits should not demand both views.
        for case in [item for item in cases if item.parent_annotation_id == instance_id]:
            suppressions.append({"parent_image": case.parent_image, "parent_annotation_id": instance_id, "suppressed_tile": case.flagged_tile, "reason": "superseded_by_deterministic_recovery_tile", "flagged_visible_fraction": case.flagged_visible_fraction, "best_existing_tile": target.image_name, "best_existing_visible_fraction": 1.0, "date_generated": datetime.now(timezone.utc).isoformat(), "script_version": "v2", "decision": "repair_add_or_adjust_recovery_tile"})

    # Materialize all v2 tile/training views from the final index and labels.
    a.output_tile_root.mkdir(parents=True, exist_ok=True); a.output_training_root.mkdir(parents=True, exist_ok=True)
    output_rows.sort(key=lambda row: (row["split"], row["image_name"]))
    for row in output_rows:
        tile = TileRow(row); image_rel = Path(row.get("image_path") or row.get("tile_path") or Path("images") / tile.split / tile.image_name)
        image_dest = a.output_tile_root / "images" / tile.split / tile.image_name
        if tile not in created: relative_link(a.tile_root / image_rel, image_dest)
        label_rel = Path("labels") / tile.split / f"{Path(tile.image_name).stem}.txt"
        row["image_path"], row["tile_path"], row["label_path"] = str(Path("images") / tile.split / tile.image_name), str(Path("images") / tile.split / tile.image_name), str(label_rel)
        label_dest = a.output_tile_root / label_rel; label_dest.parent.mkdir(parents=True, exist_ok=True); label_dest.write_text("\n".join(labels[tile.image_name]) + ("\n" if labels[tile.image_name] else ""), encoding="utf-8")
        relative_link(image_dest, a.output_training_root / "images" / tile.split / tile.image_name)
        train_label = a.output_training_root / label_rel; train_label.parent.mkdir(parents=True, exist_ok=True); train_label.write_text(label_dest.read_text(encoding="utf-8"), encoding="utf-8")
    for root, rows, fields_ in [(a.output_tile_root, output_rows, tile_fields), (a.output_training_root, output_rows, training_fields)]:
        all_fields = list(dict.fromkeys(fields_ + ["tile_path", "image_path", "label_path"]))
        write_csv(root / "dataset_index.csv", rows, all_fields)
        for split in ("train", "val"):
            subset = [row for row in rows if row["split"] == split]
            write_csv(root / "manifests" / f"{split}.csv", subset, all_fields)
            (root / "manifests" / f"{split}.txt").write_text("\n".join(str((root / row["image_path"]).absolute()) for row in subset) + "\n", encoding="utf-8")
        (root / "data.yaml").write_text(f"path: {root}\ntrain: manifests/train.txt\nval: manifests/val.txt\nnames:\n  0: ship\n", encoding="utf-8")
    for name in ("removed_annotations.csv", "kept_annotations.csv", "filter_report.csv", "dataset_manifest.json", "dataset_summary.json", "README.md"):
        source = a.training_root / name
        if source.is_file(): shutil.copy2(source, a.output_training_root / name)
    # Preserve v1 recovery provenance and append actual v2 crop metadata/maps.
    for name in ("grid_tile_policy.json",):
        source = a.tile_root / name
        if source.is_file(): shutil.copy2(source, a.output_tile_root / name)
    old_metadata = [json.loads(line) for line in (a.tile_root / "recovery_tile_metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    old_plans = [json.loads(line) for line in (a.tile_root / "recovery_tile_plan.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for tile in created:
        ids = [item.instance_id for item in by_group[tile.group_key] if item.instance_id in recovery_targets or projection_for(item, tile) is not None]
        metadata = {"image_name": tile.image_name, "image_path": tile.row["image_path"], "label_path": tile.row["label_path"], "dataset": tile.dataset, "source_split": tile.source_split, "split": tile.split, "group_key": tile.group_key, "parent_id": tile.parent_id, "parent_image_filename": tile.row.get("parent_name", ""), "tile_source": "annotation_recovery", "recovery_reason": "residual_edge_or_uncovered_repair_v2", "covered_instance_ids": ids, "covered_obb_count": len(ids), "tile_xyxy_parent": [tile.x, tile.y, tile.x + tile.width, tile.y + tile.height], "outside_manual_grid": True}
        old_metadata.append(metadata)
        old_plans.append({"parent_group_key": tile.group_key, "parent_image_filename": tile.row.get("parent_name", ""), "tile_source": "annotation_recovery", "recovery_reason": "residual_edge_or_uncovered_repair_v2", "covered_instance_ids": ids, "tile_xyxy_parent": metadata["tile_xyxy_parent"]})
    for name, values in (("recovery_tile_metadata.jsonl", old_metadata), ("recovery_tile_plan.jsonl", old_plans)):
        (a.output_tile_root / name).write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")
    map_rows, map_fields = read_csv(a.tile_root / "tile_parent_map.csv")
    write_csv(a.output_tile_root / "tile_parent_map.csv", [{field: row.get(field, "") for field in map_fields} for row in output_rows], map_fields)
    # Recovery-side filter and replacement decisions are only known after the
    # final crop contents are evaluated, so rewrite the ledger atomically here.
    deduped_suppressions = {(entry["parent_annotation_id"], entry["suppressed_tile"]): entry for entry in suppressions}
    write_json(a.output_report_root / "edge_fragment_suppression_ledger.json", {"version": "v2", "entries": list(deduped_suppressions.values())})
    # Patch-time recovery failures are only known after the final target window
    # is selected. Rewrite case artifacts and add the corresponding QC previews.
    render_previews(cases, obbs, tiles, a.active_parent_root, a.output_report_root, a.suppression_preview_count)
    write_csv(a.output_report_root / "residual_edge_cases.csv", [asdict(case) for case in cases], fields)
    write_json(a.output_report_root / "residual_edge_cases.json", [asdict(case) for case in cases])
    write_exception_queues(a.output_report_root, cases)
    summary = {"mode": "patch", "atomic_cases": len(cases), "decision_counts": dict(Counter(case.decision for case in cases)), "labels_added": len(additions), "recovery_tiles_created": len(created), "recovery_targets": len(recovery_targets), "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(a.output_report_root / "residual_edge_summary.json", summary)
    write_summary_markdown(a.output_report_root / "residual_edge_summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
