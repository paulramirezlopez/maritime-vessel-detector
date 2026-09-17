#!/usr/bin/env python3
"""Promote validated audit-fixed v2 roots through stable canonical symlinks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/dataset_promotion"
LINEAGE_ROOT = ROOT / "reports/experiment_lineage"
AUDIT_ROOT = ROOT / "reports/tile_coverage_recovery_residual_edge_v2"

ROOTS = {
    "tile": {
        "active": ROOT / "data/tiled/grid_current_with_recovery",
        "versioned": ROOT / "data/tiled/grid_current_with_recovery_audit_fixed_v2",
        "legacy": ROOT / "data/tiled/grid_current_with_recovery_pre_audit_v9",
    },
    "parent_labels": {
        "active": ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery",
        "versioned": ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery_audit_fixed_v2",
        "legacy": ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery_pre_audit_v9",
    },
    "training": {
        "active": ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3",
        "versioned": ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3_audit_fixed_v2",
        "legacy": ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3_pre_audit_v9",
    },
}
EXCLUDED_PARENTS = {"P1277", "P1591", "P1927", "P2662", "P2769"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate the promotion without changing files.")
    mode.add_argument("--apply", action="store_true", help="Archive current roots and create stable active symlinks.")
    mode.add_argument("--repair-links", action="store_true", help="Repair versioned image links after an interrupted promotion, then finalize reports.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines, ""]), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("names", {}).get(0) != "ship":
        raise ValueError(f"Invalid one-class dataset YAML: {path}")
    root = Path(payload["path"])
    for split in ("train", "val"):
        manifest = root / payload[split]
        if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
            raise ValueError(f"Missing or empty {split} manifest: {manifest}")
    return payload


def valid_yolo_obb(line: str) -> bool:
    values = line.split()
    if len(values) != 9:
        return False
    try:
        numbers = [float(value) for value in values]
    except ValueError:
        return False
    return int(numbers[0]) == 0 and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in numbers[1:])


def verify_dataset() -> dict[str, Any]:
    tile_root = ROOTS["tile"]["versioned"]
    training_root = ROOTS["training"]["versioned"]
    audit = json.loads((AUDIT_ROOT / "post_v2_audit/audit_summary.json").read_text(encoding="utf-8"))
    statuses = audit["coverage"]["status_counts"]
    if audit.get("readiness") not in {"safe_to_continue", "safe_with_minor_cautions"}:
        raise ValueError(f"Audit readiness does not permit promotion: {audit.get('readiness')}")
    if statuses.get("covered_but_missing_label", 0) or statuses.get("uncovered_by_any_tile", 0):
        raise ValueError("v2 audit has unresolved coverage failures.")
    required = [
        tile_root / "tile_parent_map.csv",
        tile_root / "recovery_tile_metadata.jsonl",
        tile_root / "recovery_tile_plan.jsonl",
        AUDIT_ROOT / "edge_fragment_suppression_ledger.json",
        AUDIT_ROOT / "valid_but_unrepresentable_large_object_ledger.json",
        ROOTS["parent_labels"]["versioned"] / "instances.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing promotion input: " + ", ".join(missing))
    tile_yaml = verify_yaml(tile_root / "data.yaml")
    train_yaml = verify_yaml(training_root / "data.yaml")
    rows = read_csv(training_root / "dataset_index.csv")
    windows: set[tuple[str, str, str, str, str]] = set()
    bad_labels = 0
    for row in rows:
        image = training_root / row["image_path"]
        label = training_root / row["label_path"]
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing paired artifact for {row['image_name']}")
        window = (row["group_key"], row["tile_x"], row["tile_y"], row["tile_width"], row["tile_height"])
        if window in windows:
            raise ValueError(f"Duplicate tile window: {window}")
        windows.add(window)
        if row["parent_id"] in EXCLUDED_PARENTS:
            raise ValueError(f"Excluded parent remains in v2: {row['parent_id']}")
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip() and not valid_yolo_obb(line):
                bad_labels += 1
    if bad_labels:
        raise ValueError(f"Found {bad_labels} invalid YOLO OBB records.")
    return {
        "audit_readiness": audit["readiness"],
        "audit_coverage_statuses": statuses,
        "tiles": len(rows),
        "labels": audit["inventory"]["annotation_count"]["total"],
        "splits": dict(Counter(row["split"] for row in rows)),
        "sources": dict(Counter(row["dataset"] for row in rows)),
        "tile_yaml": str(tile_root / "data.yaml"),
        "training_yaml": str(training_root / "data.yaml"),
        "train_manifest": str(training_root / train_yaml["train"]),
        "val_manifest": str(training_root / train_yaml["val"]),
        "test_manifest": None,
        "tile_map": str(tile_root / "tile_parent_map.csv"),
        "recovery_metadata": str(tile_root / "recovery_tile_metadata.jsonl"),
        "recovery_plan": str(tile_root / "recovery_tile_plan.jsonl"),
        "edge_suppression_ledger": str(AUDIT_ROOT / "edge_fragment_suppression_ledger.json"),
        "large_object_exclusion_ledger": str(AUDIT_ROOT / "valid_but_unrepresentable_large_object_ledger.json"),
        "tile_data_yaml_path_field": tile_yaml["path"],
    }


def metric_row(run_dir: Path, args: dict[str, Any]) -> dict[str, Any]:
    results = run_dir / "results.csv"
    metrics: dict[str, Any] = {}
    if results.is_file():
        with results.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            best = max(rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", "-inf") or "-inf"))
            metrics = {key: best.get(key) for key in ("epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)", "fitness")}
    data = str(args.get("data", ""))
    if "audit_fixed" in data:
        status = "historical_post_audit"
    elif "v9_threshold_aggressive_keep_bad_seed3" in data or "grid_current_with_recovery" in data:
        status = "historical_pre_audit"
    else:
        status = "unknown"
    return {
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "dataset_yaml_used": data or None,
        "tile_root_used": "data/tiled/grid_current_with_recovery_pre_audit_v9" if status == "historical_pre_audit" else None,
        "label_root_used": "data/annotations/second_pass_tiled/obbs_complete_with_recovery_pre_audit_v9" if status == "historical_pre_audit" else None,
        "imgsz": args.get("imgsz"), "model": args.get("model"), "seed": args.get("seed"),
        "epochs": args.get("epochs"), "patience": args.get("patience"),
        "augmentation_settings": {key: args.get(key) for key in ("degrees", "flipud", "fliplr", "scale", "translate", "mosaic", "hsv_h", "hsv_s", "hsv_v")},
        "filtering_policy": "v9 aggressive keep-bad" if "v9_threshold" in data else None,
        "box_padding_policy": "unknown", "status": status, "metrics": metrics,
    }


def experiment_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for args_path in sorted((ROOT / "models").rglob("args.yaml")):
        try:
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(args, dict):
            rows.append(metric_row(args_path.parent, args))
    return rows


def base_config(best_known: bool) -> dict[str, Any]:
    config = {
        "model": "models/yolo11m-obb.pt",
        "data": "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3/data.yaml",
        "project": "models/audit_fixed_confirmations",
        "name": "audit_fixed_best_known_1280_seed3" if best_known else "audit_fixed_baseline_1280_seed3",
        "epochs": 40, "imgsz": 1280, "batch": 4, "device": "0", "workers": 8,
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
        "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0, "cos_lr": True, "patience": 0, "seed": 3,
        "deterministic": True, "amp": True, "cache": False, "close_mosaic": 10,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 180.0 if best_known else 0.0,
        "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
        "flipud": 0.5 if best_known else 0.0, "fliplr": 0.5, "mosaic": 1.0,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "auto_augment": "randaugment",
        "erasing": 0.4, "val": True, "split": "val", "iou": 0.7, "max_det": 300,
        "save": True, "save_period": 10, "plots": True,
    }
    return config


def write_configs() -> list[str]:
    paths: list[str] = []
    targets = [
        (ROOT / "configs/training/active_second_pass_audit_fixed_1280_baseline.yaml", base_config(False)),
        (ROOT / "configs/training/active_second_pass_audit_fixed_1280_best_known.yaml", base_config(True)),
    ]
    for size in (1024, 1280):
        config = base_config(False); config["imgsz"] = size; config["name"] = f"audit_fixed_resolution_{size}_seed3"; config["project"] = "models/audit_fixed_resolution_confirmation"
        targets.append((ROOT / f"configs/experiments/audit_fixed_resolution_ablation/imgsz_{size}.yaml", config))
    for name, best in (("baseline_1280", False), ("geo_rotate_flip_1280", True)):
        config = base_config(best); config["name"] = f"audit_fixed_{name}_seed3"; config["project"] = "models/audit_fixed_augmentation_confirmation"
        targets.append((ROOT / f"configs/experiments/audit_fixed_augmentation_ablation/{name}.yaml", config))
    for size, model in (("yolo11m_1280", "models/yolo11m-obb.pt"), ("yolo11l_1280", "models/yolo11l-obb.pt")):
        config = base_config(True); config["model"] = model; config["name"] = f"audit_fixed_{size}_seed3"; config["project"] = "models/audit_fixed_model_capacity"
        targets.append((ROOT / f"configs/experiments/audit_fixed_model_capacity/{size}.yaml", config))
    for path, config in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(str(path))
    return paths


def write_lineage(inventory: list[dict[str, Any]]) -> None:
    LINEAGE_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(LINEAGE_ROOT / "experiment_inventory.json", {"runs": inventory, "historical_path_mapping": {str(ROOTS["training"]["active"]): str(ROOTS["training"]["legacy"]), str(ROOTS["tile"]["active"]): str(ROOTS["tile"]["legacy"]), str(ROOTS["parent_labels"]["active"]): str(ROOTS["parent_labels"]["legacy"])}})
    lines = ["Historical runs are immutable; this inventory maps their original pre-audit root references to archived roots.", "", "| Run | Status | Dataset YAML | imgsz | Seed | mAP50-95 |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in inventory:
        lines.append(f"| {row['run_name']} | `{row['status']}` | `{row['dataset_yaml_used'] or ''}` | {row['imgsz'] or ''} | {row['seed'] or ''} | {row['metrics'].get('metrics/mAP50-95(B)', '')} |")
    write_markdown(LINEAGE_ROOT / "experiment_inventory.md", "Experiment Inventory", lines)
    findings = [
        ("OBB geometry", "original boxes; no padding", "padding ablation reports", "historical_pre_audit", "high", "active", "yes", "Expansion did not improve the controlled results."),
        ("filtering", "v9 aggressive keep-bad policy", "v9 dataset manifest", "historical_pre_audit", "high", "active", "yes", "small=aggressive; low=low_aggressive; xview=xview_aggressive; remove_bad=false."),
        ("resolution", "imgsz=1280", "resolution ablation", "historical_pre_audit", "high", "active", "yes", "1280 exceeded 1024 and 768; 1536 was scrapped."),
        ("augmentation", "degrees=180; flipud=0.5; fliplr=0.5", "three-seed geometric augmentation confirmation", "historical_pre_audit", "medium", "best_known_hypothesis", "yes", "Mean mAP50-95 improvement was 0.00648 across seeds 0, 1, and 3."),
    ]
    payload = {"settings": [dict(zip(("setting_group", "best_known_setting", "evidence_run", "evidence_dataset_version", "confidence", "active_status", "requires_rerun_on_audit_fixed_dataset", "notes"), item)) for item in findings]}
    write_json(LINEAGE_ROOT / "best_known_settings_carry_forward.json", payload)
    lines = ["| Setting group | Best-known setting | Evidence | Status | Requires v2 rerun |", "| --- | --- | --- | --- | --- |"]
    for row in payload["settings"]:
        lines.append(f"| {row['setting_group']} | `{row['best_known_setting']}` | {row['evidence_run']} | `{row['active_status']}` | {row['requires_rerun_on_audit_fixed_dataset']} |")
    write_markdown(LINEAGE_ROOT / "best_known_settings_carry_forward.md", "Best-Known Settings Carry-Forward", lines)
    write_markdown(LINEAGE_ROOT / "audit_fixed_rerun_plan.md", "Audit-Fixed Rerun Plan", [
        "1. Run the conservative 1280 baseline on v2.",
        "2. Confirm the inherited geometric augmentation candidate on v2.",
        "3. If the baseline shifts materially, compare 1024 and 1280 only.",
        "4. Compare YOLO11m and YOLO11l at the confirmed augmentation policy.",
        "5. Defer scheduler and hyperparameter tuning until those choices are stable.",
    ])


def activate_roots() -> list[dict[str, str]]:
    ledger: list[dict[str, str]] = []
    for kind, mapping in ROOTS.items():
        active, versioned, legacy = mapping["active"], mapping["versioned"], mapping["legacy"]
        if legacy.exists() or legacy.is_symlink():
            raise FileExistsError(f"Legacy archive path already exists: {legacy}")
        if active.is_symlink():
            raise RuntimeError(f"Active path is already a symlink; refusing to replace: {active}")
        os.replace(active, legacy)
        active.symlink_to(os.path.relpath(versioned, active.parent))
        ledger.append({"kind": kind, "old_active": str(active), "archived": str(legacy), "new_target": str(versioned), "strategy": "relative_symlink"})
    return ledger


def current_ledger() -> list[dict[str, str]]:
    ledger: list[dict[str, str]] = []
    for kind, mapping in ROOTS.items():
        active, versioned, legacy = mapping["active"], mapping["versioned"], mapping["legacy"]
        if not active.is_symlink() or active.resolve() != versioned.resolve() or not legacy.is_dir():
            raise RuntimeError(f"Promotion is not in the expected interrupted state for {kind}: {active}")
        ledger.append({"kind": kind, "old_active": str(active), "archived": str(legacy), "new_target": str(versioned), "strategy": "relative_symlink"})
    return ledger


def repair_versioned_image_links() -> int:
    """Break v2->v1->old-canonical chains after the canonical pointer moves."""
    versioned = ROOTS["tile"]["versioned"] / "images"
    legacy = ROOTS["tile"]["legacy"] / "images"
    repaired = 0
    for image in versioned.rglob("*"):
        if not image.is_symlink():
            continue
        candidate = legacy / image.relative_to(versioned)
        if not candidate.is_file():
            continue  # v1/v2 recovery tiles are physical or remain valid through v1.
        image.unlink()
        image.symlink_to(os.path.relpath(candidate, image.parent))
        repaired += 1
    return repaired


def finalize_promotion(inventory: list[dict[str, Any]], ledger: list[dict[str, str]], repaired_links: int) -> dict[str, Any]:
    post_validation = verify_dataset()
    config_paths = write_configs()
    write_lineage(inventory)
    manifest = {"dataset_version": "audit_fixed_v2", "annotation_stage": "second_pass_tiled", "geometry_type": "YOLO OBB", "tiling_method": "grid_current_with_recovery", "imgsz_selected_for_training": 1280, "source_datasets": ["dota", "hrsc", "xview"], "audit_status": post_validation["audit_readiness"], "promotion_date": datetime.now(timezone.utc).isoformat(), "promotion_report": str(REPORT_ROOT / "audit_fixed_promotion.md"), "active_tile_root": str(ROOTS["tile"]["active"]), "active_label_root": str(ROOTS["parent_labels"]["active"]), "active_training_root": str(ROOTS["training"]["active"]), "legacy_previous_tile_root": str(ROOTS["tile"]["legacy"]), "legacy_previous_label_root": str(ROOTS["parent_labels"]["legacy"]), "legacy_previous_training_root": str(ROOTS["training"]["legacy"]), "best_known_settings_manifest": str(LINEAGE_ROOT / "best_known_settings_carry_forward.json"), "historical_experiment_inventory": str(LINEAGE_ROOT / "experiment_inventory.json"), "edge_fragment_suppression_ledger": post_validation["edge_suppression_ledger"], "large_object_exclusion_ledger": post_validation["large_object_exclusion_ledger"], "test_manifest": None}
    write_json(ROOT / "data/active_dataset_manifest.json", manifest)
    promotion = {"promotion_date": manifest["promotion_date"], "reason": "validated audit-fixed v2 replaces the error-prone active base", "audit_report": str(AUDIT_ROOT / "post_v2_audit/audit_summary.json"), "strategy": "relative_symlink", "validation": post_validation, "migration_ledger": ledger, "repaired_versioned_image_links": repaired_links, "new_active_configs": config_paths, "historical_runs_preserved": len(inventory)}
    write_json(REPORT_ROOT / "audit_fixed_promotion.json", promotion)
    lines = ["Validated audit-fixed v2 was promoted through stable relative symlinks.", "", "| Root | Archived legacy root | Active target |", "| --- | --- | --- |"]
    for item in ledger:
        lines.append(f"| {item['kind']} | `{item['archived']}` | `{item['new_target']}` |")
    lines.extend(["", f"Tiles: `{post_validation['tiles']}`. Tile-local labels: `{post_validation['labels']}`. Audit readiness: `{post_validation['audit_readiness']}`.", f"Repaired `{repaired_links}` inherited image symlinks to point directly at the archived legacy images.", "Historical model and report artifacts were not modified."])
    write_markdown(REPORT_ROOT / "audit_fixed_promotion.md", "Audit-Fixed Dataset Promotion", lines)
    return promotion


def main() -> int:
    args = parse_args()
    if args.repair_links:
        inventory = experiment_inventory()
        ledger = current_ledger()
        repaired_links = repair_versioned_image_links()
        promotion = finalize_promotion(inventory, ledger, repaired_links)
        print(json.dumps(promotion, indent=2, sort_keys=True))
        return 0
    validation = verify_dataset()
    preflight = {"validated_at": datetime.now(timezone.utc).isoformat(), "mode": "dry_run" if args.dry_run else "apply", "validation": validation, "roots": {key: {name: str(path) for name, path in value.items()} for key, value in ROOTS.items()}}
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    inventory = experiment_inventory()
    ledger = activate_roots()
    repaired_links = repair_versioned_image_links()
    promotion = finalize_promotion(inventory, ledger, repaired_links)
    print(json.dumps(promotion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
