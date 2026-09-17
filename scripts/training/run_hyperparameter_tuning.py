#!/usr/bin/env python3
"""Prepare and smoke-test bounded audit-fixed YOLO OBB hyperparameter trials.

Full training is intentionally manual.  This runner validates the promoted
dataset contract before it writes any experiment configuration so tuning cannot
silently fall back to the historical pre-audit training view.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from scripts.training.run_resolution_ablation import GpuMemorySampler, dataset_fingerprint, git_revision, last_metrics
except ModuleNotFoundError:  # Direct script execution has only scripts/training on sys.path.
    from run_resolution_ablation import GpuMemorySampler, dataset_fingerprint, git_revision, last_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_MANIFEST = REPO_ROOT / "data/active_dataset_manifest.json"
BASELINE_CONFIG = REPO_ROOT / "configs/training/active_second_pass_audit_fixed_1280_baseline.yaml"
HISTORICAL_BASELINE_RUN = REPO_ROOT / "models/resolution_ablation/imgsz_1280"
CONFIG_ROOT = REPO_ROOT / "configs/experiments/hyperparameter_tuning"
REPORT_ROOT = REPO_ROOT / "reports/hyperparameter_tuning"
PROJECT_ROOT = REPO_ROOT / "models/hyperparameter_tuning"
SMOKE_PROJECT_ROOT = REPO_ROOT / "models/hyperparameter_tuning_smoke"
EDGE_LEDGER = REPO_ROOT / "reports/tile_coverage_recovery_residual_edge_v2/edge_fragment_suppression_ledger.json"
LARGE_OBJECT_LEDGER = (
    REPO_ROOT / "reports/tile_coverage_recovery_residual_edge_v2/valid_but_unrepresentable_large_object_ledger.json"
)
EXCLUDED_PARENTS = ("P1277", "P1591", "P1927", "P2662", "P2769")

BASELINE_NAME = "hpo_00_baseline_confirm"
BASELINE_OUTPUT_NAME = "audit_fixed_1280_baseline_confirm"
ACTIVE_VARIANTS = (
    BASELINE_NAME,
    "hpo_01_cos_lr",
    "hpo_02_lr_low",
    "hpo_03_lr_high",
    "hpo_04_lrf_lower",
    "hpo_05_weight_decay_low",
    "hpo_06_weight_decay_high",
    "hpo_07_warmup_adjusted",
    "hpo_08_close_mosaic",
    "hpo_09_best_aug_carry_forward",
)
FROZEN_KEYS = (
    "model",
    "data",
    "imgsz",
    "epochs",
    "batch",
    "device",
    "workers",
    "optimizer",
    "momentum",
    "warmup_momentum",
    "warmup_bias_lr",
    "patience",
    "seed",
    "deterministic",
    "amp",
    "cache",
    "translate",
    "scale",
    "shear",
    "perspective",
    "fliplr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "auto_augment",
    "erasing",
    "val",
    "split",
    "iou",
    "max_det",
)

VARIANTS: dict[str, dict[str, Any]] = {
    BASELINE_NAME: {
        "hypothesis": "Confirm the compact 1280 recipe on the promoted audit-fixed dataset.",
        "changes": {},
        "kind": "baseline_confirmation",
        "output_name": BASELINE_OUTPUT_NAME,
    },
    "hpo_01_cos_lr": {
        "hypothesis": "Test the inverse scheduler after the pre-audit cosine baseline.",
        "changes": {"cos_lr": False},
        "kind": "single_factor_trial",
    },
    "hpo_02_lr_low": {
        "hypothesis": "A lower initial learning rate may improve stable OBB convergence.",
        "changes": {"lr0": 0.005},
        "kind": "single_factor_trial",
    },
    "hpo_03_lr_high": {
        "hypothesis": "A moderately higher initial learning rate may reach a stronger basin within 40 epochs.",
        "changes": {"lr0": 0.015},
        "kind": "single_factor_trial",
    },
    "hpo_04_lrf_lower": {
        "hypothesis": "A lower final learning-rate factor may improve late-epoch refinement.",
        "changes": {"lrf": 0.005},
        "kind": "single_factor_trial",
    },
    "hpo_05_weight_decay_low": {
        "hypothesis": "Less regularization may preserve small-vessel detail.",
        "changes": {"weight_decay": 0.00025},
        "kind": "single_factor_trial",
    },
    "hpo_06_weight_decay_high": {
        "hypothesis": "More regularization may improve generalization across imagery sources.",
        "changes": {"weight_decay": 0.001},
        "kind": "single_factor_trial",
    },
    "hpo_07_warmup_adjusted": {
        "hypothesis": "A modestly longer warmup may stabilize the fixed batch-4 optimization path.",
        "changes": {"warmup_epochs": 5.0},
        "kind": "single_factor_trial",
    },
    "hpo_08_close_mosaic": {
        "hypothesis": "Closing mosaic later may retain more normal-image adaptation before validation.",
        "changes": {"close_mosaic": 5},
        "kind": "single_factor_trial",
    },
    "hpo_09_best_aug_carry_forward": {
        "hypothesis": "Confirm the pre-audit three-seed-supported orientation policy on audit-fixed data.",
        "changes": {"degrees": 180.0, "flipud": 0.5},
        "kind": "inherited_pre_audit_hypothesis",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "validate", "smoke"), default="prepare")
    parser.add_argument("--variants", nargs="+", choices=ACTIVE_VARIANTS, default=list(ACTIVE_VARIANTS))
    parser.add_argument("--baseline-config", type=Path, default=BASELINE_CONFIG)
    parser.add_argument("--historical-baseline-run", type=Path, default=HISTORICAL_BASELINE_RUN)
    parser.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--smoke-project-root", type=Path, default=SMOKE_PROJECT_ROOT)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--skip-dataloader-check", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_dependencies() -> dict[str, str]:
    """Fail before configuration output when the audit/training environment is incomplete."""

    try:
        import shapely
        import torch
        import ultralytics
    except ImportError as error:
        raise RuntimeError(
            "The selected interpreter lacks an audit-fixed HPO dependency. "
            "Activate cv_practice_env before running this command."
        ) from error
    return {"shapely": shapely.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__}


def parse_obb_label(path: Path) -> tuple[int, list[str]]:
    valid = 0
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            errors.append(f"{path}:{line_number}: expected 9 fields, found {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric OBB record")
            continue
        if class_id != 0 or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{path}:{line_number}: invalid class or normalized coordinate")
            continue
        points = list(zip(values[::2], values[1::2]))
        area_twice = abs(sum(x1 * y2 - y1 * x2 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])))
        if area_twice <= 1e-12:
            errors.append(f"{path}:{line_number}: degenerate OBB polygon")
            continue
        valid += 1
    return valid, errors


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stale_sam3_references(root: Path) -> list[str]:
    """Find only absolute SAM3 paths, not historical metadata column names."""

    findings: list[str] = []
    pattern = re.compile(r"(?:/[^\s\"']*)?(?:sam3|sam_3)(?:/[^\s\"']*)?", re.IGNORECASE)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".yaml", ".yml", ".txt", ".md"}:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            for match in pattern.finditer(line):
                value = match.group(0)
                if value.startswith("/") or ":\\" in value:
                    findings.append(f"{path}:{line_number}:{value}")
    return findings


def validate_dataloader(data_yaml: Path, batch: int, imgsz: int) -> dict[str, Any]:
    """Load one Ultralytics OBB batch without beginning a training run."""

    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    data = check_det_dataset(str(data_yaml))
    hyp = deepcopy(DEFAULT_CFG)
    hyp.imgsz = imgsz
    hyp.mosaic = 0.0
    hyp.mixup = 0.0
    hyp.copy_paste = 0.0
    dataset = YOLODataset(
        img_path=data["train"],
        imgsz=imgsz,
        batch_size=batch,
        augment=False,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="HPO preflight: ",
        task="obb",
        data=data,
        fraction=1.0,
    )
    loaded = [dataset[index] for index in range(min(batch, len(dataset)))]
    batch_data = dataset.collate_fn(loaded)
    return {
        "dataset_items": len(dataset),
        "loaded_items": len(loaded),
        "image_tensor_shape": list(batch_data["img"].shape),
        "has_obb_instances": "bboxes" in batch_data,
    }


def validate_active_dataset(*, dataloader_check: bool) -> dict[str, Any]:
    """Validate the promoted, symlinked dataset contract without changing it."""

    dependency_versions = require_dependencies()
    manifest = read_json(ACTIVE_MANIFEST)
    tile_root = Path(manifest["active_tile_root"])
    label_root = Path(manifest["active_label_root"])
    training_root = Path(manifest["active_training_root"])
    stable_roots = {
        "tile": REPO_ROOT / "data/tiled/grid_current_with_recovery",
        "label": REPO_ROOT / "data/annotations/second_pass_tiled/obbs_complete_with_recovery",
        "training": REPO_ROOT / "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3",
    }
    expected = {"tile": tile_root, "label": label_root, "training": training_root}
    errors: list[str] = []
    for key, stable in stable_roots.items():
        if not stable.is_symlink():
            errors.append(f"active {key} path is not a compatibility symlink: {stable}")
        if not stable.exists() or stable.resolve() != expected[key].resolve():
            errors.append(f"active {key} path does not resolve to promoted root: {stable}")
    for root in (tile_root, label_root, training_root):
        if not root.is_dir():
            errors.append(f"active root missing: {root}")

    data_yaml = training_root / "data.yaml"
    if not data_yaml.is_file():
        errors.append(f"dataset YAML missing: {data_yaml}")
        data = {}
    else:
        data = read_yaml(data_yaml)
        if data.get("names") != {0: "ship"}:
            errors.append(f"dataset class mapping must be {{0: 'ship'}}, got {data.get('names')!r}")
        yaml_root = Path(str(data.get("path", ""))).resolve()
        if yaml_root != training_root.resolve():
            errors.append(f"dataset YAML path does not resolve to active training root: {yaml_root}")

    split_summary: dict[str, dict[str, int]] = {}
    label_errors: list[str] = []
    for split in ("train", "val"):
        images = sorted((training_root / "images" / split).glob("*"))
        labels = sorted((training_root / "labels" / split).glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        split_summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "orphan_images": len(image_stems - label_stems),
            "orphan_labels": len(label_stems - image_stems),
        }
        if image_stems != label_stems:
            errors.append(f"{split} image/label parity failure")
        for path in labels:
            _, failures = parse_obb_label(path)
            label_errors.extend(failures[: max(0, 21 - len(label_errors))])
            if len(label_errors) >= 20:
                break
    errors.extend(label_errors)

    tile_map = tile_root / "tile_parent_map.csv"
    if not tile_map.is_file():
        errors.append(f"tile map missing: {tile_map}")
        tile_rows: list[dict[str, str]] = []
    else:
        tile_rows = rows(tile_map)
    windows: dict[tuple[str, str, str, str, str], list[str]] = {}
    for row in tile_rows:
        key = (row.get("group_key", ""), row.get("tile_x", ""), row.get("tile_y", ""), row.get("tile_width", ""), row.get("tile_height", ""))
        windows.setdefault(key, []).append(row.get("tile_path", ""))
    duplicates = {key: paths for key, paths in windows.items() if len(paths) > 1}
    if duplicates:
        errors.append(f"duplicate tile windows found: {len(duplicates)}")

    index_path = training_root / "dataset_index.csv"
    index_rows = rows(index_path) if index_path.is_file() else []
    if not index_rows:
        errors.append(f"dataset index missing or empty: {index_path}")
    excluded_hits = {
        parent: sum(parent in json.dumps(row, sort_keys=True) for row in index_rows + tile_rows)
        for parent in EXCLUDED_PARENTS
    }
    for parent, count in excluded_hits.items():
        if count:
            errors.append(f"excluded parent remains in active metadata: {parent} ({count} references)")

    for ledger in (EDGE_LEDGER, LARGE_OBJECT_LEDGER):
        if not ledger.is_file():
            errors.append(f"required audit ledger missing: {ledger}")
    if LARGE_OBJECT_LEDGER.is_file():
        large_entries = read_json(LARGE_OBJECT_LEDGER).get("entries", [])
        large_names = {entry.get("parent_image") for entry in large_entries if isinstance(entry, dict)}
        for parent in ("P2641.png", "P2789.png"):
            if parent not in large_names:
                errors.append(f"large-object intentional exclusion missing: {parent}")

    sam3_paths = stale_sam3_references(training_root)
    if sam3_paths:
        errors.append(f"stale absolute SAM3 references found: {sam3_paths[:3]}")
    dataloader = validate_dataloader(data_yaml, batch=2, imgsz=1280) if dataloader_check and not errors else None
    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "dependency_versions": dependency_versions,
        "active_manifest": str(ACTIVE_MANIFEST),
        "roots": {key: str(value) for key, value in expected.items()},
        "dataset_yaml": str(data_yaml),
        "dataset": dataset_fingerprint(data_yaml) if data_yaml.is_file() else None,
        "split_summary": split_summary,
        "tile_map_rows": len(tile_rows),
        "duplicate_tile_windows": len(duplicates),
        "excluded_parent_hits": excluded_hits,
        "dataloader": dataloader,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    if errors:
        raise RuntimeError("Active audit-fixed dataset validation failed:\n- " + "\n- ".join(errors))
    return report


def historical_reference(path: Path) -> dict[str, Any]:
    path = resolve(path)
    args_path = path / "args.yaml"
    results_path = path / "results.csv"
    checkpoint = path / "weights/best.pt"
    if not (args_path.is_file() and results_path.is_file() and checkpoint.is_file()):
        raise FileNotFoundError(f"Historical compact baseline is incomplete: {path}")
    args = read_yaml(args_path)
    rows_data = rows(results_path)
    best = max(rows_data, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "run_dir": str(path),
        "checkpoint": str(checkpoint),
        "args": args,
        "best_epoch": int(float(best["epoch"])),
        "stop_epoch": int(float(rows_data[-1]["epoch"])),
        "metrics": {
            "mAP50_95": float(best["metrics/mAP50-95(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
        },
        "status": "historical_pre_audit",
    }


def read_baseline(path: Path, project_root: Path) -> tuple[dict[str, Any], Path]:
    config = read_yaml(resolve(path))
    data_yaml = resolve(config["data"])
    required = {"imgsz": 1280, "epochs": 40, "batch": 4, "patience": 0, "seed": 3, "cos_lr": True}
    mismatches = {key: (config.get(key), value) for key, value in required.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Active HPO baseline violates frozen controls: {mismatches}")
    if not data_yaml.is_file() or not resolve(config["model"]).is_file():
        raise FileNotFoundError("Baseline dataset YAML or YOLO11m-OBB checkpoint missing")
    config["data"] = str(data_yaml)
    config["model"] = str(resolve(config["model"]))
    config["project"] = str(project_root)
    config.pop("resume", None)
    return config, data_yaml


def variant_config(base: dict[str, Any], name: str, project_root: Path) -> dict[str, Any]:
    config = dict(base)
    config.update(VARIANTS[name]["changes"])
    config["project"] = str(project_root)
    config["name"] = str(VARIANTS[name].get("output_name", name))
    config.pop("resume", None)
    return config


def changed_settings(base: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ignored = {"project", "name", "exist_ok"}
    return {
        key: {"baseline": base.get(key), "variant": config.get(key)}
        for key in config
        if key not in ignored and base.get(key) != config.get(key)
    }


def validate_variant(base: dict[str, Any], name: str, config: dict[str, Any]) -> None:
    expected_changes = VARIANTS[name]["changes"]
    observed = changed_settings(base, config)
    if set(observed) != set(expected_changes):
        raise ValueError(f"{name} changes unexpected keys: expected {set(expected_changes)}, got {set(observed)}")
    for key in FROZEN_KEYS:
        if config.get(key) != base.get(key):
            raise ValueError(f"{name} changed frozen key {key}: {base.get(key)!r} -> {config.get(key)!r}")


def command(config_path: Path) -> str:
    return f"python scripts/training/train_second_pass_obb.py --config {config_path.relative_to(REPO_ROOT)}"


def write_baseline_reconstruction(report_root: Path, active: dict[str, Any], historical: dict[str, Any], base: dict[str, Any]) -> None:
    prior_aug = read_json(REPO_ROOT / "reports/augmentation_ablation/seed_confirmation_results.json")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_dataset": active,
        "historical_compact_reference": historical,
        "audit_fixed_baseline_recipe": base,
        "inherited_pre_audit_geometric_evidence": prior_aug.get("paired_geo_minus_baseline"),
        "interpretation": "Historical metrics are pre-audit evidence only and are not comparable to audit-fixed runs.",
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "baseline_reconstruction.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Audit-Fixed Baseline Reconstruction",
        "",
        f"Active dataset fingerprint: `{active['dataset']['sha256']}`.",
        "",
        "The historical compact reference is `models/resolution_ablation/imgsz_1280`; it is explicitly pre-audit.",
        f"Its best mAP50-95 was `{historical['metrics']['mAP50_95']:.5f}` at epoch `{historical['best_epoch']}`.",
        "",
        "The audit-fixed confirmation uses YOLO11m-OBB, 1280 input size, seed 3, batch 4, 40 epochs, patience 0, cosine LR, and the promoted active training YAML.",
        "The rotation/vertical-flip policy is recorded as inherited pre-audit evidence and must be reconfirmed on audit-fixed data.",
    ]
    (report_root / "baseline_reconstruction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(
    *, base: dict[str, Any], active: dict[str, Any], historical: dict[str, Any], config_root: Path, report_root: Path,
    project_root: Path, smoke_project_root: Path,
) -> dict[str, Path]:
    configs: dict[str, Path] = {}
    variants_payload: dict[str, Any] = {}
    for name in ACTIVE_VARIANTS:
        config = variant_config(base, name, project_root)
        validate_variant(base, name, config)
        path = config_root / f"{name}.yaml"
        write_yaml(path, config)
        configs[name] = path
        variants_payload[name] = {
            "variant_name": name,
            "kind": VARIANTS[name]["kind"],
            "hypothesis": VARIANTS[name]["hypothesis"],
            "changed_settings": changed_settings(base, config),
            "unchanged_controls": {key: base.get(key) for key in FROZEN_KEYS},
            "dataset_yaml": base["data"],
            "dataset_hash": active["dataset"]["sha256"],
            "active_tile_root": active["roots"]["tile"],
            "active_label_root": active["roots"]["label"],
            "model": base["model"],
            "starting_checkpoint": base["model"],
            "imgsz": base["imgsz"],
            "seed": base["seed"],
            "epochs": base["epochs"],
            "patience": base["patience"],
            "batch": base["batch"],
            "effective_batch": 64,
            "optimizer": base["optimizer"],
            "lr0": config["lr0"],
            "lrf": config["lrf"],
            "cos_lr": config["cos_lr"],
            "weight_decay": config["weight_decay"],
            "warmup_epochs": config["warmup_epochs"],
            "augmentation_settings": {key: config.get(key) for key in ("degrees", "flipud", "fliplr", "mosaic", "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "scale", "translate")},
            "full_effective_training_args": config,
            "config": str(path),
            "expected_output_dir": str(project_root / config["name"]),
            "smoke_output_dir": str(smoke_project_root / f"smoke_{config['name']}"),
            "smoke_command": f"python scripts/training/run_hyperparameter_tuning.py --phase smoke --variants {name}",
            "full_command": command(path),
            "status": "prepared",
        }
    deferred = {
        "variant_name": "hpo_10_combined_candidate",
        "kind": "combined_candidate",
        "status": "deferred",
        "reason": "Create only after individual audit-fixed trials show compatible beneficial changes.",
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "audit_fixed_bounded_hyperparameter_tuning",
        "fixed_imgsz": 1280,
        "no_larger_model_variants": True,
        "historical_reference": historical,
        "active_dataset": active,
        "variants": variants_payload,
        "deferred": deferred,
        "comparison_script": str(REPO_ROOT / "scripts/training/compare_hyperparameter_tuning_runs.py"),
        "git_revision": git_revision(),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = "\n".join(
        f"| `{name}` | `{json.dumps(variants_payload[name]['changed_settings'], sort_keys=True)}` | `{variants_payload[name]['kind']}` |"
        for name in ACTIVE_VARIANTS
    )
    plan = f"""# Audit-Fixed Bounded Hyperparameter Tuning

All active trials use audit-fixed dataset fingerprint `{active['dataset']['sha256']}`, YOLO11m-OBB, `imgsz=1280`, seed `3`, batch `4`, 40 epochs, and patience `0`.
No dataset, split, label, tile, class, model-size, or confidence-threshold change is permitted.

The historical 1280 compact run is pre-audit evidence only. Its geometric augmentation result is therefore reintroduced solely as `hpo_09_best_aug_carry_forward`.

| Variant | Changed settings | Type |
|---|---|---|
{table}

`hpo_10_combined_candidate` is deferred until completed audit-fixed trials identify compatible improvements. Gains below `+0.005` mAP50-95 are treated as likely noise; gains at or above `+0.010` are strong candidates for confirmation.
"""
    (report_root / "experiment_plan.md").write_text(plan, encoding="utf-8")

    smoke = ["# HPO Smoke Commands", "", "Run from the activated `cv_practice_env`. Each command runs two epochs and requires CUDA.", ""]
    smoke.extend(f"`python scripts/training/run_hyperparameter_tuning.py --phase smoke --variants {name}`" for name in ACTIVE_VARIANTS)
    (report_root / "smoke_commands.md").write_text("\n".join(smoke) + "\n", encoding="utf-8")
    full = [
        "# HPO Full-Run Commands",
        "",
        "Run only after the corresponding smoke test succeeds. Each command runs 40 epochs; no commands are launched automatically.",
        "Use the matching resume command only after an interrupted run and only when its `weights/last.pt` exists.",
        "",
    ]
    for name in ACTIVE_VARIANTS:
        config = read_yaml(configs[name])
        run_dir = project_root / str(config["name"])
        full.extend((
            f"## {name}",
            "",
            "Start:",
            f"`{command(configs[name])}`",
            "",
            "Resume after interruption:",
            f"`{command(configs[name])} --resume {run_dir / 'weights/last.pt'} --allow-existing`",
            "",
        ))
    (report_root / "full_run_commands.md").write_text("\n".join(full) + "\n", encoding="utf-8")
    selection = """# Final Model Selection Template

| Candidate | Dataset lineage | mAP50-95 | mAP50 | Precision | Recall | Latency | VRAM | Qualitative behavior | Deployment suitability |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Audit-fixed baseline | audit_fixed_v2 | | | | | | | | |
| Best HPO variant | audit_fixed_v2 | | | | | | | | |
| Inherited geometric policy | requires audit-fixed confirmation | | | | | | | | |
| Website/demo candidate | | | | | | | | | |

Select only from audit-fixed completed runs. Historical pre-audit metrics are explanatory evidence, not a direct benchmark.
"""
    (report_root / "final_model_selection_template.md").write_text(selection, encoding="utf-8")
    write_baseline_reconstruction(report_root, active, historical, base)
    return configs


def smoke_variant(config_path: Path, smoke_project_root: Path, smoke_epochs: int, report_root: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    config["project"] = str(smoke_project_root)
    config["name"] = f"smoke_{config['name']}"
    config["epochs"] = smoke_epochs
    smoke_config = config_path.parent / "smoke" / config_path.name
    run_dir = smoke_project_root / config["name"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite smoke run: {run_dir}")
    write_yaml(smoke_config, config)
    report_root.joinpath("smoke_logs").mkdir(parents=True, exist_ok=True)
    log_path = report_root / "smoke_logs" / f"{config['name']}.log"
    sampler = GpuMemorySampler()
    started = time.monotonic()
    sampler.start()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "scripts/training/train_second_pass_obb.py", "--config", str(smoke_config)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    sampler.stop()
    metrics, finite_losses = last_metrics(run_dir)
    return {
        "config": str(smoke_config),
        "run_dir": str(run_dir),
        "log": str(log_path),
        "return_code": result.returncode,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_memory_mb": sampler.peak_mb,
        "gpu_total_memory_mb": sampler.total_mb,
        "peak_gpu_memory_fraction": round(sampler.peak_mb / sampler.total_mb, 5) if sampler.peak_mb is not None and sampler.total_mb else None,
        "validation_completed": bool(metrics),
        "finite_losses": finite_losses,
        "metrics": metrics,
        "status": "passed" if result.returncode == 0 and finite_losses else "failed",
    }


def main() -> int:
    args = parse_args()
    config_root = resolve(args.config_root)
    report_root = resolve(args.report_root)
    project_root = resolve(args.project_root)
    smoke_project_root = resolve(args.smoke_project_root)
    active = validate_active_dataset(dataloader_check=not args.skip_dataloader_check)
    if args.phase == "validate":
        print(json.dumps(active, indent=2, sort_keys=True))
        return 0
    base, _ = read_baseline(args.baseline_config, project_root)
    historical = historical_reference(args.historical_baseline_run)
    configs = write_artifacts(
        base=base,
        active=active,
        historical=historical,
        config_root=config_root,
        report_root=report_root,
        project_root=project_root,
        smoke_project_root=smoke_project_root,
    )
    if args.phase == "prepare":
        print(report_root / "experiment_manifest.json")
        return 0
    smoke_results: dict[str, Any] = {}
    for name in args.variants:
        smoke_results[name] = smoke_variant(configs[name], smoke_project_root, args.smoke_epochs, report_root)
    (report_root / "smoke_results.json").write_text(json.dumps(smoke_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = [name for name, result in smoke_results.items() if result["status"] != "passed"]
    if failures:
        raise RuntimeError(f"HPO smoke failures: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
