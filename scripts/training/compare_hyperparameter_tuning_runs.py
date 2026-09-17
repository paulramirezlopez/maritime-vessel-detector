#!/usr/bin/env python3
"""Summarize completed audit-fixed bounded HPO runs without changing them."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports/hyperparameter_tuning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser.parse_args()


def metrics(run_dir: Path) -> dict[str, Any] | None:
    results = run_dir / "results.csv"
    if not results.is_file():
        return None
    with results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    args: dict[str, Any] = {}
    args_path = run_dir / "args.yaml"
    if args_path.is_file():
        loaded = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            args = loaded
    speed_columns = [key for key in best if "speed" in key.lower() and best[key]]
    speed_ms = None
    if speed_columns:
        try:
            speed_ms = float(best[speed_columns[0]])
        except ValueError:
            pass
    return {
        "best_epoch": int(float(best["epoch"])),
        "stop_epoch": int(float(rows[-1]["epoch"])),
        "mAP50_95": float(best["metrics/mAP50-95(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "fitness": 0.1 * float(best["metrics/mAP50(B)"]) + 0.9 * float(best["metrics/mAP50-95(B)"]),
        "training_time_seconds": float(rows[-1]["time"]),
        "inference_speed_ms": speed_ms,
        "batch": args.get("batch"),
        "imgsz": args.get("imgsz"),
    }


def smoke_result(report_root: Path, variant: str) -> dict[str, Any]:
    path = report_root / "smoke_results.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(variant, {})
    return value if isinstance(value, dict) else {}


def main() -> int:
    args = parse_args()
    report_root = args.report_root if args.report_root.is_absolute() else REPO_ROOT / args.report_root
    manifest = json.loads((report_root / "experiment_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for name, details in manifest["variants"].items():
        run_dir = Path(details["expected_output_dir"])
        value = metrics(run_dir)
        smoke = smoke_result(report_root, name)
        rows.append({
            "variant": name,
            "kind": details["kind"],
            "changed_settings": details["changed_settings"],
            "run_dir": str(run_dir),
            "status": "completed" if value else details["status"],
            "notes": details["hypothesis"],
            "peak_vram_mb": smoke.get("peak_gpu_memory_mb"),
            "smoke_status": smoke.get("status"),
            **(value or {}),
        })
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": manifest["active_dataset"], "rows": rows}
    (report_root / "hpo_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Audit-Fixed Hyperparameter Tuning Results",
        "",
        "| Variant | Changed settings | Best epoch | mAP50-95 | mAP50 | Precision | Recall | Fitness | Stop epoch | Time (s) | Inference ms | Batch | Peak VRAM MiB | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | `{changes}` | {best_epoch} | {map95} | {map50} | {precision} | {recall} | {fitness} | {stop_epoch} | {time} | {speed} | {batch} | {vram} | {status} |".format(
                variant=row["variant"], changes=json.dumps(row["changed_settings"], sort_keys=True),
                best_epoch=row.get("best_epoch", ""), map95=f"{row['mAP50_95']:.5f}" if "mAP50_95" in row else "",
                map50=f"{row['mAP50']:.5f}" if "mAP50" in row else "",
                precision=f"{row['precision']:.5f}" if "precision" in row else "",
                recall=f"{row['recall']:.5f}" if "recall" in row else "",
                fitness=f"{row['fitness']:.5f}" if "fitness" in row else "",
                stop_epoch=row.get("stop_epoch", ""), time=f"{row['training_time_seconds']:.1f}" if "training_time_seconds" in row else "",
                speed=f"{row['inference_speed_ms']:.3f}" if row.get("inference_speed_ms") is not None else "",
                batch=row.get("batch", ""), vram=row.get("peak_vram_mb") or "", status=row["status"],
            )
        )
    lines.extend(("", "Only completed audit-fixed trials are eligible for selection. Historical pre-audit results are not included in this comparison table."))
    (report_root / "hpo_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_root / "hpo_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
