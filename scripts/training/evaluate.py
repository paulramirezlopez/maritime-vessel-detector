"""Run reproducible validation on the assembled maritime dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any

from experiment_log import build_manifest, write_manifest
from reporting.report_training_dataset import load_index

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/training_artifacts/assembled_legacy/maritime_ship"
DEFAULT_DATA_YAML = DEFAULT_DATASET_ROOT / "data.yaml"
DEFAULT_MODEL = REPO_ROOT / "models/ship_detector_v2/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the maritime ship detector.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Merged dataset YAML for the overall validation pass.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Assembled dataset root containing images, labels, and the provenance index.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional path to dataset_index.csv. Defaults to <dataset-root>/dataset_index.csv.",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "models/validation",
        help="Directory that will contain the validation run outputs.",
    )
    parser.add_argument(
        "--name",
        default="maritime_ship_eval",
        help="Validation run name.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset names to validate individually. Defaults to all datasets in the index.",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exist-ok", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_empty_row(row: dict[str, str]) -> bool:
    return row.get("empty", "").strip().lower() in {"true", "1", "yes"}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    return value


def metrics_to_dict(result: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if isinstance(result, dict):
        metrics.update(result)
    else:
        results_dict = getattr(result, "results_dict", None)
        if isinstance(results_dict, dict):
            metrics.update(results_dict)
        else:
            for key in ("fitness", "map", "map50", "map75", "mp", "mr", "box", "speed"):
                if hasattr(result, key):
                    metrics[key] = getattr(result, key)

    save_dir = getattr(result, "save_dir", None)
    if save_dir is not None:
        metrics["save_dir"] = str(save_dir)

    return jsonable(metrics)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_subset_dataset(
    dataset_root: Path,
    subset_rows: list[dict[str, str]],
    output_root: Path,
    subset_name: str,
) -> dict[str, Any]:
    ensure_clean_dir(output_root)
    images_dir = output_root / "images" / "val"
    labels_dir = output_root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    used_stems: Counter[str] = Counter()

    for row in subset_rows:
        dataset = row["dataset"]
        image_name = row["image_name"]
        image_path = dataset_root / "images" / row["split"] / image_name
        label_path = (
            dataset_root / "labels" / row["split"] / f"{Path(image_name).stem}.txt"
        )

        stem_base = f"{dataset}__{Path(image_name).stem}"
        used_stems[stem_base] += 1
        suffix = "" if used_stems[stem_base] == 1 else f"__{used_stems[stem_base]}"
        subset_stem = f"{stem_base}{suffix}"
        subset_image_name = f"{subset_stem}{image_path.suffix}"
        subset_label_name = f"{subset_stem}.txt"

        subset_image_path = images_dir / subset_image_name
        subset_label_path = labels_dir / subset_label_name
        subset_image_path.symlink_to(image_path.resolve())
        subset_label_path.symlink_to(label_path.resolve())

        manifest_rows.append(
            {
                "dataset": dataset,
                "group_key": row["group_key"],
                "image_name": image_name,
                "split": row["split"],
                "empty": row["empty"],
                "source_xml": row["source_xml"],
                "source_image": row["source_image"],
                "eval_image_name": subset_image_name,
                "eval_label_name": subset_label_name,
            }
        )

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/val",
                "val: images/val",
                "names:",
                "  0: ship",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = output_root / "manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        fieldnames=[
            "dataset",
            "group_key",
            "image_name",
            "split",
            "empty",
            "source_xml",
            "source_image",
            "eval_image_name",
            "eval_label_name",
        ],
    )

    return {
        "subset_name": subset_name,
        "subset_root": str(output_root),
        "data_yaml": str(yaml_path),
        "manifest": str(manifest_path),
        "row_count": len(manifest_rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in manifest_rows)),
        "empty_count": sum(1 for row in manifest_rows if is_empty_row(row)),
    }


def run_validation(
    model,
    data_yaml: Path,
    project: Path,
    name: str,
    imgsz: int,
    batch: int,
    device: str,
    plots: bool,
    exist_ok: bool,
) -> tuple[dict[str, Any], Path]:
    save_dir = project / name
    if save_dir.exists():
        shutil.rmtree(save_dir)
    result = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        plots=plots,
        exist_ok=exist_ok,
    )
    save_dir = Path(getattr(result, "save_dir", save_dir))
    metrics = metrics_to_dict(result)
    metrics_path = save_dir / "metrics.json"
    write_json(metrics_path, metrics)
    return metrics, metrics_path


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    dataset_root = resolve_path(args.dataset_root)
    index_path = (
        resolve_path(args.index) if args.index else dataset_root / "dataset_index.csv"
    )
    project_path = resolve_path(args.project)
    run_dir = project_path / args.name

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")

    rows = load_index(index_path)
    dataset_names = args.datasets or sorted({row["dataset"] for row in rows})
    val_rows_by_dataset = {
        dataset: [
            row for row in rows if row["dataset"] == dataset and row["split"] == "val"
        ]
        for dataset in dataset_names
    }

    manifest = build_manifest(
        mode="validate",
        repo_root=REPO_ROOT,
        script="scripts/training/evaluate.py",
        args=vars(args),
        extra={
            "model_path": model_path,
            "data_path": data_path,
            "dataset_root": dataset_root,
            "index_path": index_path,
            "project_path": project_path,
            "run_dir": run_dir,
            "datasets": dataset_names,
        },
    )
    write_manifest(run_dir, manifest)

    model = YOLO(str(model_path))

    validation_root = run_dir / "validation_runs"

    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "dataset_root": str(dataset_root),
        "index_path": str(index_path),
        "run_dir": str(run_dir),
        "validation_root": str(validation_root),
        "overall": {},
        "datasets": {},
    }

    overall_metrics, overall_metrics_path = run_validation(
        model,
        data_path,
        validation_root,
        "overall",
        args.imgsz,
        args.batch,
        args.device,
        args.plots,
        args.exist_ok,
    )
    summary["overall"] = {
        "metrics_path": str(overall_metrics_path),
        "metrics": overall_metrics,
    }

    for dataset in dataset_names:
        subset_rows = val_rows_by_dataset.get(dataset, [])
        subset_dir = run_dir / "subsets" / dataset
        if not subset_rows:
            summary["datasets"][dataset] = {
                "skipped": True,
                "reason": "no val rows in dataset_index.csv",
            }
            continue

        subset_info = materialize_subset_dataset(
            dataset_root, subset_rows, subset_dir, dataset
        )
        subset_metrics, subset_metrics_path = run_validation(
            model,
            Path(subset_info["data_yaml"]),
            validation_root,
            dataset,
            args.imgsz,
            args.batch,
            args.device,
            args.plots,
            args.exist_ok,
        )
        summary["datasets"][dataset] = {
            **subset_info,
            "metrics_path": str(subset_metrics_path),
            "metrics": subset_metrics,
        }

    summary_path = run_dir / "validation_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote validation summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
