#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from experiment_log import build_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the maritime ship detector.")
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models/yolo11m-obb.pt",
        help="Base model checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data/training_artifacts/assembled_legacy/maritime_ship/data.yaml",
        help="Dataset YAML produced by the assembly step.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--name")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--exist-ok", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    data_path = args.data if args.data.is_absolute() else REPO_ROOT / args.data
    model_path = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    project_path = (
        args.project if args.project.is_absolute() else REPO_ROOT / args.project
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Run scripts/data/build_training_dataset.py first."
        )

    run_dir = project_path / args.name
    manifest = build_manifest(
        mode="train",
        repo_root=REPO_ROOT,
        script="scripts/training/train.py",
        args=vars(args),
        extra={
            "model_path": model_path,
            "data_path": data_path,
            "project_path": project_path,
            "run_dir": run_dir,
        },
    )
    write_manifest(run_dir, manifest)

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_path),
        name=args.name,
        cache=args.cache,
        patience=args.patience,
        cos_lr=args.cos_lr,
        resume=args.resume,
        plots=args.plots,
        save=args.save,
        exist_ok=args.exist_ok,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
