"""Report basic quality and composition stats for the assembled maritime dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/training_artifacts/assembled_legacy/maritime_ship"
DEFAULT_REPORT_DIR = "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on the assembled maritime dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Assembled dataset root.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional path to dataset_index.csv. Defaults to <dataset-root>/dataset_index.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for saved snapshot artifacts. Defaults to <dataset-root>/reports.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many groups and xml sources to show in the summary.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_empty_row(row: dict[str, str]) -> bool:
    value = row.get("empty", "")
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_report(rows: list[dict[str, str]], dataset_root: Path, index_path: Path, top_n: int) -> dict[str, Any]:
    total = len(rows)
    by_split: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    by_empty: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    dataset_stats: dict[str, dict[str, Any]] = {}

    for row in rows:
        split = row["split"]
        dataset = row["dataset"]
        by_split[split] += 1
        by_dataset[dataset] += 1
        by_empty["empty" if is_empty_row(row) else "positive"] += 1
        source_counts[row["source_xml"]] += 1
        groups[row["group_key"]].add(split)

    for dataset in sorted(by_dataset):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_split_counts = Counter(row["split"] for row in dataset_rows)
        dataset_empty = sum(1 for row in dataset_rows if is_empty_row(row))
        dataset_total = len(dataset_rows)
        dataset_stats[dataset] = {
            "total": dataset_total,
            "splits": dict(sorted(dataset_split_counts.items())),
            "positive": dataset_total - dataset_empty,
            "empty": dataset_empty,
            "empty_ratio": (dataset_empty / dataset_total) if dataset_total else 0.0,
        }

    split_empty_ratios: dict[str, float] = {}
    split_empty_counts: dict[str, int] = {}
    for split in sorted(by_split):
        split_rows = [row for row in rows if row["split"] == split]
        split_empty = sum(1 for row in split_rows if is_empty_row(row))
        split_empty_counts[split] = split_empty
        split_empty_ratios[split] = (split_empty / len(split_rows)) if split_rows else 0.0

    leaking_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    missing_images, missing_labels = file_parity_report(dataset_root)

    return {
        "dataset_root": str(dataset_root),
        "index_path": str(index_path),
        "total_examples": total,
        "split_counts": dict(sorted(by_split.items())),
        "split_empty_counts": split_empty_counts,
        "dataset_counts": dict(sorted(by_dataset.items())),
        "example_mix": {
            "positive": by_empty["positive"],
            "empty": by_empty["empty"],
            "empty_ratio": (by_empty["empty"] / total) if total else 0.0,
        },
        "split_empty_ratios": split_empty_ratios,
        "dataset_stats": dataset_stats,
        "group_leakage": {
            "count": len(leaking_groups),
            "sample_groups": leaking_groups[:top_n],
        },
        "source_xml_counts": source_counts.most_common(top_n),
        "filesystem_parity": {
            "image_files_without_labels": len(missing_labels),
            "label_files_without_images": len(missing_images),
            "sample_image_only_stems": missing_labels[:top_n],
            "sample_label_only_stems": missing_images[:top_n],
        },
    }


def file_parity_report(dataset_root: Path) -> tuple[list[str], list[str]]:
    missing_images: list[str] = []
    missing_labels: list[str] = []

    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue

        image_stems = {path.stem for path in image_dir.iterdir() if path.is_file()}
        label_stems = {path.stem for path in label_dir.iterdir() if path.is_file()}

        for stem in sorted(image_stems - label_stems):
            missing_labels.append(f"{split}:{stem}")
        for stem in sorted(label_stems - image_stems):
            missing_images.append(f"{split}:{stem}")

    return missing_images, missing_labels


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Dataset root: {report['dataset_root']}")
    lines.append(f"Index: {report['index_path']}")
    lines.append(f"Total examples: {report['total_examples']}")

    lines.append("\nSplit counts:")
    for split, count in report["split_counts"].items():
        lines.append(f"  {split}: {count}")

    lines.append("\nDataset counts:")
    for dataset, count in report["dataset_counts"].items():
        lines.append(f"  {dataset}: {count}")

    lines.append("\nDataset breakdown:")
    for dataset, stats in report["dataset_stats"].items():
        split_bits = ", ".join(f"{split}={count}" for split, count in stats["splits"].items()) or "none"
        lines.append(
            f"  {dataset}: total={stats['total']} positive={stats['positive']} empty={stats['empty']} "
            f"empty_ratio={stats['empty_ratio']:.4f} splits[{split_bits}]"
        )

    mix = report["example_mix"]
    lines.append("\nExample mix:")
    lines.append(f"  positives: {mix['positive']}")
    lines.append(f"  empty: {mix['empty']}")
    lines.append(f"  empty ratio: {mix['empty_ratio']:.4f}")

    lines.append("\nEmpty ratio by split:")
    for split, ratio in report["split_empty_ratios"].items():
        split_total = report["split_counts"][split]
        split_empty = report["split_empty_counts"][split]
        lines.append(f"  {split}: {ratio:.4f} ({split_empty}/{split_total})")

    leakage = report["group_leakage"]
    lines.append("\nGroup leakage:")
    lines.append(f"  groups spanning multiple splits: {leakage['count']}")
    if leakage["sample_groups"]:
        lines.append(f"  sample groups: {', '.join(leakage['sample_groups'])}")

    lines.append("\nTop source XMLs:")
    for source_xml, count in report["source_xml_counts"]:
        lines.append(f"  {count:5d}  {source_xml}")

    parity = report["filesystem_parity"]
    lines.append("\nFilesystem parity:")
    lines.append(f"  image files without labels: {parity['image_files_without_labels']}")
    lines.append(f"  label files without images: {parity['label_files_without_images']}")
    if parity["sample_image_only_stems"]:
        lines.append(
            f"  sample image-only stems: {', '.join(parity['sample_image_only_stems'])}"
        )
    if parity["sample_label_only_stems"]:
        lines.append(
            f"  sample label-only stems: {', '.join(parity['sample_label_only_stems'])}"
        )

    return "\n".join(lines) + "\n"


def write_report_artifacts(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "snapshot_report.json"
    text_path = output_dir / "snapshot_report.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(format_report(report), encoding="utf-8")
    return json_path, text_path


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    index_path = resolve_path(args.index) if args.index else dataset_root / "dataset_index.csv"
    output_dir = resolve_path(args.output_dir) if args.output_dir else dataset_root / DEFAULT_REPORT_DIR

    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")

    rows = load_index(index_path)
    report = build_report(rows, dataset_root, index_path, top_n=args.top_n)
    write_report_artifacts(report, output_dir)
    print(format_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
