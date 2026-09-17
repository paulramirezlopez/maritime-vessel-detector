#!/usr/bin/env python3
"""Create local-only evidence for preparing a clean public repository mirror.

This command never edits datasets, model artifacts, Git history, or source files.
It aggregates the large local tree rather than listing every image individually and
redacts secret-like values from every report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".sh"}
LARGE_ARTIFACT_SUFFIXES = {".pt", ".onnx", ".engine", ".mp4", ".avi", ".mov", ".zip", ".tar", ".gz", ".7z", ".npy", ".npz"}
SECRET_PATTERNS = {
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "generic_secret_assignment": re.compile(r"(?i)\b(?:token|secret|api[_-]?key|password)\b\s*[:=]\s*(?:['\"][^'\"]{8,}['\"]|[A-Za-z0-9_-]{20,})"),
}
SKIP_SCAN_PARTS = {".git", "data", "datasets", "models", "outputs", ".venv", "venv", "env", ".conda", "__pycache__"}
PRIVATE_BULK_ROOTS = {"data", "datasets", "models", "outputs", "reports"}


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_path(path: str) -> str:
    if path.startswith(("data/", "datasets/")):
        return "private_dataset_or_annotation"
    if path.startswith(("models/", "outputs/")):
        return "private_model_or_generated_output"
    if path.startswith("reports/"):
        return "historical_or_local_report"
    if path.startswith("release/"):
        return "release_metadata_or_local_artifact"
    if path.startswith("ship-detector-api/"):
        return "deployment_candidate"
    if path.startswith(("scripts/", "configs/", "tests/", "docs/")):
        return "code_or_public_documentation_candidate"
    return "requires_manual_classification"


def aggregate_tree(root: Path, max_depth: int = 3) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current)
        rel_parts = current_path.relative_to(root).parts
        if current_path == root:
            bulk_dirs = [name for name in dirs if name in PRIVATE_BULK_ROOTS]
            dirs[:] = [name for name in dirs if name not in {*bulk_dirs, ".git", "__pycache__", ".pytest_cache"}]
            for name in bulk_dirs:
                rows[name] = private_root_summary(root / name, root)
        else:
            dirs[:] = [name for name in dirs if name not in {".git", "__pycache__", ".pytest_cache"}]
        visible_parts = rel_parts[:max_depth]
        key = "." if not visible_parts else "/".join(visible_parts)
        row = rows.setdefault(key, {"path": key, "file_count": 0, "directory_count": 0, "size_bytes": 0})
        if len(rel_parts) <= max_depth:
            row["directory_count"] += len(dirs)
        for name in files:
            file_path = current_path / name
            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            row["file_count"] += 1
            row["size_bytes"] += size
            for depth in range(0, min(len(rel_parts), max_depth) + 1):
                parent_key = "." if depth == 0 else "/".join(rel_parts[:depth])
                if parent_key == key:
                    continue
                parent = rows.setdefault(parent_key, {"path": parent_key, "file_count": 0, "directory_count": 0, "size_bytes": 0})
                parent["file_count"] += 1
                parent["size_bytes"] += size
    return sorted(rows.values(), key=lambda item: item["path"])


def private_root_summary(path: Path, root: Path) -> dict[str, Any]:
    """Summarize a bulk private root without traversing every image/label file."""
    size_output = command_output(["du", "-sb", str(path)], root)
    try:
        size_bytes = int(size_output.split()[0])
    except (IndexError, ValueError):
        size_bytes = None
    return {
        "path": path.name,
        "file_count": "not enumerated (private bulk root)",
        "directory_count": "not enumerated (private bulk root)",
        "size_bytes": size_bytes,
        "classification": "private bulk root excluded from publication scan",
    }


def release_relevant_file(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return bool(parts) and parts[0] in {"docs", "configs", "scripts", "release", "ship-detector-api", "tests"}


def inventory_files(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return large files globally and duplicate candidates only for public code.

    Hashing repeated tile labels is not meaningful for public-release cleanup and can
    consume substantial memory/time in a private vision archive.
    """
    large: list[dict[str, Any]] = []
    candidates: dict[tuple[int, str], list[str]] = defaultdict(list)
    for current, dirs, files in os.walk(root, topdown=True):
        current_path = Path(current)
        if current_path == root:
            dirs[:] = [name for name in dirs if name not in {*PRIVATE_BULK_ROOTS, ".git"}]
        else:
            dirs[:] = [name for name in dirs if name != ".git"]
        for name in files:
            path = current_path / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            rel = relative(path, root)
            if size >= 10 * 1024 * 1024 or path.suffix.lower() in LARGE_ARTIFACT_SUFFIXES:
                large.append({"path": rel, "size_bytes": size, "classification": classify_path(rel)})
            if release_relevant_file(path, root):
                candidates[(size, path.name)].append(rel)
    duplicates = [
        {"size_bytes": size, "filename": name, "paths": paths, "content_hash_checked": size <= 10 * 1024 * 1024}
        for (size, name), paths in candidates.items() if len(paths) > 1
    ]
    for candidate in duplicates:
        if candidate["content_hash_checked"]:
            candidate["sha256"] = sorted({sha256(root / item) for item in candidate["paths"]})
    return large, duplicates


def command_output(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No findings.\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def write_report(output_dir: Path, name: str, payload: Any, markdown: str) -> None:
    (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / f"{name}.md").write_text(markdown, encoding="utf-8")


def text_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_SCAN_PARTS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 10 * 1024 * 1024:
            found.append(path)
    return found


def audit_security(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    absolute_paths: list[dict[str, Any]] = []
    for path in text_files(root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            for kind, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({"path": relative(path, root), "line": number, "kind": kind, "value": "[REDACTED]"})
            if "/home/" in line or re.search(r"[A-Za-z]:\\\\Users\\", line):
                absolute_paths.append({"path": relative(path, root), "line": number, "kind": "absolute_local_path"})
    return {"secret_like_findings": findings, "absolute_path_references": absolute_paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("reports/repo_cleanup"))
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs without writing reports.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()

    tree = aggregate_tree(root)
    top_level = [row for row in tree if "/" not in row["path"] and row["path"] != "."]
    large, duplicate_candidates = inventory_files(root)
    script_rows = []
    for path in sorted(root.glob("scripts/**/*.py")):
        content = path.read_text(encoding="utf-8", errors="replace")
        script_rows.append({"path": relative(path, root), "classification": classify_path(relative(path, root)), "has_cli": "ArgumentParser" in content, "absolute_path_reference": "/home/" in content})
    config_rows = []
    for path in sorted([*root.glob("configs/**/*.yaml"), *root.glob("configs/**/*.yml"), *root.glob("configs/**/*.json")]):
        content = path.read_text(encoding="utf-8", errors="replace")
        config_rows.append({"path": relative(path, root), "classification": classify_path(relative(path, root)), "absolute_path_reference": "/home/" in content, "active_candidate": "/active/" in relative(path, root) or path.name.startswith("active_")})
    security = audit_security(root)
    git_pack = command_output(["git", "count-objects", "-vH"], root)
    tracked = command_output(["git", "ls-files"], root).splitlines()
    unreachable = command_output(["git", "fsck", "--unreachable", "--no-reflogs"], root)
    active = {
        "dataset_line": "audit_fixed_v2",
        "stable_tile_root": "data/tiled/grid_current_with_recovery",
        "stable_parent_obb_root": "data/annotations/second_pass_tiled/obbs_complete_with_recovery",
        "stable_training_root": "data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3",
        "selection_policy": "v9 aggressive keep-bad",
        "selected_model": "YOLO11m-OBB at imgsz=1280 with seed 3 and geometric augmentation",
        "public_status": "summary only; no data, annotation, or model artifact is eligible for mirror publication",
    }
    payloads = {
        "full_tree_inventory": {"generated_at": datetime.now(UTC).isoformat(), "aggregation_depth": 3, "top_level": top_level, "tree": tree},
        "active_pipeline_map": active,
        "dataset_artifact_audit": {
            "private_roots": [private_root_summary(root / name, root) for name in sorted(PRIVATE_BULK_ROOTS)],
            "scanned_large_file_count_outside_private_bulk_roots": len(large),
            "large_artifacts_outside_private_bulk_roots": sorted(large, key=lambda row: row["size_bytes"], reverse=True),
        },
        "duplicate_data_report": {"method": "same basename and byte size; hashes only for candidates <=10 MiB", "candidates": duplicate_candidates},
        "script_inventory": {"scripts": script_rows},
        "config_inventory": {"configs": config_rows},
        "large_file_audit": {"files": sorted(large, key=lambda row: row["size_bytes"], reverse=True)},
        "security_audit": security,
        "github_publish_audit": {
            "tracked_file_count": len(tracked),
            "git_object_store": git_pack,
            "unreachable_object_summary": unreachable[:4000],
            "absolute_path_reference_count": len(security["absolute_path_references"]),
            "recommendation": "create_clean_public_mirror",
            "reason": "The development archive contains large private artifacts and historical absolute-path evidence. Do not rewrite its history for publication.",
        },
    }
    readiness = "# Publication Readiness\n\n**Recommendation:** `create_clean_public_mirror`.\n\n"
    readiness += "The private development archive remains authoritative. The mirror must be materialized only from the vetted allowlist in `docs/public_mirror_manifest.json`; no datasets, weights, videos, raw reports, or Git history should be copied.\n\n"
    readiness += "## Open Items\n\n- Confirm source-dataset redistribution and attribution obligations before publishing examples.\n- Confirm which public example imagery is permitted.\n- Verify the Gradio Space API build before presenting deployment as live.\n- Review any redacted secret-like findings before publication.\n"
    if args.dry_run:
        print(json.dumps({"repo_root": str(root), "output_dir": str(output_dir), "reports": sorted(payloads)}, indent=2))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        if isinstance(payload, dict) and len(payload) == 1:
            primary = next(iter(payload.values()))
            rows = primary if isinstance(primary, list) else [payload]
        else:
            rows = [payload]
        write_report(output_dir, name, payload, f"# {name.replace('_', ' ').title()}\n\n" + markdown_table(rows[:200], list(rows[0]) if rows and isinstance(rows[0], dict) else ["value"]))
    (output_dir / "publication_readiness_check.md").write_text(readiness, encoding="utf-8")
    print(f"Wrote {len(payloads) + 1} local-only reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
