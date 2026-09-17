#!/usr/bin/env python3
"""Materialize a vetted public mirror from an explicit allowlist.

Dry-run is the default. This command never initializes Git, pushes a remote, or
changes the private development archive.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


FORBIDDEN_SUFFIXES = {".pt", ".onnx", ".engine", ".mp4", ".avi", ".mov", ".zip", ".tar", ".gz", ".7z", ".npy", ".npz"}
FORBIDDEN_PARTS = {"data/raw", "data/tiled", "data/annotations", "data/training", "datasets", "models", "outputs", "reports"}
SECRET_RE = re.compile(r"\b(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{20,})\b|(?i:(?:token|secret|api[_-]?key|password)\s*[:=]\s*(?:['\"][^'\"]{8,}['\"]|[A-Za-z0-9_-]{20,}))")


def fail(message: str) -> None:
    raise ValueError(message)


def validate_entry(root: Path, entry: dict[str, str]) -> tuple[Path, Path]:
    source_rel = Path(entry["source"])
    destination_rel = Path(entry.get("destination", entry["source"]))
    if source_rel.is_absolute() or destination_rel.is_absolute() or ".." in source_rel.parts or ".." in destination_rel.parts:
        fail(f"unsafe mirror path: {entry}")
    source = (root / source_rel).resolve()
    if root not in source.parents and source != root:
        fail(f"source escapes repository: {source_rel}")
    if not source.is_file():
        fail(f"allowlisted source is missing or not a file: {source_rel}")
    source_text = source_rel.as_posix()
    if source.suffix.lower() in FORBIDDEN_SUFFIXES or any(source_text == part or source_text.startswith(part + "/") for part in FORBIDDEN_PARTS):
        fail(f"forbidden public artifact: {source_rel}")
    if source.suffix.lower() in {".md", ".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".ini"}:
        content = source.read_text(encoding="utf-8", errors="replace")
        if SECRET_RE.search(content):
            fail(f"secret-like value found in allowlisted source: {source_rel}")
        # Compare against this machine's resolved home prefix. This avoids treating
        # the generic path-pattern text inside this validator as an actual leak.
        home_prefix = Path.home().as_posix() + "/"
        if home_prefix in content or re.search(r"[A-Za-z]:\\\\Users\\[^\\]+\\", content):
            fail(f"absolute local path found in allowlisted source: {source_rel}")
    return source, destination_rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("docs/public_mirror_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Copy allowlisted files. Default is validation-only dry-run.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest_path = (root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("include", [])
    if not entries:
        fail("mirror manifest has no allowlisted files")
    validated = [validate_entry(root, entry) for entry in entries]
    output = args.output_dir.resolve()
    print(f"Validated {len(validated)} allowlisted files for {output}.")
    if not args.apply:
        print("Dry-run complete. Re-run with --apply to materialize files; no Git repository will be created.")
        return 0
    if output.exists() and any(output.iterdir()):
        fail(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for source, destination_rel in validated:
        destination = output / destination_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output / "PUBLIC_MIRROR_NOTICE.md").write_text(
        "# Public Mirror\n\nThis mirror was materialized from an explicit allowlist. It intentionally excludes datasets, annotations, weights, videos, generated outputs, and private historical reports.\n",
        encoding="utf-8",
    )
    print(f"Materialized {len(validated)} files. No Git repository was initialized or pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
