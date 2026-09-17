"""Ensure public-safe allowlisted text does not contain obvious credential values."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"\b(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{20,})\b|(?i:(?:token|secret|api[_-]?key|password)\s*[:=]\s*(?:['\"][^'\"]{8,}['\"]|[A-Za-z0-9_-]{20,}))")


class PublicSecretScanTests(unittest.TestCase):
    def test_allowlisted_text_is_portable_and_secret_free(self) -> None:
        manifest = json.loads((ROOT / "docs/public_mirror_manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["include"]:
            source_path = ROOT / entry["source"]
            destination_path = ROOT / entry.get("destination", entry["source"])
            path = source_path if source_path.is_file() else destination_path
            self.assertTrue(path.is_file(), entry)
            if path.suffix.lower() not in {".md", ".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".ini"}:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            self.assertIsNone(SECRET.search(content), path.as_posix())
            self.assertNotIn(Path.home().as_posix() + "/", content, path.as_posix())


if __name__ == "__main__":
    unittest.main()
