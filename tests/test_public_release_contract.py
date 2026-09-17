"""Public-mirror contract checks without private datasets or model dependencies."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseContractTests(unittest.TestCase):
    def test_required_public_documents_exist(self) -> None:
        for relative in (
            "README.md",
            "MODEL_ARTIFACTS.md",
            "docs/public_mirror_plan.md",
            "docs/public_mirror_manifest.json",
            "docs/reproducibility.md",
            "docs/active_dataset_operations.md",
            "data/README.md",
            "data/DATASET_REGENERATION.md",
            "data/public_active_dataset_manifest.json",
            "configs/active/audit_fixed_v2_1280_geo.yaml",
            "environment.yml",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_mirror_manifest_has_only_relative_allowlist_paths(self) -> None:
        manifest = json.loads((ROOT / "docs/public_mirror_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertTrue(manifest["include"])
        for entry in manifest["include"]:
            for key in ("source", "destination"):
                if key in entry:
                    value = Path(entry[key])
                    self.assertFalse(value.is_absolute(), entry)
                    self.assertNotIn("..", value.parts, entry)

    def test_portable_active_config_uses_placeholders(self) -> None:
        config = (ROOT / "configs/active/audit_fixed_v2_1280_geo.yaml").read_text(encoding="utf-8")
        self.assertIn("${MARITIME_DATASET_YAML}", config)
        self.assertIn("imgsz: 1280", config)
        self.assertNotIn(Path.home().as_posix() + "/", config)


if __name__ == "__main__":
    unittest.main()
