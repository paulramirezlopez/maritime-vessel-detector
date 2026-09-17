# Private Data Area

`data/` is intentionally local-only. It contains derived images, annotations,
training views, audit outputs, and metadata that are not included in the public
mirror. Do not treat this directory as redistributable source material.

The active dataset lineage is `audit_fixed_v2`. Stable local paths point to the
versioned audit-fixed roots through compatibility symlinks. The local active manifest
may contain development-machine paths and is not a public artifact; the sanitized
summary is [public_active_dataset_manifest.json](public_active_dataset_manifest.json).

For a high-level regeneration boundary, see [DATASET_REGENERATION.md](DATASET_REGENERATION.md).
