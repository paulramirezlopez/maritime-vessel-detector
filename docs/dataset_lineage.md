# Dataset Lineage

The private development archive contains source datasets, parent images, curated
annotations, tiled training views, recovery metadata, and historical experiments.
They are intentionally excluded from the public mirror.

## Active Line

- Dataset status: `audit_fixed_v2`
- Tile policy: grid-constrained fixed 1024 px windows with deterministic recovery
  tiles where a valid parent annotation cannot otherwise be represented
- Annotation source: second-pass parent-coordinate OBBs projected to the active tile
  inventory
- Training policy: v9 aggressive size filtering while retaining the `bad` cohort
- Split policy: existing train/validation membership is preserved; no test split is
  implied

## Important Boundaries

The project does not redistribute DOTA, xView, HRSC2016, derived image tiles,
annotations, CVAT exports, or parent imagery. A public user must acquire permissible
source data independently and regenerate the private artifacts using the documented
pipeline. Before any release that includes imagery or examples, verify each source's
redistribution and attribution terms.

See [data/DATASET_REGENERATION.md](../data/DATASET_REGENERATION.md) for the intended
high-level regeneration order.
