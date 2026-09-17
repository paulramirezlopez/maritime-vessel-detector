# Reproducibility Notes

## Scope

The repository is reproducible as a **pipeline and experiment record**, not as a
self-contained dataset release. It excludes source imagery, curated annotations,
tiles, CVAT exports, weights, videos, and generated reports. Reproduction requires
independent, authorized access to each source dataset and a local recreation of the
documented private layout.

## Software Baseline

Create the base environment with:

```bash
conda env create -f environment.yml
conda activate maritime_vessel_detector_env
```

The development environment used Python 3.12.13, Ultralytics 8.4.37, OpenCV 4.13.0,
Shapely 2.1.2, Pillow 12.1.1, and PyYAML 6.0.3. Training additionally requires a
CUDA-enabled PyTorch build matched to the local driver and GPU. Do not let
Ultralytics silently select CPU for a controlled training run.

## Active Dataset Contract

The current local line is `audit_fixed_v2`:

- YOLO OBB labels in the active training view
- grid-constrained 1024 px source tiles and deterministic recovery windows
- v9 aggressive `small`/`low`/xView filtering while retaining the `bad` cohort
- unchanged train/validation membership and no separate test manifest
- selected training size 1280, seed 3, YOLO11m-OBB
- selected geometric augmentation: `degrees=180`, `flipud=0.5`, `fliplr=0.5`

Read `data/active_dataset_manifest.json` locally for the resolved paths and ledgers.
It intentionally contains machine-specific paths and is not a portable input.

## Determinism Limits

The repository records seed, configuration, dataset manifests, run metadata, and
effective training configs. Exact GPU training results can still differ slightly by
CUDA driver, GPU architecture, PyTorch/Ultralytics version, dataloader behavior, and
non-deterministic kernels. Treat a rerun as comparable only when the source data,
active symlinks, config, seed, dependency versions, and checkpoint all match.

## Validation Before Training

```bash
python scripts/reporting/report_dataset_lineage_counts.py
python scripts/audit/audit_tile_coverage_recovery.py --dry-run
```

For a new candidate dataset, follow the staging instructions in
[active_dataset_operations.md](active_dataset_operations.md). Do not overwrite the
stable roots just to test a change.
