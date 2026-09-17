# Maritime Vessel Detection With Oriented Bounding Boxes

This repository is the development archive for an aerial and satellite
maritime vessel detector. The project combines DOTA, xView, and HRSC2016 imagery,
manual annotation curation, ROI-aware fixed-window tiling, and YOLO OBB training.

The active dataset lineage is **audit-fixed v2**. It uses 1024 px grid-constrained
tiles, parent-space coverage auditing, deterministic recovery tiles where needed,
and the v9 aggressive keep-bad filtering policy. The selected training recipe is
YOLO11m-OBB at `imgsz=1280`, seed 3, with geometric augmentation (`degrees=180`,
`flipud=0.5`, `fliplr=0.5`). Reported validation metrics are validation-split
evidence, not a claim of independent held-out test performance.

## Quick Start

This is a private development archive. It is reproducible only for users who have
legal access to the source datasets and the curated annotation inputs; those assets
are intentionally not committed.

1. Create the non-GPU base environment:

   ```bash
   conda env create -f environment.yml
   conda activate cv_practice_env
   ```

2. Install a CUDA-enabled PyTorch build appropriate for the target driver/GPU using
   the official PyTorch instructions. Training refuses an implicit CPU fallback.
3. Acquire DOTA, xView, HRSC2016, and any other approved source data under their
   own terms. Recreate the local source layout described in
   [data/DATASET_REGENERATION.md](data/DATASET_REGENERATION.md).
4. Verify the active local contract before training:

   ```bash
   python scripts/reporting/report_dataset_lineage_counts.py
   python scripts/audit/audit_tile_coverage_recovery.py --dry-run
   ```

5. Use the current controlled training wrapper and config, not the legacy generic
   training defaults:

   ```bash
   conda run -n cv_practice_env python scripts/training/train_second_pass_obb.py \
     --config configs/training/active_second_pass_audit_fixed_1280_best_known.yaml
   ```

The training command intentionally refuses an existing output directory. Copy the
config and give it a new `name` before starting a new experiment.

## Repository Roles

- `data/`, `datasets/`, `models/`, `outputs/`, and detailed `reports/` are private
  development artifacts and are intentionally not public-release inputs.
- `scripts/` contains dataset, audit, training, CVAT, QC, and release utilities.
- `configs/` contains private historical configurations plus public-safe templates.
- `release/` contains metadata for the model release; model binaries are distributed
  separately through the model host.
- `ship-detector-api/` contains a lightweight Gradio Space candidate. Its public API
  must be validated before being described as a live deployment.

## Key Paths

| Path | Purpose | Status |
| --- | --- | --- |
| `data/sources/active_parent_images/` | Accepted full-size maritime parent images | Private source inventory |
| `data/metadata/roi_grids/` | Per-parent ROI/grid allowlists | Editable source metadata |
| `data/annotations/recovered_parent/current/` | Parent-coordinate recovered annotations | Upstream annotation source |
| `data/tiled/grid_current_with_recovery/` | Stable active grid-tile root | Audit-fixed v2 symlink |
| `data/annotations/second_pass_tiled/obbs_complete_with_recovery/` | Stable parent-space OBB source | Audit-fixed v2 symlink |
| `data/training/ship_size_threshold_variants/v9_threshold_aggressive_keep_bad_seed3/` | Active Ultralytics training view | Audit-fixed v2 symlink |
| `configs/training/active_second_pass_audit_fixed_1280_best_known.yaml` | Selected training recipe | Current local config |
| `models/active/` | Selected model metadata/checkpoint location | Local-only artifact root |
| `reports/` | Audit, experiment, and QC evidence | Local-only generated evidence |

See [active dataset operations](docs/active_dataset_operations.md) for the safe
staging and validation sequence. See [scripts layout](scripts/README.md) for the
current CLI taxonomy.


## Documentation

- [Project methodology](docs/project_methodology.md)
- [Dataset lineage and limitations](docs/dataset_lineage.md)
- [Experiment summary](docs/experiments_summary.md)
- [Deployment notes](docs/deployment.md)
- [Model artifacts](MODEL_ARTIFACTS.md)
- [Dataset regeneration boundary](data/DATASET_REGENERATION.md)
- [Reproducibility notes](docs/reproducibility.md)
- [Active dataset operations](docs/active_dataset_operations.md)

## License and Source Data

Code is licensed under the repository license. Source imagery and annotations are
not redistributed here; obtain DOTA, xView, HRSC2016, and any other source material
from their respective owners and comply with their licenses, access terms, and
attribution requirements.
