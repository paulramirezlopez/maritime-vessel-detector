# Active Dataset Operations

This is the operational guide for the current `audit_fixed_v2` line. Stable paths
are compatibility symlinks; treat them as read-only inputs for normal work.

## Authoritative Inputs

| Input | Role |
| --- | --- |
| `data/sources/active_parent_images/` | Active parent membership inventory |
| `data/metadata/roi_grids/` | Grid allowlists; edit deliberately and review diffs |
| `data/annotations/recovered_parent/current/` | Recovered parent-coordinate baseline annotations |
| `data/annotations/second_pass_tiled/obbs_complete_with_recovery/instances.jsonl` | Current parent-space OBB source for active tile labels |

The active training dataset is not the source of truth for parent annotations. Do
not edit tile-local labels to make a broad curation change unless the corresponding
parent-space source and audit record are updated.

## Routine Validation

These commands are read-only against the active line:

```bash
python scripts/reporting/report_dataset_lineage_counts.py
python scripts/audit/audit_tile_coverage_recovery.py --dry-run
```

For a full report, choose a new output directory instead of replacing the prior
audit evidence:

```bash
python scripts/audit/audit_tile_coverage_recovery.py \
  --output-root reports/tile_coverage_recovery_audit_candidate \
  --preview-count 50
```

## Safe Dataset-Change Sequence

1. Update only the intended parent annotation source and/or ROI JSON allowlist.
2. Build tiles into a **new staging root**, never the stable active path:

   ```bash
   python scripts/data/build_grid_tiled_dataset.py \
     --output-root data/tiled/grid/candidate_YYYYMMDD \
     --canonical-root data/tiled/grid_current_with_recovery
   ```

3. Reproject the current parent-space OBB source onto those candidate tile windows:

   ```bash
   python scripts/training/assemble_second_pass_grid_recovery.py \
     --source-tiles data/tiled/grid/candidate_YYYYMMDD \
     --source-obbs data/annotations/second_pass_tiled/obbs_complete_with_recovery/instances.jsonl \
     --output-root data/training/obb_second_pass_grid_recovery_candidate_YYYYMMDD \
     --reports-root reports/training/grid_recovery_candidate_YYYYMMDD
   ```

4. Apply the selected v9 size policy to a new variant root. Review the generated
   policy report before treating that variant as a candidate training view:

   ```bash
   python scripts/qc/analyze_ship_size_thresholds.py \
     --dataset-root data/training/obb_second_pass_grid_recovery_candidate_YYYYMMDD \
     --variant-root data/training/ship_size_threshold_variants/candidate_YYYYMMDD \
     --output-root outputs/qc/ship_size_threshold_candidate_YYYYMMDD \
     --policies v9_threshold_aggressive_keep_bad_seed3 \
     --write-variants
   ```

5. Audit the candidate with explicit roots. Keep the resulting reports and previews
   separate from the active audit:

   ```bash
   python scripts/audit/audit_tile_coverage_recovery.py \
     --tile-root data/tiled/grid/candidate_YYYYMMDD \
     --label-root data/annotations/second_pass_tiled/obbs_complete_with_recovery/instances.jsonl \
     --training-dataset-root data/training/ship_size_threshold_variants/candidate_YYYYMMDD/v9_threshold_aggressive_keep_bad_seed3 \
     --output-root reports/tile_coverage_recovery_candidate_YYYYMMDD
   ```

6. Promote only after the candidate has passed audit and training validation. The
   existing `scripts/audit/promote_audit_fixed_dataset.py` is the one-time v2
   promotion utility with fixed v2 roots; do **not** point it at a new candidate.
   Write a versioned promotion plan for any successor rather than overwriting the
   active symlinks ad hoc.

## Current Training and Evaluation

Use the selected local recipe:

```bash
conda run -n cv_practice_env python scripts/training/train_second_pass_obb.py \
  --config configs/training/active_second_pass_audit_fixed_1280_best_known.yaml
```

To resume an interrupted run, use the same config and its `last.pt` checkpoint:

```bash
conda run -n cv_practice_env python scripts/training/train_second_pass_obb.py \
  --config path/to/copied_experiment_config.yaml \
  --resume models/<project>/<run_name>/weights/last.pt \
  --allow-existing
```

The general `scripts/training/train.py` and `scripts/training/evaluate.py` retain
legacy defaults. Pass every dataset/model argument explicitly if using them, or
prefer `train_second_pass_obb.py` for the active controlled recipe.
