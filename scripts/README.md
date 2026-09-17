# Scripts Layout

The `scripts/` tree is now organized by workflow instead of by iteration history.

## Active areas

- `scripts/data/`: dataset builders, recovery, relabeling, and tiling pipelines
- `scripts/cvat/`: CVAT export/import packaging and batch XML generation
- `scripts/qc/`: ROI audits and preview / validation tooling
- `scripts/audit/`: coverage audits, versioned audit repairs, and audited dataset promotion
- `scripts/reporting/`: dataset-lineage and training-dataset reports
- `scripts/inference/`: local model benchmarks and video inference export
- `scripts/training/`: training, evaluation, and run-manifest entrypoints
- `scripts/preprocess/`: low-level geometry / conversion helpers
- `scripts/archive/legacy/`: legacy scripts kept only for reference

## Shared helpers retained at the root

These are still imported by multiple workflows and remain at the top level for now:

- `dataset_utils.py`
- `experiment_log.py`
- `parent_recovery_utils.py`
- `roi_smart_retile_utils.py`

## Current entrypoints

- Training: `python scripts/training/train.py`
- Validation: `python scripts/training/evaluate.py`
- Dataset builds and recoveries: `python scripts/data/<name>.py`
- CVAT exports: `python scripts/cvat/<name>.py`
- ROI QC / previews: `python scripts/qc/<name>.py`
- Coverage audits and fixes: `python scripts/audit/<name>.py`
- Reports: `python scripts/reporting/<name>.py`
- Video inference: `python scripts/inference/process_flagship_marina_video.py`
- SixOpen benchmark: `python -m scripts.inference.maritime_model_benchmark.run_sixopen_inference`

## Cleanup manifest

See `scripts/cleanup_manifest.csv` for the first cleanup and
`scripts/reorganization_manifest_v2.csv` for this follow-up sort.

## Public mirror boundary

This tree is the development archive. Only the explicit allowlist in
`docs/public_mirror_manifest.json` is eligible for a public mirror. Run
`python scripts/release/audit_public_release.py` for local-only release evidence,
then run `python scripts/release/materialize_public_mirror.py --output-dir <empty-dir>`
as a dry run before adding `--apply`.
