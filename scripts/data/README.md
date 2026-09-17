# Data Pipeline Entry Points

The active parent inventory is under `data/sources/active_parent_images/`, ROI
metadata is under `data/metadata/roi_grids/`, and stable current dataset roots are
described by `data/active_dataset_manifest.json`.

Use these scripts with a new staging output for any material dataset change:

- `build_grid_tiled_dataset.py`: fixed-stride, ROI-grid constrained tile build
- `recover_parent_annotations.py`: recover first-pass annotation geometry into
  parent coordinates
- `apply_second_pass_xview_parent_overrides.py`: apply targeted second-pass xView
  parent annotation replacements
- `merge_curated_pass_annotations.py`: combine curated annotation passes in parent
  coordinates
- `project_recovered_parents_to_smart_tiles.py`: inspect or stage parent-to-tile
  projection against grid windows

`build_training_dataset.py` and raw-baseline builders are historical/experimental
paths. They do not produce the active audit-fixed v2 training contract. See
[the active dataset operations guide](../../docs/active_dataset_operations.md) for
the current safe sequence.
