# SixOpen Y8Naval ONNX Inference

This is a focused local inference integration for `SixOpen/Y8NavalONNX`. It is not a training or serving path.

## Environment

Use the existing `cv_practice_env` Conda environment. It needs `onnx`, `onnxruntime-gpu`, and `huggingface_hub` in addition to the project's existing NumPy, Pillow, OpenCV, Ultralytics, and PyTorch packages.

The model is cached at `models/pretrained_maritime/sixopen/model.onnx`. The companion `model_metadata.json` records the upstream filename, revision, commit, size, and SHA-256. The source class map is in `class_map.json`.

## Model Contract

The inspected graph uses a fixed `1 x 3 x 640 x 640` float input named `images` and emits `1 x 55 x 8400` float output named `output0`. Repository preprocessing metadata specifies RGB direct resize to 640x640, float32 rescale to 0-1, and no normalization or padding.

The verified output layout is `xywh + 50 class scores + rotation radians`; the model provides oriented detections. The integration preserves OBB corner points and also records the horizontal envelope for comparison. It does not synthesize masks.

Original model classes are retained. Actual vessel classes normalize to `vessel`; the model's `Dock` class is preserved as `non_vessel` and excluded from the missing-ship candidate review.

## Inspect

```bash
conda run -n cv_practice_env python -m scripts.inference.maritime_model_benchmark.inspect_sixopen_model
```

The graph report is written to `outputs/pretrained_model_benchmark/sixopen/model_inspection.json`.

## Smoke Test

```bash
conda run -n cv_practice_env python -m scripts.inference.maritime_model_benchmark.run_sixopen_inference \
  --images-dir data/tiled/grid/current/images \
  --manifest data/tiled/grid/current/tile_parent_map.csv \
  --output-dir outputs/pretrained_model_benchmark/sixopen \
  --max-images 6 --include-empty --save-previews --confidence 0.05 --device auto
```

The CLI uses `tile_parent_map.csv` as the authoritative tile-to-parent map. Parent coordinates are `tile coordinates + tile_x/tile_y`; filename parsing is not used for this mapping. Results are isolated under the SixOpen output root and do not modify source tiles or annotations.

Each saved detection preserves tile-relative `obb` and `bbox_xyxy` values and parent-relative `parent_obb` and `parent_bbox_xyxy` values. The parent geometry is formed solely from the authoritative tile offset in `tile_parent_map.csv`.

For a different, reproducible sample, use a new output directory and select a seed:

```bash
conda run -n cv_practice_env python -m scripts.inference.maritime_model_benchmark.run_sixopen_inference \
  --images-dir data/tiled/grid/current/images \
  --manifest data/tiled/grid/current/tile_parent_map.csv \
  --output-dir outputs/pretrained_model_benchmark/sixopen_random_20260719 \
  --max-images 6 --include-empty --save-previews --random-sample --seed 20260719 \
  --confidence 0.05 --device auto
```

## Later Full Run

Only run this deliberately after reviewing smoke outputs:

```bash
conda run -n cv_practice_env python -m scripts.inference.maritime_model_benchmark.run_sixopen_inference \
  --images-dir data/tiled/grid/current/images \
  --manifest data/tiled/grid/current/tile_parent_map.csv \
  --output-dir outputs/pretrained_model_benchmark/sixopen \
  --max-images 3600 --include-empty --no-smoke-selection --confidence 0.05 --device auto
```

`detections.json` is the completion marker. Resume is enabled by default and skips completed images. `--overwrite` intentionally replaces results for selected images and should only be used when a clean rerun is intended.

## Full Count-Only Run

To count SixOpen detections over every current grid tile without writing per-image detections, candidate files, previews, summaries, or any output directory, use count-only mode. `--max-images 0` means the full tile inventory in this mode. Include unlabeled tiles because they may contain vessels absent from the current annotations.

```bash
conda run -n cv_practice_env python -m scripts.inference.maritime_model_benchmark.run_sixopen_inference \
  --images-dir data/tiled/grid/current/images \
  --manifest data/tiled/grid/current/tile_parent_map.csv \
  --count-only --max-images 0 --include-empty --no-smoke-selection \
  --confidence 0.05 --iou-threshold 0.50 --device auto
```

The command prints an in-memory JSON report with total post-NMS detections and breakdowns by source dataset, active split, original SixOpen class, and confidence range. The total includes all model classes, including `Dock`; use the class breakdown to distinguish vessel detections. Per-tile detections are capped by `--max-detections` (default `300`).

## Providers and Limits

The integration requests CUDA before CPU for `--device auto`; it reports the provider actually used. If CUDA cannot load, it safely falls back to CPU. No CUDA, PyTorch, or SAM3 environment repair is performed automatically.

The upstream model card describes academic/non-commercial constraints for portions of its training material. Review the [model card](https://huggingface.co/SixOpen/Y8NavalONNX) before any use outside that scope.
