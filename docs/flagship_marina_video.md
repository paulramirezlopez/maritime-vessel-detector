# Flagship Marina Video Export

`scripts/inference/process_flagship_marina_video.py` runs the selected YOLO11 OBB vessel model over the marina video one frame at a time. It writes a frontend-ready detection JSON in source-video coordinates and can render a separate QA preview video.

The exporter never modifies the source MP4, labels, training data, or website repository. It filters to the model's `ship`/`vessel` class by default, assigns stable track IDs with deterministic OBB matching, fills gaps of at most two frames, and applies a centered five-frame smoother.

## GPU smoke test

Run this from the normal CUDA-enabled project environment:

```bash
python scripts/inference/process_flagship_marina_video.py \
  --video data/video/11159118-hd_1920_1080_30fps.mp4 \
  --model models/best_vessel_yolo_obb.pt \
  --out-dir outputs/flagship_marina_smoke \
  --imgsz 1280 --conf 0.25 --iou 0.50 --device 0 \
  --max-frames 60 --render-preview
```

## Full export

```bash
python scripts/inference/process_flagship_marina_video.py \
  --video data/video/11159118-hd_1920_1080_30fps.mp4 \
  --model models/best_vessel_yolo_obb.pt \
  --out-dir outputs/flagship_marina \
  --imgsz 1280 --conf 0.25 --iou 0.50 --device 0 \
  --render-preview
```

The default outputs are `marina_detections.json`, `processing_summary.json`, and, only with `--render-preview`, `marina_detected_preview.mp4`. Existing declared outputs make the command fail rather than overwrite them. Add `--overwrite` to replace only those declared output files.

The JSON uses zero-based `f` frame indices and `t` timestamps in seconds. Each final detection includes an original-frame `poly`, `track_id`, class, confidence, and whether it was interpolated. With `--save-raw`, unsmoothed model detections are saved separately in each frame's `raw_detections` field.

Copy only the final JSON and optional QA preview into the website repository when the output has passed visual review. The model may still produce dock, wake, or shoreline false positives; model-based ship filtering does not guarantee semantic correctness.
