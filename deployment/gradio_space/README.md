---
title: Ship Detector Demo
emoji: "🚢"
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.25.0
python_version: 3.10
app_file: app.py
pinned: false
---

# Maritime Vessel Detection Demo

Gradio Space for the released YOLO11m-OBB maritime ship detector. Upload a JPEG or PNG image to receive an annotated preview and normalized oriented-bounding-box polygons.

## Runtime

For ZeroGPU, use the released PyTorch OBB checkpoint through Ultralytics. `@spaces.GPU` allocates a GPU only while a prediction is running; the model is placed on CUDA during module startup as required by ZeroGPU. This is the recommended hosted mode.

The older `onnx` mode remains available as a CPU-only fallback, using the static-640 ONNX artifact through ONNX Runtime. It does not use ZeroGPU acceleration.

The adapter supports the validated YOLO11 OBB export contracts: a raw `[1, 6, 8400]` head, decoded as `cx, cy, width, height, ship confidence, angle` with deterministic polygon-based rotated NMS, or an NMS-enabled `[1, 300, 7]` graph. The local release artifact currently reports the latter contract. Both paths reverse Ultralytics-compatible letterboxing and return four-corner `ship` polygons.

## Space Variables

For a ZeroGPU Space, configure these in the Space **Variables** settings:

```text
API_MODE=ultralytics
HF_MODEL_REPO_ID=pdramirezlopez/maritime-ship-detector-yolo11m-obb
HF_MODEL_FILENAME=model/maritime_ship_detector_yolo11m_obb_v1.pt
HF_MODEL_REVISION=main
MODEL_INPUT_SIZE=640
MODEL_CONF_THRESHOLD=0.25
MODEL_IOU_THRESHOLD=0.50
MAX_DET=300
MAX_UPLOAD_MB=8
```

Set `HF_TOKEN` as a Space **Secret** only while the model repository is private. Remove the secret after the model repository is public.

Set `API_MODE=mock` to start the UI without downloading a model. Mock mode is intended only to validate the upload and response contract.

For the non-accelerated ONNX fallback, set `API_MODE=onnx` and `HF_MODEL_FILENAME=model/maritime_ship_detector_yolo11m_obb_v1_640.onnx` instead.

## API Endpoint

The prediction event is exposed as the named Gradio endpoint `predict`. It accepts one image and returns:

1. an annotated image;
2. a concise detection summary; and
3. normalized JSON with four-corner OBB polygons.

The endpoint is Gradio-specific; it is not compatible with a FastAPI `/predict` client.

## Local Verification

Use Python 3.10 through 3.12. Python 3.13 is not supported by this pinned Gradio runtime.

```bash
cd ship-detector-api
python -m venv .venv
.venv/bin/pip install -r requirements.txt
API_MODE=mock .venv/bin/python app.py
```

For ZeroGPU mode, select **ZeroGPU** as the Space hardware and use `API_MODE=ultralytics`. The model uses the GPU only inside the decorated prediction function. The adapter enforces a 640px model input.

## Limits

Only JPEG and PNG uploads are accepted. `MAX_UPLOAD_MB` defaults to 8 MB, and decoded images are capped at 24 megapixels to protect the CPU Space from oversized uploads.
