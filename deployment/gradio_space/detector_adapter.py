"""ONNX, ZeroGPU PyTorch, and mock detector adapters for the Gradio Space."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


MAGENTA = "#ff3df2"
ACCEPTED_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_PIXELS = 24_000_000
EXPECTED_OUTPUT_SHAPE = [1, 6, 8400]
EXPECTED_NMS_OUTPUT_SHAPE = [1, 300, 7]
MAX_PRE_NMS_CANDIDATES = 3000

logger = logging.getLogger("ship_detector.gradio")


class PredictionError(RuntimeError):
    """An actionable prediction failure that is safe to display in the UI."""


@dataclass(frozen=True)
class SpaceSettings:
    api_mode: Literal["mock", "onnx", "ultralytics"]
    model_repo_id: str | None
    model_filename: str
    model_revision: str
    hf_token: str | None
    input_size: int
    confidence_threshold: float
    iou_threshold: float
    max_det: int
    max_upload_mb: int

    @classmethod
    def from_environment(cls) -> "SpaceSettings":
        api_mode = os.getenv("API_MODE", "mock").lower()
        if api_mode not in {"mock", "onnx", "ultralytics"}:
            raise PredictionError("API_MODE must be 'mock', 'onnx', or 'ultralytics'.")

        try:
            input_size = int(os.getenv("MODEL_INPUT_SIZE", "640"))
            max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "8"))
            max_det = int(os.getenv("MAX_DET", "300"))
            confidence_threshold = float(os.getenv("MODEL_CONF_THRESHOLD", "0.25"))
            iou_threshold = float(os.getenv("MODEL_IOU_THRESHOLD", "0.50"))
        except ValueError as error:
            raise PredictionError("Model and upload environment values must be numeric.") from error

        if input_size != 640:
            raise PredictionError("This release requires MODEL_INPUT_SIZE=640.")
        if max_upload_mb <= 0 or not 1 <= max_det <= 300:
            raise PredictionError("MAX_UPLOAD_MB must be positive and MAX_DET must be between 1 and 300.")
        if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise PredictionError("Model thresholds must be between 0 and 1.")

        return cls(
            api_mode=api_mode,
            model_repo_id=os.getenv("HF_MODEL_REPO_ID") or None,
            model_filename=os.getenv("HF_MODEL_FILENAME")
            or (
                "model/maritime_ship_detector_yolo11m_obb_v1.pt"
                if api_mode == "ultralytics"
                else "model/maritime_ship_detector_yolo11m_obb_v1_640.onnx"
            ),
            model_revision=os.getenv("HF_MODEL_REVISION", "main"),
            hf_token=os.getenv("HF_TOKEN") or None,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            max_upload_mb=max_upload_mb,
        )


@dataclass(frozen=True)
class Detection:
    id: str
    class_name: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]
    obb: dict[str, float] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "className": self.class_name,
            "confidence": round(self.confidence, 4),
            "polygon": [[round(x, 6), round(y, 6)] for x, y in self.polygon],
        }
        if self.obb is not None:
            payload["obb"] = self.obb
        return payload


@dataclass(frozen=True)
class RawObbCandidate:
    anchor_index: int
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    angle: float
    polygon: np.ndarray


def load_uploaded_image(upload_path: str | None, max_upload_mb: int) -> Image.Image:
    if not upload_path:
        raise PredictionError("Upload a JPEG or PNG image before running prediction.")

    path = Path(upload_path)
    if not path.is_file():
        raise PredictionError("The uploaded image is no longer available. Upload it again and retry.")
    if path.stat().st_size > max_upload_mb * 1024 * 1024:
        raise PredictionError(f"Upload a JPEG or PNG image under {max_upload_mb} MB.")

    try:
        with Image.open(path) as source:
            if source.format not in ACCEPTED_FORMATS:
                raise PredictionError("Upload a JPEG or PNG image.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except PredictionError:
        raise
    except (OSError, ValueError) as error:
        raise PredictionError("The uploaded image could not be decoded.") from error

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise PredictionError("The uploaded image is too large. Use an image below 24 megapixels.")
    return image


def mock_detections() -> list[Detection]:
    return [
        Detection(
            id="det_001",
            class_name="ship",
            confidence=0.94,
            polygon=((0.15, 0.28), (0.31, 0.24), (0.34, 0.33), (0.18, 0.37)),
            obb={"cx": 0.245, "cy": 0.305, "width": 0.19, "height": 0.09, "angle": -12.0},
        ),
        Detection(
            id="det_002",
            class_name="ship",
            confidence=0.88,
            polygon=((0.55, 0.43), (0.72, 0.40), (0.75, 0.50), (0.58, 0.53)),
            obb={"cx": 0.65, "cy": 0.465, "width": 0.2, "height": 0.1, "angle": 9.0},
        ),
        Detection(
            id="det_003",
            class_name="ship",
            confidence=0.81,
            polygon=((0.36, 0.68), (0.48, 0.64), (0.52, 0.72), (0.40, 0.77)),
            obb={"cx": 0.44, "cy": 0.705, "width": 0.14, "height": 0.09, "angle": -18.0},
        ),
    ]


class OnnxDetector:
    """Lazily loaded CPU-only detector for the released YOLO11 OBB model."""

    def __init__(self, settings: SpaceSettings) -> None:
        self.settings = settings
        self._session: Any | None = None
        self._input_name: str | None = None

    def predict(self, image: Image.Image) -> list[Detection]:
        session, input_name = self._ensure_session()
        tensor, scale, pad_x, pad_y = letterbox(image, self.settings.input_size)
        try:
            output = np.asarray(session.run(None, {input_name: tensor})[0])
        finally:
            del tensor

        logger.info("ONNX output tensor shape=%s", list(output.shape))
        return parse_onnx_obb_output(
            output,
            image_width=image.width,
            image_height=image.height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence_threshold=self.settings.confidence_threshold,
            iou_threshold=self.settings.iou_threshold,
            max_det=self.settings.max_det,
        )

    def _ensure_session(self) -> tuple[Any, str]:
        if self._session is not None and self._input_name is not None:
            return self._session, self._input_name
        if not self.settings.model_repo_id:
            raise PredictionError("Set HF_MODEL_REPO_ID before using API_MODE=onnx.")

        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
        except ImportError as error:
            raise PredictionError("ONNX dependencies are unavailable. Rebuild the Space from requirements.txt.") from error

        try:
            model_path = hf_hub_download(
                repo_id=self.settings.model_repo_id,
                filename=self.settings.model_filename,
                revision=self.settings.model_revision,
                repo_type="model",
                token=self.settings.hf_token,
            )
        except Exception as error:
            raise PredictionError(
                "The ONNX model could not be downloaded. Check HF_MODEL_REPO_ID, "
                "HF_MODEL_FILENAME, HF_MODEL_REVISION, and HF_TOKEN."
            ) from error

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.log_severity_level = 3
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            raise PredictionError("The ONNX model could not be loaded with CPUExecutionProvider.") from error

        if session.get_providers() != ["CPUExecutionProvider"]:
            raise PredictionError("The Space must use ONNX Runtime CPUExecutionProvider only.")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise PredictionError("The ONNX model must expose exactly one image input and one OBB output.")

        input_shape = list(inputs[0].shape)
        expected_input = [1, 3, self.settings.input_size, self.settings.input_size]
        if input_shape != expected_input:
            raise PredictionError(f"The ONNX input must be fixed at {expected_input}; received {input_shape}.")
        output_shape = list(outputs[0].shape)
        if tuple(output_shape) not in {tuple(EXPECTED_OUTPUT_SHAPE), tuple(EXPECTED_NMS_OUTPUT_SHAPE)}:
            raise PredictionError(
                "The ONNX output must be either the raw OBB head "
                f"{EXPECTED_OUTPUT_SHAPE} or NMS-enabled OBB records {EXPECTED_NMS_OUTPUT_SHAPE}; "
                f"received {output_shape}."
            )

        self._session = session
        self._input_name = inputs[0].name
        return session, inputs[0].name


class UltralyticsDetector:
    """ZeroGPU-only adapter for the released PyTorch OBB checkpoint."""

    def __init__(self, settings: SpaceSettings) -> None:
        self.settings = settings
        self._model: Any | None = None

    def preload_to_cuda(self) -> None:
        """Download and place the model on CUDA during ZeroGPU module startup."""
        if self._model is not None:
            return
        if not self.settings.model_repo_id:
            raise PredictionError("Set HF_MODEL_REPO_ID before using API_MODE=ultralytics.")

        try:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
        except ImportError as error:
            raise PredictionError("ZeroGPU dependencies are unavailable. Rebuild from requirements.txt.") from error

        try:
            model_path = hf_hub_download(
                repo_id=self.settings.model_repo_id,
                filename=self.settings.model_filename,
                revision=self.settings.model_revision,
                repo_type="model",
                token=self.settings.hf_token,
            )
            model = YOLO(model_path, task="obb")
            if model.task != "obb":
                raise PredictionError(f"Expected an OBB model; received task={model.task!r}.")
            model.to("cuda")
        except PredictionError:
            raise
        except Exception as error:
            raise PredictionError(
                "The PyTorch OBB model could not be loaded for ZeroGPU. Check HF_MODEL_FILENAME and HF_TOKEN."
            ) from error
        self._model = model

    def predict(self, image: Image.Image) -> list[Detection]:
        if self._model is None:
            raise PredictionError("The ZeroGPU model was not initialized during Space startup.")

        # Ultralytics treats ndarray inputs as BGR images, so convert the PIL RGB upload.
        source = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        try:
            result = self._model.predict(
                source,
                imgsz=self.settings.input_size,
                conf=self.settings.confidence_threshold,
                iou=self.settings.iou_threshold,
                max_det=self.settings.max_det,
                device=0,
                verbose=False,
            )[0]
        except Exception as error:
            raise PredictionError("ZeroGPU inference failed while running the OBB model.") from error

        if result.obb is None or len(result.obb) == 0:
            return []
        polygons = result.obb.xyxyxyxy.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()
        xywhr = result.obb.xywhr.cpu().numpy()
        detections: list[Detection] = []
        for index, (polygon, confidence, box) in enumerate(zip(polygons, confidences, xywhr), start=1):
            if not math.isfinite(float(confidence)):
                continue
            center_x, center_y, width, height, angle = map(float, box)
            detections.append(
                Detection(
                    id=f"det_{index:03d}",
                    class_name="ship",
                    confidence=float(confidence),
                    polygon=normalize_polygon(np.asarray(polygon), image.width, image.height),
                    obb={
                        "cx": round(clamp(center_x / image.width), 6),
                        "cy": round(clamp(center_y / image.height), 6),
                        "width": round(clamp(width / image.width), 6),
                        "height": round(clamp(height / image.height), 6),
                        "angle": round(math.degrees(angle), 4),
                    },
                )
            )
        return detections


def letterbox(image: Image.Image, input_size: int) -> tuple[np.ndarray, float, int, int]:
    """Apply the centered fixed-shape letterbox convention used by Ultralytics."""
    scale = min(input_size / image.width, input_size / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    pad_left = round((input_size - resized_width) / 2 - 0.1)
    pad_top = round((input_size - resized_height) / 2 - 0.1)

    source = np.asarray(image)
    resized = cv2.resize(source, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_right = input_size - resized_width - pad_left
    pad_bottom = input_size - resized_height - pad_top
    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, pad_left, pad_top


def parse_raw_obb_output(
    output: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> list[Detection]:
    """Decode [1, 6, 8400] YOLO11 OBB predictions and apply deterministic rotated NMS."""
    if list(output.shape) != EXPECTED_OUTPUT_SHAPE:
        raise PredictionError(
            f"Unsupported ONNX output shape {list(output.shape)}; expected {EXPECTED_OUTPUT_SHAPE}."
        )
    rows = output[0].transpose(1, 0)
    candidates: list[RawObbCandidate] = []
    for anchor_index, row in enumerate(rows):
        center_x, center_y, width, height, confidence, angle = map(float, row)
        if (
            not all(math.isfinite(value) for value in row)
            or confidence < confidence_threshold
            or width <= 0
            or height <= 0
        ):
            continue
        polygon = xywhr_to_polygon_pixels(center_x, center_y, width, height, angle)
        if polygon_area(polygon) < 1.0:
            continue
        candidates.append(RawObbCandidate(anchor_index, center_x, center_y, width, height, confidence, angle, polygon))

    candidates.sort(key=lambda item: (-item.confidence, item.anchor_index))
    candidates = candidates[:MAX_PRE_NMS_CANDIDATES]
    kept: list[RawObbCandidate] = []
    for candidate in candidates:
        if any(rotated_iou(candidate.polygon, selected.polygon) > iou_threshold for selected in kept):
            continue
        kept.append(candidate)
        if len(kept) == max_det:
            break

    detections: list[Detection] = []
    for index, candidate in enumerate(kept, start=1):
        center_x = (candidate.center_x - pad_x) / scale
        center_y = (candidate.center_y - pad_y) / scale
        width = candidate.width / scale
        height = candidate.height / scale
        detections.append(
            Detection(
                id=f"det_{index:03d}",
                class_name="ship",
                confidence=candidate.confidence,
                polygon=normalize_polygon(
                    xywhr_to_polygon_pixels(center_x, center_y, width, height, candidate.angle),
                    image_width,
                    image_height,
                ),
                obb={
                    "cx": round(clamp(center_x / image_width), 6),
                    "cy": round(clamp(center_y / image_height), 6),
                    "width": round(clamp(width / image_width), 6),
                    "height": round(clamp(height / image_height), 6),
                    "angle": round(math.degrees(candidate.angle), 4),
                },
            )
        )
    return detections


def parse_onnx_obb_output(
    output: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> list[Detection]:
    """Dispatch only the two validated YOLO11 OBB export contracts."""
    output_shape = list(output.shape)
    if output_shape == EXPECTED_OUTPUT_SHAPE:
        return parse_raw_obb_output(
            output,
            image_width=image_width,
            image_height=image_height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )
    if output_shape != EXPECTED_NMS_OUTPUT_SHAPE:
        raise PredictionError(
            "Unsupported ONNX output shape "
            f"{output_shape}; expected {EXPECTED_OUTPUT_SHAPE} or {EXPECTED_NMS_OUTPUT_SHAPE}."
        )

    # NMS is embedded in this graph. Rows are x, y, width, height, confidence,
    # class ID, and angle in the letterboxed model coordinate frame.
    candidates: list[RawObbCandidate] = []
    for anchor_index, row in enumerate(output[0]):
        center_x, center_y, width, height, confidence, class_id, angle = map(float, row)
        if (
            not all(math.isfinite(value) for value in row)
            or int(class_id) != 0
            or confidence < confidence_threshold
            or width <= 0
            or height <= 0
        ):
            continue
        polygon = xywhr_to_polygon_pixels(center_x, center_y, width, height, angle)
        if polygon_area(polygon) < 1.0:
            continue
        candidates.append(RawObbCandidate(anchor_index, center_x, center_y, width, height, confidence, angle, polygon))

    candidates.sort(key=lambda item: (-item.confidence, item.anchor_index))
    detections: list[Detection] = []
    for index, candidate in enumerate(candidates[:max_det], start=1):
        center_x = (candidate.center_x - pad_x) / scale
        center_y = (candidate.center_y - pad_y) / scale
        width = candidate.width / scale
        height = candidate.height / scale
        detections.append(
            Detection(
                id=f"det_{index:03d}",
                class_name="ship",
                confidence=candidate.confidence,
                polygon=normalize_polygon(
                    xywhr_to_polygon_pixels(center_x, center_y, width, height, candidate.angle),
                    image_width,
                    image_height,
                ),
                obb={
                    "cx": round(clamp(center_x / image_width), 6),
                    "cy": round(clamp(center_y / image_height), 6),
                    "width": round(clamp(width / image_width), 6),
                    "height": round(clamp(height / image_height), 6),
                    "angle": round(math.degrees(candidate.angle), 4),
                },
            )
        )
    return detections


def xywhr_to_polygon_pixels(
    center_x: float, center_y: float, width: float, height: float, angle: float
) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    half_width, half_height = width / 2, height / 2
    width_vector = np.array([half_width * cosine, half_width * sine], dtype=np.float32)
    height_vector = np.array([-half_height * sine, half_height * cosine], dtype=np.float32)
    center = np.array([center_x, center_y], dtype=np.float32)
    return np.array(
        [center - width_vector - height_vector, center + width_vector - height_vector,
         center + width_vector + height_vector, center - width_vector + height_vector],
        dtype=np.float32,
    )


def polygon_area(polygon: np.ndarray) -> float:
    return abs(float(cv2.contourArea(polygon.astype(np.float32))))


def rotated_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_area, second_area = polygon_area(first), polygon_area(second)
    if first_area < 1.0 or second_area < 1.0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    except cv2.error:
        return 0.0
    union = first_area + second_area - max(0.0, float(intersection))
    return max(0.0, float(intersection)) / union if union > 0 else 0.0


def normalize_polygon(polygon: np.ndarray, image_width: int, image_height: int) -> tuple[tuple[float, float], ...]:
    return tuple((clamp(float(x) / image_width), clamp(float(y) / image_height)) for x, y in polygon)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(annotated.size) / 480))
    for detection in detections:
        points = [(round(x * annotated.width), round(y * annotated.height)) for x, y in detection.polygon]
        draw.line(points + [points[0]], fill=MAGENTA, width=line_width, joint="curve")
        label_x, label_y = points[0]
        label = f"{detection.class_name} {detection.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((label_x, label_y), label)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=MAGENTA)
        draw.text((label_x, label_y), label, fill="black")
    return annotated


def build_response(
    image: Image.Image,
    detections: list[Detection],
    *,
    mode: Literal["mock", "onnx", "ultralytics"],
    runtime_ms: float,
) -> dict[str, object]:
    return {
        "image": {"width": image.width, "height": image.height},
        "model": {
            "name": "ship-detector-obb",
            "version": {
                "mock": "mock-gradio",
                "onnx": "onnx-640-gradio",
                "ultralytics": "zerogpu-pytorch-640-gradio",
            }[mode],
        },
        "runtimeMs": round(runtime_ms, 2),
        "detections": [detection.as_dict() for detection in detections],
    }


def format_summary(response: dict[str, object], mode: Literal["mock", "onnx", "ultralytics"]) -> str:
    detections = response["detections"]
    assert isinstance(detections, list)
    confidences = [float(item["confidence"]) for item in detections if isinstance(item, dict)]
    confidence_summary = "No detections" if not confidences else (
        f"Highest confidence: **{max(confidences):.2f}**  |  Average confidence: **{sum(confidences) / len(confidences):.2f}**"
    )
    return (
        f"### {len(detections)} detections\n\n{confidence_summary}\n\n"
        f"Mode: **{mode}**  |  Runtime: **{response['runtimeMs']} ms**"
    )


def predict_uploaded_image(
    upload_path: str | None, settings: SpaceSettings, detector: OnnxDetector | UltralyticsDetector
) -> tuple[Image.Image | None, str, dict[str, object]]:
    try:
        image = load_uploaded_image(upload_path, settings.max_upload_mb)
    except PredictionError as error:
        return None, f"### Upload error\n\n{error}", {"error": str(error), "mode": settings.api_mode}

    started_at = perf_counter()
    try:
        detections = mock_detections() if settings.api_mode == "mock" else detector.predict(image)
        response = build_response(image, detections, mode=settings.api_mode, runtime_ms=(perf_counter() - started_at) * 1000)
        return draw_detections(image, detections), format_summary(response, settings.api_mode), response
    except PredictionError as error:
        logger.warning("Prediction failed: %s", error)
        return image, f"### Inference error\n\n{error}", {"error": str(error), "mode": settings.api_mode}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
