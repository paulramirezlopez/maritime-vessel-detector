"""SixOpen Y8Naval ONNX artifact management, inspection, and inference helpers."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import valid_xyxy

REPOSITORY_ID = "SixOpen/Y8NavalONNX"
DEFAULT_MODEL_DIR = Path("models/pretrained_maritime/sixopen")
DEFAULT_CHECKPOINT_NAME = "model.onnx"
NON_VESSEL_CLASSES = {"Dock"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_onnx_file(api: Any, revision: str | None) -> tuple[str, str]:
    info = api.model_info(REPOSITORY_ID, revision=revision, files_metadata=True, token=False)
    candidates = [item for item in info.siblings if item.rfilename.lower().endswith(".onnx")]
    if not candidates:
        raise FileNotFoundError(f"No ONNX artifact found in {REPOSITORY_ID}")
    # Prefer the repository's primary root ONNX artifact, then the largest non-quantized candidate.
    preferred = next((item for item in candidates if item.rfilename == "Y8Naval.onnx"), None)
    if preferred is None:
        non_quantized = [item for item in candidates if "quant" not in item.rfilename.lower()]
        preferred = max(non_quantized or candidates, key=lambda item: item.size or 0)
    return preferred.rfilename, info.sha


def download_model_artifact(
    model_dir: Path | None = None,
    *,
    revision: str | None = None,
    force_redownload: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Fetch the exact public ONNX checkpoint and persist reproducibility metadata."""
    target_dir = (model_dir or repository_root() / DEFAULT_MODEL_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = target_dir / DEFAULT_CHECKPOINT_NAME
    metadata_path = target_dir / "model_metadata.json"
    existing_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        checkpoint.exists()
        and checkpoint.stat().st_size > 0
        and not force_redownload
        and existing_metadata.get("repository_id") == REPOSITORY_ID
        and existing_metadata.get("sha256") == sha256_file(checkpoint)
    ):
        return checkpoint, existing_metadata

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=False)
    filename, commit_hash = _discover_onnx_file(api, revision)

    downloaded = Path(
        hf_hub_download(
            repo_id=REPOSITORY_ID,
            filename=filename,
            revision=commit_hash,
            local_dir=target_dir,
            force_download=force_redownload,
            token=False,
        )
    )
    if not downloaded.exists() or downloaded.stat().st_size == 0:
        raise FileNotFoundError(f"Downloaded checkpoint is missing or empty: {downloaded}")
    if downloaded.resolve() != checkpoint.resolve():
        temporary = checkpoint.with_suffix(".onnx.tmp")
        os.replace(downloaded, temporary)
        os.replace(temporary, checkpoint)
    metadata = {
        "repository_id": REPOSITORY_ID,
        "filename": filename,
        "local_path": str(checkpoint),
        "revision": revision or "main",
        "commit_hash": commit_hash,
        "download_timestamp": datetime.now(UTC).isoformat(),
        "file_size_bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint, metadata


def download_repository_json(filename: str, model_dir: Path, commit_hash: str) -> dict[str, Any] | None:
    """Fetch small repository metadata files when they exist; never require auth."""
    from huggingface_hub import hf_hub_download

    local = model_dir / filename
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    try:
        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY_ID,
                filename=filename,
                revision=commit_hash,
                local_dir=model_dir,
                token=False,
            )
        )
    except Exception:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def class_map_from_config(config: dict[str, Any] | None) -> dict[int, str]:
    if not config:
        return {}
    raw_map = config.get("id2label") or config.get("label2id") or {}
    if "id2label" in config:
        return {int(key): str(value) for key, value in raw_map.items()}
    return {int(value): str(key) for key, value in raw_map.items()}


def write_class_map(path: Path, class_map: dict[int, str]) -> None:
    payload = {
        "repository_id": REPOSITORY_ID,
        "normalization": "Vessel classes normalize to vessel; known non-vessel classes remain explicit.",
        "classes": [
            {
                "original_class_id": class_id,
                "original_class_name": class_map[class_id],
                "normalized_class_name": normalized_class_name(class_map[class_id]),
            }
            for class_id in sorted(class_map)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_class_name(original_class_name: str) -> str:
    return "non_vessel" if original_class_name in NON_VESSEL_CLASSES else "vessel"


def inspect_onnx_model(checkpoint: Path) -> dict[str, Any]:
    """Inspect the graph with ONNX and ONNX Runtime without making decoder assumptions."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(checkpoint), load_external_data=False)
    session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
    graph = model.graph
    model_meta = session.get_modelmeta()
    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()

    def tensor_detail(value: Any) -> dict[str, Any]:
        shape = [dimension if isinstance(dimension, int) else str(dimension) for dimension in value.shape]
        return {
            "name": value.name,
            "shape": shape,
            "dtype": value.type,
            "dimensions": len(shape),
            "dynamic_dimensions": [index for index, dimension in enumerate(shape) if not isinstance(dimension, int)],
        }

    node_types = [node.op_type for node in graph.node]
    return {
        "checkpoint": str(checkpoint),
        "inputs": [tensor_detail(item) for item in session_inputs],
        "outputs": [tensor_detail(item) for item in session_outputs],
        "onnx": {
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "ir_version": model.ir_version,
            "opset_imports": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "graph_name": graph.name,
            "node_count": len(graph.node),
            "node_type_counts": {name: node_types.count(name) for name in sorted(set(node_types))},
            "has_embedded_nms": any(name in {"NonMaxSuppression", "EfficientNMS_TRT", "BatchedNMSDynamic_TRT"} for name in node_types),
        },
        "runtime": {
            "available_providers": ort.get_available_providers(),
            "inspection_session_providers": session.get_providers(),
            "model_description": model_meta.description,
            "graph_description": model_meta.graph_description,
            "domain": model_meta.domain,
            "version": model_meta.version,
            "custom_metadata_map": model_meta.custom_metadata_map,
        },
    }


def choose_providers(device: str = "auto") -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is unavailable in this ONNX Runtime installation")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def nms_xyxy(detections: list[dict[str, Any]], threshold: float, max_detections: int) -> list[dict[str, Any]]:
    """Class-aware greedy NMS for raw horizontal box outputs."""
    kept: list[dict[str, Any]] = []
    for candidate in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if max_detections > 0 and len(kept) >= max_detections:
            break
        suppress = False
        for accepted in kept:
            if candidate["class_id"] != accepted["class_id"]:
                continue
            ax1, ay1, ax2, ay2 = accepted["box"]
            bx1, by1, bx2, by2 = candidate["box"]
            inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
            intersection = inter_w * inter_h
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
            if union > 0 and intersection / union > threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)
    return kept


def decode_yolo_raw_output(
    output: np.ndarray,
    *,
    confidence_threshold: float,
    input_width: int,
    input_height: int,
    class_count: int,
) -> list[dict[str, Any]]:
    """Decode a YOLOv8-style raw xywh+class tensor after its shape was verified."""
    values = np.asarray(output)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 raw output after batch removal, got {values.shape}")
    expected_features = 4 + class_count
    if values.shape[0] == expected_features:
        values = values.T
    if values.shape[1] != expected_features:
        raise ValueError(f"Raw output shape {values.shape} does not match xywh+{class_count} class scores")
    rows: list[dict[str, Any]] = []
    for row in values:
        if not np.isfinite(row).all():
            continue
        class_id = int(np.argmax(row[4:]))
        confidence = float(row[4 + class_id])
        if confidence < confidence_threshold:
            continue
        center_x, center_y, width, height = (float(value) for value in row[:4])
        box = [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2]
        if not valid_xyxy(box):
            continue
        rows.append({"class_id": class_id, "confidence": confidence, "box": box})
    return rows


def decode_yolo_obb_raw_output(
    output: np.ndarray,
    *,
    confidence_threshold: float,
    class_count: int,
) -> list[dict[str, Any]]:
    """Decode verified Y8Naval output: xywh, class scores, then clockwise rotation radians."""
    values = np.asarray(output)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 output after batch removal, got {values.shape}")
    expected_features = 5 + class_count
    if values.shape[0] == expected_features:
        values = values.T
    if values.shape[1] != expected_features:
        raise ValueError(
            f"Y8Naval output {values.shape} does not match xywh + {class_count} scores + rotation"
        )
    rows: list[dict[str, Any]] = []
    for row in values:
        if not np.isfinite(row).all():
            continue
        class_id = int(np.argmax(row[4 : 4 + class_count]))
        confidence = float(row[4 + class_id])
        if confidence < confidence_threshold:
            continue
        center_x, center_y, width, height = (float(value) for value in row[:4])
        box = [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2]
        if not valid_xyxy(box):
            continue
        rows.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "box": box,
                "xywhr": [center_x, center_y, width, height, float(row[-1])],
            }
        )
    return rows
