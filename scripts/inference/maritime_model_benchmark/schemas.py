"""Serializable data structures for SixOpen inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TransformRecord:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
    uniform_scale: float | None
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    resize_mode: str
    channel_order: str
    input_range: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TileMetadata:
    tile_path: str
    source_image_relative_path: str
    parent_image: str
    parent_id: str
    dataset: str
    source_split: str
    split: str
    tile_offset_x: int
    tile_offset_y: int
    tile_width: int
    tile_height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Detection:
    detection_id: str
    model_name: str
    model_repository: str
    model_checkpoint: str
    checkpoint_sha256: str
    source_image: str
    source_image_filename: str
    source_image_relative_path: str
    image_width: int
    image_height: int
    original_class_id: int
    original_class_name: str
    normalized_class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_xywh: list[float]
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    inference_backend: str
    execution_provider: str
    input_tensor_size: list[int]
    confidence_threshold: float
    nms_iou_threshold: float
    obb: list[float] | None = None
    parent_image: str | None = None
    parent_bbox_xyxy: list[float] | None = None
    parent_obb: list[float] | None = None
    tile_offset_x: int | None = None
    tile_offset_y: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
