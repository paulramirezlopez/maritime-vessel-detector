"""In-memory image preprocessing and reversible box coordinate transforms."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

from .schemas import TransformRecord


def letterbox_image(
    image: Image.Image,
    input_width: int,
    input_height: int,
    *,
    channel_order: Literal["RGB", "BGR"] = "RGB",
    normalize: bool = True,
    padding_value: int = 114,
) -> tuple[np.ndarray, TransformRecord]:
    """Letterbox an image and return an NCHW float32 tensor plus inverse metadata."""
    source = image.convert("RGB")
    original_width, original_height = source.size
    uniform_scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(round(original_width * uniform_scale)))
    resized_height = max(1, int(round(original_height * uniform_scale)))
    pad_total_x = input_width - resized_width
    pad_total_y = input_height - resized_height
    pad_left = pad_total_x // 2
    pad_top = pad_total_y // 2
    pad_right = pad_total_x - pad_left
    pad_bottom = pad_total_y - pad_top

    resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_width, input_height), (padding_value,) * 3)
    canvas.paste(resized, (pad_left, pad_top))
    array = np.asarray(canvas, dtype=np.float32)
    if channel_order == "BGR":
        array = array[..., ::-1]
    if normalize:
        array /= 255.0
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    transform = TransformRecord(
        original_width=original_width,
        original_height=original_height,
        input_width=input_width,
        input_height=input_height,
        scale_x=uniform_scale,
        scale_y=uniform_scale,
        uniform_scale=uniform_scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        resize_mode="letterbox",
        channel_order=channel_order,
        input_range="0-1" if normalize else "0-255",
    )
    return tensor, transform


def resize_image(
    image: Image.Image,
    input_width: int,
    input_height: int,
    *,
    channel_order: Literal["RGB", "BGR"] = "RGB",
    normalize: bool = True,
) -> tuple[np.ndarray, TransformRecord]:
    """Directly resize an image, matching models whose metadata disables padding."""
    source = image.convert("RGB")
    original_width, original_height = source.size
    resized = source.resize((input_width, input_height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if channel_order == "BGR":
        array = array[..., ::-1]
    if normalize:
        array /= 255.0
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    transform = TransformRecord(
        original_width=original_width,
        original_height=original_height,
        input_width=input_width,
        input_height=input_height,
        scale_x=input_width / original_width,
        scale_y=input_height / original_height,
        uniform_scale=(input_width / original_width)
        if input_width / original_width == input_height / original_height
        else None,
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        resize_mode="resize",
        channel_order=channel_order,
        input_range="0-1" if normalize else "0-255",
    )
    return tensor, transform


def map_xyxy_to_source(box: np.ndarray | list[float], transform: TransformRecord) -> list[float]:
    """Map a model-space xyxy box back to the original image and clamp bounds."""
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = (x1 - transform.pad_left) / transform.scale_x
    x2 = (x2 - transform.pad_left) / transform.scale_x
    y1 = (y1 - transform.pad_top) / transform.scale_y
    y2 = (y2 - transform.pad_top) / transform.scale_y
    x1 = min(max(x1, 0.0), float(transform.original_width))
    x2 = min(max(x2, 0.0), float(transform.original_width))
    y1 = min(max(y1, 0.0), float(transform.original_height))
    y2 = min(max(y2, 0.0), float(transform.original_height))
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def valid_xyxy(box: list[float]) -> bool:
    return bool(np.isfinite(np.asarray(box, dtype=np.float32)).all() and box[2] > box[0] and box[3] > box[1])
