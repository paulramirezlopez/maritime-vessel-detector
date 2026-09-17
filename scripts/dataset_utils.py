#!/usr/bin/env python3
"""
Shared helpers for maritime dataset assembly.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

SHIP_LABEL = "ship"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SOURCE_TILE_PATTERNS = (
    re.compile(r"^(?P<source>.+?)_row\d+_col\d+$"),
    re.compile(r"^(?P<source>.+?)_\d+_\d+$"),
)


@dataclass(frozen=True)
class CVATBox:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    rotation: float


@dataclass(frozen=True)
class CVATImage:
    name: str
    width: int
    height: int
    boxes: tuple[CVATBox, ...]


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def parse_cvat_xml(
    xml_path: Path, allowed_labels: Iterable[str] = (SHIP_LABEL,)
) -> list[CVATImage]:
    allowed = {normalize_whitespace(label) for label in allowed_labels}
    root = ET.parse(xml_path).getroot()

    images: list[CVATImage] = []
    for image_elem in root.findall("image"):
        name = image_elem.get("name")
        width = int(float(image_elem.get("width", "0")))
        height = int(float(image_elem.get("height", "0")))
        boxes: list[CVATBox] = []

        for box_elem in image_elem.findall("box"):
            label = normalize_whitespace(box_elem.get("label", ""))
            if allowed and label not in allowed:
                continue

            try:
                boxes.append(
                    CVATBox(
                        label=label,
                        xtl=float(box_elem.get("xtl", "0")),
                        ytl=float(box_elem.get("ytl", "0")),
                        xbr=float(box_elem.get("xbr", "0")),
                        ybr=float(box_elem.get("ybr", "0")),
                        rotation=float(box_elem.get("rotation", "0")),
                    )
                )
            except (TypeError, ValueError):
                continue

        if name:
            images.append(
                CVATImage(
                    name=name,
                    width=width,
                    height=height,
                    boxes=tuple(boxes),
                )
            )

    return images


def source_group_key(image_name: str) -> str:
    stem = Path(image_name).stem
    for pattern in SOURCE_TILE_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group("source")
    return stem


def resolve_image_path(image_name: str, search_roots: Iterable[Path]) -> Optional[Path]:
    rel = Path(image_name)
    for root in search_roots:
        candidate = root / rel.name
        if candidate.is_file():
            return candidate
    return None


def rotated_box_to_points(
    box: CVATBox, *, clockwise: bool = True
) -> list[tuple[float, float]]:
    cx = (box.xtl + box.xbr) / 2.0
    cy = (box.ytl + box.ybr) / 2.0
    width = abs(box.xbr - box.xtl)
    height = abs(box.ybr - box.ytl)
    # CVAT stores positive rotation in the clockwise screen-space direction.
    # Use that sign convention directly when reconstructing the corner points.
    rotation_rad = math.radians(box.rotation if clockwise else -box.rotation)

    cos_a = math.cos(rotation_rad)
    sin_a = math.sin(rotation_rad)

    corners = [
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ]
    points = []
    for dx, dy in corners:
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        points.append((x, y))
    return points


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _point_inside(x: float, y: float, width: int, height: int, *, eps: float = 1e-6) -> bool:
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    return (-eps <= x <= max_x + eps) and (-eps <= y <= max_y + eps)


def _interval_for_linear(
    base: float, slope: float, low: float, high: float
) -> tuple[float, float] | None:
    eps = 1e-12
    if abs(slope) <= eps:
        return (-math.inf, math.inf) if low <= base <= high else None

    first = (low - base) / slope
    second = (high - base) / slope
    return (min(first, second), max(first, second))


def _intersect_intervals(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float] | None:
    low = max(left[0], right[0])
    high = min(left[1], right[1])
    if low > high:
        return None
    return low, high


def _rotated_axes(rotation_rad: float) -> tuple[tuple[float, float], tuple[float, float]]:
    cos_a = math.cos(rotation_rad)
    sin_a = math.sin(rotation_rad)
    width_axis = (cos_a, sin_a)
    height_axis = (-sin_a, cos_a)
    return width_axis, height_axis


def _rectangle_points(
    cx: float,
    cy: float,
    width_half: float,
    height_half: float,
    width_axis: tuple[float, float],
    height_axis: tuple[float, float],
) -> list[tuple[float, float]]:
    wx, wy = width_axis
    hx, hy = height_axis
    return [
        (cx - width_half * wx - height_half * hx, cy - width_half * wy - height_half * hy),
        (cx + width_half * wx - height_half * hx, cy + width_half * wy - height_half * hy),
        (cx + width_half * wx + height_half * hx, cy + width_half * wy + height_half * hy),
        (cx - width_half * wx + height_half * hx, cy - width_half * wy + height_half * hy),
    ]


def _crop_axis_candidate(
    box: CVATBox,
    width: int,
    height: int,
    *,
    crop_axis: str,
    anchor_sign: int,
    clockwise: bool = True,
) -> tuple[list[tuple[float, float]], float] | None:
    if crop_axis not in {"width", "height"}:
        raise ValueError(f"unknown crop axis: {crop_axis}")
    if anchor_sign not in {-1, 1}:
        raise ValueError(f"invalid anchor sign: {anchor_sign}")
    if width <= 0 or height <= 0:
        return None

    box_width = abs(box.xbr - box.xtl)
    box_height = abs(box.ybr - box.ytl)
    if box_width <= 0 or box_height <= 0:
        return None

    cx = (box.xtl + box.xbr) / 2.0
    cy = (box.ytl + box.ybr) / 2.0
    rotation_rad = math.radians(box.rotation if clockwise else -box.rotation)
    width_axis, height_axis = _rotated_axes(rotation_rad)

    if crop_axis == "width":
        axis_half = box_width / 2.0
        orth_half = box_height / 2.0
        axis_vec = width_axis
        orth_vec = height_axis
    else:
        axis_half = box_height / 2.0
        orth_half = box_width / 2.0
        axis_vec = height_axis
        orth_vec = width_axis

    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))

    # The side we keep unchanged must already be inside the image bounds.
    for orth_sign in (-1, 1):
        fixed_x = (
            cx
            + anchor_sign * axis_half * axis_vec[0]
            + orth_sign * orth_half * orth_vec[0]
        )
        fixed_y = (
            cy
            + anchor_sign * axis_half * axis_vec[1]
            + orth_sign * orth_half * orth_vec[1]
        )
        if not _point_inside(fixed_x, fixed_y, width, height):
            return None

    interval = (0.0, axis_half)

    for orth_sign in (-1, 1):
        base_x = (
            cx
            + anchor_sign * axis_half * axis_vec[0]
            + orth_sign * orth_half * orth_vec[0]
        )
        base_y = (
            cy
            + anchor_sign * axis_half * axis_vec[1]
            + orth_sign * orth_half * orth_vec[1]
        )
        slope_x = -2.0 * anchor_sign * axis_vec[0]
        slope_y = -2.0 * anchor_sign * axis_vec[1]

        ix = _interval_for_linear(base_x, slope_x, 0.0, max_x)
        iy = _interval_for_linear(base_y, slope_y, 0.0, max_y)
        if ix is None or iy is None:
            return None

        interval = _intersect_intervals(interval, ix)
        if interval is None:
            return None
        interval = _intersect_intervals(interval, iy)
        if interval is None:
            return None

    new_half = interval[1]
    if new_half <= 0.0:
        return None

    shift = anchor_sign * (axis_half - new_half)
    cx_new = cx + shift * axis_vec[0]
    cy_new = cy + shift * axis_vec[1]

    if crop_axis == "width":
        new_width_half = new_half
        new_height_half = box_height / 2.0
    else:
        new_width_half = box_width / 2.0
        new_height_half = new_half

    points = _rectangle_points(
        cx_new,
        cy_new,
        new_width_half,
        new_height_half,
        width_axis,
        height_axis,
    )
    if not all(_point_inside(x, y, width, height) for x, y in points):
        return None

    kept_area = (new_width_half * 2.0) * (new_height_half * 2.0)
    original_area = box_width * box_height
    if original_area <= 0.0:
        return None
    return points, kept_area / original_area


def truncate_rotated_box_to_fit(
    box: CVATBox, width: int, height: int, *, clockwise: bool = True
) -> list[tuple[float, float]] | None:
    """Truncate a rotated box by shortening only the protruding axis.

    The side of the rectangle that is already inside the image is left
    untouched. Only one axis is shortened for a given candidate, so the
    resulting label stays a true rectangle instead of a uniform shrink.
    """

    original_points = rotated_box_to_points(box, clockwise=clockwise)
    if all(_point_inside(x, y, width, height) for x, y in original_points):
        return original_points

    box_width = abs(box.xbr - box.xtl)
    box_height = abs(box.ybr - box.ytl)
    if box_width <= 0 or box_height <= 0:
        return None

    x_overflow = 0.0
    y_overflow = 0.0
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    for x, y in original_points:
        x_overflow += max(0.0, -x) + max(0.0, x - max_x)
        y_overflow += max(0.0, -y) + max(0.0, y - max_y)

    preferred_axis = "width" if x_overflow >= y_overflow else "height"

    candidates: list[tuple[str, list[tuple[float, float]], float]] = []
    for crop_axis in ("width", "height"):
        for anchor_sign in (-1, 1):
            candidate = _crop_axis_candidate(
                box,
                width,
                height,
                crop_axis=crop_axis,
                anchor_sign=anchor_sign,
                clockwise=clockwise,
            )
            if candidate is None:
                continue
            points, retained_fraction = candidate
            candidates.append((crop_axis, points, retained_fraction))

    if not candidates:
        return None

    preferred_bonus = 0.05
    best_axis, best_points, _ = max(
        candidates,
        key=lambda item: item[2] + (preferred_bonus if item[0] == preferred_axis else 0.0),
    )
    return best_points


def shrink_rotated_box_to_fit(
    box: CVATBox, width: int, height: int, *, clockwise: bool = True
) -> list[tuple[float, float]] | None:
    return truncate_rotated_box_to_fit(box, width, height, clockwise=clockwise)


def clamp_points(
    points: Iterable[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        return list(points)

    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    clamped = []
    for x, y in points:
        clamped.append((min(max(x, 0.0), max_x), min(max(y, 0.0), max_y)))
    return clamped


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def points_to_yolo_obb_line(
    class_id: int, points: list[tuple[float, float]], width: int, height: int
) -> str:
    if width <= 0 or height <= 0 or len(points) < 3:
        raise ValueError("invalid geometry")

    normalized = []
    for x, y in points:
        normalized.extend([x / width, y / height])
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized)


def read_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size
