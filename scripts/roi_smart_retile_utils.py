"""Geometry helpers for ROI/grid-driven smart retiling."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dataset_utils import (
    CVATBox,
    polygon_area,
    points_to_yolo_obb_line,
    truncate_rotated_box_to_fit,
)
from preprocess.obb_utils import polygon_to_rotated_box, rotated_box_to_polygon

GRID_TO_ROW_COL: dict[int, tuple[int, int]] = {
    1: (2, 0),
    2: (2, 1),
    3: (2, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (0, 0),
    8: (0, 1),
    9: (0, 2),
}

NEIGHBORS_4: dict[int, tuple[int, ...]] = {
    1: (2, 4),
    2: (1, 3, 5),
    3: (2, 6),
    4: (1, 5, 7),
    5: (2, 4, 6, 8),
    6: (3, 5, 9),
    7: (4, 8),
    8: (5, 7, 9),
    9: (6, 8),
}


Rect = tuple[float, float, float, float]
Origin = tuple[int, int]


@dataclass(frozen=True)
class GridComponent:
    grids: tuple[int, ...]
    rects: tuple[Rect, ...]
    bounds: Rect


@dataclass(frozen=True)
class BoxProjection:
    points: tuple[tuple[float, float], ...]
    overlap_ratio: float
    retained_fraction: float


def load_grid_mappings(mapping_path: Path) -> dict[str, tuple[int, ...]]:
    """Load a grid-selection JSON file into an image->sorted grid mapping."""

    raw_text = mapping_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        # Some manually edited ROI maps contain trailing commas or similarly
        # benign JSON formatting artifacts. Normalize only that class of issue.
        relaxed_text = re.sub(r",(\s*[\]}])", r"\1", raw_text)
        payload = json.loads(relaxed_text)
    mappings: dict[str, tuple[int, ...]] = {}

    if isinstance(payload, dict):
        if "image" in payload and "grids" in payload:
            payload = [payload]
        else:
            payload = [
                {"image": image_name, "grids": grids}
                for image_name, grids in payload.items()
            ]

    if not isinstance(payload, list):
        raise TypeError(f"Unsupported ROI mapping structure in {mapping_path}")

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        image_name = str(entry.get("image", "")).strip()
        grids_raw = entry.get("grids", [])
        if not image_name:
            continue
        try:
            grids_set: set[int] = set()
            for grid in grids_raw:
                if isinstance(grid, int):
                    value = grid
                elif isinstance(grid, str) and grid.strip().isdigit():
                    value = int(grid)
                else:
                    continue
                if value in GRID_TO_ROW_COL:
                    grids_set.add(value)
            grids = sorted(grids_set)
        except (TypeError, ValueError):
            continue
        if grids:
            mappings[image_name] = tuple(grids)

    return mappings


def axis_edges(length: int) -> tuple[int, int, int, int]:
    """Return monotonic 3-way partitions for a parent dimension."""

    if length <= 0:
        return (0, 0, 0, 0)

    # Floor-based thirds stay monotonic and deterministic across odd sizes.
    edges = [0, length // 3, (2 * length) // 3, length]
    for idx in range(1, len(edges)):
        if edges[idx] < edges[idx - 1]:
            edges[idx] = edges[idx - 1]
    return tuple(edges)  # type: ignore[return-value]


def grid_cell_rect(width: int, height: int, grid: int) -> Rect:
    if grid not in GRID_TO_ROW_COL:
        raise ValueError(f"Unsupported grid id: {grid}")

    row, col = GRID_TO_ROW_COL[grid]
    x_edges = axis_edges(width)
    y_edges = axis_edges(height)
    return (
        float(x_edges[col]),
        float(y_edges[row]),
        float(x_edges[col + 1]),
        float(y_edges[row + 1]),
    )


def rect_area(rect: Rect) -> float:
    x0, y0, x1, y1 = rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def rect_intersection_area(left: Rect, right: Rect) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def rect_iou(left: Rect, right: Rect) -> float:
    inter = rect_intersection_area(left, right)
    if inter <= 0.0:
        return 0.0
    union = rect_area(left) + rect_area(right) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def rect_bounds(rects: Iterable[Rect]) -> Rect:
    rects = tuple(rects)
    if not rects:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def order_points(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = list(points)
    if len(pts) <= 2:
        return pts
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    return sorted(pts, key=lambda point: math.atan2(point[1] - cy, point[0] - cx))


def polygon_overlap_ratio(points: Iterable[tuple[float, float]], tile_rect: Rect) -> float:
    pts = order_points(points)
    area = polygon_area(pts)
    if area <= 0.0:
        return 0.0

    try:
        from shapely.geometry import Polygon, box as shapely_box

        polygon = Polygon(pts)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return 0.0
        x0, y0, x1, y1 = tile_rect
        return polygon.intersection(shapely_box(x0, y0, x1, y1)).area / area
    except ImportError:
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        x0, y0, x1, y1 = tile_rect
        inter_x0 = max(min(xs), x0)
        inter_y0 = max(min(ys), y0)
        inter_x1 = min(max(xs), x1)
        inter_y1 = min(max(ys), y1)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0
        return min(1.0, ((inter_x1 - inter_x0) * (inter_y1 - inter_y0)) / area)


def componentize_grids(grid_ids: Iterable[int]) -> list[tuple[int, ...]]:
    """Group selected cells into 4-connected components."""

    remaining = {int(grid) for grid in grid_ids if int(grid) in GRID_TO_ROW_COL}
    components: list[tuple[int, ...]] = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = {start}

        while stack:
            current = stack.pop()
            for neighbor in NEIGHBORS_4[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        components.append(tuple(sorted(component)))

    return sorted(components)


def build_grid_components(width: int, height: int, grids: Iterable[int]) -> list[GridComponent]:
    components = []
    for component_grids in componentize_grids(grids):
        rects = tuple(grid_cell_rect(width, height, grid) for grid in component_grids)
        components.append(
            GridComponent(
                grids=component_grids,
                rects=rects,
                bounds=rect_bounds(rects),
            )
        )
    return components


def clamp_origin(x: int, y: int, width: int, height: int, tile_size: int) -> Origin:
    max_x = max(width - tile_size, 0)
    max_y = max(height - tile_size, 0)
    return (min(max(x, 0), max_x), min(max(y, 0), max_y))


def stepped_positions(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        return [start]
    if stop < start:
        return [start]
    positions = list(range(start, stop + 1, step))
    if not positions:
        return [start]
    if positions[-1] != stop:
        positions.append(stop)
    return positions


def candidate_origins_for_component(
    width: int,
    height: int,
    component: GridComponent,
    *,
    tile_size: int,
    stride: int,
) -> list[Origin]:
    """Generate a narrow ROI-bounded origin set for one connected ROI component."""

    seeds: set[tuple[float, float]] = set()
    bounds = component.bounds
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bounds

    for rect in component.rects:
        x0, y0, x1, y1 = rect
        cell_cx = (x0 + x1) / 2.0
        cell_cy = (y0 + y1) / 2.0
        seeds.update(
            {
                (x0, y0),
                (x1 - tile_size, y0),
                (x0, y1 - tile_size),
                (x1 - tile_size, y1 - tile_size),
                (cell_cx - tile_size / 2.0, cell_cy - tile_size / 2.0),
                (bbox_x0, bbox_y0),
                (bbox_x1 - tile_size, bbox_y0),
                (bbox_x0, bbox_y1 - tile_size),
                (bbox_x1 - tile_size, bbox_y1 - tile_size),
                ((bbox_x0 + bbox_x1) / 2.0 - tile_size / 2.0, (bbox_y0 + bbox_y1) / 2.0 - tile_size / 2.0),
            }
        )

    origins: set[Origin] = set()
    for seed_x, seed_y in seeds:
        ox, oy = clamp_origin(
            int(round(seed_x)),
            int(round(seed_y)),
            width,
            height,
            tile_size,
        )
        origins.add((ox, oy))

    return sorted(origins)


def tile_bbox(origin: Origin, tile_width: int, tile_height: int) -> Rect:
    x0, y0 = origin
    return (float(x0), float(y0), float(x0 + tile_width), float(y0 + tile_height))


def tile_roi_coverage(tile_rect: Rect, roi_rects: Iterable[Rect]) -> float:
    tile_area = rect_area(tile_rect)
    if tile_area <= 0.0:
        return 0.0
    covered = sum(rect_intersection_area(tile_rect, roi_rect) for roi_rect in roi_rects)
    return covered / tile_area


def candidate_center_bonus(tile_rect: Rect, component_bounds: Rect) -> float:
    tile_cx = (tile_rect[0] + tile_rect[2]) / 2.0
    tile_cy = (tile_rect[1] + tile_rect[3]) / 2.0
    comp_cx = (component_bounds[0] + component_bounds[2]) / 2.0
    comp_cy = (component_bounds[1] + component_bounds[3]) / 2.0
    diag = math.hypot(component_bounds[2] - component_bounds[0], component_bounds[3] - component_bounds[1])
    if diag <= 0.0:
        return 1.0
    distance = math.hypot(tile_cx - comp_cx, tile_cy - comp_cy)
    return max(0.0, 1.0 - (distance / diag))


def project_box_to_tile(
    box: CVATBox,
    *,
    tile_origin: Origin,
    tile_width: int,
    tile_height: int,
    clockwise: bool = True,
    min_retained_fraction: float = 0.25,
) -> BoxProjection | None:
    """Project a recovered parent box into tile-local coordinates."""

    if tile_width <= 0 or tile_height <= 0:
        return None

    tile_rect = tile_bbox(tile_origin, tile_width, tile_height)
    center_x = (box.xtl + box.xbr) / 2.0
    center_y = (box.ytl + box.ybr) / 2.0
    width = abs(box.xbr - box.xtl)
    height = abs(box.ybr - box.ytl)
    angle_rad = math.radians(box.rotation if clockwise else -box.rotation)
    original_points = order_points(
        rotated_box_to_polygon(center_x, center_y, width, height, angle_rad)
    )
    original_area = polygon_area(list(original_points))
    if original_area <= 0.0:
        return None

    overlap = polygon_overlap_ratio(original_points, tile_rect)
    if overlap <= 0.0:
        return None

    local_points = [(x - tile_origin[0], y - tile_origin[1]) for x, y in original_points]
    box_params = polygon_to_rotated_box(local_points, clockwise=False)
    if box_params is None:
        return None

    xtl, ytl, xbr, ybr, rotation = box_params
    fitted_points = truncate_rotated_box_to_fit(
        CVATBox(
            label=box.label,
            xtl=xtl,
            ytl=ytl,
            xbr=xbr,
            ybr=ybr,
            rotation=rotation,
        ),
        tile_width,
        tile_height,
        clockwise=True,
    )
    if fitted_points is None:
        return None

    fitted_area = polygon_area(fitted_points)
    if fitted_area <= 0.0:
        return None

    retained_fraction = fitted_area / original_area
    return BoxProjection(
        points=tuple(fitted_points),
        overlap_ratio=overlap,
        retained_fraction=retained_fraction,
    )


def projection_to_yolo_line(
    class_id: int,
    projection: BoxProjection,
    tile_width: int,
    tile_height: int,
) -> str:
    return points_to_yolo_obb_line(class_id, list(projection.points), tile_width, tile_height)
