"""Shared OBB and polygon conversion helpers for preprocessing scripts."""

from __future__ import annotations

import math
from typing import Iterable


def parse_points(points_str: str) -> list[tuple[float, float]]:
    return [tuple(map(float, pair.split(","))) for pair in points_str.split(";")]


def points_to_cvat_string(points: Iterable[tuple[float, float]]) -> str:
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in points)


def polygon_bbox(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_to_rotated_box(
    pts: list[tuple[float, float]], *, clockwise: bool = False
) -> tuple[float, float, float, float, float] | None:
    """
    Given 4 arbitrary corners of a rectangle, return
    (xtl, ytl, xbr, ybr, rotation) where width >= height.
    """
    if len(pts) != 4:
        return None

    center_x = sum(x for x, _ in pts) / 4.0
    center_y = sum(y for _, y in pts) / 4.0
    center = (center_x, center_y)

    vectors = [(x - center_x, y - center_y) for x, y in pts]
    angles = [math.atan2(vy, vx) for vx, vy in vectors]
    order = sorted(range(4), key=lambda idx: angles[idx])
    pts_sorted = [pts[idx] for idx in order]

    edges = []
    for i in range(4):
        x1, y1 = pts_sorted[i]
        x2, y2 = pts_sorted[(i + 1) % 4]
        edges.append((x2 - x1, y2 - y1))

    e1 = edges[0]
    e1_len = math.hypot(e1[0], e1[1])
    if e1_len == 0:
        return None
    e1_unit = (e1[0] / e1_len, e1[1] / e1_len)

    e2 = None
    e2_len = 0.0
    for vx, vy in edges[1:]:
        v_len = math.hypot(vx, vy)
        if v_len == 0:
            continue
        if abs(vx * e1_unit[0] + vy * e1_unit[1]) < 0.01 * v_len:
            e2 = (vx, vy)
            e2_len = v_len
            break
    if e2 is None:
        e2 = edges[1]
        e2_len = math.hypot(e2[0], e2[1])

    if e1_len >= e2_len:
        width = e1_len
        height = e2_len
        angle_rad = math.atan2(e1[1], e1[0])
    else:
        width = e2_len
        height = e1_len
        angle_rad = math.atan2(e2[1], e2[0])

    angle_deg = math.degrees(angle_rad) % 360.0
    if clockwise:
        angle_deg = (-angle_deg) % 360.0

    xtl = center[0] - width / 2.0
    ytl = center[1] - height / 2.0
    xbr = center[0] + width / 2.0
    ybr = center[1] + height / 2.0

    return (
        round(xtl, 2),
        round(ytl, 2),
        round(xbr, 2),
        round(ybr, 2),
        round(angle_deg, 2),
    )


def rotated_box_to_polygon(
    cx: float, cy: float, width: float, height: float, angle_rad: float
) -> list[tuple[float, float]]:
    hw, hh = width / 2.0, height / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    points = []
    for dx, dy in corners:
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        points.append((x, y))
    return points
