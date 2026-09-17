"""Deterministic polygon tracking and smoothing helpers for offline OBB video export."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import count
from typing import Iterable

import cv2
import numpy as np


Polygon = np.ndarray


def _as_polygon(points: Iterable[Iterable[float]]) -> Polygon | None:
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return None
    return polygon


def signed_area(polygon: Polygon) -> float:
    return float(0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)))


def polygon_is_valid(points: Iterable[Iterable[float]], width: int | None = None, height: int | None = None) -> bool:
    polygon = _as_polygon(points)
    if polygon is None:
        return False
    if width is not None and height is not None:
        if np.any(polygon[:, 0] < -1e-3) or np.any(polygon[:, 0] > width + 1e-3):
            return False
        if np.any(polygon[:, 1] < -1e-3) or np.any(polygon[:, 1] > height + 1e-3):
            return False
    contour = polygon.reshape(-1, 1, 2)
    return abs(signed_area(polygon)) >= 1.0 and bool(cv2.isContourConvex(contour))


def canonicalize_polygon(points: Iterable[Iterable[float]], width: int, height: int) -> Polygon | None:
    """Clamp a convex quad, normalize its winding, and choose a stable initial vertex."""
    polygon = _as_polygon(points)
    if polygon is None:
        return None
    polygon[:, 0] = np.clip(polygon[:, 0], 0.0, float(width))
    polygon[:, 1] = np.clip(polygon[:, 1], 0.0, float(height))
    center = polygon.mean(axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    polygon = polygon[np.argsort(angles)]
    # Image coordinates increase downward. Positive signed area is clockwise here.
    if signed_area(polygon) < 0:
        polygon = polygon[::-1]
    start = min(range(4), key=lambda index: (float(polygon[index, 1]), float(polygon[index, 0])))
    polygon = np.roll(polygon, -start, axis=0)
    return polygon if polygon_is_valid(polygon, width, height) else None


def align_polygon(points: Polygon, reference: Polygon | None) -> Polygon:
    """Rotate a canonical quad to minimize corner displacement from its track predecessor."""
    if reference is None:
        return points.copy()
    candidates = [np.roll(points, -shift, axis=0) for shift in range(4)]
    return min(candidates, key=lambda candidate: float(np.square(candidate - reference).sum()))


def polygon_iou(first: Polygon, second: Polygon) -> float:
    """Convex polygon IoU using OpenCV, returning zero for malformed intersections."""
    first_area = abs(signed_area(first))
    second_area = abs(signed_area(second))
    if first_area < 1.0 or second_area < 1.0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    except cv2.error:
        return 0.0
    union = first_area + second_area - max(0.0, float(intersection))
    return max(0.0, float(intersection)) / union if union > 0 else 0.0


def polygon_center(polygon: Polygon) -> Polygon:
    return polygon.mean(axis=0)


def polygon_diagonal(polygon: Polygon) -> float:
    minimum, maximum = polygon.min(axis=0), polygon.max(axis=0)
    return float(np.linalg.norm(maximum - minimum))


def match_score(first: Polygon, second: Polygon) -> float:
    """Combine rotated IoU with an object-aware center-distance term."""
    iou = polygon_iou(first, second)
    distance = float(np.linalg.norm(polygon_center(first) - polygon_center(second)))
    radius = max(64.0, 2.5 * max(polygon_diagonal(first), polygon_diagonal(second)))
    center_score = max(0.0, 1.0 - distance / radius)
    return 0.7 * iou + 0.3 * center_score


@dataclass(frozen=True)
class TrackDetection:
    frame_index: int
    cls: int
    name: str
    confidence: float
    polygon: Polygon
    track_id: int | None = None
    source: str = "model"
    interpolated: bool = False


@dataclass
class Track:
    track_id: int
    cls: int
    name: str
    observations: list[TrackDetection]

    @property
    def last(self) -> TrackDetection:
        return self.observations[-1]


class PolygonTracker:
    """Lightweight deterministic tracker for stable/offline nadir footage."""

    def __init__(self, max_gap: int, match_threshold: float) -> None:
        self.max_gap = max_gap
        self.match_threshold = match_threshold
        self.tracks: dict[int, Track] = {}
        self._ids = count(1)

    def update(self, frame_index: int, detections: list[TrackDetection]) -> list[TrackDetection]:
        active = [
            track for track in self.tracks.values()
            if frame_index - track.last.frame_index <= self.max_gap + 1
        ]
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track in active:
                if track.cls != detection.cls:
                    continue
                score = match_score(track.last.polygon, detection.polygon)
                if score >= self.match_threshold:
                    candidates.append((score, track.track_id, detection_index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        assigned: dict[int, TrackDetection] = {}
        for _, track_id, detection_index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[detection_index]
            aligned = replace(detection, polygon=align_polygon(detection.polygon, track.last.polygon), track_id=track_id)
            track.observations.append(aligned)
            assigned[detection_index] = aligned
            used_tracks.add(track_id)
            used_detections.add(detection_index)
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned:
                continue
            track_id = next(self._ids)
            tracked = replace(detection, track_id=track_id)
            self.tracks[track_id] = Track(track_id, detection.cls, detection.name, [tracked])
            assigned[detection_index] = tracked
        return [assigned[index] for index in range(len(detections))]


def interpolate_short_gaps(track: Track, max_gap: int) -> list[TrackDetection]:
    observations = sorted(track.observations, key=lambda item: item.frame_index)
    completed: list[TrackDetection] = []
    for previous, current in zip(observations, observations[1:]):
        completed.append(previous)
        gap = current.frame_index - previous.frame_index - 1
        if 0 < gap <= max_gap:
            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                completed.append(TrackDetection(
                    frame_index=previous.frame_index + offset,
                    cls=track.cls,
                    name=track.name,
                    confidence=min(previous.confidence, current.confidence),
                    polygon=(previous.polygon + ratio * (current.polygon - previous.polygon)).astype(np.float32),
                    track_id=track.track_id,
                    source="interpolated",
                    interpolated=True,
                ))
    if observations:
        completed.append(observations[-1])
    return completed


def smooth_observations(observations: list[TrackDetection], window: int) -> list[TrackDetection]:
    """Centered moving-average smoothing without crossing a long absence."""
    if window <= 1:
        return list(observations)
    ordered = sorted(observations, key=lambda item: item.frame_index)
    result: list[TrackDetection] = []
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end].frame_index == ordered[end - 1].frame_index + 1:
            end += 1
        segment = ordered[start:end]
        radius = window // 2
        for index, observation in enumerate(segment):
            left, right = max(0, index - radius), min(len(segment), index + radius + 1)
            polygon = np.mean([item.polygon for item in segment[left:right]], axis=0).astype(np.float32)
            result.append(replace(observation, polygon=align_polygon(polygon, observation.polygon)))
        start = end
    return result


def final_detections_by_frame(tracks: Iterable[Track], max_gap: int, smooth_window: int) -> dict[int, list[TrackDetection]]:
    frames: dict[int, list[TrackDetection]] = {}
    for track in tracks:
        for detection in smooth_observations(interpolate_short_gaps(track, max_gap), smooth_window):
            frames.setdefault(detection.frame_index, []).append(detection)
    for detections in frames.values():
        detections.sort(key=lambda item: (item.track_id or 0, item.cls, -item.confidence))
    return frames
