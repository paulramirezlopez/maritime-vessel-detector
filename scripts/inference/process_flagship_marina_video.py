#!/usr/bin/env python3
"""Export tracked YOLO11 OBB detections from the flagship marina video.

Smoke test:
    python scripts/inference/process_flagship_marina_video.py \
      --video data/video/11159118-hd_1920_1080_30fps.mp4 \
      --model models/best_vessel_yolo_obb.pt \
      --out-dir outputs/flagship_marina_smoke \
      --imgsz 1280 --conf 0.25 --iou 0.50 --device 0 \
      --max-frames 60 --render-preview

Full export:
    python scripts/inference/process_flagship_marina_video.py \
      --video data/video/11159118-hd_1920_1080_30fps.mp4 \
      --model models/best_vessel_yolo_obb.pt \
      --out-dir outputs/flagship_marina \
      --imgsz 1280 --conf 0.25 --iou 0.50 --device 0 \
      --render-preview
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.video_obb_tracking import (
        PolygonTracker,
        TrackDetection,
        canonicalize_polygon,
        final_detections_by_frame,
    )
else:
    from ..lib.video_obb_tracking import (
        PolygonTracker,
        TrackDetection,
        canonicalize_polygon,
        final_detections_by_frame,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
DEFAULT_VIDEO = REPO_ROOT / "data/video/11159118-hd_1920_1080_30fps.mp4"
DEFAULT_MODEL = REPO_ROOT / "models/best_vessel_yolo_obb.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/flagship_marina"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-json", type=Path, default=None, help="Defaults to <out-dir>/marina_detections.json.")
    parser.add_argument("--out-preview", type=Path, default=None, help="Defaults to <out-dir>/marina_detected_preview.mp4.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Defaults to <out-dir>/processing_summary.json.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="0", help="Ultralytics device, such as 0, cpu, or cuda:0.")
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--match-threshold", type=float, default=0.15)
    parser.add_argument("--render-preview", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--classes", nargs="*", default=None, help="Class names or IDs. Defaults to ship/vessel when present.")
    parser.add_argument("--overwrite", action="store_true", help="Replace only declared final output files.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    video, model, out_dir = resolve(args.video), resolve(args.model), resolve(args.out_dir)
    out_json = resolve(args.out_json) if args.out_json else out_dir / "marina_detections.json"
    out_preview = resolve(args.out_preview) if args.out_preview else out_dir / "marina_detected_preview.mp4"
    summary = resolve(args.summary_json) if args.summary_json else out_dir / "processing_summary.json"
    return video, model, out_dir, out_json, out_preview, summary


def validate_args(args: argparse.Namespace, video: Path, model: Path, output_paths: list[Path]) -> None:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")
    if args.imgsz < 32 or args.max_gap < 0 or args.smooth_window < 1 or not 0 <= args.match_threshold <= 1:
        raise ValueError("Invalid imgsz, max-gap, smooth-window, or match-threshold.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive when provided.")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists; use --overwrite to replace it: {joined}")


def resolve_class_ids(classes: list[str] | None, names: dict[int, str]) -> tuple[list[int] | None, list[str]]:
    if classes is None:
        preferred = [class_id for class_id, name in names.items() if name.casefold() in {"ship", "vessel"}]
        return (preferred or None), ([] if preferred else ["No ship/vessel class was found; retaining all model classes."])
    requested = [value.strip() for item in classes for value in item.split(",") if value.strip()]
    if not requested:
        return None, []
    by_name = {name.casefold(): class_id for class_id, name in names.items()}
    selected: set[int] = set()
    for value in requested:
        if value.isdigit() and int(value) in names:
            selected.add(int(value))
        elif value.casefold() in by_name:
            selected.add(by_name[value.casefold()])
        else:
            raise ValueError(f"Unknown class filter {value!r}; available classes: {dict(sorted(names.items()))}")
    return sorted(selected), []


def compact_detection(detection: TrackDetection) -> dict[str, Any]:
    return {
        "track_id": detection.track_id,
        "cls": detection.cls,
        "name": detection.name,
        "conf": round(float(detection.confidence), 4),
        "poly": [[round(float(x), 2), round(float(y), 2)] for x, y in detection.polygon],
        "source": detection.source,
        "interpolated": detection.interpolated,
    }


def raw_detections_from_result(result: Any, width: int, height: int, names: dict[int, str]) -> list[TrackDetection]:
    if not hasattr(result, "obb") or result.obb is None or result.obb.xyxyxyxy is None:
        raise RuntimeError("Ultralytics prediction did not expose OBB polygons; expected result.obb.xyxyxyxy.")
    polygons = result.obb.xyxyxyxy.cpu().numpy()
    confidences = result.obb.conf.cpu().numpy()
    class_ids = result.obb.cls.cpu().numpy().astype(int)
    detections: list[TrackDetection] = []
    for polygon, confidence, class_id in zip(polygons, confidences, class_ids, strict=True):
        canonical = canonicalize_polygon(polygon, width, height)
        if canonical is None:
            continue
        detections.append(TrackDetection(
            frame_index=-1,
            cls=int(class_id),
            name=names.get(int(class_id), str(class_id)),
            confidence=float(confidence),
            polygon=canonical,
        ))
    return detections


def open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"OpenCV returned invalid video metadata for: {path}")
    return capture, {"fps": fps, "width": width, "height": height, "source_frame_count": frame_count}


def require_requested_device(device: str) -> None:
    import torch

    requested = str(device).strip().casefold()
    if requested not in {"", "cpu", "mps"} and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing an implicit CPU fallback.")


def infer_video(
    video: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[int, list[TrackDetection]], dict[int, list[TrackDetection]], dict[str, Any], list[str]]:
    require_requested_device(args.device)
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "obb":
        raise RuntimeError(f"Expected an OBB model, received task={model.task!r}")
    raw_names = model.names
    names_source = raw_names if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    names = {int(class_id): str(name) for class_id, name in names_source.items()}
    class_ids, warnings = resolve_class_ids(args.classes, names)
    capture, metadata = open_video(video)
    tracker = PolygonTracker(args.max_gap, args.match_threshold)
    raw_by_frame: dict[int, list[TrackDetection]] = {}
    inference_times: list[float] = []
    failed_frames: list[int] = []
    raw_count = 0
    frame_index = 0
    limit = min(metadata["source_frame_count"], args.max_frames) if args.max_frames else metadata["source_frame_count"]
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (args.max_frames is not None and frame_index >= args.max_frames):
                break
            started = time.perf_counter()
            try:
                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    classes=class_ids,
                    max_det=1000,
                    verbose=False,
                )[0]
                detections = raw_detections_from_result(result, metadata["width"], metadata["height"], names)
                detections = [replace(item, frame_index=frame_index) for item in detections]
                tracked_detections = tracker.update(frame_index, detections)
                raw_by_frame[frame_index] = tracked_detections
                raw_count += len(tracked_detections)
            except Exception as exc:  # Continue processing remaining frames and preserve the failure in the summary.
                LOGGER.exception("Inference failed for frame %d", frame_index)
                warnings.append(f"Frame {frame_index} inference failed: {type(exc).__name__}: {exc}")
                failed_frames.append(frame_index)
                raw_by_frame[frame_index] = []
            inference_times.append((time.perf_counter() - started) * 1000)
            frame_index += 1
            if frame_index % 25 == 0 or frame_index == limit:
                LOGGER.info("Processed %d/%d frames", frame_index, limit)
    finally:
        capture.release()
    final_by_frame = final_detections_by_frame(tracker.tracks.values(), args.max_gap, args.smooth_window)
    metadata.update({
        "model_task": model.task,
        "model_classes": names,
        "selected_class_ids": class_ids,
        "selected_class_names": [names[class_id] for class_id in class_ids] if class_ids else list(names.values()),
        "requested_device": str(args.device),
        "source_duration_seconds": metadata["source_frame_count"] / metadata["fps"] if metadata["source_frame_count"] else None,
        "processed_duration_seconds": frame_index / metadata["fps"],
        "processed_frame_count": frame_index,
        "raw_detection_count": raw_count,
        "failed_frames": failed_frames,
        "track_count": len(tracker.tracks),
        "average_inference_time_ms": statistics.mean(inference_times) if inference_times else None,
        "median_inference_time_ms": statistics.median(inference_times) if inference_times else None,
    })
    return final_by_frame, raw_by_frame, metadata, warnings


def render_preview(video: Path, output_path: Path, metadata: dict[str, Any], detections_by_frame: dict[int, list[TrackDetection]]) -> None:
    capture, _ = open_video(video)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), metadata["fps"], (metadata["width"], metadata["height"])
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create preview video: {temporary}")
    frame_index = 0
    try:
        while frame_index < metadata["processed_frame_count"]:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not re-decode frame {frame_index} while rendering preview")
            for detection in detections_by_frame.get(frame_index, []):
                points = np.rint(detection.polygon).astype(np.int32).reshape(-1, 1, 2)
                color = (90, 180, 255) if detection.interpolated else (70, 255, 100)
                cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
                x, y = points[0, 0]
                text = f"#{detection.track_id} {detection.confidence:.2f}"
                cv2.putText(frame, text, (int(x), max(16, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    os.replace(temporary, output_path)


def build_payload(
    video: Path,
    model: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    final_by_frame: dict[int, list[TrackDetection]],
    raw_by_frame: dict[int, list[TrackDetection]],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frames = []
    interpolated = 0
    final_count = 0
    for frame_index in range(metadata["processed_frame_count"]):
        detections = final_by_frame.get(frame_index, [])
        interpolated += sum(item.interpolated for item in detections)
        final_count += len(detections)
        frame = {"f": frame_index, "t": round(frame_index / metadata["fps"], 6), "detections": [compact_detection(item) for item in detections]}
        if args.save_raw:
            frame["raw_detections"] = [compact_detection(item) for item in raw_by_frame.get(frame_index, [])]
        frames.append(frame)
    track_lengths = Counter()
    for detections in final_by_frame.values():
        for detection in detections:
            track_lengths[detection.track_id] += 1
    per_frame = [len(frame["detections"]) for frame in frames]
    payload = {
        "schema_version": "1.0",
        "source_video": str(video),
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "frame_count": metadata["processed_frame_count"],
        "source_frame_count": metadata["source_frame_count"],
        "model": str(model),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_gap": args.max_gap,
        "smooth_window": args.smooth_window,
        "frames": frames,
    }
    summary = {
        "source_video": str(video),
        "model": str(model),
        "command": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        **metadata,
        "total_raw_detections": metadata["raw_detection_count"],
        "total_final_detections": final_count,
        "interpolated_detections": interpolated,
        "average_final_detections_per_frame": statistics.mean(per_frame) if per_frame else 0.0,
        "per_frame_detection_count": {"min": min(per_frame, default=0), "mean": statistics.mean(per_frame) if per_frame else 0.0, "max": max(per_frame, default=0)},
        "track_length": {"min": min(track_lengths.values(), default=0), "mean": statistics.mean(track_lengths.values()) if track_lengths else 0.0, "max": max(track_lengths.values(), default=0)},
        "warnings": warnings,
    }
    return payload, summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    video, model, out_dir, out_json, out_preview, summary_json = normalize_paths(args)
    output_paths = [out_json, summary_json] + ([out_preview] if args.render_preview else [])
    validate_args(args, video, model, output_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    final_by_frame, raw_by_frame, metadata, warnings = infer_video(video, model, args)
    payload, summary = build_payload(video, model, args, metadata, final_by_frame, raw_by_frame, warnings)
    summary["total_processing_time_seconds"] = time.perf_counter() - started
    if args.render_preview:
        render_preview(video, out_preview, metadata, final_by_frame)
    atomic_json(out_json, payload)
    atomic_json(summary_json, summary)
    LOGGER.info("Completed %d frames, %d tracks, %d final detections", metadata["processed_frame_count"], metadata["track_count"], summary["total_final_detections"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
