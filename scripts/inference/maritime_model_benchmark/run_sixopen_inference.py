"""Run bounded SixOpen Y8Naval OBB inference on the current grid-tile dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .preprocessing import map_xyxy_to_source, resize_image, valid_xyxy, xyxy_to_xywh
from .schemas import Detection, TileMetadata
from .sixopen_detector import (
    DEFAULT_MODEL_DIR,
    REPOSITORY_ID,
    choose_providers,
    class_map_from_config,
    decode_yolo_obb_raw_output,
    download_model_artifact,
    download_repository_json,
    nms_xyxy,
    normalized_class_name,
    repository_root,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_GRID_ROOT = repository_root() / "data/tiled/grid/current"
DEFAULT_IMAGES_DIR = DEFAULT_GRID_ROOT / "images"
DEFAULT_MANIFEST = DEFAULT_GRID_ROOT / "tile_parent_map.csv"
DEFAULT_OUTPUT_DIR = repository_root() / "outputs/pretrained_model_benchmark/sixopen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=300, help="Per-tile post-NMS cap; use 0 for no cap.")
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=[],
        help="Original model class name to discard before NMS and output; repeat as needed.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Run inference and print aggregate detection counts without writing per-image results or previews.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--smoke-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Select a deterministic random sample instead of the curated smoke-selection set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used with --random-sample.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else repository_root() / path


def load_tile_metadata(manifest_path: Path) -> list[TileMetadata]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        TileMetadata(
            tile_path=row["tile_path"],
            source_image_relative_path=row["tile_path"],
            parent_image=row["parent_name"],
            parent_id=row["parent_id"],
            dataset=row["dataset"],
            source_split=row["source_split"],
            split=row["split"],
            tile_offset_x=int(float(row["tile_x"])),
            tile_offset_y=int(float(row["tile_y"])),
            tile_width=int(float(row["tile_width"])),
            tile_height=int(float(row["tile_height"])),
        )
        for row in rows
    ]


def label_count(grid_root: Path, tile: TileMetadata) -> int:
    label = grid_root / tile.tile_path.replace("images/", "labels/").replace(".png", ".txt")
    if not label.exists():
        return 0
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def _tile_area_stats(grid_root: Path, tile: TileMetadata) -> tuple[int, float, float]:
    label = grid_root / tile.tile_path.replace("images/", "labels/").replace(".png", ".txt")
    areas: list[float] = []
    if label.exists():
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 9:
                continue
            points = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(4, 2)
            areas.append(float((points[:, 0].max() - points[:, 0].min()) * (points[:, 1].max() - points[:, 1].min())))
    return len(areas), min(areas, default=0.0), max(areas, default=0.0)


def select_smoke_tiles(grid_root: Path, tiles: list[TileMetadata], limit: int) -> list[TileMetadata]:
    """Choose a deterministic diversity-oriented smoke set without model-dependent filtering."""
    stats = [(tile, *_tile_area_stats(grid_root, tile)) for tile in tiles]
    selected: list[TileMetadata] = []

    def add(candidate: TileMetadata | None) -> None:
        if candidate is not None and candidate.tile_path not in {item.tile_path for item in selected}:
            selected.append(candidate)

    positives = [item for item in stats if item[1] > 0]
    add(max(positives, key=lambda item: (item[1], item[2]))[0] if positives else None)  # dense
    add(min(positives, key=lambda item: (item[1], item[2]))[0] if positives else None)  # sparse
    xview = [item for item in positives if item[0].dataset == "xview"]
    add(min(xview, key=lambda item: item[2])[0] if xview else None)  # small xView vessel
    add(max(positives, key=lambda item: item[3])[0] if positives else None)  # large vessel envelope
    empty = [item for item in stats if item[1] == 0]
    add(next((item[0] for item in empty if item[0].dataset == "dota"), empty[0][0] if empty else None))
    for tile, *_ in sorted(stats, key=lambda item: item[0].tile_path):
        add(tile)
        if len(selected) >= limit:
            break
    return selected[:limit]


def select_random_tiles(tiles: list[TileMetadata], limit: int, seed: int) -> list[TileMetadata]:
    if limit < 1:
        return []
    return random.Random(seed).sample(tiles, min(limit, len(tiles)))


def confidence_bucket(value: float) -> str:
    if value < 0.01:
        return "<0.01"
    if value < 0.025:
        return "0.01-0.025"
    if value < 0.05:
        return "0.025-0.05"
    if value < 0.10:
        return "0.05-0.10"
    if value < 0.25:
        return "0.10-0.25"
    return ">=0.25"


def select_tiles_for_run(args: argparse.Namespace, tiles: list[TileMetadata], grid_root: Path) -> list[TileMetadata]:
    """Select tiles consistently; count-only mode uses zero as the full-inventory sentinel."""
    if args.count_only and args.max_images == 0:
        return tiles
    if args.random_sample:
        return select_random_tiles(tiles, args.max_images, args.seed)
    if args.smoke_selection:
        return select_smoke_tiles(grid_root, tiles, args.max_images)
    return tiles[: args.max_images]


def image_path_for_tile(images_dir: Path, tile: TileMetadata) -> Path:
    root = images_dir.parent if images_dir.name == "images" else images_dir
    candidate = root / tile.tile_path
    if candidate.exists():
        return candidate
    fallback = images_dir / tile.split / Path(tile.tile_path).name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Unable to resolve image for {tile.tile_path}")


def output_dir_for_tile(output_root: Path, tile: TileMetadata) -> Path:
    return output_root / Path(tile.tile_path).with_suffix("")


def is_completed(result_file: Path) -> bool:
    """A final detections JSON is the only completion marker required for resume."""
    if not result_file.exists() or not result_file.is_file():
        return False
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("detections"), list)


def map_box_to_parent(box: list[float], tile: TileMetadata) -> list[float]:
    """Apply authoritative tile offsets to a source-tile xyxy box."""
    return [
        box[0] + tile.tile_offset_x,
        box[1] + tile.tile_offset_y,
        box[2] + tile.tile_offset_x,
        box[3] + tile.tile_offset_y,
    ]


def map_obb_to_parent(points: list[float], tile: TileMetadata) -> list[float]:
    """Apply authoritative tile offsets to four tile-relative OBB corners."""
    return [
        coordinate + (tile.tile_offset_x if index % 2 == 0 else tile.tile_offset_y)
        for index, coordinate in enumerate(points)
    ]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def obb_points(
    xywhr: list[float], scale_x: float, scale_y: float, image_width: int, image_height: int
) -> list[float]:
    center_x, center_y, width, height, angle = xywhr
    cos_angle, sin_angle = float(np.cos(angle)), float(np.sin(angle))
    corners = np.asarray(
        [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)],
        dtype=np.float32,
    )
    rotation = np.asarray([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
    points = corners @ rotation.T + np.asarray([center_x, center_y], dtype=np.float32)
    points[:, 0] /= scale_x
    points[:, 1] /= scale_y
    points[:, 0] = np.clip(points[:, 0], 0.0, image_width)
    points[:, 1] = np.clip(points[:, 1], 0.0, image_height)
    return [float(value) for value in points.reshape(-1)]


def draw_preview(image_path: Path, detections: list[Detection], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        points = detection.obb
        if points and len(points) == 8:
            draw.polygon([(points[index], points[index + 1]) for index in range(0, 8, 2)], outline="lime", width=3)
        else:
            draw.rectangle(detection.bbox_xyxy, outline="lime", width=3)
        draw.text((detection.bbox_xyxy[0], max(0, detection.bbox_xyxy[1] - 14)), f"{detection.original_class_name} {detection.confidence:.2f}", fill="lime")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def load_gt_boxes(grid_root: Path, tile: TileMetadata) -> list[list[float]]:
    label = grid_root / tile.tile_path.replace("images/", "labels/").replace(".png", ".txt")
    boxes: list[list[float]] = []
    if not label.exists():
        return boxes
    for line in label.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 9:
            continue
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(4, 2)
        points[:, 0] *= tile.tile_width
        points[:, 1] *= tile.tile_height
        boxes.append([float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())])
    return boxes


def iou_xyxy(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions: list[Detection], ground_truth: list[list[float]], threshold: float) -> dict[str, Any]:
    available = set(range(len(ground_truth)))
    matches = []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        best = max(available, key=lambda index: iou_xyxy(prediction.bbox_xyxy, ground_truth[index]), default=None)
        if best is not None and iou_xyxy(prediction.bbox_xyxy, ground_truth[best]) >= threshold:
            matches.append({"detection_id": prediction.detection_id, "ground_truth_index": best})
            available.remove(best)
    return {"matched": len(matches), "missed_ground_truth": len(available), "unmatched_predictions": len(predictions) - len(matches), "matches": matches}


def run_one(
    session: Any,
    input_name: str,
    tile: TileMetadata,
    image_path: Path,
    class_map: dict[int, str],
    metadata: dict[str, Any],
    providers: list[str],
    confidence: float,
    iou_threshold: float,
    max_detections: int,
    excluded_classes: set[str],
) -> tuple[dict[str, Any], list[Detection]]:
    with Image.open(image_path) as source:
        started = time.perf_counter()
        tensor, transform = resize_image(source, 640, 640, channel_order="RGB", normalize=True)
        preprocessing_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    raw_output = session.run(None, {input_name: tensor})[0]
    inference_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    raw_detections = decode_yolo_obb_raw_output(raw_output, confidence_threshold=confidence, class_count=len(class_map))
    excluded_casefold = {name.casefold() for name in excluded_classes}
    raw_detections = [
        item
        for item in raw_detections
        if class_map.get(item["class_id"], str(item["class_id"])).casefold() not in excluded_casefold
    ]
    raw_detections = nms_xyxy(raw_detections, iou_threshold, max_detections)
    detections: list[Detection] = []
    for index, item in enumerate(raw_detections):
        points = obb_points(
            item["xywhr"],
            transform.scale_x,
            transform.scale_y,
            transform.original_width,
            transform.original_height,
        )
        xs, ys = points[0::2], points[1::2]
        box = [max(0.0, min(xs)), max(0.0, min(ys)), min(float(transform.original_width), max(xs)), min(float(transform.original_height), max(ys))]
        if not valid_xyxy(box):
            continue
        parent_box = map_box_to_parent(box, tile)
        parent_obb = map_obb_to_parent(points, tile)
        detections.append(
            Detection(
                detection_id=f"{Path(tile.tile_path).stem}:{index}",
                model_name="SixOpen Y8NavalONNX",
                model_repository=REPOSITORY_ID,
                model_checkpoint=str(metadata["local_path"]),
                checkpoint_sha256=metadata["sha256"],
                source_image=str(image_path),
                source_image_filename=image_path.name,
                source_image_relative_path=tile.tile_path,
                image_width=transform.original_width,
                image_height=transform.original_height,
                original_class_id=item["class_id"],
                original_class_name=class_map[item["class_id"]],
                normalized_class_name=normalized_class_name(class_map[item["class_id"]]),
                confidence=float(item["confidence"]),
                bbox_xyxy=box,
                bbox_xywh=xyxy_to_xywh(box),
                inference_time_ms=inference_ms,
                preprocessing_time_ms=preprocessing_ms,
                postprocessing_time_ms=0.0,
                inference_backend="onnxruntime_direct",
                execution_provider=providers[0],
                input_tensor_size=[1, 3, 640, 640],
                confidence_threshold=confidence,
                nms_iou_threshold=iou_threshold,
                obb=points,
                parent_image=tile.parent_image,
                parent_bbox_xyxy=parent_box,
                parent_obb=parent_obb,
                tile_offset_x=tile.tile_offset_x,
                tile_offset_y=tile.tile_offset_y,
            )
        )
    postprocessing_ms = (time.perf_counter() - started) * 1000
    detections = [
        Detection(**{**item.to_dict(), "postprocessing_time_ms": postprocessing_ms}) for item in detections
    ]
    payload = {
        "source_image": str(image_path),
        "source_image_relative_path": tile.tile_path,
        "tile_metadata": tile.to_dict(),
        "image_width": transform.original_width,
        "image_height": transform.original_height,
        "model": metadata,
        "preprocessing": transform.to_dict(),
        "inference": {
            "backend": "onnxruntime_direct",
            "execution_providers": providers,
            "confidence_threshold": confidence,
            "nms_iou_threshold": iou_threshold,
            "max_detections": max_detections,
            "excluded_original_classes": sorted(excluded_classes),
        },
        "detection_count": len(detections),
        "detections": [item.to_dict() for item in detections],
        "timing_ms": {
            "preprocessing": preprocessing_ms,
            "inference": inference_ms,
            "postprocessing": postprocessing_ms,
        },
        "warnings": [],
    }
    return payload, detections


def run_count_only(
    session: Any,
    input_name: str,
    *,
    selected: list[TileMetadata],
    images_dir: Path,
    class_map: dict[int, str],
    metadata: dict[str, Any],
    providers: list[str],
    args: argparse.Namespace,
    model_load_time_ms: float,
) -> int:
    """Aggregate post-NMS detections in memory without writing image-level artifacts."""
    summary: Counter[str] = Counter(total_selected_images=len(selected))
    durations: list[float] = []
    detections_by_class: Counter[str] = Counter()
    detections_by_dataset: Counter[str] = Counter()
    detections_by_dataset_split: Counter[str] = Counter()
    detections_by_confidence: Counter[str] = Counter()
    failures: list[dict[str, str]] = []

    for index, tile in enumerate(selected, start=1):
        try:
            image_path = image_path_for_tile(images_dir, tile)
            payload, detections = run_one(
                session,
                input_name,
                tile,
                image_path,
                class_map,
                metadata,
                providers,
                args.confidence,
                args.iou_threshold,
                args.max_detections,
                set(args.exclude_class),
            )
            summary["processed_images"] += 1
            summary["total_detections"] += len(detections)
            summary["images_with_detections" if detections else "images_without_detections"] += 1
            durations.append(float(payload["timing_ms"]["inference"]))
            detections_by_class.update(item.original_class_name for item in detections)
            detections_by_dataset.update({tile.dataset: len(detections)})
            detections_by_dataset_split.update({f"{tile.dataset}/{tile.split}": len(detections)})
            detections_by_confidence.update(confidence_bucket(item.confidence) for item in detections)
        except Exception as exc:
            LOGGER.exception("Failed %s", tile.tile_path)
            summary["failed_images"] += 1
            failures.append({"source_image": tile.tile_path, "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0 or index == len(selected):
            LOGGER.info("Counted %d/%d tiles", index, len(selected))

    report = {
        "mode": "count_only",
        "persistence": "none",
        "total_selected_images": len(selected),
        "processed_images": summary["processed_images"],
        "failed_images": summary["failed_images"],
        "images_with_detections": summary["images_with_detections"],
        "images_without_detections": summary["images_without_detections"],
        "total_detections": summary["total_detections"],
        "detections_by_original_class": dict(sorted(detections_by_class.items())),
        "detections_by_dataset": dict(sorted(detections_by_dataset.items())),
        "detections_by_dataset_split": dict(sorted(detections_by_dataset_split.items())),
        "detections_by_confidence_range": dict(sorted(detections_by_confidence.items())),
        "average_inference_time_ms": statistics.mean(durations) if durations else None,
        "median_inference_time_ms": statistics.median(durations) if durations else None,
        "checkpoint_path": str(metadata["local_path"]),
        "checkpoint_sha256": metadata["sha256"],
        "backend": "onnxruntime_direct",
        "execution_provider": providers,
        "input_tensor_shape": [1, 3, 640, 640],
        "model_load_time_ms": model_load_time_ms,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.iou_threshold,
        "max_detections_per_tile": args.max_detections,
        "excluded_original_classes": sorted(args.exclude_class),
        "include_empty_tiles": args.include_empty,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    images_dir, manifest, output_root = resolve(args.images_dir), resolve(args.manifest), resolve(args.output_dir)
    grid_root = images_dir.parent if images_dir.name == "images" else images_dir
    if not manifest.exists():
        raise FileNotFoundError(f"Tile manifest not found: {manifest}")
    checkpoint, metadata = download_model_artifact() if args.checkpoint_path is None else (resolve(args.checkpoint_path), json.loads((resolve(args.checkpoint_path).parent / "model_metadata.json").read_text()))
    config = download_repository_json("config.json", checkpoint.parent, metadata["commit_hash"])
    class_map = class_map_from_config(config)
    if not class_map:
        raise RuntimeError("No class map found in the model repository configuration")
    import onnxruntime as ort

    model_load_started = time.perf_counter()
    requested_providers = choose_providers(args.device)
    session = ort.InferenceSession(str(checkpoint), providers=requested_providers)
    actual_providers = session.get_providers()
    model_load_time_ms = (time.perf_counter() - model_load_started) * 1000
    if args.device == "cuda" and actual_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA was requested but ONNX Runtime created {actual_providers}")
    tiles = load_tile_metadata(manifest)
    if not args.include_empty:
        tiles = [tile for tile in tiles if label_count(grid_root, tile) > 0]
    selected = select_tiles_for_run(args, tiles, grid_root)
    if args.count_only:
        if args.save_previews:
            LOGGER.warning("--save-previews is ignored in --count-only mode.")
        return run_count_only(
            session,
            session.get_inputs()[0].name,
            selected=selected,
            images_dir=images_dir,
            class_map=class_map,
            metadata=metadata,
            providers=actual_providers,
            args=args,
            model_load_time_ms=model_load_time_ms,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary: Counter[str] = Counter(total_discovered_images=len(tiles))
    durations: list[float] = []
    smoke_rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for tile in selected:
        result_dir = output_dir_for_tile(output_root, tile)
        result_file = result_dir / "detections.json"
        if args.resume and is_completed(result_file) and not args.overwrite:
            summary["skipped_images"] += 1
            continue
        image_path = image_path_for_tile(images_dir, tile)
        try:
            payload, detections = run_one(
                session, session.get_inputs()[0].name, tile, image_path, class_map, metadata, actual_providers,
                args.confidence, args.iou_threshold, args.max_detections, set(args.exclude_class),
            )
            atomic_json(result_file, payload)
            if args.save_previews:
                draw_preview(image_path, detections, result_dir / "preview.png")
            ground_truth = load_gt_boxes(grid_root, tile)
            match_025 = match_boxes(detections, ground_truth, 0.25)
            match_050 = match_boxes(detections, ground_truth, 0.50)
            matched_ids = {item["detection_id"] for item in match_025["matches"]}
            unmatched.extend(
                [
                    item.to_dict()
                    for item in detections
                    if item.detection_id not in matched_ids and item.normalized_class_name == "vessel"
                ]
            )
            summary["processed_images"] += 1
            summary["total_detections"] += len(detections)
            summary["images_with_detections" if detections else "images_without_detections"] += 1
            durations.append(float(payload["timing_ms"]["inference"]))
            smoke_rows.append({
                "source_image": tile.tile_path,
                "ground_truth_annotation_count": len(ground_truth),
                "prediction_count": len(detections),
                "predicted_classes": sorted({item.original_class_name for item in detections}),
                "confidence_range": [min((item.confidence for item in detections), default=None), max((item.confidence for item in detections), default=None)],
                "inference_time_ms": payload["timing_ms"]["inference"],
                "parent_mapping_succeeded": True,
                "matching_iou_025": {key: value for key, value in match_025.items() if key != "matches"},
                "matching_iou_050": {key: value for key, value in match_050.items() if key != "matches"},
            })
        except Exception as exc:
            LOGGER.exception("Failed %s", tile.tile_path)
            summary["failed_images"] += 1
            smoke_rows.append({"source_image": tile.tile_path, "error": f"{type(exc).__name__}: {exc}"})
    class_counts = Counter(item["original_class_name"] for item in unmatched)
    atomic_json(output_root / "sixopen_annotation_candidates.json", {"matching_iou": 0.25, "unmatched_predictions": unmatched})
    run_summary = {
        **summary,
        "total_discovered_images": len(tiles),
        "average_inference_time_ms": statistics.mean(durations) if durations else None,
        "median_inference_time_ms": statistics.median(durations) if durations else None,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "backend": "onnxruntime_direct",
        "execution_provider": actual_providers,
        "input_tensor_shape": [1, 3, 640, 640],
        "model_load_time_ms": model_load_time_ms,
        "command_arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "smoke_test": smoke_rows,
        "unmatched_candidates_by_original_class": dict(sorted(class_counts.items())),
    }
    atomic_json(output_root / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
