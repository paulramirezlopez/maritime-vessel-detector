"""Focused unit tests for the CPU-only raw YOLO11 OBB decoder."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector_adapter import (  # noqa: E402
    EXPECTED_OUTPUT_SHAPE,
    PredictionError,
    SpaceSettings,
    letterbox,
    mock_detections,
    parse_onnx_obb_output,
    parse_raw_obb_output,
)


def raw_output(*rows: tuple[int, float, float, float, float, float, float]) -> np.ndarray:
    """Build a synthetic raw `[1, 6, 8400]` output from indexed detections."""
    output = np.zeros(EXPECTED_OUTPUT_SHAPE, dtype=np.float32)
    for index, cx, cy, width, height, confidence, angle in rows:
        output[0, :, index] = [cx, cy, width, height, confidence, angle]
    return output


class RawObbDecoderTests(unittest.TestCase):
    def test_letterbox_and_reverse_mapping_on_non_square_image(self) -> None:
        image = Image.new("RGB", (1280, 720), color="white")
        tensor, scale, pad_x, pad_y = letterbox(image, 640)

        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(scale, 0.5)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 140)

        detections = parse_raw_obb_output(
            raw_output((0, 320, 320, 100, 20, 0.9, 0.0)),
            image_width=1280,
            image_height=720,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            max_det=300,
        )

        self.assertEqual(len(detections), 1)
        polygon = detections[0].polygon
        center_x = sum(point[0] for point in polygon) / 4
        center_y = sum(point[1] for point in polygon) / 4
        self.assertAlmostEqual(center_x, 640 / 1280, places=4)
        self.assertAlmostEqual(center_y, 360 / 720, places=4)

    def test_confidence_filtering_and_deterministic_rotated_nms(self) -> None:
        output = raw_output(
            (4, 300, 300, 100, 30, 0.95, 0.1),
            (8, 300, 300, 100, 30, 0.90, 0.1),
            (12, 500, 300, 100, 30, 0.80, 0.1),
            (16, 100, 100, 40, 40, 0.10, 0.0),
        )
        _, scale, pad_x, pad_y = letterbox(Image.new("RGB", (640, 640), color="white"), 640)

        detections = parse_raw_obb_output(
            output,
            image_width=640,
            image_height=640,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            max_det=300,
        )

        self.assertEqual([round(item.confidence, 2) for item in detections], [0.95, 0.8])

    def test_invalid_output_shape_is_rejected(self) -> None:
        _, scale, pad_x, pad_y = letterbox(Image.new("RGB", (640, 640)), 640)
        with self.assertRaises(PredictionError):
            parse_raw_obb_output(
                np.zeros((1, 7, 8400), dtype=np.float32),
                image_width=640,
                image_height=640,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                confidence_threshold=0.25,
                iou_threshold=0.5,
                max_det=300,
            )

    def test_nms_enabled_export_rows_are_supported(self) -> None:
        output = np.zeros((1, 300, 7), dtype=np.float32)
        output[0, 0] = [320, 320, 80, 20, 0.9, 0, 0.0]
        _, scale, pad_x, pad_y = letterbox(Image.new("RGB", (640, 640)), 640)

        detections = parse_onnx_obb_output(
            output,
            image_width=640,
            image_height=640,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            max_det=300,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "ship")
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=5)

    def test_normalized_polygons_are_bounded(self) -> None:
        _, scale, pad_x, pad_y = letterbox(Image.new("RGB", (640, 640)), 640)
        detections = parse_raw_obb_output(
            raw_output((0, 5, 5, 40, 40, 0.9, 0.0)),
            image_width=640,
            image_height=640,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            max_det=300,
        )
        self.assertEqual(len(detections), 1)
        for point in detections[0].polygon:
            self.assertGreaterEqual(point[0], 0.0)
            self.assertLessEqual(point[0], 1.0)
            self.assertGreaterEqual(point[1], 0.0)
            self.assertLessEqual(point[1], 1.0)

    def test_mock_mode_is_available_without_model_settings(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = SpaceSettings.from_environment()
        self.assertEqual(settings.api_mode, "mock")
        detections = mock_detections()
        self.assertGreater(len(detections), 0)
        self.assertEqual(detections[0].class_name, "ship")

    def test_zerogpu_mode_uses_the_pytorch_checkpoint_by_default(self) -> None:
        with patch.dict(os.environ, {"API_MODE": "ultralytics"}, clear=True):
            settings = SpaceSettings.from_environment()
        self.assertEqual(settings.api_mode, "ultralytics")
        self.assertEqual(settings.model_filename, "model/maritime_ship_detector_yolo11m_obb_v1.pt")


if __name__ == "__main__":
    unittest.main()
