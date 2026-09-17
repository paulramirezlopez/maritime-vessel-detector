"""Download and inspect the SixOpen Y8Naval ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sixopen_detector import (
    DEFAULT_MODEL_DIR,
    class_map_from_config,
    download_model_artifact,
    download_repository_json,
    inspect_onnx_model,
    repository_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=repository_root() / DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root() / "outputs/pretrained_model_benchmark/sixopen/model_inspection.json",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force-redownload", action="store_true")
    args = parser.parse_args()

    checkpoint, metadata = download_model_artifact(
        args.model_dir, revision=args.revision, force_redownload=args.force_redownload
    )
    config = download_repository_json("config.json", args.model_dir, metadata["commit_hash"])
    preprocessor = download_repository_json("preprocessor_config.json", args.model_dir, metadata["commit_hash"])
    class_map = class_map_from_config(config)
    from .sixopen_detector import write_class_map

    write_class_map(args.model_dir / "class_map.json", class_map)
    inspection = inspect_onnx_model(checkpoint)
    inspection.update(
        {
            "artifact": metadata,
            "repository_config": config,
            "preprocessor_config": preprocessor,
            "class_map": {str(key): value for key, value in class_map.items()},
            "inference_contract": {
                "channel_order": "RGB",
                "resize_mode": "direct_resize",
                "input_range": "0-1",
                "raw_output_interpretation": "xywh + 50 class scores + clockwise rotation radians",
                "coordinates": "pixel coordinates in the fixed 640x640 model image",
            },
            "backend_decision": {
                "ultralytics_probe": {
                    "task_detect": "Rejected: treats the rotation channel as a 51st class score.",
                    "task_obb": "Produces plausible OBB results with the embedded 50-class metadata.",
                },
                "selected_backend": "onnxruntime_direct",
                "reason": (
                    "The raw graph contract is verified and direct ONNX Runtime preserves explicit provider "
                    "selection, raw-class provenance, OBB rotation decoding, NMS, and parent-coordinate mapping."
                ),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"checkpoint: {checkpoint}")
    for item in inspection["inputs"]:
        print(f"input  {item['name']}: {item['shape']} {item['dtype']}")
    for item in inspection["outputs"]:
        print(f"output {item['name']}: {item['shape']} {item['dtype']}")
    print(f"embedded NMS: {inspection['onnx']['has_embedded_nms']}")
    print(f"inspection: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
