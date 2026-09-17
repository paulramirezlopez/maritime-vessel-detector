"""Gradio entry point for the standalone maritime vessel detector Space."""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
import spaces

from detector_adapter import OnnxDetector, SpaceSettings, UltralyticsDetector, predict_uploaded_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

settings = SpaceSettings.from_environment()
detector = UltralyticsDetector(settings) if settings.api_mode == "ultralytics" else OnnxDetector(settings)
if settings.api_mode == "ultralytics":
    # ZeroGPU emulates CUDA at startup and switches to a real GPU inside predict_gpu.
    detector.preload_to_cuda()


def predict_cpu(upload_path: str | None):
    return predict_uploaded_image(upload_path, settings, detector)


@spaces.GPU(duration=30)
def predict_gpu(upload_path: str | None):
    return predict_uploaded_image(upload_path, settings, detector)


predict = predict_gpu if settings.api_mode == "ultralytics" else predict_cpu


def example_images() -> list[list[str]]:
    examples_dir = Path(__file__).with_name("examples")
    return [[str(path)] for path in sorted(examples_dir.glob("*")) if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]


with gr.Blocks(title="Maritime Vessel Detection Demo") as demo:
    gr.Markdown(
        "# Maritime Vessel Detection Demo\n\n"
        "Upload aerial or maritime imagery to run an oriented ship detector. "
        "Detection polygons are normalized in the JSON response and drawn in magenta on the preview."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Aerial or maritime image",
                type="filepath",
                sources=["upload"],
            )
            predict_button = gr.Button("Run detection", variant="primary")
        with gr.Column():
            annotated_output = gr.Image(label="Annotated result", type="pil")
            summary_output = gr.Markdown("Upload an image, then run detection.")
            json_output = gr.JSON(label="Normalized detection response")

    examples = example_images()
    if examples:
        gr.Examples(examples=examples, inputs=image_input, label="Examples")

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[annotated_output, summary_output, json_output],
        api_name="predict",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
