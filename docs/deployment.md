# Deployment Notes

The distributable model artifacts are documented in [MODEL_ARTIFACTS.md](../MODEL_ARTIFACTS.md).
The development archive also contains `ship-detector-api/`, a lightweight Gradio
Space candidate that supports a CPU-oriented ONNX path and a mock mode for UI checks.

Deployment is **not** considered validated merely because source files exist. Before
advertising live inference, verify that the public Space builds, exposes its named
prediction endpoint, accepts an image upload, returns normalized detection JSON, and
has no startup/API errors. Do not expose model-host tokens in a browser frontend.

The 640 px ONNX artifact reduces fixed input activation cost, but it is not a
quantized replacement for the 1280 px training configuration. Validate accuracy,
latency, memory, and polygon alignment in the actual serving environment before
selecting it for production.
