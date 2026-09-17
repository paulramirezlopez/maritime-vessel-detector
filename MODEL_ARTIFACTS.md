# Model Artifacts

Model binaries are intentionally excluded from this repository and from the public
source mirror. The selected release is hosted separately at
`pdramirezlopez/maritime-ship-detector-yolo11m-obb`.

## Selected Release

- Architecture: YOLO11m-OBB
- Class: `ship`
- Training input size: 1280
- Dataset line: audit-fixed v2
- Training seed: 3
- Selected geometric augmentation: `degrees=180`, `flipud=0.5`, `fliplr=0.5`
- Validation evidence: mAP50-95 `0.70666`, mAP50 `0.89688`, precision `0.95672`,
  recall `0.86197`

The release also documents a 640 px ONNX alternative for constrained inference. It
uses the same model family but has a different fixed input path; serving validation
is required before treating it as interchangeable with the 1280 px training setup.

Always verify artifact checksums from the release metadata before loading a model.
Review Ultralytics and source-dataset licenses before redistribution or deployment.
