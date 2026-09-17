# Limitations

- Validation results use the project validation split; no independent held-out test
  set is represented by the release metrics.
- DOTA, xView, HRSC2016, and other sources have different acquisition conditions,
  resolutions, labeling conventions, and redistribution terms.
- Very small vessels, blurred/censored regions, dense marinas, dock-adjacent vessels,
  and large objects that cannot fit safely in a 1024 px tile remain difficult cases.
- The audit line documents accepted intentional exclusions. It does not make absent
  ground truth provably correct.
- The release model is a detector, not a vessel identity, type, tracking, or safety
  system. It must not be used for navigation, surveillance decisions, or other
  high-impact decisions without domain-specific validation and safeguards.
- The Gradio deployment path is a deployment candidate until its public API has been
  verified in its target environment.
