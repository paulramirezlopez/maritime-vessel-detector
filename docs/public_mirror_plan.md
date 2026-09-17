# Public Mirror Plan

## Decision

This repository remains the private development archive. Create a fresh public
mirror from an explicit allowlist rather than publishing this Git history or copying
the full working tree. The mirror contains code, portable templates, release
metadata, and concise methodology summaries only.

## Excluded by Design

The mirror must not include raw datasets, parent images, tiles, annotations, CVAT
exports, videos, model binaries, ONNX files, caches, generated inference outputs,
detailed QC images, detailed reports, environment files, or secrets. Historical
experiments are represented by compact summaries and regeneration references.

## Procedure

1. Generate local-only evidence:

   ```bash
   python scripts/release/audit_public_release.py
   ```

2. Inspect `reports/repo_cleanup/` locally. Resolve any secret-like findings and
   licensing questions before materialization.
3. Validate the allowlist without writing a mirror:

   ```bash
   python scripts/release/materialize_public_mirror.py \
     --output-dir /tmp/maritime-vessel-detector-public
   ```

4. Materialize only after review:

   ```bash
   python scripts/release/materialize_public_mirror.py \
     --output-dir /tmp/maritime-vessel-detector-public \
     --apply
   ```

5. Initialize and push a new public Git repository manually from the resulting
   directory. This tooling intentionally never initializes Git or contacts a remote.

## Release Gates

- Confirm all source-dataset redistribution and attribution obligations.
- Decide whether any public sample imagery is permitted; none is allowlisted now.
- Review the redacted secret scan and all portability findings.
- Verify the model-host release and Gradio Space API separately.
- Preserve the private archive unchanged for detailed historical evidence.
