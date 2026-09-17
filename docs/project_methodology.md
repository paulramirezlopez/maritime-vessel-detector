# Methodology

The project trains an oriented-bounding-box vessel detector for mixed aerial and
satellite imagery. Parent images were manually triaged for maritime context, then
processed through a grid-aware 1024 px tiling workflow. Explicit ROI grids restrict
partial parents; full-parent coverage is retained where all grid cells are allowed.

Annotations were curated in CVAT, reconstructed into parent coordinates for audit,
and projected back into tiles using clipped, rectangle-preserving OBB geometry. A
coverage audit validates tile windows, label syntax, parent-to-tile projection,
recovery-tile provenance, duplicate windows, and intentional exclusions. The active
line is `audit_fixed_v2`; accepted cautions are documented as ledgers rather than
silently discarded findings.

Training uses Ultralytics YOLO OBB. The selected controlled recipe uses YOLO11m-OBB
at 1280 input pixels. It retains original OBB geometry, uses the v9 aggressive
keep-bad filtering policy, and applies rotation/flip augmentation appropriate for
overhead vessel headings. All reported measurements are validation results from the
project split, not a substitute for an independent external benchmark.
