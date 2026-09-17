# Dataset Regeneration Boundary

This repository does not include source imagery, tiles, labels, CVAT exports, or
weights in its public mirror. A user who has valid access to the source datasets can
recreate a compatible private workflow in this order:

1. Obtain DOTA, xView, HRSC2016, and other approved sources under their own terms.
2. Build and document an active parent-image inventory; do not redistribute parent
   imagery without explicit permission.
3. Create or import curated parent-coordinate OBB annotations.
4. Define ROI grid metadata and produce 1024 px grid-constrained tiles.
5. Project parent OBBs to tiles, preserving valid visible geometry and recording tile
   offsets/provenance.
6. Run tile coverage/recovery auditing before producing a training view.
7. Apply a documented filtering policy and freeze train/validation membership.
8. Train from a relative-path configuration and record the dataset fingerprint,
   software versions, and model checksum.

The private scripts are implementation references, not a promise that every source
dataset can be downloaded or redistributed automatically.
