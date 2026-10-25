# Status

M0–M2: published and tested starter.

## M3 — Dataset software delivered; independent-reference audit open
15 checksummed LROC NAC observations and 12 repeat pairs from three geographically separate scenes have been acquired and visually inspected. All source metadata, grouping, nodata masks and map-grid loading are implemented. The release does not supply independent checkpoints or map-alignment uncertainty; those acceptance checks remain open. The local review/import tools support completing them without deriving truth from an evaluated matcher. See DATASET_CARD.md.

M4: implemented and executed. Pinned DISK + LightGlue, verified weight hashes, explicit missing-dependency errors, shared robust geometry and mask handling. Pretrained controlled check: 396 inliers, 0.0689px median error. 13 matcher/integration tests passed.

M5–M6: implementation underway. Later software milestones may proceed with provisional map-reference diagnostics, while independent scientific validation remains explicitly open.
