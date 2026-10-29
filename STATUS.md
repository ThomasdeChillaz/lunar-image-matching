# Status

## Implemented and executed
- **M0–M2:** original benchmark contract, deterministic procedural checks and classical baselines.
- **M3 software:** checksummed acquisition of 15 real LROC NAC observations, 12 repeat pairs, 3 geographically separate scenes; native nodata masks; shared-map-grid loading; split/provenance validation; offline checkpoint review and validated import. All 12 pairs were visually inspected. Fresh downloads verified all source hashes.
- **M4:** pinned pretrained DISK + LightGlue, weight checksums, real forward-pass validation, masked features and shared RANSAC geometry. Controlled validation: 396 inliers and 0.0689px median endpoint error.
- **M5 software/pilot:** frozen evaluation on all three regions; 126 evaluations (72 repeat-observation comparisons and 54 exact-transform controls), none/CLAHE ablation, failed-fit accounting, threshold sensitivity, CSV/JSON, three figures, failure examples and a two-page PDF report. Source/input hashes and grouped summaries verified.
- **M6 software:** clean-environment installation, 33 passing tests including pretrained inference, dependency consistency check, runnable reproduction command, offline interactive viewer, automated CI, code/model/data notices, citation and pilot release.

## Scientific acceptance still open
Independent point correspondences and their uncertainty are not supplied by the published dataset. The map reference is therefore provisional. M3 independent-reference acceptance, M5 verified navigation-accuracy conclusions and the corresponding research-grade M6 acceptance remain open in [issue #1](https://github.com/ThomasdeChillaz/lunar-image-matching/issues/1). No separate terrain-model experiment or LIMA reproduction is claimed. The released pilot implements the full software workflow; it does not claim completed independent scientific validation.

The existing MILESTONES.md acceptance criteria remain the definition of research completion. Use the checkpoint tools to add independent references and rerun before making navigation-accuracy claims. No future background execution or automatic outreach is scheduled.

## Research extension in progress — 26 October 2026

The requested five-part extension is tracked in [RESEARCH_EXTENSION.md](RESEARCH_EXTENSION.md). All 42 upstream images and 321 pairs are now frozen in a separate manifest, with crop-local Sun direction sourced from archived LROC product metadata. Bidirectional matching diagnostics, conservative correctness labels, development-only policy calibration, source-bound manual-review tooling, range-based MoonAnything acquisition and an original CNSFM Windows execution workflow are implemented. The expanded evaluation is running; physical Hapke renders and metadata have downloaded with CRC/SHA provenance. Human landmark annotations and an independent second review have not been obtained. No new scientific acceptance or improvement claim is made.

## Verified progress — 29 October 2026

- **Expanded real experiment executed:** 321 pairs × three methods × two preprocessing settings = 1,926 bidirectional evaluations. Exact source snapshots, every saved map-reference metric and every geometry diagnostic were audited. Their reference uncertainty remains unknown.
- **Physical adaptation executed:** 48 train, 16 dev and 16 test patches, one source acquisition per patch, geographically separated Tycho bands. The 16,384-parameter projection trained on 358 co-visible training descriptor pairs and selected checkpoint 300 on 126 development pairs. The complete evaluation includes 1,152 method/case measurements.
- **Held-out physical measurements:** frozen DISK/LightGlue correct on 93/128 pairs; adapter 98/128, at the same three-native-pixel threshold. Default SIFT 1/128; CLAHE learned baseline 90/128. These are 128px synthetic/rendered cases, not independently verified real navigation results.
- **Rejection experiment:** the frozen learned policy retains 17 correct held-out fits and accepts zero incorrect fits; the adapter’s policy retains 41 correct fits but accepts three incorrect fits. Both policies reject all 16 known non-overlap negatives. Zero-error development selection did not guarantee zero held-out errors.
- **CNSFM native execution verified:** original binary succeeded on one native pair in every scene, with pinned binary hashes and audited original-output coordinate conventions. Its 512px common-grid detector has produced 2,213 crater predictions across all 42 matched-input images. Local staging confirms exact pixel/mask identity; common-grid Windows execution and final comparison remain open.
- **Reference review:** the user has agreed to review the blank source-bound landmark packet. No completed human annotation has been imported; a second review has not yet been arranged. No synthetic or detector coordinates are treated as independent real checkpoints.
- **Evidence:** [measured report](reports/EXTENSION_REPORT.md) and [saved-measurement audit](reports/extension_audit.json). The audit covers executed geometry/splits/calibration, not full scientific completion. Further real repeat-observation regions, independent reference scoring, real-data gate calibration and adaptation transfer remain open.
