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
