# Matched lunar relighting pilot

Searching 24 physically rendered reference lighting states recovered three of
four positive poses on the same eight development pairs. It recovered one of
two changed-light poses. SIFT recovered two positives, and frozen compact and
MatchAnything each recovered one. The 2,000-update compact response recovered
zero. This comparison uses absolute camera pose, native 512-pixel images,
common reference geometry, and the common 3-pixel AP3P estimator.

| Method | Correct / 4 | Correct changed light / 2 | Wrong positive poses | Raw poses on nonoverlap / 4 | Median measured inference |
|---|---:|---:|---:|---:|---:|
| Direct render and SIFT | 3 | 1 | 1 | 0 | 1.36 s |
| SIFT2048 | 2 | 0 | 0 | 0 | 0.14 s |
| MatchAnything–EfficientLoFTR | 1 | 0 | 2 | 1 | 3.65 s |
| Frozen compact with shared Sun and active patches | 1 | 0 | 0 | 0 | 9.30 s |
| Distributional compact response, 2,000 updates | 0 | 0 | 3 | 1 | 11.28 s |

These are uncalibrated outputs on four positive and four nonoverlap pairs.
They measure separate two-thread CPU jobs. The MatchAnything timings span its
original and recovery jobs. The direct timings exclude the original render
preparation. Its 32-camera library contains 768 images and carries a
conservative 6,247.81-second preparation accounting bound, including the
original acquisition's query work. The inventory reused these existing bytes.
The shared simulator favors methods that use a rendered reference.

On the changed-light pair that the compact system rejected at its image
information guard, direct rendering found 17 correspondences within one pixel
and recovered the pose with 0.192° rotation error and 6.12 m translation error.
It chose the correct lighting index from the generic library without receiving
the query Sun label. On the other changed-light pair it found ten matches within
two pixels and selected the correct lighting index, but the 21.54 m translation
error exceeded the 16.65 m tolerance. These cases motivate testing a pose
initializer from rendered reference hypotheses and using further patch
measurements to improve translation geometry.

The nearest literature comparison is Ansar and colleagues' LIMA work. Its
fine-tuned LoFTR reports 70.6% of returned matches within one pixel and 85.9%
within five pixels on unseen 0°-elevation illumination. Those correspondence
metrics and the input setup differ from the absolute pose criterion here.
An outreach-ready performance claim requires a reproducible gain on the
larger matched cohort, evidence that the gain transfers beyond the shared
renderer, and the original acceptance and efficiency targets.
[LIMA, Table 1](https://doi.org/10.2514/6.2025-2073)

MatchAnything's initial run halted after four sealed rows because the scorer
combined a full visibility mask with a compact list of projected coordinates.
The repair scatters the coordinates back to their original rows before
scoring. A separate charged run completed only the four unsealed cases.
Independent replay verified all eight across both runs.
[MatchAnything replay](solar_matched_matchanything_recovery_results_v91.json)

The direct run took 29.15 seconds and its independent replay verified all eight
rows. The existing measured-cost rule projects 1,256 seconds for all 384
development cases under the same profile. The 12,000-second full ceiling
permits that evaluation. The pilot establishes which hypothesis to pursue.
The calibration and final terrain groups remain reserved.
[Direct replay](solar_matched_direct_render_results_v91.json) ·
[Compact checkpoint pilot](SOLAR_COARSE_RESPONSE_RESULTS_V90.md) ·
[SIFT pilot](SOLAR_DIAGNOSTICS_V86.md) ·
[Original research requirements](requirements_and_sota_v69.json)
