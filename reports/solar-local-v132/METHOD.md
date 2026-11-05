# Native radiance, subpixel peaks and physical terrain templates

The shared-camera engineering control fixes the previous zero-pose result,
but ties Direct at 4/16 while taking about 6.4 times longer. Its photometric
camera proposals can be precise on the bright condition, yet their native
patch handoff loses precision. The faint condition still has too few
independently measured image matches.

This diagnostic integrates original floating-point radiance over a two-pixel
Gaussian, searches 64-pixel patches using common-support masked normalized
correlation, and fits a local two-dimensional quadratic around each strongest
peak. It never averages competing peaks. Reverse matching uses its own
observed template. Keep correlation ≥0.55, alternative-peak margin ≥0.015,
and reciprocal error ≤1 original pixel. Missing terrain remains unknown after
smoothing. Candidate scores use identical known taps in both images and
require at least 2,048 taps and 256 per quadrant. No image percentile clipping or integer brightness conversion
occurs in the new measurement stage.

Three equal-budget arms search at most 36 anchors: planar integer peaks,
planar subpixel peaks, and full reference-3D terrain subpixel peaks. The first
pair isolates fractional peak fitting; the second isolates terrain warping.
Compared with the historical 12-anchor pipeline, radiance handling, filtering
and anchor count also change. Do not attribute that aggregate gain to any
single change. All seven earlier methods are rerun and must reproduce their
saved coordinates, cameras and estimator evidence exactly. Direct receives
the same 25-image reference library and its established settings.

Every final camera still comes from actual reciprocal image endpoints and
the unchanged three-pixel AP3P estimator. Optimized camera projections never
become observed landmarks. Query camera/depth/Sun are not inference inputs.
All 16 positives and 16 certified nonoverlaps remain; eight near-uniform
positive stress cases stay in the denominator. These are four exposed
conditions on three terrain components with matching light supplied, not
fresh geographic or unknown-Sun evidence. Calibration/final groups stay shut.

No neural weights are added. Direct photometric optimization, patch
correlation, subpixel refinement and terrain rendering have established
precedents. This is an engineering diagnostic, with no novelty claim:
[Direct Sparse Odometry](https://arxiv.org/abs/1607.02565),
[RoMa v2](https://arxiv.org/html/2511.15706), and
[lunar offline/online rendering and masked NCC](https://doi.org/10.1016/j.cja.2026.104149).
The original learned Sun/response/active-selection contribution remains open.

A 60-second generated-only control/cost pilot precedes the ten-method study,
capped at 900 additional seconds within the unchanged cumulative allowance.
Sources and case lists are published before inference. Seal predictions
before loading query truth; replay saved metrics independently. A preliminary
positive signal requires at least two additional correct poses over Direct,
with no increase in wrong positive or nonoverlap outputs. This does not
replace the original accepted-pose, uncertainty, transfer and efficiency targets.
