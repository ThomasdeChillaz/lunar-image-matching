# Reference-camera coordinates for every method

The v132 cost pilot failed its generated pose gate before lunar inference.
All 36 image measurements were retained on its flat-scene handoff, but the
unchanged world-frame AP3P/refit returned no front-facing camera. All three
measurement arms recovered the layered scene. Preserve the failure, source,
deadline and full 60-second charge.

This amendment forms 3D points directly from the reference camera's depth and
calibrated rays, runs the identical frozen AP3P/RANSAC/refit in that reference
frame, then transforms the estimated camera back to the body frame. It changes
no image endpoint, threshold, iteration count or pose criterion. All ten methods
receive this same optional frame control, including Direct. Retain all ten
original world-frame outputs, for twenty methods in total. The seven historical
methods must reproduce their original coordinates, poses and estimator evidence.

The three new image arms remain exactly as frozen in
[v132](../solar-local-v132/METHOD.md): native linear radiance, Gaussian integration,
masked correlation, integer/subpixel peaks and planar/3D templates, with equal
36-anchor budgets. All final cameras come from independently measured reciprocal
image endpoints. Query depth, true pose and Sun remain scoring-only inputs.

Unit checks include arbitrary body-frame rotations and million-metre translations,
approximate camera handoffs, missing support, fractional-pixel displacements,
blanks and nonoverlap. The new generated-only 60-second pilot must recover both
camera motions within two metres for every framed new arm, using image-estimated
initialization. Original world-frame failures remain reported negative controls.
No old gate is loosened or erased. A new 900-second full-study reservation is
admitted only after the measured cost forecast passes. Both charges count against
the unchanged cumulative allowance.

Use the same four exposed conditions, sixteen positives and sixteen disjoint
nonoverlaps; all near-uniform failures stay in the denominator. Matching illumination
is supplied. These are three terrain components, not sixteen independent sites.
Calibration and final terrain remain closed. This is a coordinate-conditioning
and measurement engineering study, not a new learned model or novelty claim.

Primary comparison: framed native-3D subpixel versus framed Direct. A preliminary
signal requires at least two additional correct poses beyond the strongest of
all seven preexisting controls in either frame, without more wrong positive or
nonoverlap poses than framed Direct. No uncalibrated raw result establishes the
full accepted-pose, uncertainty, transfer, speed or modern superiority targets.
