# Plane-initialized patch selection by terrain parallax

In the prior archived study, the same first-ranked camera from the 3D
and fitted-plane arms put at least six projected anchors within 16 pixels
on every render-seed pairing of the faint and bright conditions. Its
orientation and position still missed the frozen pose tolerance. The
original coarse-pose advancement gate failed. The new hypothesis is
that the shared plane initializer places useful search windows and that
patches at different heights can help distinguish the camera poses.
[Sealed prior results](../solar-warp-v125/RESULTS.md)

Planar pose estimation can admit multiple plausible solutions. The
saved benchmark also shows failure mechanisms that a depth-aware
selector would not directly address. On one changed-light miss, RoMa
v2's saved pose retained none of 28 accurate matches in its three-pixel
consensus. On another, Direct's six accurate pairs included two exact
duplicates. Direct's ten distinct accurate pairs in a third condition
all fell in one reference quadrant and supported both the saved and
true poses within three pixels. These diagnostics use scoring truth
after inference. They motivate a patch selection control without
establishing that these lunar cases have two distinct pose solutions.
[Planar pose estimation paper](https://encov.ip.uca.fr/publications/pubfiles/2014_Collins_etal_IJCV_plane.pdf) ·
[Modern benchmark replay](../solar-five-arm-results-v123/README.md) ·
[Saved conditioning analysis](saved_conditioning_v126.json)

## Matched native measurements

Use the top-ranked plane camera from the sealed v125 primary 32-pixel
ranking to place windows. Keep up to eight of its frozen plane
hypotheses for the policy. Sample 36 fixed reference anchors on a
6 × 6 native grid. Retain the same reference-side visibility and patch
support checks for both arms. Each arm gets at most 12 native patch
searches, with the same quota in each occupied spatial cell. It does not
replace failed searches.

The control prefers the anchor closest to its spatial cell centre. The
parallax policy prefers the anchor for which the projected difference
between the true terrain point and its global-plane counterpart best
separates the frozen pose hypotheses. It computes a conservative
separation of projection envelopes under deterministic 0, ±5, ±10 and
±20 m height offsets. The offsets affect reference geometry inside the
policy. The test does not rerender images or demonstrate robustness to
measured map errors. The policy receives no query radiance, depth,
visibility, or true pose.

Both arms use the same native plane template, ±16-pixel reciprocal
search, and original three-pixel AP3P estimator. The solver receives
original 3D reference anchors and image-measured query endpoints.
Keep failures and the same minimum of six distinct matches distributed
across three or more quadrants of each image. Measure each arm
independently and sort its selected anchors before measurement so
search priority does not change AP3P ordering.
[Policy](parallax_policy.py) · [Measurement](measure_v126.py) ·
[Original native estimator](../solar-warp-handoff-v120/handoff_localizer.py)

## Known-light development scope

The earlier ranker chose references under matched illumination. The
same four seed pairings per camera remain repeated renderings of that
camera pair. An archived result can establish whether this choice of
patches improves measured pose recovery on those development images.
Estimated-Sun performance, nonoverlap risk and broader generalization
need separate preregistered evidence. Calibration and final terrain
stay reserved under the original research requirements.
[Original research targets](../requirements_and_sota_v69.json)
