# Repairing native measurement before testing patch selection

The prior v126 run completed all 16 exposed development pairings with zero
poses and zero retained image measurements in both policies. On the faint
condition, 27 attempts passed coarse correlation and failed when fine
scoring returned to individual noisy pixels. On the bright condition,
24 attempts reached a fine forward match and failed reverse support or
boundary checks. These are distinct measurement failures. The frozen v126
output and its failed feasibility rule remain the historical control.

The corrected matcher averages the same valid pixels in the reference
and target patch for each displacement, and uses eight-pixel cell means
through the fine search. A comparison halo supplies neighbours for peaks
near the 16-pixel return limit. The returned location remains within
that limit. The ambiguity margin compares distinct pooled correlation
basins. Reciprocal error remains at most one native pixel, correlation
at least 0.55, and the margin at least 0.015.
[Corrected matcher](masked_patch.py) · [Regression tests](test_masked_patch.py)

Four patch arms isolate the changes. Legacy repeats the old measurement
under the common input encoding, halo adds comparison context with the
old pixel score and strict support, pooled-uniform repairs the support
and fine representation, and pooled-parallax adds the v126 selection
policy. They share the same initializer, native reference template and
pose estimator. The initializer is recomputed from the current input
pair for positives and nonoverlaps, without query geometry.

The Direct comparison uses the published SIFT settings, 2,048 features
and 0.75 ratio, over all 24 physical reference renders plus the matched
reference supplied to the patch methods. It uses the original selection
of up to eight pose candidates and common three-pixel AP3P estimator.
All measurements use native 512-pixel images. A fixed independent
1st–99th percentile linear encoding gives every method the same
eight-bit reference and query content at its measurement stage. Direct's
reference library receives that encoding too. The coarse physical
initializer uses the original float radiance and reference geometry.
[Pipeline and baseline](pipeline.py)

The comparison includes all 16 original positive render pairings and
16 cross-terrain negatives. Negative endpoints are a deterministic
derangement with certified disjoint full scene bounds in the original
lunar body frame. Both endpoints are recorded. The conditions reuse
three terrain groups, including two cases from the same group.
This is development evidence under supplied matching illumination.

The 60-second synthetic cost pilot must admit the full 32-pair comparison
under a 600-second reservation. A raw improvement passes the pilot rule
only if a corrected arm recovers at least two more positive poses than
Direct while increasing neither wrong positive poses nor nonoverlap
proposals. All five arms run on the same CPU and inputs, with all failures
in the denominators. Calibration and final terrain remain reserved under
the original fourteen requirements.
[Frozen experiment](experiment.py)
