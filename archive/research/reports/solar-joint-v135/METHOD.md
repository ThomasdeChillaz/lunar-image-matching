# Fit a shared camera before collapsing ambiguous image matches

The native diagnostic recovered more two-pixel matches without improving
localization. Faint conditions still had only 2–5 of 8–10 matches within one pixel
and 22–82m translation errors. This changes the estimator's use of image evidence,
rather than fitting another pair-acceptance threshold.

For each of the same 36 reference anchors, build the frozen 64-pixel physical
terrain template and 49×49 native correlation field, including its comparison
halo. Keep all finite supported candidates. Form the correlation potential
exp(20 * (NCC - 0.55)). Marginalize locations with a two-pixel Gaussian and a 0.2
outlier component. Fit one six-parameter camera to the sum of these patch
energies. Missing terrain supplies zero inlier evidence. Projections outside
the declared 16-pixel returned window add a barrier. No query truth is used.

After fitting, choose each image-grid posterior mode using appearance and the
two-pixel geometric density. Fit its fractional peak locally, without averaging
multiple modes. Keep appearance≥0.55, outside-basin margin≥0.015 and independent
reverse-search error≤1 pixel. Use the unchanged three-pixel AP3P/refit.

These are explicitly **camera-conditioned correspondences**, not independent
landmarks or validation checkpoints. Pure projected camera coordinates are not
returned as image measurements. Independent simulation geometry stays with the
scorer. Report the soft camera proposal separately: geometric self-consistency
is not verified localization.

The fresh hard-peak control uses exactly the same fields and image-only peaks.
Its endpoints, AP3P pose and estimator must equal the frozen v134 native3D
control. Both methods use at most 36 forward searches and the same reverse
checks. Keep all original twenty predictions and the cached physical camera
as reference controls. Their recorded times are not fresh measurements.

## Prior art and limits

Probabilistic PnP and pose-driven correspondence learning are established by
[EPro-PnP](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html).
Direct learned feature alignment and joint keypoint/pose refinement are central
to [PixLoc](https://openaccess.thecvf.com/content/CVPR2021/html/Sarlin_Back_to_the_Feature_Learning_Robust_Camera_Localization_From_Pixels_CVPR_2021_paper.html).
This tests a particular masked native-correlation implementation with lunar low
signal and terrain geometry. It establishes no novel general probabilistic pose
formulation. Correlation potentials are uncalibrated.

The eventual hypothesis is that retaining image ambiguity can help a compact
illumination-aware matcher under viewpoint and lighting changes. This classical
pilot supplies no learned-response, unknown-Sun, active-selection, new-geography,
orbital or flight-readiness result.

## Frozen pilot and resource gate

Use exactly the four exposed conditions, reference/query noise seeds0/0, and
their certified nonoverlaps: four positives and four negatives on three terrain
components. Near-uniform cases remain. Initialization is the first camera
already predicted in sealed v134 outputs, bound to the same images and geometry.
It is not the true query pose. Matching illumination is supplied.

A charged 15-second generated pilot must pass image-to-AP3P controls and forecast
the eight-case study within 80 seconds, extrapolating camera fitting to the
fixed 40-step maximum. These are the final 95 seconds of the original baseline
allowance. No charge is refunded or deadline reset. Publish sources, cases and
settings before inference. Seal predictions and correlation-field archives
before query-truth scoring.

A preliminary signal needs one additional correct positive beyond the strongest
matched completed control and fresh hard-peak control, without more wrong
positive/nonoverlap outputs than Direct. Four positives cannot establish a
population improvement or 1% risk. All fourteen original targets, including 70%
correctly accepted poses and the full fresh/transfer comparisons, stay unchanged.
Calibration/final terrain remains closed. No training, model promotion, email or
paper submission is authorized by this pilot.
