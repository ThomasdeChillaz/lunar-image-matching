# Lunar camera localization under changing illumination

**Thomas de Chillaz — independent research prototype and reproducible study**

## Question

Does retaining multiple lighting and camera hypotheses during native patch
selection improve camera localization over selecting one hypothesis early?

Image similarity is only an intermediate objective. A method can select the
correct illumination or produce correspondences within two pixels while still
estimating an inaccurate camera pose. The study evaluates the full path from
image evidence through correspondences to three-dimensional pose.

## Method and experimental design

- Keep up to eight camera/light branches across four catalogue illuminations.
- Select native-resolution reference patches using disagreement between their
  predicted structural and geometric observations; acquire and update sequentially.
- Give the strong single-camera and beam controls the same total 128-field allowance,
  common reference candidate universe and final AP3P/RANSAC estimator.
- Isolate retention from patch-policy effects with uniform and geometry beam controls.
- Separate reference-only inference from query-truth scoring. Seal every prediction
  before scoring, preserve failures, and independently replay geometry and accounting.

The implementation adds zero trainable parameters. Lighting comes from a discrete
24-image catalogue. The preview's height perturbations are a proxy for map uncertainty,
not certified solar visibility or a new learned illumination model.

## Measured result

All seven methods recover **1/4 raw positive poses** on the same eight-case pilot.
Active beam returns zero wrong positives and 0/4 nonoverlap poses, as does the
old no-extra control. Median positive time is **54.52s versus 21.77s**. There is no
recovery or efficiency advantage; the new method is not promoted.

## What the negative result taught us

**Proposal coverage:** the faint query retains its correct catalogue light, but
its closest retained initial camera is still 364.51m and 12.11° from truth.
Preserving that branch does not recover the pose.

**Geometric precision:** two nearly uniform cases yield inaccurate final poses
despite precise-looking correspondences. The single control's saved trajectory
even visits accurate intermediate soft cameras, which its final discrete matches
and AP3P estimate do not preserve. Those oracle states are diagnostics, not successes.

These observations separate camera-candidate coverage from precision and conditioning
at the final geometry handoff. Follow-up experiments should test those mechanisms
independently at matched cost, using pose recovery rather than descriptor loss or
inlier count to select a method.

## Evidence and limits

The four positives cover three already exposed terrain components; repeated views
are not independent geography. All **56 method/case cells** and **24 historical
control identities** independently replay. Calibration and final terrain remain
unused. Acceptance risk, fresh-terrain performance, modern-model superiority and
real orbital navigation accuracy remain unestablished.

[Measured results and figures](RESULTS.md) · [Readable implementation map](ARCHITECTURE.md) ·
[Reproduction](REPRODUCING.md) · [Original detailed report](../reports/solar-beam-v144/README.md)
