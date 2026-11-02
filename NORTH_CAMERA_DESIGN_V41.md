# What the new data can establish

The additional northern dataset has completed 576 images. The unchanged
geometry auditor recovers all 72 development poses and checks all 243,360
fractional supervision points. The separate supplemental render took 726.75
calendar seconds; the original v37 deadline remains recorded as failed.
This establishes data/software controls, not learned matching accuracy.

## A measured camera-design limitation

The metadata-only v41 diagnostic fits three camera templates using the 48
training geometries from 16 terrain groups. Fitting receives training camera
extrinsics only; no query-view IDs, images or depth arrays enter the predictor.
Given a selection reference-camera pose, it returns three possible query poses.

The **oracle best of these three** reconstructs the intended poses of all
24 selection geometries from eight terrain groups exactly to the reported
floating-point precision: maximum measured rotation and translation error zero.
The first fixed template is close on 8/24 geometries. The oracle chooses using
query truth in scoring and is **not a usable matcher or a 100% localization
result**. Intended camera metadata are reconstructed from the frozen generator
and acquired terrain-clearance metadata; this is not a new independent check
of the renderer's measured camera rays.

[Diagnostic evidence](reports/north_pose_prior_diagnostic_v41.json).
Three new isolated metadata regressions pass: held-out reconstruction, refusing
selection metadata during template fitting, and proposal independence from
query truth. The previous full repository suite passed 307 tests/four skips;
do not sum isolated counts into a new full-suite claim. A new complete regression
and fresh software smoke are queued after the active scientific controller;
new neural diagnostics remain gated on those checks.

Terrain separation remains useful, but fixed relative camera motions make the
pose distribution predictable. The ongoing three-seed v39 run remains an
expanded-data architectural control. Its outcome alone cannot establish broad
unseen-pose or trajectory generalization, even if its absolute score is high.

## Consequences for the next experiment

Keep the active renderer/training source, all case lists and original outcomes
unchanged. Preserve the four calibration and twelve final supports. After the
current controlled comparison:

1. Inspect full positive/nonoverlap results and correspondence precision by
   lighting and viewpoint. Retain initialization if training worsens pose.
2. Add an image-selected pose-prior comparator using the training-derived bank.
   Selection must use images and reference geometry only; report the oracle
   upper bound separately. Never select using the query pose or case/view ID.
3. Freeze a genuinely different camera/roll family before any inference on it.
   Validate it first on the excluded pilot terrains. Use new deterministic
   camera samples and overlap checks, preserving existing fixed-template cases
   as historical controls. Do not tune on the untouched reserve.
4. Compare learned precision/reliability against deterministic refinement and
   modern matchers under that stronger protocol. A prior-driven score is not
   sufficient evidence for the original contribution.

No additional terrain or reserve imagery was read for this diagnostic, and no
neural forward, pose-estimator call or optimizer update was run alongside the
active scientific job.

## Related-work implications checked on 2 November 2026

The team's 2024 SIW paper fine-tunes an approximately 11.5-million-parameter
LoFTR on 8,920 perspective training pairs. Its PerspA/PerspB tests contain
460/507 pairs; training pitch/roll reaches 15 degrees, while PerspB reaches
45 degrees. Figure 6's 180-degree azimuth / two-degree elevation-difference
cell reports 67.2% one-pixel reprojection accuracy for the fine-tuned model;
Figure 7 reports 42.9% for the corresponding PerspB cell. These are percentages
of returned correspondences, not accepted camera-pose success rates. Our existing
scorable-only correspondence precision also has a different denominator.
Native-pixel thresholds alone do not make the results comparable. This
publication already tests broader camera variation than our three fixed
templates, strengthening the need for an unseen-camera control.
[Primary seven-page paper, Figures 6–7 and evaluation](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf).

The same team's 2026 LIMA-based map-generation paper is listed on the lead
author's publication page. Its publisher page was inaccessible during this
check; only the author-listed scope was inspected, so no quantitative comparison
or reproduced-method claim is made.
[Author publication record](https://noahzegna.github.io/),
[paper DOI](https://doi.org/10.2514/6.2026-2244).

RoMa v2 already includes subpixel refinement and predicted covariance. On
MegaDepth-1500, its reported pose AUC@5 drops from 62.8 to 46.3 when its matches
are perturbed to reduce precision. Its Hypersim covariance-weighted refinement
raises AUC@1 from 54.9 to 75.8. These are different benchmarks and metrics,
not numerical targets directly comparable to our lunar pose-success rate.
They motivate testing precision and uncertainty but do not make those ideas
new contributions. [Primary paper, Sections 4.8–4.9](https://arxiv.org/html/2511.15706).

LoMa-R is the authors' rotation-invariant model for aerial imagery. The author
repository distinguishes LoMa-B and larger models; model/profile settings must
remain explicit rather than treating one small configuration as the whole
state of the art. [Author implementation](https://github.com/davnords/LoMa).

RoMa-Ω reports stronger cross-benchmark matching after replacing RoMa v2's
backbone with reconstruction-model features. It remains a literature reference
until the required baseline comparisons finish within budget.
[Primary paper](https://arxiv.org/abs/2609.09507).

StereoLunar's author guidance distinguishes raw pair reconstruction from its
interactive demo's further optimization. Our principal matching comparison
must continue to use the common supplied reference geometry and pose estimator;
image-only reconstruction is a separate track.
[Author implementation](https://github.com/clementinegrethen/StereoLunar).

The desired contribution remains a measured compact-model localization advantage
with correct rejection, uncertainty and defensible transfer—not a reinterpretation
of these metadata controls as a trained result. No state-of-the-art claim,
submission or email follows from this diagnostic.
