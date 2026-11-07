# What the compact study establishes

The clearest result is improved confidence calibration under changing rendered
lighting. A retrained 28,272-parameter network correctly accepts 376/384 positive
pairs, with no incorrect acceptance and 0/288 nonoverlap acceptances. The original
checkpoint with the same new calibration procedure retains 368/384. Both return
correct raw proposals for all 384 positives at the declared coarse-registration
tolerance. New descriptor or architecture superiority is not established by
those raw results.

The old frozen gate retains only 100/384 correct pairs. A post-study diagnosis,
using the saved original alignments without inference or threshold fitting,
finds that **all 284 rejected correct proposals fail only its global-correlation
threshold of 0.4**. Every other old gate condition passes for those cases.
This explains why a gate can discard otherwise robust descriptor matches under
changing lighting. See [case-level diagnosis](compact_gate_diagnosis_v2.json).

| Source lighting pair | Correct raw proposals | Old gate accepts | Recalibrated old checkpoint accepts |
|---|---:|---:|---:|
| 0–1 | 128/128 | 2/128 | 122/128 |
| 0–2 | 128/128 | 91/128 | 126/128 |
| 1–2 | 128/128 | 7/128 | 120/128 |

These are zero-based within-geometry lighting ordinals (the figure labels use
one-based ordinals); Sun angles and cross-region Sun-setting equivalence are
unavailable. This diagnosis
explains the consumed test outcomes and does not select a replacement threshold
from them. The new confidence rules were fitted and calibrated on separate
development groups before any final inference.

The primary retrained seed improves correct acceptance over the recalibrated
old checkpoint by 2.08 percentage points (exploratory spatial-component interval
+0.25 to +4.31). The other predefined seeds do not show a consistent gain:
their correct acceptance rates are 91.41% and 98.96%, versus 97.92% for the primary.
The mean across all three is 96.09%, close to the recalibrated old checkpoint's
95.83%. Seed 20261032 accepts one nonoverlap pair. All outcomes remain reported;
the primary seed was fixed before test. See [secondary comparisons](compact_secondary_comparisons_v2.json).

The added reliability head, difficult-negative training and shared multiscale
refinement did not improve the predefined raw-success development criterion in
the single-seed ablations. That is a result under this training budget, rather
than evidence that those ideas can never help.

## Where performance remains weak

- Compact meets the secondary pose criterion in only **2/288 stereo cases**;
  the old CNN also meets 2/288. SIFT meets 7/288 with much lower match/pose
  coverage. Strong controlled registration does not establish strong 3D pose.
- Pose RANSAC uses a **3-original-pixel** inlier tolerance. The selected compact
  matcher samples candidate locations every four benchmark pixels, equivalent
  to 16 original pixels. The correspondence-scoring tolerance is looser at
  12 original pixels. This precision mismatch is a plausible failure mechanism
  for future development experiments; no pose threshold was changed after test.
- Exact projected correspondences recover the camera poses closely on three
  metadata-first development examples. This post-study software control checks
  conventions; it does not repair or validate the matcher's predictions.
- New NASA POLAR configurations yield 23/40 compact acceptances within nominal
  fixed-camera geometry, plus one outside. The old gate gives 8/40 within and
  none outside; the recalibrated old checkpoint gives 22/40 within and none
  outside. Increased nominal coverage therefore comes with a remaining outlier.
  Unknown reference drift/calibration uncertainty prevents a verified sensor
  accuracy claim.

The modern comparison, runtime measurements and complete operating-point curves
are in the [measured report](RESEARCH_ITERATION_6.md). They compare identical
low-resolution inputs and separately calibrated frozen rules. Compact networks
are lunar-trained; the large pretrained baselines are not adapted to this sample.
The comparison is specific to these inputs and CPU profiles.

## Scope and the next useful experiment

Inference uses 128px grids and a tolerance of **12 original 512px source pixels**.
The test has 12 selected spatial components, not 384 independent terrain samples.
Trajectory IDs are absent, shared renderer/terrain dependence remains, and the
explicit trajectory shortfall was recorded after training, before final inference.
Calibration contains only 12 positives and 12 negatives across four components;
observed low error is not a deployment risk guarantee. Real orbital navigation
accuracy remains unverified.

The next defensible study would obtain more independent trajectory/terrain
groups, train with explicit 3D visibility/occlusion supervision, and evaluate
correspondence precision and pose before expanding network size. Confidence
calibration needs broader groups and sensor-transfer evidence. Any new tuning
requires a new test reserve; this study's final cases remain consumed.
