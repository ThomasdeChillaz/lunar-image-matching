# Native pose pilots: measured negative result

Both trained response models completed their predeclared cost pilots and
independent saved-metric replays. **There is no measured pose-recovery gain
across training checkpoints in this pilot.** The full studies were refused by
the existing admission rule before any additional scientific reservation.

## Fixed experiment

- Native 512-pixel images; 95,808 parameters; two CPU threads.
- Primary track estimates query Sun. Query depth, true pose and true Sun are
  scoring labels, not inference inputs.
- Shared Sun, learned response and active selection are enabled; 128 native
  searches, four proposal rotations, eight hypotheses, the original 4,096
  proposal samples and common 3-pixel final pose estimator are retained.
- Each mode evaluates checkpoints 0/500/1,000/1,500/2,000 on the same eight
  pairs. These are **four unique positive and four unique negative pairs**,
  repeated across five checkpoints, not 40 independent terrain tests.
- There are four reference terrain groups and eight groups including both
  negative endpoints. Calibration and final reserve terrain remain closed.

## Results

| Mode | Completed rows | Correct raw poses at every checkpoint | Changed-light positives | Raw poses on nonoverlap at every checkpoint | External pilot time |
|---|---:|---:|---:|---:|---:|
| Directional SH response | 40/40 | 1/4 | 0/2 | 0/4 | 471.95 s |
| Matched pooled response | 40/40 | 1/4 | 0/2 | 0/4 | 474.11 s |

Both modes recover the same unchanged-light pair. All rows complete without a
backend failure. The changed-light pairs also have roll-varied cameras, so
their failures do not isolate illumination as the sole cause. Native matches
can change across checkpoints even though the binary pose result does not.
This subset selects no checkpoint or model. There is no matched A/B/C-off
comparison yet, so these results cannot establish any module's benefit.

These are **raw research proposals**, not calibrated acceptance results.
Public outputs remain `pose: null` without a validated acceptance policy.
Zero outputs on four mismatches cannot establish a 1% deployment-error bound.
The historical attention result of 15/96 is preserved and uses different
cases; subtracting the percentages would not measure an improvement.

Sun inference is also unverified as a reliable physical estimate. The three
positive pairs that emit a Sun vector have saved angular errors of **39.99°,
58.80° and 2.36°**, unchanged across these checkpoints and modes. All carry
`solar_supported: true`; that flag records map/observation support, not correct
Sun direction. A correct camera pose can coexist with an incorrect Sun
estimate. The vector arithmetic and prediction hashes are in the results JSON.

## What fails

The saved-array diagnosis identifies a bootstrap problem:

1. One positive pair exits the blank/uninformative-image guard before proposals.
   It remains a failure in the denominator.
2. Two others initialize zero of eight poses and retain zero through all rounds.
   Their learned response refresh and usage counts are zero at every checkpoint.
   More response training alone cannot change those recorded executions.
3. Their top-eight coarse candidates are weak and spatially concentrated.
   Only 30/233 and 7/225 visible anchors have a candidate within a 16-pixel disk
   of the saved projected location. This disk is a diagnostic, not an exact
   certificate of support for the square native search window.
4. All eight retained beams have `initial_candidate_rank=0`. In one failed
   pair their assignment vectors are identical at every saved update; the
   other has very small assignment differences. Premeasurement vectors were
   not recorded, so the diagnosis does not reconstruct them.

See [the saved bootstrap diagnosis](SOLAR_POSE_BOOTSTRAP_DIAGNOSIS_V84.md).
Candidate coverage and pose initialization need a distinct controlled test.
Threshold calibration cannot fix missing correspondences or create a pose.

## Compute gate

The unchanged rule sums, over five checkpoints,
`1.5 * (model load + worst measured complete-case time * 384)`.

| Mode | Derived complete-study cap | Predeclared maximum | Admission |
|---|---:|---:|---|
| Directional SH | 130,720 s | 12,000 s | Refused |
| Pooled | 130,295 s | 12,000 s | Refused |

These are conservative admission estimates, not observed full-study runtimes.
All 384 eligible cases and all five checkpoints remain required. No smaller
subset, renewed deadline, refunded charge or partial winner substitutes for
the full selection. The original remaining reserved-cap allowances are
15,867 training seconds and 53,100 baseline seconds after both pilots/replays.

An explicit, separately tested cache implementation is under development to
avoid repeated geometry, feature and response calculations. It does not add
parameters, change the scientific search policy or demonstrate an accuracy
gain. Its actual speed must be measured with a fresh published protocol.
Separately, pose-independent response proposals and genuinely diverse
correspondence hypotheses are candidate follow-up experiments; neither has
been shown to improve recovery.

## Evidence and replay scope

- [Curated results and exact source hashes](solar_pose_pilot_results_v83.json)
- [Directional saved-metric replay](solar-sh-checkpoint-selection-cost-pilot-replay-v79.json)
- [Pooled saved-metric replay](solar-pooled-checkpoint-selection-cost-pilot-replay-v79.json)
- [Completed reference/data review](solar_reference_completed_review_v83.json)
- [Original saved prediction archive](solar_pose_pilot_predictions_v83.tar.gz)

The archive preserves both summaries, numeric prediction arrays, compact raw
traces, execution and termination receipts, and replay evidence. Its 266 files
retain their original relative paths. Repeated worker-log text is omitted;
the original logs remain local. The physical dataset/scoring-source assets
are not bundled. Full independent source-label replay still requires those
unchanged generated inputs. Saved-label arithmetic alone is a narrower check.

The research goal remains active. Full selection, eight ablations plus pooled,
seed repetitions, modern comparisons, transfer, calibration, final freeze/test
and the final paper/release package remain unfinished. No superior-model,
novelty, orbital-navigation accuracy or NASA/JPL affiliation claim is made.
