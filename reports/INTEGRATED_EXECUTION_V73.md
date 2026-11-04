# Integrated matcher: execution and remaining evidence

The three candidate techniques are implemented in one 95,808-parameter matcher:
shared-Sun correspondence/pose inference, learned spherical-harmonic descriptor
responses, and sequential native-patch selection. This is **implemented research
software, not a completed or successful lunar localization study**. The new
response head has zero lunar optimizer updates at this publication.

## What now runs

- `localize_terrain.py` preserves native reference-to-query coordinates and
  uncalibrated public `pose: null`; raw hypotheses require the research flag.
  `match_terrain.py` is unchanged.
- Source-bound evaluation supports all eight module combinations, pooled
  response, shared/off/independent Sun controls, four selection policies and
  three patch budgets. Every method uses the common final reference-geometry
  AP3P estimator, 2,000 final iterations and a three-native-pixel threshold.
- Native pretrained adapters now include MatchAnything–EfficientLoFTR and
  MOONSt3R, alongside the existing modern baseline adapters. Actual source,
  weights, runtime settings and saved pretrained controls must qualify each
  integration. A prepared profile is not a completed comparison.
- Historical adapters distinguish the original CNN, frozen expanded surface
  matcher and completed attention study. No checkpoint may be silently
  substituted. The attention study remains **15/96 raw correct poses** with
  no calibrated acceptance result.
- The limited direct-render control uses reference-camera physical relighting,
  not query-camera renders. It includes preparation costs and reports an
  unavailable exact known-Sun render as missing comparison coverage, never as
  successful rejection.

## Serial scientific execution

1. The already-running missing-only completion retains its original deadline
   of **2026-11-04 13:23:15.913367 UTC**, source contract and failed history.
   `run_solar_pipeline_v70.py` follows it with the full 96-unit independent
   geometry audit. No new learning begins unless all units are inventoried
   and exact-correspondence pose recovery reaches the required 95% gate.
2. Fill the prepared learning settings with actual completed data/audit paths,
   finite pilot caps and explicitly declared trial-pose perturbations. Keep
   all eligible training cameras and all 16 training groups. Teacher image
   access and hashing are even-light-only; held-out targets remain unavailable.
3. `build_solar_learning_protocol_v73.py initialize` freezes that contract;
   `prepare-next` produces its first concrete protocol. Publish the exact
   sources and protocol before scientific execution. Then
   `run_solar_learning_pipeline_v73.py` dispatches the published stage through
   the existing external watchdog and canonical compute journal.
4. The sequence is measured teacher pilot/full bank, SH and pooled 100-update
   pilots with exact replay, both 2,000-update studies, actual independent
   pose-based checkpoint selection, then the two selected-seed repetitions.
   Dependent protocols stop at a publication barrier. Timeouts and failures
   retain their charges; existing deadlines cannot be reset. Publication
   queue allowances are explicit and are not scientific compute allowances.
5. Build qualified reference assets and all eligible selection cases; perform
   measured-cost admissions before the ablations and pretrained comparisons.
   Calibrate only after model selection. Keep the four calibration and twelve
   final terrain groups closed until their prescribed stages.

The cumulative remaining ceilings declared before these new learning jobs are
20,220 training seconds and 66,900 baseline seconds. They may only decrease;
the measured cost projections must fit before jobs are admitted. No new
allowance is created by these execution tools.

## A deliberately separate engineering control

An optional `pose_initialization: quantization_aware` configuration interprets
16-pixel coarse proposals at their quantization scale, before native refinement.
It distinguishes unobserved search windows, actual native measurements and
failed measurements. Paid measurements remain available to later RANSAC;
current three-pixel support separately verifies a hypothesis. The historical
`strict3` default and four-angle bank are unchanged.

This control does not change the final error criterion or add learned
parameters. It must be held identical across a matched A/B/C comparison and
its compact control. It is not one of the proposed research contributions.
Procedural quantization tests diagnose an initial-pose bias, but optional
expanded-bank tests also expose many wrong tentative correspondences. The
45-degree software control fails its original combined pose/median-match gate
for all eight toggles, despite four small translation errors. All failures are
retained. Neither those initial-pose diagnostics nor software smoke results
are evidence of lunar generalization or a learned improvement. Correspondence
precision must include the raw tentative outputs; selecting geometric inliers
must not silently improve the reported matching metric.

## What remains necessary

Actual training, full development comparisons, perturbation studies, selected
seed repetitions, fresh laboratory transfer, acceptance calibration and a
published final freeze/test remain incomplete. A complete research package
requires the measured figures, downloadable checkpoints, two-page brief,
demonstration and paper draft. The absolute and relative success targets in
the accepted plan remain unchanged. Negative findings will be reported if
they are missed; real orbital navigation accuracy remains unverified.

Prior-art review must delimit any claim. Spherical-harmonic light transfer,
learned response fields, feature distillation and active matching have close
antecedents. Implementing them together is not by itself proof of novelty.

See the [targeted primary-source review](solar_prior_art_review_v73.md) for
closest antecedents, access limits and the specific evidence each candidate
contribution would need.
