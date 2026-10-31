# Compact localization study - accepted iteration 7

## Objective and current gate

Build a compact matcher/refiner for camera localization under simultaneous lunar
illumination and viewpoint change, in 8-12 weeks using existing free resources.
The combined model must remain below 100,000 parameters. No human review, paid
compute, automatic outreach, or verified orbital-navigation claim is authorized.

The [executed diagnostic report](reports/LOCALIZATION_DIAGNOSTICS_V3.md) is the
current result. **The fresh-data gate failed before training.** Per the accepted
plan, do not train compact-v3 or run its final evaluation until a separately
audited data source/split satisfies the gate. This conditional stop is not
completion of the learned-model objective.

## Accepted milestones

- L1: audit all released geometry metadata; group overlapping/shared terrain,
  lighting variants, cameras and generation batches. Require twelve genuinely
  fresh test footprint groups outside earlier exposed terrain. Keep train,
  model selection, confidence calibration and test separate. Missing trajectory
  IDs must never become unsupported independence claims.
- L2: diagnose exact geometry, coordinate rounding/noise/outliers, current raw
  matches and deterministic native refinement. Use the unchanged v2 pose
  estimator for the primary diagnostic. Require >=95% oracle success. Preserve
  later estimator experiments as separate development ablations.
- L3, only after L1/L2 pass: freeze the coarse model; add a shared fine structural
  encoder, 33x33-native-pixel local search, pair-conditioned scoring/no-match
  head and local-peak subpixel refinement. Train with exact 3D correspondences,
  visibility, incorrect proposals, occlusions and nonoverlap. Darkness is not a
  geometric occlusion label. Total parameters <100k.
- L4: ablate frozen coarse, deterministic refinement, true 3D supervision,
  learned refinement and pair-specific reliability. Seed 20261101; selected
  configuration also seeds 20261102/20261103, all outcomes reported. Select by
  model-selection unfiltered pose success. Fit confidence on development,
  threshold on calibration, give baselines the same fitting opportunity.
- L5: compare on common original image content against SIFT, old/recalibrated
  CNN, deterministic refinement, MatchAnything-EfficientLoFTR, LoMa-R, RoMa-v2
  and MOONSt3R. Check MOONSt3R training overlap. Review RoMa-Omega; integrate
  only after required baselines fit budget. Use common reference geometry and
  estimator; predicted image-only 3D is separate. Freeze LuSNAR scenes 7-9 for
  untouched transfer after actual file/metadata verification.
- L6: freeze checkpoints, thresholds, source bytes and case lists before a single
  final run. Targets: >=10 percentage-point correct accepted-pose gain over the
  strongest simple compact baseline, positive grouped paired interval, >=70%
  correct accepted poses, <=1% empirical accepted-error/nonoverlap rates. These
  are targets, not guarantees. Report stricter thresholds, absolute errors,
  component/lighting/seed strata and both-endpoint negative dependence.
- L7: preserve the existing command; expose an actually trained compact-v3 and
  separate native-coordinate pose output only after verification. Reject with
  pose:null; raw proposals opt-in. Test depth, visibility, pixel centers,
  boundaries, leakage, blanks, nonoverlap and degenerate geometry; independently
  replay results. Deliver a 6-8-page paper draft, four principal figures, a
  two-page brief, tagged artifacts and an unsent outreach draft.

## Budgets and resume condition

Training cap 48 local CPU wall-hours; pretrained comparison cap 72 CPU wall-hours,
two threads per job, development pilots, cached outputs and explicit timeouts.
No silent model/profile substitution or test-driven tuning.

The diagnostic does not implement/publish an untrained model as compact-v3,
invent a final test, fit confidence on diagnostic outcomes, or fill a superiority
manuscript with pending experiments. Its two-page document is a diagnostic brief,
not the completed method paper.

Resume only after an audited manifest establishes verified grouping and twelve
fresh test footprint groups. If public data cannot support the contract, record
the shortfall. Additional suitable data does not require human landmark review.
Any change to the scientific split contract must be explicit; relabeling modes
as trajectories is not a repair.

Related work: [Ansar et al.](https://doi.org/10.2514/6.2025-2073),
[RoMa-v2](https://arxiv.org/html/2511.15706),
[matching uncertainty](https://arxiv.org/abs/2608.08685),
[MOONSt3R](https://github.com/clementinegrethen/StereoLunar),
[LunarGeo](https://zenodo.org/records/18415119).
