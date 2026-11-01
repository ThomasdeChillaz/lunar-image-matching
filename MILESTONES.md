# Milestones

## Native LoMa-R / RoMa-v2 common comparisons complete — 1 November 2026

Both fixed36+36 native comparisons now pass all72 cases and paired controls:
LoMa-R42/72 (28 wrong raw proposals), RoMa-v2 fast/common51253/72 (19 wrong).
The original compact control is45/72, with median paired CPU speedups20.6× and
16.8× respectively. These speedups apply to the old compact pipeline, not the
new surface control. Native surface normalization is54/72 (three wrong), or
57/72 with SIFT (two wrong); a reliable advantage over RoMa-v2 is unestablished.
The surface-minus-RoMa terrain interval includes zero. Only four exposed
terrain components are available; generic modern weights and lunar-trained
small weights are not a matched adaptation comparison.

See reports/NATIVE_MODERN_V10.md. Exact source, weights, native pixel controls,
full72 denominators and all288 assigned output records replay. All original
partial/duplicate attempts are retained with verified ZIP hashes. Original
jobs stopped64/72, but both declared tail workers completed36/36. No cherry-
picking is used in the principal assignment. Author-precise native settings,
EfficientLoFTR/MOONSt3R native runs, calibrated acceptance, fresh final and
transfer targets remain missing. A direct same-worker surface/RoMa timing and
full-score workflow is prepared; no surface speedup is yet claimed.

Full v8 coverage-training winner repeats remain live. The completed v7 model
study and geometry corrections are frozen development evidence, not the full
goal. The native capsule preserves all completed cohorts and every failure.
Earlier entries below are historical snapshots.


## Completed native study and 3D normalization control — 1 November 2026

All10000 v7 optimizer steps and five runs are complete. Independent replay
passes360 candidate rows,234 supervised cases, optimizer/sampling/source and
seed checks. Actual selected-model quarter-turn/blank controls pass for all
five models. The selected pair-refinement seeds give45/72,42/72,42/72 versus
frozen45/72: no consistent learned advantage. See reports/NATIVE_TRAINING_V7.md.

The deterministic affine correction gives49/72 versus45, but the same
SIFT-first fallback remains53/72 and wrong poses increase6→9. It does not
replace the default or establish a gain over the stronger simple pipeline.

Explicit reference-surface normalization gives54/72 raw correct poses versus45,
with wrong poses12→3 and strict poses17→16. The same SIFT-first fallback gives
57/72 with two wrong proposals, versus53/72 with six wrong. The terrain-bootstrap
intervals include zero. This is a frozen28,272-parameter engineering correction,
not a newly learned contribution, calibrated acceptance or final validation.
Query depth and true query pose enter scoring only. Source/input/checkpoint,
analytic controls, native metrics and pose-estimator replay pass. See
reports/SURFACE_WARP_V11.md. The stronger simple control must be included in
future learning claims. Do not equate raw57/72 with70% reliable acceptance.

Full v8 uniform-positive training remains live. Completed primary encoder/pair
readouts48/72 and47/72 are preliminary; the failed no-match head again collapses.
The predefined winner repeats remain required. No calibration/test pixels.

The original LoMa-R/RoMa-v2 jobs each timed out after64/72 cases. Both verified
ZIP hashes and all256 partial output rows/control metrics replay successfully;
they remain incomplete comparisons. The predeclared last36-case workers are
live. Preserve duplicates and use the fixed36+36 combiner once all assigned
cases exist. No modern ranking, fresh-final, transfer,1% error or SOTA target
is achieved. No paid compute, human review or automatic outreach.

Earlier entries below are historical execution snapshots.


## Native training continuation — 1 November 2026

The v7 three primary runs are complete: encoder44/72, pair refinement45/72,
and dustbin45/72 (initial checkpoint selected). The second pair-refinement
seed gives42/72; seed20261103 is still running. Frozen native orientation
matching remains45/72, and the stronger simple SIFT-first fallback53/72 with
six wrong poses. A consistent learned gain is not established.

The full v8 uniform-positive experiment is now running: three2000-step
ablations, followed by the bank-selected configuration with both additional
predefined seeds. Its earlier100-step pilot gave45/72, so no gain is assumed.
All source was published before launch; train/selection separation and native
inference remain unchanged. Do not edit either cohort's pinned sources.

A failed final dustbin checkpoint rejects all1727 sampled train/selection
windows before pose estimation, despite515 decoded matches within2px.
Match-score ranking AUC for window validity is0.337 on sampled train proposals
and0.290 on selection proposals (chance0.5). These are unaugmented,
class-stratified, dependent coarse-proposal diagnostics, not the augmented
training distribution or pose acceptance. The failure is not confined to
unseen groups, and adjusting the null-vs-single-bin decision alone does not
repair the discriminative ranking. No threshold was fitted.

Native LoMa-R and RoMa-v2 comparisons are active on standard free CPU runners.
The predeclared36+36 assignment now has all four worker IDs recorded in
reports/native_execution_continuation_v9.json. The new combiner refuses missing
assigned cases, incorrect source runs and accuracy-based duplicate selection.
Complete-case replay, calibration, fresh final validation, transfer and paper
claims remain outstanding. No paid compute, human review or automatic email.


## Current native research checkpoint

V6 completed16000 steps without a pose gain; all controls exactly replicate v5.
V7 native pilot47/72 versus45, but full encoder/pair ablations44/45 so far; the
complete study and repeat seeds remain active. Uniform-positive pilot45/72,
with no raw-pose gain. Oracle exact-in-window67/72 is diagnostic only. Native
LoMa-R/RoMa-v2 comparisons are executing with mandatory complete-case replay.
No full research target, calibrated acceptance or SOTA claim is achieved.
See STATUS and NATIVE_REFINEMENT_V7.md. Older sections are historical snapshots.

## Active goal: corrected training and continued hypotheses

Corrected-depth coarse study complete: four4000-step runs, all predefined
seeds, primary/repeat raw poses 44/72, 45/72, 45/72 versus frozen45/72. No consistent
learned advantage. SIFT-first engineering fallback53/72 has six wrong poses;
not calibrated acceptance. Independent288-record training replay and actual
candidate controls pass. Exact-correspondence diagnostic26/26; all173 software
tests/four optional skips and fresh smoke pass.

V6 orientation-regime fitting is active; its100-step software pilot passed
replay without a learned gain. Sources pinned publicly before launch. The
[learning study](reports/COARSE_TRAINING_V5.md) retains every seed/failure and
stronger-baseline comparison. Earlier ray-based3D results/labels are superseded
by the [depth correction](reports/GEOMETRY_CORRECTION_V5.md). The original fresh
final-data gate and acceptance/uncertainty/modern comparison/paper targets remain
unmet. Do not declare goal completion from software checks or development scores.

## Supplemental training continuation completed

Ten 2,000-step runs now complete across the preserved primary and supplemental
balanced cohorts. Balanced reliability seeds:27/72,31/72,36/72 versus native+LM
26/72. Primary gain +1.39 points; descriptive interval [-11.11,+7.78], so a
reliable pose advantage remains inconclusive. All weights, failed heads and
seeds retained; 147 tests/four skips and both independent replays pass.
The original fresh-data gate remains for final validation. See
[balanced study](reports/LOCALIZATION_BALANCED_V3.md).

## Development training continuation - iteration 8

User subsequently authorized "Continue with the training". Implement and run
3D-supervised development-only refinement ablations on the original train split,
select using separate model-selection groups, and repeat the winner with the
predefined seeds. Do not access calibration/test pixels or claim fresh validation.
The historical iteration7 data gate remains failed for final generalization.

Completed: three 2,000-step ablations and two repeat seeds of the selected
descriptor_3d configuration (10,000 total optimizer steps). Combined models
contain 56,544 /63,505 /64,098 parameters including the frozen coarse network.
Primary/repeat outcomes26/72,25/72,26/72 versus native+LM baseline26/72; no
consistent learned pose gain. Reliability selected its initial checkpoint after
all later selection checks returned zero poses. Independent replay validates
234 supervision cases,360 candidate outcomes, selection, source/weight hashes
and seed completeness. All calibration/test pixels excluded. See
[training report](reports/LOCALIZATION_TRAINING_V3.md).

## Localization research - iteration 7

- L1: all released metadata audited: 76,802 cameras / 12,974 stereo geometries.
  Only intrinsics/cam2world fields; Tycho directory empty. Fresh split cannot be
  certified: whole-folder fallback has three already exposed groups and zero
  fresh versus twelve required. Modes are not trajectory IDs. Data gate failed
  before training; no full-depth fresh-footprint independence is claimed.
- L2: complete: 312 diagnostic records, eight geometries / four spatial
  components, 72 predicted stereo/light cases. Oracle 8/8 passes software gate.
  Coarse 0/72 and native refinement 7/72 using unchanged EPNP; separate
  post-diagnostic LM refit gives 3/72 and 26/72. Independent replay passes.
  These are engineering development results, not a new trained method.
- L3-L7: deferred by the explicitly accepted data gate. No compact-v3 training,
  learned checkpoint/CLI alias, final inference, new baseline ranking, transfer
  result or completed superiority paper is claimed. A diagnostic brief/report
  and reproducible measurement capsule are delivered instead.

Resume conditions and the accepted method/evaluation contract are in
[LOCALIZATION_RESEARCH.md](LOCALIZATION_RESEARCH.md).

## Compact research execution - iteration 6

- C1: executed actual LunarGeo regrouping and ray-depth audit,28 selected spatial components with8/4/4/12 partitions. Explicit trajectory membership is absent, so strict trajectory-held-out independence is not established; record this limitation without relabelling it as a pretraining preregistration.
- C2: complete. All40 development and40 selected final shards completed/audited. Five modern baselines use identical common cases and paired CPU controls. RoMa-v2 precise author control hit its600s budget; the declared fast512 comparison is complete.
- C3: completed all four2000-step ablations and three predefined selected-configuration seeds. Expanded architecture control selected by declared development criterion;28,272 parameters. Exact numerical development replay passed.
- C4: complete. Ten policies/checkpoints/cases/source hashes publicly frozen before final inference; all core and modern final outcomes scored without retuning. Thirteen policy fits and14 aggregate tables independently replayed.
- C5: complete bounded study and research package. All40 laboratory cases, three3D methods, two-page brief, principal figures, interactive demo, unsent outreach draft and checksummed release assets prepared. Global SOTA, reliable3D pose and verified orbital accuracy remain unestablished; research issue #2 stays open.

The accepted protocol is [COMPACT_RESEARCH.md](COMPACT_RESEARCH.md). No current-SOTA, target-achievement or real-navigation claim is implied by software completion.

## Active research iteration milestones

- M12: pinned LoFTR on identical original Sun pairs, complete outcomes and pixel/metric audit. Executed.
- M13: expanded training and source/terrain-disjoint geographic reserves. Dataset/training completed; all 64 reserve patches in 12 footprint/acquisition components were evaluated and all 32,832 method outcomes passed audit. Raw CNN8,414/9,216 correct. Both reserves are consumed. Cross-crater known-warps use real photos from three earlier scenes and remain distinct from fresh physical-render transfer.
- M14: explicit translation/rotation/scale/combined experiments and literature-inspired development hypotheses. Development experiments completed: standalone CNN1,072/1,152; original RoMa1,032; tested MatchAnything4×995; baseline ensemble1,083; ensemble+CNN1,103. Full13-predictor ensemble fresh evaluation is not established. NASA real-sensor/laser-reference diagnostics expose a relighting rejection failure.
- M15: strict standalone development rejection and checkpoint bound publicly before reserve access; primary evaluation and audits completed. Frozen gated CNN4,603 correct/zero incorrect accepted of 9,216 positives,0/1,728 negatives accepted. Empirical preregistered targets met at 49.95% positive coverage. No retrospective reserve calibration. The full ensemble has a development-only public export; its fresh validation and genuine real-illumination gate transfer remain pending.
- M16: substantial advantage against relevant current baselines on a common protocol, with uncertainty, failures, negatives and reproducibility. Not established; do not infer completion from development gains.

## Scope amendment — 29 October 2026

The user has declined all human landmark review. The original scientific acceptance criteria below are retained as historical criteria for verified real-image accuracy, not as blockers for the current automated benchmark. Current deliverables use known-geometry procedural and physical-render accuracy, and clearly labelled unverified real-image diagnostics. Human annotation, real-development correctness calibration and independently verified real-image transfer are optional future validation, outside current scope. No pending user annotation is required.

## M0 — Repository and experiment contract
Deliver AI_PLAN, data protocol, status tracking and an explicit transformation/error convention. Acceptance: another agent can identify the next task and distinguish synthetic from real evaluation.

## M1 — Controlled pairs and metric correctness
Deliver seeded procedural images, known affine transforms, gain/gamma/noise cases, common-support evaluation points and unit tests. Acceptance: identity and known translations give analytically correct errors; blank and invalid inputs are handled; repeated seeds reproduce inputs. Commit and push after tests pass.

## M2 — Classical baselines and runnable benchmark
Deliver NCC, SIFT/RANSAC, CLAHE ablation, CLI, per-pair CSV, failures, environment capture, three figures and a gallery. Acceptance: translation recovery tests pass, blank pairs fail explicitly, rotation cases are separately labeled, clean CLI run produces all outputs. Commit and push with measured smoke results labeled procedural.

## M3 — Audited lunar dataset
Select 3–5 distinct regions. Obtain publicly accessible LROC repeat imagery and suitable terrain products; verify product geometry, units, valid-data masks and redistribution conditions. Record source IDs, acquisition/solar angles when present, URLs, SHA256, footprints, preprocessing and grouped train/dev/test split. Implement manifest validation, image loading and invalid-pixel masks. Acceptance: ten inspected overlapping pairs; no split leakage; independently checked reference points for real pairs with residual/annotation uncertainty. Do not close this milestone on metadata alone.

## M4 — Pretrained learned matcher
Integrate SuperPoint + LightGlue (or justify a different model), pin upstream revision and weight hashes, document licenses, and use the same geometry estimator where meaningful. Acceptance: actual pretrained forward pass, known-transform check, controlled failures, timing protocol and explicit missing-dependency errors. No placeholder metrics. Commit and push.

## M5 — Frozen evaluation and scientific report
Freeze methods/parameters using development regions. Run held-out real regions, controlled transforms and real repeat observations separately. Report registration success, endpoint errors, failures, runtime, illumination metadata strata and preprocessing ablation. Quantify reference uncertainty and avoid unsupported planar assumptions for relief/viewpoint changes. Acceptance: reproducible table, three plots, failure gallery, two-page report and no claims exceeding evidence. Commit and push.

## M6 — Portfolio release
Review installation from a clean environment, documentation, citations, licenses and data provenance. Add a concise demo and tagged release after validation. Prepare an outreach summary only from measured real-data results. Do not email anyone without authorization.
