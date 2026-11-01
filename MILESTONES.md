# Milestones

## Full context cohort running; first ablation audited — 1 November 2026

The complete2,000-update pointwise capacity control finished. Its trained
checkpoints yield48/49/44/47 correct raw poses at500/1000/1500/2000 updates,
versus54 at initialization. The predefined selection rule retains step0:
54/72 correct,16 strict, three wrong raw estimates. Training this control did
not improve localization. Independent replay verifies all360 checkpoint rows,
97,344 projected label checks and the recorded2,000-update sampler/optimizer
budget. This audit does not claim independent full retraining.

Self-context and cross-image context, then two predefined repeats of the
selected primary configuration, remain in progress/pending. Keep the live
cohort running from public freeze b2d749f; do not restart it or mutate its
pinned sources. The full10,000-update cohort is not complete. See
reports/global_context_pointwise_progress_v16.md for this partial snapshot.

The completed surface-study prerelease surface-supervision-development-v13
is public. All four assets were redownloaded byte-identically and ZIP CRCs
pass; compact-v2 remains Latest. The source tag points to b2d749f.

Zenodo latest/version metadata still identify the same MoonAnything sample;
no new released data or batch identities were established. The fresh-final
gate remains failed. LuSNAR7-9 remain reserved with zero inspected pixels or
inference. No new dataset or accuracy claim is substituted for that gate.

210 repository tests/four optional skips, a fresh100-row software smoke, the
actual partial-cohort audit and reviewed report figure pass. Fresh evaluation,
calibrated acceptance, required remaining native baselines, transfer, timing,
uncertainty, paper and compact-v3 pose interface remain required. No paid
compute, human review or automatic email. Full goal active and unmet; older
entries are historical snapshots.

## Surface cohort complete; full context cohort prepared — 1 November 2026

All five v13 runs completed10,000 optimizer updates. The selected geometry-
normalized pair model gives55/72 correct raw poses at all three predefined
seeds, versus54/72 frozen surface. The same SIFT fallback gives58/72 at every
seed versus57/72, with one wrong raw pose. The1.39 percentage-point gain has
an exposed-component descriptive interval[0,4.76] and remains inconclusive.
The10-point improvement and calibrated1% error targets are not achieved.
All1,800 checkpoint rows,360 selected rows, training samplers, native errors
and poses independently replay. Actual selected-model forward inference also
reproduces all72 saved rows exactly from a clean public checkout.

The global-context100-update train-only cost pilot completed and independently
retrained bit-exactly, including optimizer and random states. It provides no
localization result. The complete capacity-controlled cohort runner is now
implemented: pointwise/self/cross variants at2,000 updates, then two repeat
seeds of the selected variant. Sources must be public before launch. No full
context fitting has started at this snapshot. See GLOBAL_COHORT_V16.md.

The broad-initialization negative finding remains unchanged:15 added wrong
poses and no added correct ones. No adoption. Current software validation:
210 passed/four optional skips, plus a fresh100-row procedural smoke and
exact interruption/restart controls. Fresh final data, calibrated acceptance,
remaining required native baselines, transfer, uncertainty, paper and the
compact-v3 pose interface remain incomplete. Existing4-component raw-pose
comparisons have a saturated-group bootstrap limitation; see
reports/development_raw_ceiling_v16.json. This conditional arithmetic does
not establish a ceiling on future data or calibrated acceptance.

No calibration/test pixels, paid compute, human review or email. The full
research objective remains active and unmet. Older entries are history.

## Global match failure diagnosed; context controls pass — 1 November 2026

The v14 broad-initialization control is rejected: 54/72 correct poses remain
unchanged, while wrong poses increase from 3 to 18. Independent replay covers
all 72 final poses and all 60 coarse/broad branches on the 15 missing cases.
No branch produces a correct broad pose. Median best genuine 12-pixel match
fraction is 2.83%; seven cases have fewer than six such matches in every
branch. The tested local refinement did not repair these global proposals.
See reports/POSE_RECOVERY_V14.md. The default matcher is unchanged.

The next v15 proposal experiment adds a small image-pair context module.
Its 25,664 trainable parameters plus the frozen backbone total 53,936;
combining it later with the existing separate learned fine model would total
89,169. No context weights have been trained. Actual native controls and exact
72-case native-point/pose equivalence pass, preserving 54 primary and 16 strict
successes. Capacity-controlled bidirectional 3D supervision and the bounded
training-pilot runner are implemented; 205 tests/four optional skips and a
fresh 100-row smoke pass. The pilot must run after source publication; the
full fitting cohort still needs implementation. See GLOBAL_CONTEXT_V15.md.
No context model gain is claimed.

The v13 surface-training cohort remains live. Its completed primary runs give
54/72 raw-control, 55/72 normalized-encoder and 55/72 normalized-pair poses;
repeat seeds and the full independent cohort audit are still pending. The
stronger fixed SIFT baseline remains 57/72. A one-pair exposed-development gain
is not the requested research target. Fresh validation, reliable acceptance,
remaining native comparisons, transfer and the paper remain incomplete.
No calibration/test pixels, paid compute, human review or automatic email.
Earlier entries are historical snapshots; the full goal remains active.

## Geometry-normalized training active — 1 November 2026

The initial gate passed: both initialized variants reproduce every historical
v11 primary/strict case status,54/72 correct and16 strict, with identical2px
point counts. An independent144-row replay verifies native errors and reruns
the common pose estimator. Sources were public at19a4357 before fitting.

The full v13 cohort is now running: three2000-update ablations plus two repeat
seeds of the selected configuration. The raw/normalized encoder ablation
shares10340 train points, component/light/jitter draws and sensor perturbations;
the third variant adds pair scoring. Models remain below100000 combined
parameters. Select on all72 exposed development pairs, retaining initialization
and every failed checkpoint. No completed new learned result is claimed yet.

193 tests/four optional skips and a corrected fresh100-row software smoke pass.
The previous study is published as native-precision-development-v11; all11
public assets were redownloaded and verified, with compact-v2 still Latest.
No calibration/test pixels, paid compute, human review, automatic email, fresh
final or SOTA claim. Full research targets remain active and unmet. Earlier
entries are historical snapshots. See SURFACE_LEARNING_V13.md.

## Geometry-normalized supervision prepared — 1 November 2026

New v13 ablations compare raw3D encoder supervision, normalized encoder
supervision and normalized pair refinement, with identical common training
support:10340 visible integer-reference points from18 geometries/12 training
components. Fractional crop jitter preserves subpixel labels. Query truth is
training/scoring only; inference retains estimated query pose/reference depth.
Each primary run has2000 updates; the predefined winner receives two repeat
seeds. All72 exposed selection pairs select checkpoints for this new cohort.

The initial gate must reproduce every v11 primary/strict status before fitting.
A feature-batch shape error was caught before fitting; the failed preparation
is retained with zero optimizer steps and covered by an actual-forward
regression test.193 tests/four optional skips and a fresh100-row software smoke
pass after correction. No new learned result is available at this snapshot.
Sources/protocol are pinned; no calibration/test pixels, fresh final, paid
compute, human review or automatic email. See SURFACE_LEARNING_V13.md. The
full research goal remains active and unmet. Earlier entries are history.

## Surface CPU validation complete — 1 November 2026

The direct same-worker run41262874858 completed and independently replays all
88 records:72 surface accuracy cases and16 RoMa/paired timing cases. Surface
54/72,16 strict, three wrong; all72 primary/strict statuses match the local
study. On the fixed16-case bank both methods give12/16 correct/six strict.
Surface median inference4.43s versus53.31s for RoMa-v2 fast/common512; median
paired speedup12.03x, descriptive component interval[11.80,12.10]. Two threads,
both warmed/loaded, alternating case order, fresh paired feature maps.

This speedup does not apply to the57/72 SIFT variant. Joint3.74GB process memory
cannot rank isolated model memory. The72-pair accuracy interval against RoMa
includes zero; data remain four exposed terrain components, not a fresh test.
No1% acceptance/nonoverlap, default-precise RoMa, orbital accuracy or global
SOTA claim. See reports/SURFACE_CPU_V12.md. No fitting or remote job remains
live; all handles from the preceding cohorts are terminal. The next learned
experiment is geometry-normalized supervision against this stronger control.
The full original research goal remains active and unmet.


## Uniform native training complete — 1 November 2026

The full v8 cohort completed10000 optimizer steps/five runs and passes exact
360-result/234-label/uniform-projection/checkpoint/sampler/selection replay.
All five selected models pass actual quarter-turn and blank controls. The
selected pair-refinement seeds give47/72,45/72,47/72, versus45/42/42 in v7 and45
for the frozen native control. The encoder-only primary gives48 but is not
substituted for the predefined bank-selected configuration. The no-match head
again selects initialization after trained checkpoints reject everything.

Uniform positive coverage and exact-query centering both change. Their modest
gains over the previous trained configuration do not establish a reliable or
large gain over the frozen model, and remain below the stronger deterministic
surface pipeline54/72 (57 with SIFT). The original research targets remain
unmet. No fresh final, calibrated1% error/nonoverlap or orbital accuracy claim.
See reports/UNIFORM_NATIVE_V8.md. Both completed native cohorts and every failed
checkpoint are packaged separately. No local fitting process remains live.

The bounded direct surface/RoMa CPU comparison is live in run41262874858,
using two threads on the standard public ARM worker. It tests surface accuracy
on all72 and paired latency on the existing fixed16-case bank. Do not borrow
the old compact speedup for the new surface method. The next learned hypothesis
is geometry-normalized supervision against the stronger deterministic control.
Earlier entries below are historical snapshots.


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
