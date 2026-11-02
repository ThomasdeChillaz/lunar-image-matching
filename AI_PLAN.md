# AI coding-agent execution brief

## Native fine training underway; both pilots replayed — 2 November 2026

Both matched100update pilots COMPLETED and independently replayed every optimizer
update, exact input hash, final weight, Adam state and random state. Aligned took
42.25seconds; fractional40.96seconds. Both preserved the known-transform and
blank-image controls; the original coarse weights remained frozen. All800
physical examples/reference inputs matched between variants. Training-only bank
loss decreased in each variant; this is not a held-out pose-accuracy result.

The measured control/cost gates admitted both2000update development studies.
The aligned study is now running; the fractional study is queued after its
completion and audit. Each study reuses its100pilot updates and has its own
original90minute deadline, without resetting a failed attempt. All96 positives
are evaluated at five checkpoints and96 negatives at the selected checkpoint.
No improved model is adopted. Four calibration/12final groups remain untouched.
Evidence: reports/north_fine_pilot_results_v58.json and both pilot audit receipts.
438tests/four optional skips and a fresh100-row procedural smoke passed before
training; source was public and byte-verified at6cc8a4c before either pilot.

## Native fine-encoder training control prepared — 2 November 2026

The next experiment freezes the original coarse proposal network and trains only
its native-resolution fine encoder using exact visible BVH3D correspondences.
The combined model has **56,544 parameters**, of which **28,272 are trainable**.
No new pose gain or improved model is claimed before measured development results.

Two matched variants use integer-aligned versus fractional query-image crops.
Both draw identical physical examples, lighting, jitter, fractional phases and
sensor augmentation; only the query sampling intervention and its exact label
change. Reference features retain their fractional native coordinates after
encoding. Full-image augmentation precedes cropping. Native feature maps are
invalidated whenever the fine checkpoint changes. The frozen128px coarse model
and native512px pose inference remain unchanged.

Predeclared sequence: each variant runs a100update TRAINING-ONLY pilot with a
600-second cap and exact optimizer replay. Both actual controls and the measured
cost gate must pass before either2000update study. Studies reuse their100pilot
updates, add1900updates, and each retain a separate5400-second absolute cap.
Seed20261101, batch8, AdamW1e-4, decay1e-4, gradient clip5, two CPU threads.
Evaluate all96 positive development pairs at0/500/1000/1500/2000 updates; select
by the existing raw-pose rule with initialization eligible, then evaluate all96
nonoverlap pairs. Saved576metric rows and2000sampler/input records must replay.
Training-patch loss and precision are diagnostics, not held-out camera accuracy.
No no-match/reliability head or confidence calibration is added in this ablation.

Source and software checks will be published and byte-verified before training.
No v58 optimizer has run yet. Four calibration and12final groups remain untouched.
The70%/10pp/1% research targets, modern and transfer comparisons and paper/release
remain incomplete. Earlier failed and negative experiments are preserved.

## Native-grid control passed; no improved pose model yet — 2 November 2026

The full-development larger-rate study remains a negative result: every learned
checkpoint scores 12/96 correct raw poses, below the original selected 14/96.
The predeclared rule retains initialization, not learned weights. The sampler,
optimizer and 576 saved metric rows independently replayed. No model is promoted.

The v57 native-grid control now COMPLETED 100 neural forwards in 4.32 seconds,
with 32 training-image control points and independent saved-feature replay.
At identical native coordinates, integer-origin image crops plus fractional
reference-feature sampling agree with full-image feature extraction within
**1.1921e-7 maximum absolute feature-channel error** across all four conditions.
That is safely below the declared 1e-4 tolerance.

| Training image condition | Aligned max feature error | Fractional path mean feature-vector L2 |
|---|---:|---:|
|Identity|1.1921e-7|0.3618|
|Relighting|1.1921e-7|0.4175|
|Viewpoint|1.1921e-7|0.3062|
|Relighting + viewpoint|1.1921e-7|0.3778|

The fractional path compares interpolation of images before encoding with
interpolation of encoded features at the SAME fractional coordinates. These
operations need not commute in a nonlinear encoder. The measured difference
is a representation/sampling effect, not a pixel correspondence error, software
bug, proven cause of earlier model failure, pose gain or independent-terrain
validation. In particular, the v56 coarse-only experiment used full 128px images;
this crop diagnostic does NOT explain its failure to learn on varied pairs.
It informs the next NATIVE FINE training experiment. No optimizer ran in v57.

Next: implement a matched native fine-encoder training control using the north
training split, exact fractional BVH correspondences and all six rendered
lighting settings. Compare integer-aligned query sampling with a separately
controlled fractional-query sampling variant; keep physical pairs, initial
weights, sensor augmentation, updates and inference identical. The reference
point stays fractional AFTER encoding; darkness remains distinct from occlusion.
Augment full images before integer slicing so blur does not create crop-border
artifacts. Reset native feature caches whenever fine weights change.

Start with encoder-only refinement and the original frozen proposal generator,
under 100k combined parameters. Gate on actual crop/full-image agreement and a
short cost/gradient pilot before full development pose evaluation. Neither
variant is implemented or running yet. Pair-specific reliability/refinement
heads remain a later controlled change. More coarse-match progress and unseen-
camera controls are also needed before the complete research targets can hold.

Evidence: `reports/native_crop_results_v57.json` and
`reports/native_crop_audit_v57.json`. All 434 repository tests/four optional skips
and fresh 100-row procedural smoke passed before this neural control. The final
reserve remains untouched: 4 calibration/12 final groups. The full 70%/10pp/1%
acceptance, modern/fresh/transfer/paper/release goal remains ACTIVE, not achieved.

## Larger-rate study complete: no pose gain; native-grid control next — 2 November 2026

The v56 matched experiment COMPLETED 2,000 updates in 1,539.63 seconds and passed
all 576 saved native metric replays, 2,000 original sampler/label records, matched
initialization/first gradient and optimizer-setting checks. Both clocks stayed
within its separate one-hour allowance. No model is adopted or promoted.

| Update | Original 1e-5 correct/wrong | Higher 1e-3 correct/wrong |
|---|---:|---:|
|0|13/28|13/28|
|500|13/15|12/7|
|1000|14/8|12/5|
|1500|14/15|12/8|
|2000|13/14|12/6|

All counts are out of 96 positive DEVELOPMENT pairs at every checkpoint.
The unchanged selection rule keeps initialization for the higher-rate run:
13 correct, 28 wrong and 11/96 selected nonoverlap poses. The original run selects
step 1500: 14 correct, 15 wrong and 4/96 nonoverlap poses. No confidence calibration
was performed. The learning improvement on repeated training fixtures did NOT
transfer to greater pose coverage on varied development terrain. Changed-light
positive loss over the last 500 updates is 5.3123 versus 5.3009 originally; changing
step size alone did not resolve learning on the varied training distribution.
Do not repeat this failed higher-rate configuration across additional seeds.

Next v57 is a small diagnostic of native-pixel fine supervision. The older fine
training code resamples fractional image crops before encoding. Native inference
uses an integer query grid and samples fractional reference features after
encoding. These operations need not commute. A difference is not by itself a
software bug or proof of why earlier models failed. Integer-origin crops can
provide a directly comparable control without changing shadow/occlusion labels.

The control uses one TRAINING terrain and four image conditions, eight geometry-
supported points each. It compares full-image versus aligned-crop features at
identical coordinates, plus a separately measured fractional-resampling path.
100 neural forwards, zero optimizer updates, two threads, a fixed 300-second cap.
An independent saved-feature arithmetic replay is prepared. Eight focused
array/coordinate/error tests and the full 434 tests/four optional skips/fresh
100-row procedural smoke PASSED. No crop-control neural inference has run yet;
publish/verify source first. Feature-vector error is not correspondence or pose
accuracy, and this diagnostic cannot establish geographic generalization.

Evidence: `reports/north_rate_development_results_v56.json` and
`reports/north_rate_development_audit_v56.json`; crop preparation and software
receipts are separately versioned. GitHub CI is repaired and passed: the core
workflow now installs the pinned CPU neural runtime, leaving every executed
experiment source unchanged. The original six-hour attempt remains FAILED.
All 4 calibration/12 final groups remain untouched. Full 70%/10pp/1%, modern-model,
fresh test, transfer, paper and release targets remain unmet; the goal is active.

## Full-development pose experiment prepared — 2 November 2026

The previous goal turn made progress: 800 new training updates and 40 actual
checkpoint replays showed better fixed-example fitting at coarse rate 1e-3.
The v56 experiment now tests actual DEVELOPMENT camera poses, with one seed
20261101, 2,000 updates and checkpoint steps 0/500/1000/1500/2000. Only coarse
AdamW rate changes from 1e-5 to 1e-3; context stays 1e-4. Architecture, initial
weights, all six physical lights, original sampler/appearance augmentation,
loss, clipping, fine encoder, native refinement and pose selection stay fixed.

All five checkpoint evaluations are FRESH on all 96 positive pairs; selected
nonoverlap evaluation covers all 96 negative pairs. 576 fresh native inference
rows plus all 2,000 sampler/geometry-label records must replay. No reuse of old
checkpoint case caches. Compare to the original same-seed full 2,000-update
control's selected checkpoint (14/96 correct, 15 wrong, 4/96 nonoverlap proposals),
not just its 13/96 frozen initialization. Counts remain raw and uncalibrated.

The resource gate independently checked the existing measured fitting pilot,
full common baselines and cost bank. A factor-two projection including all 576
cases and control overhead is 2,902.91 seconds, below the separate 3,600-second
limit. One scientific job/two threads; atomic per-update checkpoints and
per-case inference caches; explicit resume only inside the original deadline.
The original six-hour attempt stays FAILED, with every old file unchanged.

Twelve focused controls pass: finite budget/clock checks, full-denominator
inference failure handling, public-source guard, and AST identity of the
numerical inference and training loops with the original implementation.
Full current software checks PASSED426tests/four optional skips and a fresh
100-row procedural smoke. Public source and actual training are pending. The saved replay checks same initialization, optimizer settings,
first pre-intervention gradient, every original sampled draw and target-label
count, and all saved checkpoint/selected-negative pose metrics.

No candidate is promoted automatically. Three-camera-template dependence
remains; any gain needs repeated seeds and unseen-camera controls before a
strong claim. All 4 calibration/12 final groups stay untouched. Full 70%/10pp/1%
acceptance, modern comparisons, fresh test, transfer and paper targets remain
unmet. No paid compute, human review or automatic email.

## Larger coarse learning rate improves fitting; pose test next — 2 November 2026

The v55 one-factor comparison COMPLETED 800 new logical training updates.
Both rates passed all 40 actual checkpoint forward/decoder replays and the
independent objective replay. Inputs, initial tensors, step-zero outputs and
first pre-intervention gradients match v54; optimizer settings differ only in
the declared coarse rate. The original control and its files remain unchanged.

| Coarse AdamW rate | Relighting | Viewpoint | Relighting + viewpoint |
|---|---:|---:|---:|
| Original 1e-5 |5.33%|98.05%|4.66%|
| 1e-4 |6.95%|99.55%|7.07%|
| 1e-3 |31.21%|100.0%|32.78%|

Each cell is the fraction of highest-scoring coarse locations within the
native 33-pixel refinement search after 100 updates on ONE fixed TRAINING
example. Identity stays 100% at every rate. These are neither precise matches
nor held-out camera poses. Returned reachable matches also increase: under
relighting, 16/181 at 1e-5 becomes 143/301 at 1e-3; for relighting+viewpoint,
7/182 becomes 120/247. Many matches remain wrong. No candidate is promoted.

This supports the hypothesis that the original 1e-5 coarse rate limited fitting;
it does not prove that learning rate alone explains the original pose failures.
The 1e-3 setting is an exploratory candidate, chosen on training diagnostics.
Next run ONE matched full-development seed 20261101 at 1e-3 for 2,000 updates,
using the original data sampler, loss, initialization, architecture, native
refinement, checkpoint steps 0/500/1000/1500/2000 and all 96 positive/96 negative
cases. Compare with the original same-seed/full-budget run, including its best
selected checkpoint. Keep context rate 1e-4 and every other training setting
fixed. Predeclare a separate 3,600-second cap; publish current source/checks and
cost admission before training. This next experiment is NOT yet implemented
or running. Repeat other seeds only if development pose evidence justifies it.

Important data fact: north v39 ALREADY samples all six rendered lighting
settings (`north_training_bundle_v38.py`); do not infer a three-light shortage
from the older v18 sampler. Existing three-camera-template dependence remains
a limitation. Unseen-camera controls are required before a strong claim.
All 4 calibration/12 final terrain groups remain untouched.

Evidence: `reports/north_rate_results_v55.json`, both rate audits/numeric
replays, and `reports/north_rate_training_curves_v55.png`. The figure is derived
only from saved audited metrics; no matching/scoring code changed. Training
source at `de231381424ba14efe47079b227eafd0719c4847` passed 414 repository tests,
four optional skips and a fresh 100-row procedural smoke before inference.
All research targets remain unmet: no adopted model, SOTA, fresh-test pose
advantage, calibrated 1% risk, orbital accuracy or outreach claim.

## One-factor learning-rate comparison prepared — 2 November 2026

The previous turn made scientific progress: it completed the original cohort's
missing evaluation, replayed all 1,728 rows, diagnosed coarse reachability, and
completed/replayed 400 fixed-example updates. It did not produce a strong
localization model. The full research goal remains active.

The next v55 control compares coarse AdamW rates **1e-4 and 1e-3** against the
completed v54 **1e-5** control. Keep all four training fixtures, 100 updates each,
initial weights, context rate1e-4, loss, clipping, augmentation and decoder fixed.
No architecture or supervision change. The original v54 files stay unchanged.
Two serial jobs, two threads, each at most900calendar seconds including its
10-update cost pilot; independent replay follows each. Total800new logical
updates, zero reserve cases. Both alternatives will be published, including
failures. These are optimization diagnostics, not model-selection pose results.

The v55 verifier additionally checks the same inputs, initial tensors, step-zero
scores and first pre-intervention gradients, and verifies that only the coarse
optimizer rate changed. It reuses the unchanged v54 twenty-checkpoint actual
forward/decoder replay and independent NumPy objective replay. Ten focused
controls pass. Full414tests/four optional skips and a fresh100-row procedural
smoke PASSED. Public source freeze and actual training are still pending.

Proceed only after publishing and byte-verifying the source. If either rate
improves fitting, test any proposed change on development geometry before a
larger cohort; fixed-example learning cannot establish generalization.
All4calibration/12final groups remain untouched. No candidate promotion,
SOTA, orbital accuracy or1%risk claim; no paid compute, human review or email.

## Fixed-example learning completed: relighting remains weak — 2 November 2026

The v54 diagnostic COMPLETED all 400 logical training updates in 123.22 seconds.
Independent NumPy objective replay and actual 20-checkpoint forward/decoder
replay PASSED. Both coarse encoder and context weights changed; protected
fine weights stayed identical. No additional full cohort or candidate promotion.

Highest-scoring coarse locations within native 33-pixel refinement reach changed:

| Fixed TRAINING example | Before | After 100 updates |
|---|---:|---:|
| Identity |100.0%|100.0%|
| Physical relighting, same camera |2.81%|5.33%|
| Same-light viewpoint change |85.11%|98.05%|
| Relighting and viewpoint |1.20%|4.66%|

These are coarse search-window diagnostics on four TRAINING examples, NOT
precise correspondences, camera pose, held-out generalization or a learned
upgrade. Loss decreased for every fixture, but relighting ranking stayed weak.
This rules out completely inactive learning; it does not establish that the
architecture cannot fit relighting or identify the best learning rate.

Next: compare original coarse AdamW learning rate 1e-5 with 1e-4 and 1e-3 on
these same metadata-fixed examples, keeping context rate 1e-4, initialization,
objective, steps, decoding and all other settings fixed. Publish both outcomes.
Predeclare finite serial compute allowance and current checks/source before
new fitting. If increased step size does not help, inspect illumination
representation and supervision rather than repeating the same large cohort.
No full new cohort starts from this diagnostic alone.

Evidence: `reports/north_optimization_results_v54.json` and
`reports/north_optimization_audit_v54.json`. Reproduce with the source at
`819f9cbe6534ca4c3ef77f413645a9dd97fd5204` using
`python north_optimization_probe_v54.py --publication-commit <that SHA>`;
then `python verify_north_optimization_v54.py`. Keep existing output roots;
use an explicitly new output/audit path for another replay. All 404 repository
tests/four optional skips and fresh 100-row smoke passed before fitting.
Original six-hour attempt stays FAILED; completion amendment is separately
complete. All 4 calibration/12 final groups remain untouched. Full research
targets remain unmet; no SOTA, orbital or 1% risk claim.

## Diagnose coarse learning before another cohort — 2 November 2026

The v53 coarse diagnostic and independent replay COMPLETE: 768 proposal
branches / 192 case summaries. The ground-truth-assisted ideal local
refinement/rejection control recovers **54/96 frozen** and **49/96 trained
seed1 endpoint** poses. These are oracle diagnostics, NOT usable matcher
accuracy or a strict mathematical bound. No reachable proposal exists in
28 frozen cases and 31 endpoint cases. Under light pair1–5, both ideal
controls recover only8/24; same-light controls recover18/24. The current
local33px strategy therefore needs better coarse proposals as well as better
local refinement. These endpoint comparisons do not evaluate the selected
step1500 checkpoint and do not establish a causal architectural effect.

The next v54 experiment is a TRAIN-ONLY optimization diagnostic: four fixed
examples (identity, physical relighting, viewpoint, both), each100updates
from identical original initialization using the original objective and
optimizer. This is400logical updates total, not a new adopted candidate.
No appearance augmentation; no holdout, calibration, final or transfer
inference. Record per-part gradients, parameter changes, full score matrices
and20checkpoint snapshots. A separate NumPy objective and actual snapshot
forward replay are prepared. Seventeen focused controls pass. Current full software checks PASSED
404 tests/four optional skips and a fresh100-row procedural smoke.
Public source freeze and actual fitting remain pending.
Fixed-example improvement cannot establish generalization or localization gains.

Completed cohort evidence remains14/96,15/96,15/96 correct raw selected poses,
versus13/96frozen; this is far below the research targets. Original fitting
deadline staysFAILED; the v52 evaluation-only amendment remains separately
complete and audited. Preserve all original files and the4cal/12final reserve.

## Completed three-seed evaluation; coarse failure diagnosis next — 2 November 2026

The separate v52 completion finished in 319.68 seconds with ZERO additional
fitting. Independent replay PASSED all 1,728 saved rows and 6,000 original
sampler/update records. All 1,652 original files remain byte-identical; the
original six-hour attempt remains FAILED. The new allowance completed only
62 missing endpoint positives and 96 selected-negative evaluations.

Selected correct raw poses are **14/96, 15/96, 15/96** for seeds 20261101–3,
compared with **13/96** for the frozen baseline on the same development set.
Wrong positive-pair poses are 15, 10, 13; raw nonoverlap poses are 4/96, 7/96,
5/96. These are uncalibrated proposals, not correct acceptance. The gains are
small, post-selection development evidence; no strong model is adopted. Seed3
selected step1500; its step2000 endpoint scored14/96. Historical unequal speed
conditions prevent treating these saved timings as a fair speed comparison.

`north_coarse_reachability_v53.py` is the explicit completion-aware successor
to the unexecuted v43 diagnostic. Its geometric measurement functions are
AST-identical to v43. It compares frozen and seed1 endpoint proposals against
exact labels, then measures a separate ground-truth-assisted ideal local
refinement/rejection control. This control is NOT model accuracy or a usable
branch selector. Ten focused geometry/replay controls pass. Current full software validation
PASSED 387 tests/four optional skips and a fresh 100-row procedural smoke.
Public source freeze and actual diagnostic remain pending.
Run the diagnostic only after its source-bound software and completion gates.

Evidence: `reports/north_amended_completion_audit_v52.json` and
`reports/north_amended_numeric_metric_replay_v52.json`. All 4 calibration and
12 final groups remain untouched. No extra training, SOTA, fresh-test,
orbital accuracy or 1% risk claim. The full research goal remains active.

## Original fitting deadline failed; completion-only amendment prepared — 2 November 2026

Current software update: the latest full suite PASSED 377 tests/four optional
skips and a fresh 100-row procedural smoke/audit. This supersedes neither the
original 363-test source state nor the failed original smoke. No completion-only
inference has run yet; publish/verify source before the separate missing-case
pass. `reports/north_completion_software_checks_v52.json` records exact hashes.

The original v39/v40 six-hour attempt is TERMINAL FAILED after a large observed
clock gap. All original processes ended. All 6,000 fitting updates are intact:
three 2,000-update histories, Adam counters and endpoint tensors independently
agree. The third endpoint has 34/96 valid saved case files; 62 positive and
96 selected-negative evaluations remain. The original full regression passed
363 tests/four skips before the latest two fixture-cache tests; its fresh smoke
hit its own calendar cap and produced no output. The waiting v43 probe refused
inference because its prerequisites failed. Preserve every original state/log.

`north_completion_v52.py` prepares a separate 1,800-second inference-only
allowance in a NEW root. All old files are copied and verified without hard
links; the original protocol/deadline stay byte-identical and FAILED. Reuse
all weights, updates and completed cases; execute no optimizer updates. Keep
the original checkpoint selection/metrics/cases. An unchanged numerical cohort
replayer plus an amendment-aware byte-preservation audit are prepared. Twelve
focused controls pass; fresh complete regression/smoke, public source freeze
and actual completion/audit are PENDING. Do not restart the original training.

Evidence: `reports/north_original_timeout_inventory_v52.json` and
`reports/north_completion_contract_v52.json`. New v49/v51 experiments remain
PREPARED, not executed. Their old success-only gates need an explicit amended
handoff after recovery/coarse diagnostics; never relabel old failures as passes.
The v51 fixed-pair fixture/metric/cache code passes 15 isolated checks; this is
not an optimization result. All 70%/10pp/1%/modern/fresh/transfer/paper targets
remain unmet; no model upgrade, paid compute, landmark review or email.

## Saved modern-control replay hardened; full comparison still pending — 2 November 2026

The prepared v49 verifier now refits the saved known-translation control,
recomputes its pixel/pose errors and the compact known-rotation errors, and
rejects absent/altered control records or invalid rigid camera poses. All 25
focused numerical/interface/evidence tests pass (no neural forward). The earlier
15-check preparation is preserved as history. Latest source hashes and scope:
`reports/north_modern_replay_hardening_v50.json`.

Full repository regression/fresh smoke, public source verification, actual
pretrained controls and common-case inference/replay remain PENDING. Do not
create a passed admission receipt, run another neural job, or claim a modern
ranking from these software checks. Current third-seed fitting/evaluation and
all three original controller/source seals remain unchanged.

## New northern modern-comparison adapter prepared — 2 November 2026

`north_modern_comparison_v49.py` prepares standalone SIFT and the pinned
MatchAnything, LoMa-R, RoMa-v2 and MOONSt3R common profiles on the original
96-positive/96-negative northern selection cases. The paired control preserves
the frozen compact/native/surface pipeline. The earlier SIFT-first fallback
is still labelled separately; it is not standalone SIFT. Query pose/depth enter
scoring only. Complete/partial cases, controls, source provenance and the
original absolute time allowance are retained; no precision/profile fallback.

Fifteen isolated interface/cache/process/numerical-PnP checks pass. The new
complete repository regression/fresh smoke, public source verification,
actual pretrained controls and benchmark/replay are PENDING. This is prepared
software, not measured modern accuracy. No new comparison is autoqueued.
The program refuses inference until all three existing controllers end cleanly,
the v43 independent diagnostic passes, and a source-bound software receipt exists.
LoMa/RoMa keep their native-BF16 CPU requirement; use existing free resources,
never an undeclared local precision replacement. One two-thread science job.

Evidence: `reports/north_modern_preparation_v49.json`. Keep all current fitting
and v43 sources unchanged. Third seed is live; full research targets, four
calibration/twelve final supports and all no-paid-compute/no-email limits remain.

## Two seeds complete; diagnostic sources frozen before inference — 2 November 2026

The first two expanded-data seeds completed 2,000 updates each. Their selected
checkpoints recover 14/96 and 15/96 correct raw poses, versus 13/96 frozen;
selected nonoverlap proposals are 4/96 and 7/96 versus 11/96. These small gains
are far below the targets; the third seed and complete cohort audit are running.
Keep every seed, checkpoint and endpoint; no trained upgrade is adopted.

Prepared v43 sources test coarse-proposal reachability and a separately labelled
perfect-local-refinement oracle, followed by independent grid/window/pose replay.
The oracle uses query truth and is never model accuracy. All 18 new isolated
metadata/metric/cache checks pass. The new full repository regression and fresh
smoke remain queued after the current scientific controller; do not claim the
sum of old/new isolated counts as a completed suite. Publish and byte-verify
source before the new probe, and execute it only after all upstream gates and
process-exit checks pass. [Failure diagnosis](NORTH_FAILURE_DIAGNOSIS_V43.md),
[camera-design limitation](NORTH_CAMERA_DESIGN_V41.md).

All running v38/v39/v40 fitting/inference sources stay unchanged. Keep four
calibration/twelve final supports unused. Full acceptance, modern comparisons,
unseen-camera transfer, uncertainty, paper and compact-v3 targets remain open.
No paid compute, human landmark review, email or SOTA claim.

## Northern data and training gates complete; full fitting live — 2 November 2026

All 576 images are complete; the unchanged audit passes 72/72 oracle poses and
243,360 exact label points. The separate recovery took 726.75 seconds; the
original v37 calendar deadline stays failed. The train-only 100-update pilot
completed, and all model/optimizer/loss/gradient/RNG states independently replay
exactly. No calibration or final supports were loaded.

The full common baselines and all 384 saved metric rows replay. Frozen compact/
surface and SIFT-first/surface each recover 13/96 correct raw positive poses,
with 28 wrong positive proposals, 55 failed poses, and 11/96 nonoverlap proposals.
These are raw development proposals, not calibrated acceptances or SOTA results.
The northern dataset is different from the old 72-pair benchmark.

The fixed three-seed architectural control is live: 2,000 updates per seed,
all 96 positive selection cases, initialization eligible. Keep its sealed source
unchanged, retain all outcomes, and inspect the actual controller before another
scientific CPU job. [Verified data, pilot and baseline evidence](NORTH_DATA_AND_PILOT_V41.md).
Full research targets remain active and unmet; no email or paid compute.

## Original render timeout; separate recovery prepared — 2 November 2026

The original v37 render terminated at its fixed calendar cap after an observed
wall-clock gap; 16/24 terrains are complete. Both original processes ended.
The exact cause of that clock gap is unknown. Original timeout, partial outputs
and deadlines remain preserved. No new-data matching or training ran.

A separate explicit v40 amendment permits up to 30 additional calendar minutes
for the eight incomplete components. All sixteen completed components are
byte-verified and copied, not linked; the rendering/supervision kernel is
identical. There is no camera, lighting, terrain, metric or case-selection change.
307 tests/four optional skips and a fresh 100-row procedural smoke pass.

After public source verification, queue recovery, the unchanged full geometry
audit, six-light bundle, 100-update pilot/exact replay, cost bank, full common
baselines/replay and the prepared three-seed architectural control. Each gate
must pass; no final/calibration pixels, paid compute or email. New model
accuracy remains unmeasured. [Amendment](NORTH_RENDER_RECOVERY_V40.md).

## Expanded-data training continuation prepared — 2 November 2026

The corrected v37 data pipeline is active; its train-only 100-update pilot and
exact replay remain queued behind the complete geometry audit. The separate
v39 control is now implemented: unchanged architecture/loss, three fixed seeds,
2,000 updates each, all 96 positive selection cases, initialization eligible,
and selected-checkpoint nonoverlap diagnostics. Eight targeted failure/resume/
freeze/cache regressions pass. The full suite and fresh smoke for this new code
are queued after the current scientific job, before full fitting.

Publish/verify this prepared source, then serialize: upstream completion,
repository tests/smoke, full frozen/SIFT baselines and metric replay, measured
cost admission, three-seed fitting and saved-result audit. Keep calibration and
final supports unused. No new model gain, accepted-pose or SOTA result is yet
measured. [Protocol and commands](NORTH_EXPANDED_CONTROL_V39.md).

## Objective

## Solar correction before expanded-data fitting — 1 November 2026

All 24 development terrains/meshes were acquired. The first render was stopped
and preserved after a necessary celestial-geometry check found 96/144 old Sun
settings outside a generous lunar envelope. No matching/fitting outcomes were
used. The fixed solar-body latitude/hour-angle amendment keeps source regions,
cameras, case-index pairs and metrics unchanged. Actual Blender lamp controls
pass all 144 configurations. 295 tests/four optional skips and fresh 100-row
smoke pass. The original six-hour render budget/absolute deadline is retained.
[Evidence and gated commands](NORTH_SOLAR_AMENDMENT_V37.md).

Next publicly verify and launch the corrected v37 render/audit, then prepare
v38 data and a train-only 100-update cost pilot with exact independent replay.
The v38 programs are PREPARED; no new fitting or new model accuracy is measured.
Keep four calibration/twelve final supports untouched. Historical v35 output
is stress data, not the primary lunar lighting study; do not restart it. Full
research targets remain active and unmet. No paid compute, human review or email.

## Expanded development generation prepared — 1 November 2026

Only the original16training/eightselection terrain supports are allowed by the
new v35cache/acquirer.576renders and243,360exact fractional-grid labels are
prepared, not generated. Actual pinned Blender BVHlabels match independent
analytic triangles in both directions/allfourrotations.288tests/four optional
skips and fresh100row smoke pass. Publish and byte-verify before new height
reads. Keep calibration/final/pilots out; use explicit rawZ/axial/ray fields.
See [generation/training handoff](NORTH_DEVELOPMENT_DATA_V35.md). No new fitting
or model gain is measured. Original full research targets remain active.

## Verified northern data pilots; no learned upgrade — 1 November 2026

All96pilot images rendered in425.32s; independent exact-correspondence recovery
passes12/12camera poses. The initial pilot audit caught an implementation
mistake: a raw Blender camera-Zpass was calledray_depth, despite the prior
analytic controls measuring axialZ. Original failed audit/source/arrays remain
preserved. The corrected auditor is bound to those earlier measured controls,
computes true ray range separately, and retains every scoring tolerance.
282tests/four optional skips and fresh100row smoke pass. This is a data/software
gate, not learned accuracy. [Measured outcome and limitations](NORTH_PILOT_RESULTS_V34.md).

All four excluded pilot supports are acquired. Metadata checks separate all44
candidate supports from named historical project image ledgers and known height
probes; unknown external pretraining overlap remains uncharacterized. The16train/
8selection/4calibration/12final supports remain unread. Next implement only new
training/selection generation and exact supervision before the next expanded-
data fitting control. Keep old datasets, failed6kcohort, transport failures and
raw mislabeled fields unchanged; future records must label axial/ray explicitly.
No new learned checkpoint, acceptance/fresh-final/SOTA/orbital claim or email.
Full research goal remains active. Earlier entries are history.

Current transport update: original v32four-pilot acquisition FAILED before any
window returned; no renderer/fitting started. Full-stream raw download stopped
partial159,383,552bytes because observed cost exceeded its600second allowance;
preserved. Same-source v33exact-range cache covers only100pilot tiles (~54MB).
Publish/verify source before decoding; original renderer and oracle thresholds
unchanged. See NORTH_TRANSPORT_RECOVERY_V33.md. Do not restart old attempts.

Current update: all6,000 coupled-assignment updates/audit/actual selected controls
are TERMINAL complete; selected54/54/54, endpoints47/47/49. No adopted gain.
Original v30 exact replay failed on7floating-point values (max4.66e-10m);
separate numeric recovery passes, failure preserved.272tests/four skips and
fresh100row smoke pass. Actual unchanged-source north analytic renderer
controls and independent triangle/common AP3P audit PASSED; inspect
runs/localization-v32-north-control-recovery-state.json and actual process
before more science. No northern terrain pixels decoded or new training yet.
Next prioritize actual verified additional3D terrain, not repeating failed loss
sweeps. The older LIVE entries below are superseded by this update.

Continue improving lunar camera localization under the full user-approved
research targets. Do not replace the goal with software checks or a small
exposed-development gain. Fresh final validation, reliable acceptance,
remaining required native baselines, transfer, uncertainty and the paper are
still required. No global SOTA or orbital-navigation accuracy is established.

Current two-thread study is LIVE: train_mutual_cohort_v24.py, parent52307,
child52459, pipeline state runs/localization-v27-mutual-pipeline-state.json,
root runs/localization-v24-mutual-development. Public freeze
4978ed2b564049a57ee97edb85d202afbb6b9302. Do not restart or alter its pinned
sources. Three predefined seeds ×2000 updates, initialization eligible,72
exposed selection cases, shared6h cap. Actual100-update new-loss pilot passed
in16.90s with exact optimizer/model/RNG/gradient replay and eight independent
NumPy scalar checks. No pose gain yet; first seed500/1000 checkpoints45/72
versus54/72 initialization. All three outcomes must be retained. Full saved
cohort audit and actual selected controls/exact72forward are queued serially.

MOONSt3R original71-case calendar-expired run remains TERMINAL INCOMPLETE.
A separately public600s fixed-case amendment completed the single missing case
in42.48s, preserving71rows byte-identically. Amended72 independently replay:
28correct raw,21strict,44wrong. Original calendar protocol was not completed
or reset. Exact pretraining overlap unknown. See reports/native_pose_comparison_v29.md;
stronger simple pipeline remains57/72, RoMa-v2 fast/common53, frozen-surface54.
Profiles/adaptation differ; no globalSOTA, paired MOON speed or calibrated
accepted-risk claim. Dual-softmax is standard machinery, not a new operator.

v18 fitting TERMINAL12000updates/no adopted gain; all selected54/72 at step0.
Clean exact public-checkout replay now PASSED2160rows,6actual controls/exact72
forward,144MatchAnythingrows and CRESENTmetadata. The native-window-development-v18 prerelease is now published at exact clean-replayed6561f02; all four public assets match/ZIP CRC pass, compact-v2 remains latest.
Scale bank also negative12/16 allprofiles; do not expand or rerun old cohorts.
One scientific CPU job at a time; lightweight saved-only reporting/publication
may proceed during training.249tests/four skips and fresh100row smoke pass.
Fresh final data/calibration/nonoverlap/authorprofiles/transfer/uncertainty/paper/
compact-v3 CLI remain unmet; CRESENT metadata is not a fresh-depth gate. No
email. Goal active. Do not equate a completed software check with the research target.

Additional data work: LOLA_REFERENCE_V30.md records a completely acquired2.12GB
northern512ppd map (ignored data/lola-reference-v30),3official labels/8height
samples and11isolated metadata tests. Same40exposed metadata-control craters
show311.68m median/1125.09m max nominal-radius discrepancy. Exploratory halfaxes
84.79m median/1259.45m max do not establish a correction. No reference camera
depth, optical image inference, renderer agreement or fresh gate. New5metadata
source bytes are sealed by waiting53249, state runs/localization-v30-data-software-state.json;
it waits actual parent52307 clean completion, then fullpytest/freshsmoke/mapreplay.
Do not mutate these queued sources. Do not run another neural job before it.

Prospective next-data work: NORTH_RENDER_STUDY_V31.md, north_render_plan_v31.py
and reports/north_render_prospective_plan_v31.json define44disjoint fullterrain
supports before any northern height decode.16train/8select/4cal/12candidatefinal/
4pilots. Freshness not certified.6isolated analytic tests and independent GDAL
projection proof pass; official Blender4.5.9checksum/signature/build/APImetadata
checked. Actual Blender analytic render+independent triangle/commonAP3P oracle
are prepared, notrun. Publish/byteverify before forward; wait52307and53249clean
terminal before another renderer/science job. No northern optical/height pixels
or model inference yet. Original missing12releasedLunarGeogroups remain failed.

## Historical checkpoints (retained; current state above takes precedence)

The v16 cohort is TERMINAL10,000 updates, all selected54/72 at initialization;
final primaries47/43/39. Do not restart or change its pinnedM20 sources. Complete
negative capsule is public and clean-checkout replay passed. The audited v17
diagnosis motivates search-window supervision/anchors/coarse adaptation; it does
not prove the new model helps. See SEARCH_WINDOW_V18.md and STATUS.md.

Native512 MatchAnything is TERMINAL complete:39/72 correct/five strict/30wrong
versus same SIFT-first surface57/25/two. All144 native metrics/common poses replay.
Median4.40s versus2.98s on the same warmed two-thread CPU. See
reports/native_matchanything_comparison_v17.md. Remaining author-profile and
MOONSt3R baselines are required; no new baseline CPU job while fitting is live.
The prepared native_moonst3r_v19.py/control/tests are unvalidated locally; no
actual forward yet. Finish the active cohort and audit it before its control.

MOONSt3R acquired at f339a2c46f4e37ba7ed0ad3b7df51a1cb6aa8664, published full
LFS SHA4ecc74a6f8190d2cdc3837c2cc1b101a0feffa84b114a5a27c1aaa9ee91d1725.
Safe local tensor conversion passed after preserving the first strict-stride
failure. No model forward yet. Its source is ed6ea257475abb8a5235dff3a4d6167214b6f697.
Top-level Apache2 does not replace nested MASt3R/DUSt3R CC BY-NC-SA4 code.
Weight-card license unspecified; do not redistribute external weights. Actual
checkpoint metadata names split files, but they are absent in published source;
exact training-pair overlap remains unknown.

The v13 surface study remains55/55/55, same SIFT fallback58/58/58 versus57,
one wrong raw estimate; its one-pair gain is inconclusive. Fresh final gate
still fails: StereoLunar v2's3 archives exactly duplicate current source files.
No calibration/test pixels, paid compute, human review or email.225 tests/four
optional skips and fresh100-row smoke pass. Full research targets remain unmet;
no global SOTA, fresh-generalization,1%risk or orbital-navigation claim.

The currently authorized extension is [LOCALIZATION_RESEARCH.md](LOCALIZATION_RESEARCH.md).
The user's subsequent instruction, "Continue with the training", authorizes
development-only training on the existing train split with model-selection
groups kept separate. This supersedes the training stop for that narrow scope;
the failed fresh-data gate still prohibits a new final-test/generalization claim.
Do not load calibration/test pixels, alter old results, or imply a fresh split.
The earlier compact-v2 study remains
historical and unchanged. Preserve every earlier outcome, and distinguish
implemented software from executed measurements and achieved research targets.
Build a scientifically honest, reproducible benchmark of lunar image matching across illumination and geometric conditions. Deliver runnable code, tested metrics, provenance, frozen geographically separated evaluation, plots and a short report. Do not optimize for attractive numbers.

## Workflow for every milestone
1. Read README, this file, MILESTONES and STATUS. Inspect existing work before editing.
2. Implement the next incomplete milestone after its software dependencies pass. If an external dataset lacks independent reference accuracy, proceed with explicitly provisional map-reference diagnostics and keep real-accuracy claims unverified; never invent checkpoint accuracy. Human review is optional and must not block the current automated benchmark.
3. Run relevant numerical tests and a small end-to-end experiment. Record exact commands, dataset/version, configuration, seed, environment, outcomes and limitations.
4. Update STATUS with evidence. Distinguish implemented, executed and scientifically validated.
5. Commit with `M<number>: <concrete result>` and push to the existing default branch for this new solo project. Never force-push or rewrite history. Never commit credentials, personal emails, virtual environments, downloaded source rasters or model weights.
6. Update the corresponding GitHub issue; close only after every acceptance check passes. Do not contact researchers automatically.

## Scientific rules
- All transforms map reference pixels to query pixels; use x,y, not row,column.
- Synthetic gain, gamma and noise are appearance perturbations, not lunar illumination simulation.
- Split by terrain region AND source observation before extracting patches. Adjacent or overlapping footprints must remain in the same group. Freeze manifests before evaluation.
- Do not estimate ground truth using the matcher being evaluated. Claims of real repeat-observation accuracy require independent reference measurements and uncertainty. Without them, report diagnostic results only; human annotation is outside the current scope.
- Count failed registrations in denominators. Report applicability exclusions separately. Report errors only where reference points remain within the query image and valid-data masks.
- NCC is a translation baseline. Compare it fairly on translation-only cases; explicitly label out-of-model rotation stress tests.
- Use consistent image sizes and preprocessing. Time inference separately from model initialization/download; warm up learned models and synchronize accelerators.
- Do not infer metric accuracy from pixels unless a validated image scale supports the conversion.
- Preserve negative results. Never invent data, citations, metrics or lunar-science conclusions from procedural tests.

## Architecture
Keep generation, matchers, geometry/metrics, orchestration and reporting separate. Use configuration files and a deterministic seed. Real-data manifests must include checksums and provenance. Keep large assets outside Git.

## Scope boundaries
First deliver a verified CPU smoke benchmark. Then acquire and audit real lunar data. Add a pretrained learned matcher only after checking upstream API and code/weight licenses, pinning its revision and proving a real forward pass. Missing weights or unavailable data must be explicit errors or blockers, never silently replaced with another method.

## Final report
Two pages: question; dataset and split; baselines; metrics; main figure/table; failure cases; limitations; next experiment. Include a reproducibility appendix. Do not claim reproduction of LIMA or the cited lunar TRN paper unless its full method is actually implemented and verified.
