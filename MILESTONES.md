## M37 — Correct celestial lighting before model fitting

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

24development meshes acquired; v35render stopped/preserved before any model
inference after96/144solar settings failed a necessary lunar geometry screen.
New fixed body-latitude/hour-angle contract is independent of model results.
144actual Blender lamp controls and295tests/four skips/fresh100row smoke pass.
Original render allowance and absolute deadline remain. Corrected render,
72geometry audit and gated v38train-only100update/exact replay are next. No
model gain, fresh-final or SOTA result; full objective remains active.

## M35 — Expanded development data/software freeze prepared

16training/eightselection original supports only;576renders and exact
fractional-grid forward/reverse visibility banks prepared. Actual Blender BVH
controls8/8match independent triangles,288tests/four skips/fresh100row smoke
pass. Public source verification required before decoding. Full72geometry
oracle and actual generation remain pending; no new fitting/model result.

## M34 — Actual terrain and depth-semantics controls complete

96renders/four excluded pilot components in425.32s. Original audit failed on
our mistaken ray-vs-axial interpretation; preserved. Prior analyticAXIALZcontrol
anchors the corrected audit, with unchanged tolerances;12/12geometry oracle
poses pass.282tests/four skips/fresh100row smoke pass. Named own-project image
and height-probe supports are separated from prospective northern supports;
external pretraining remains unknown. No new learned model or final inference.
Next acquire only16train/8selection supports with explicitly labelled depth
and exact 3D supervision; keep calibration/final untouched. Full targets unmet.

## M33 — Same-source pilot transport recovery prepared

Original remote BigTIFF extraction failed; full-stream attempt interrupted and
preserved159MBpartial. New bounded strict-range cache covers identical four
pilot supports (100tiles/54MB), no reserve access. Six transport tests pass;
actual replacement cache/decoding/render/oracle not run. No new fitting/gain.

## M32 — Complete coupled-assignment negative cohort / data software recovery

Executed6,000updates, three seeds all selected initialization54/72; endpoints
47/47/49. All1,080saved rows and6,000draws audit; three actual selected controls
and exact primary72forward pass. Original byte-exact map replay failed on seven
floating-point values; preserved alongside separate <=1.87e-9m numeric recovery.
272tests/four skips and fresh100row smoke pass. Actual renderer and independent triangle/AP3P controls pass;
no height pixels decoded/new model gain/fresh benchmark. Full research targets
remain incomplete. Package/public-source replay are pending.

# Milestones

## Two seeds complete; prospective northern simulation controls prepared — 1 November 2026

The first two coupled-assignment seeds completed2000updates each, both ending
47/72correct raw poses and retaining initialization54/72. Third seed remains
live; its500/1000 checkpoints45/41 trail initialization54. The complete6000-
update result, independent1080rowaudit and actual selected controls remain
pending. Do not claim an adopted learned improvement or stop/restart this run.

A separate additional northern-terrain study is prospectively specified before
any source-height decoding:16training,8selection,4calibration,12candidatefinal
components and4pilot components, with disjoint20.48kmfullterrain supports.
Historical optical/training overlap still needs audit before calling them fresh.
Official Blender4.5.9package checksum/signature/build/API and actual COGheader
are checked; independent GDAL-versus-analytic projection error is2.85e-14degrees.
Six isolated geometry/triangle tests pass. Actual renderer camera/depth/occlusion
and independent common-AP3P controls are PREPARED, not run. No new terrain
pixels, training pairs, renderer forward or final matching result exists.
See [prospective data/control contract](NORTH_RENDER_STUDY_V31.md). The original
fresh released-LunarGeo shortfall remains recorded.

Serialize actual renderer controls only after live52307and waiting53249finish
all phases cleanly; first publicly freeze and byte-verify their source. No
competing neural or renderer jobs. Data/camera/oracle/cost gates must pass before
full generation or new fitting. The original fresh-test/acceptance/modern-profile/
transfer/uncertainty/paper/CLI targets remain active and incomplete. No email.
Earlier milestones are history.

## Training continues; additional terrain-reference check reveals uncertainty — 1 November 2026

The first coupled-assignment seed finished2000updates: checkpoints54/45/45/49/47
correct raw poses, so selection retains initialization54/72. The second seed
remains live; its500/1000 checkpoints44/42 are below initialization54. The third
seed and full independent cohort/actual selected controls remain queued. These
are partial development outcomes, not a completed all-seed result. All fitting
source bytes still match public freeze4978ed2. Do not restart or edit them.

An official2.12GB northern LOLA512ppd height tile is completely acquired and
locally hashed; source size/ETag match the fixed range query. No publisher
checksum is claimed. Three official labels/eight actual samples pass an
independent byte/unit audit and11focused no-neural controls. A predeclared
40crater/360rim-position check on already exposed CRESENTmetadata frame0 finds
311.68m median absolute radius discrepancy, maximum1125.09m. An explicitly
exploratory half-axis alternative gives84.79m median but1259.45m maximum; it
does not settle semantics or renderer registration. Original annotations and
all outcomes remain preserved. [Data/reference findings](LOLA_REFERENCE_V30.md).
No source optical pixels or matching inference, camera-depth map or fresh-test
gate is created. CRESENT cannot silently replace missing fresh LunarGeo groups.

The full repository neural regression, fresh100row smoke and actual map-probe
replay are queued behind the verified live fitting parent52307; waiting
process53249 must not start until all fitting/audit/control phases exit cleanly.
Metadata controls run without neural inference. No learned upgrade, calibrated
acceptance, fresh-test or SOTA claim; the full research goal remains active.
Earlier milestones are history.

## Coupled-assignment fitting live; amended MOONSt3R complete — 1 November 2026

The publicly frozen new objective passed its actual100-update train-only pilot
in16.90seconds. Independent retraining exactly reproduces model, optimizer,
RNG, losses and gradients; eight float64 NumPy scalar checks pass. This is a
reproducibility gate, not a pose gain. The **6,000-update, three-seed study is
RUNNING** from public freeze `4978ed2b564049a57ee97edb85d202afbb6b9302`.
All72 exposed model-selection cases and initialization remain eligible; all
three seeds will be retained. One two-thread science job, shared six-hour cap.
Inspect `runs/localization-v27-mutual-pipeline-state.json` and
`runs/localization-v27-mutual_cohort.log`; do not restart or mutate frozen code.
Initial54/72 remains ahead of the first seed's500/1000-update45/72 checkpoints.
This is interim evidence, not a completed negative or positive cohort result.

The explicit public single-case recovery contract completed in42.48seconds.
The **amended MOONSt3R comparison is complete and independently replayed72/72:
28 correct raw poses,21 strict,44 wrong**. All71 original rows are byte-identical.
The original clock-interrupted four-calendar-hour run remains incomplete; this
separate600-second resource amendment does not reset or complete that protocol.
Exact training overlap is unknown. [Consolidated native results and limits](reports/native_pose_comparison_v29.md)
compare SIFT-first+surface57/72, frozen-surface54, RoMa-v2 fast/common53,
LoMa-R42 and MatchAnything39. MOON's21strict exceeds frozen-surface16; these
profiles do not establish general precision, speed or SOTA dominance. Raw
wrong proposals are not calibrated accepted errors.

249tests/four optional skips and fresh100-row smoke replay pass. The complete
v18 negative study's fresh public checkout replay also passes, including actual
six selected controls and exact72-case neural forward. The [v18 negative-study prerelease](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/native-window-development-v18) now targets the exact clean-replayed6561f02 source. All four public assets are byte-identical; both ZIP CRCs pass. compact-v2 remains latest. No source/test changes were made to live fitting. Remaining
fresh-data, calibration/nonoverlap, author-profile, transfer, uncertainty,
paper and compact-v3 requirements are still unmet. No paid compute, human
review, automatic email or research-goal completion. Earlier entries are history.

## New coupled-assignment study prepared; calendar interruption preserved — 1 November 2026

A small algebra fixture using the full v18 loss shows an objective decrease
0.09809→0.09448 while mutual proposals fall2→1. The new coupled positive mass
penalizes that fixture (0.55291→1.02631). This is a counterexample to a guarantee,
not evidence that this caused the actual lunar regressions. No new fitting or
pose improvement is measured. See [hypothesis and complete prepared commands](MUTUAL_ASSIGNMENT_V23.md).

Prepared: a train-only100-update cost/exact-replay pilot, then a gated three-
seed6,000-update cohort with unchanged native inference,81,696 total parameters,
all72 exposed selection cases and initialization eligible. Full software gates
pass: **249 tests/four skips**, including actual new-objective gradient/protected-
weight/exact-resume integration; a fresh100-row smoke independently replays.
Fresh seeded weights match the previously actually controlled graph exactly.
Publish/byte-verify all source before pilot fitting; independent100-update
retraining/new-scalar audit is required before the full cohort. Neither software
checks nor the toy fixture establish a localization gain.

The original MOONSt3R run was stopped incomplete at **71/72** after observed
calendar/monotonic clock divergence. Calendar elapsed41,891s exceeded its14,400s
contract; completed-case inference totalled3,239s. Preserve its original data,
protocol and timeout. The exact cause of the divergence is not proven. A
[separate publicly frozen600-second recovery contract](MOONST3R_RECOVERY_V28.md)
permits only the single missing case with identical source/weights/inputs and
copies the original71rows byte-identically. It stays within the unchanged
72CPUwall-hour project allowance. This is an amended two-phase comparison,
not a reset or completed original four-calendar-hour study. Supplemental
inference has not run yet and no completed baseline accuracy is claimed.

The complete v18 negative capsule now passes a fresh public checkout/install:
all2,160 saved checkpoint rows, six actual selected-model controls, exact72-
case primary forward,144 saved MatchAnything rows and separate CRESENT metadata
replay. Tagged assets/public download verification remain pending. The CRESENT
reference-depth/freshness gates remain false; zero new source raster pixels.

No current science job is active; prior fitting/comparison/replay queues are
terminal. Source publication, bounded recovery and the new pilot are next.
Full fresh final data, calibrated acceptance/nonoverlap, remaining comparisons,
transfer, uncertainty, paper and compact-v3 interface remain incomplete. No
paid compute, human review, email or goal completion. Earlier entries are history.

## Selected-checkpoint replay complete; scale bank negative — 1 November 2026

All six selected v18 checkpoints pass actual transform/blank/invalid controls.
A fresh primary72-case neural forward exactly reproduces every saved field.
The full negative development capsule is packaged/checksummed locally:
`reports/native-window-v18-capsule.zip`, SHA256
`1a68508aead95bcd86f3fcd6a372be2e18f25425d104a5464b3e64859f437ca6`.
Its tagged release and clean public-checkout replay remain pending.

The source-frozen shared-scale diagnostic also completed: native /native+half /
native+half+quarter each12/16 correct raw poses, strict5/4/4 and one wrong
estimate each. Successful case identities are unchanged. Point precision at
2nativepixels rises77.96%→81.49%, but pose accuracy does not improve; the
historical original-border control has six strict poses. No scale method is
adopted and the conditional full72-case run is not triggered. All48 saved rows
independently replay. See [measured scale findings](reports/fine_pyramid_bank_v22.md).

MOONSt3R passes actual pretrained controls (3,973 known-translation matches,
median0px error; both blank/invalid empty). Its **full72-case common-native
lunar comparison is RUNNING**, as the only two-thread research CPU job, from
public source `fac436adf00c9c46d7f24f40b793154830574f9f`. A shared four-hour
comparison deadline survives resume. Inspect
`runs/localization-v26-next-native-state.json` and the live case files/log;
do not launch another training or baseline job while it is live. Partial
case counts are not a completed baseline accuracy result.

238 tests/four optional skips and fresh100-row smoke replay pass. All new code
and the completed v18 evidence are public;38 changed files and all788 current
public tree files were byte-verified before a metadata-only local fast-forward.
Fresh final data, acceptance/nonoverlap calibration, transfer, remaining
comparisons, uncertainty, paper and compact-v3 interface remain incomplete.
No paid compute, human review, outreach or research-goal completion.

## Native-window training complete: no learned gain — 1 November 2026

All six v18 runs completed **12,000 optimizer updates** from public fitting
freeze `76ec8bf1e3ecdf356bbcc3293bebd12e0e4ce189`. Every selected checkpoint is
initialization: **54/72 correct raw poses**, 16 strict and three wrong estimates.
The four primary trained endpoints give 49/50/53/43; the window-only repeats
end at 43/45. The same stronger SIFT-first baseline stays at 57/72. These
trained variants are not adopted as an improvement.

Independent audit passes all **2,160 saved checkpoint rows**, 12,000 sampler/
update records, 97,344 projected label points and 65,804,544 window memberships.
The full cohort is not independently retrained; its separate 100-update pilot
was exactly retrained. Both initialization graphs pass actual transform,
blank/invalid controls and exact 72-case baseline equivalence. See
[complete results](reports/window_cohort_complete_v18.md) and
[audit](reports/native_window_complete_audit_v18.json). All failures and seeds
are retained. Fitting sources remain immutable.

Validation: 238 tests pass, four optional skips; a fresh 100-row software smoke
independently replays. The optional CRESENT source-metadata fixture is installed
locally; without it, one integration check explicitly skips. Report-only label
layout was regenerated from the unchanged saved metrics after retaining both
earlier renders. MOONSt3R's strict safe-tensor model load and actual pretrained
translation/pose, blank and invalid-mask controls pass (116.97 seconds total). Full lunar
baseline results are not available yet. The selected-checkpoint controls and
exact primary forward replay are now running as the only research CPU job,
followed by the complete development capsule. Inspect live state files before launching work.

Next: the [shared-scale fine diagnostic](FINE_PYRAMID_V22.md) is prepared and its
coordinate/actual-encoder tests pass. Run its actual checkpoint controls and
fixed 16-case exposed bank only after existing jobs finish and source is
publicly frozen. It tests context at native/half/quarter scale with identical
search and conservative support, before another learned cohort. No scale gain
has been measured. [CRESENT metadata](CRESENT_CAMERA_METADATA_V21.md) provides a
candidate additional source, with original broad failure preserved; zero image
pixels or matching results, and no completed depth/fresh-data gate.

All pose numbers remain development results on four exposed terrain groups.
Fresh final data, calibrated accepted-error/nonoverlap risk, remaining native
comparisons, transfer, uncertainty, paper and compact-v3 pose interface remain
unmet. No SOTA/orbital-accuracy claim, paid compute, human review or email.
Earlier entries below are historical snapshots.

## Full native-window cohort active; both fitting gates passed — 1 November 2026

The v18 initialization gate passed for both distinct inference graphs on all72
exposed selection pairs: identical native points and poses,54/72 correct raw,
16 strict. Actual known-transform, blank and invalid-mask controls pass. This
checks unchanged initialization, not a learned gain.

The train-only100-update coarse-adaptation pilot completed in23.05 seconds.
Independent audit checks97,344 projected label points and65,804,544 candidate
window memberships, then exactly retrains all100 updates: gradients, weights,
optimizer and RNG match. The fine encoder is unchanged. The measured pilot is
not warm-started into the scientific cohort. Gates are in
reports/native_window_initial_control_v18.json and reports/native_window_pilot_audit_v18.json.

The full12,000-update cohort is RUNNING from public fitting freeze76ec8bf:
four2,000-update controls and two predefined repeats of the selected control.
One two-thread CPU job, with a shared six-hour cohort deadline. All72 exposed
selection pairs choose checkpoints by unfiltered pose count, then2px count,
then earlier update; initialization remains eligible. No completed trained
v18 localization outcome exists yet. Do not restart or modify frozen sources.
Authoritative log:runs/localization-v18-window-pipeline.log; stage state:
runs/localization-v18-window-pipeline-state.json.

The native512 MatchAnything comparison is complete and independently replayed:
39/72 correct raw/five strict/30 wrong, versus SIFT-first surface57/72/25/two.
Median4.40s versus2.98s, same warmed CPU/order-alternated full pipeline. See
reports/native_matchanything_comparison_v17.md. This is a common-profile
exposed-development comparison, not a fresh/SOTA or calibrated-acceptance claim.
229 tests/four optional skips and a fresh100-row smoke pass. The MOONSt3R
adapter is prepared locally but has no actual forward/control yet; do not
substitute its software readiness for a completed baseline. Full fresh-data,
acceptance, transfer, remaining comparisons and paper targets remain unmet.
No paid compute, human review or email. Earlier entries are preserved history.

## Native-window training runner prepared — 1 November 2026

The complete v18 runner now implements four2,000-update ablations and two
predefined-seed repeats of the selected configuration:12,000 updates. Search-
window set supervision, genuine-proposal anchors, coarse adaptation and paired
context are separated. The native fine CNN stays frozen; total81,696–82,208
parameters. Resumable checkpoints preserve optimizer/sampler/model state and
initialization remains eligible. This is implemented software, not a trained
result. Actual all72-case initialization and100-update independently replayed
pilot gates are required before scientific fitting. See SEARCH_WINDOW_V18.md.

The native512 MatchAnything comparison completed from public3b0cc2d:39/72
correct raw poses/five strict/30 wrong, versus SIFT-first surface57/72/25
strict/two wrong. Median paired-worker inference4.40s versus2.98s. Independent
replay verifies all144 native metrics/pose fits. No simultaneous research CPU
job ran during timing. This profile is not a fresh test or a global SOTA claim.
All229 repository tests pass, four optional skips; fresh100-row smoke replays.

The complete v16 negative cohort is published as global-context-development-v16.
Both public assets redownload byte-identically and ZIP CRC passes. A fresh
public checkout reproduces all1,800 saved checkpoint rows, five selected-model
controls and exact primary72-case forward output. This verifies reproducibility,
not a new terrain test. Fine-grained failure diagnosis remains the reason for
the new loss; training has not demonstrated an improvement. Original fresh-
data, calibrated acceptance, transfer, remaining baselines and paper targets
remain unmet. No paid compute, human review or email. Earlier entries are history.

## Context study complete; objective mismatch diagnosed — 1 November 2026

All five v16 runs completed10,000 updates. Every selected checkpoint is step0:
54/72 correct raw poses,16 strict, three wrong estimates; the same SIFT-first
fallback remains57/72. Final trained primary checkpoints regress to47/43/39.
This context hypothesis did not improve localization and is not adopted.
Independent audit replays all1,800 checkpoint rows and every sampler record.
Actual selected-weight controls pass for all five models, and the primary
model exactly reproduces all72 saved outputs. The completed reproducibility
capsule is checksummed; this is not a new terrain test.

A separate2,160-record diagnosis finds better target likelihood but fewer
geometrically useful proposals on both training fixtures and exposed selection.
Independent geometry/window replay passes. Oracle filtering of the saved
failed matches yields at most two extra correct cases across the tested rules;
all252 fits replay. These use query truth and are not deployable model results
or a universal upper bound. See reports/context_failure_diagnosis_v17.md.

The next untrained hypothesis is search-window positives, genuine-proposal
anchors and coarse-feature learning while native fine descriptors stay frozen.
The81,696–82,208-parameter core and five meaningful controls are implemented.
The full runner, actual72-case initial gate and measured training pilot remain
pending; no v18 fitting or gain exists. See SEARCH_WINDOW_V18.md.

Native512px MatchAnything passes actual pretrained software controls:0.0602px
median known-translation error, blank/invalid-mask outputs empty. Its full
common-pose comparison is implemented and must start only after public source
verification and while no other research CPU job is active. No lunar result
or isolated runtime claim comes from that one-thread control. MOONSt3R source
and its full published checkpoint are pinned and verified;1,017 tensors were
extracted using inert metadata parsing. No MOONSt3R forward pass is claimed.
Its exact training overlap remains unknown; no external weights are bundled.

StereoLunar v2's three public archive checksums match the consumed LunarGeo
archives exactly, so they provide no new reserve. The fresh-final gate still
fails.225 tests/four optional skips and a fresh100-row software smoke pass.
Fresh evaluation, calibrated acceptance/nonoverlap, remaining native baselines,
transfer, uncertainty, paper and compact-v3 pose interface remain required.
No paid compute, human review or automatic email. Full research goal active
and unmet. Older entries are historical snapshots.

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

## M58 — matched native fine supervision

Prepared encoder-only aligned/fractional sampling controls, exact100update pilot optimizer audit and full development metric replay. Execute source-published pilots, cost/control gates and sequential2000update studies. No learned improvement or final evaluation claimed.

M58 update: both100update pilots completed and exact optimizer replays passed. Full aligned study running; fractional study queued serially. No pose gain or final result claimed.
