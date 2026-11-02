# Status

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

## Coupled-assignment cohort complete; next-data controls running — 1 November 2026

All6,000updates and three predefined seeds completed. Every run selects
initialization54/72correct raw poses; trained endpoints47/47/49. No learned
upgrade is adopted. The full independent audit replays1,080saved checkpoint
rows and6,000update records; all three selected models pass actual controls,
and a fresh primary72case forward exactly reproduces the saved outputs.
[Complete negative result](reports/mutual_cohort_complete_v24.md).

The repository passes272tests/four optional skips and a fresh100row smoke.
Original byte-exact map replay failed on seven floating-point values, maximum
4.66e-10m. The original failure is preserved; a separate numeric recovery check
passes within eight representable increments at lunar radius (1.87e-9m).
This does not repair the311.68m terrain-registration discrepancy. All sources
and both probe outputs remain unchanged. Actual public-source Blender analytic
controls and independent triangle/common AP3P audit PASSED via the recovered precondition, with unchanged
renderer sources/tolerances and no northern height decoding. Check
runs/localization-v32-north-control-recovery-state.json before another science
job. The original waiting queue failed upstream before rendering.

Next: preserve/publish the negative result and verify exact camera/depth controls,
then only the four predeclared northern terrain pilots, cost and actual terrain
oracle gates. More loss sweeps on the same four exposed terrain components are
not evidence of progress. Original fresh-data, calibrated acceptance, modern
profiles, transfer, uncertainty, paper and compact-v3 targets remain unmet.
No paid compute, human review or email. Earlier entries are history.

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


## Native precision research active; no verified target achievement

The completed v6 orientation ablation adds16000 optimizer steps with no pose
improvement. The narrowed regime retains initial weights (45/72); the wide
control and repeat seeds give44/72,45/72,45/72 and exactly reproduce v5 tensors.
Independent288-record replay passes. See reports/ORIENTATION_TRAINING_V6.md.

A native-refinement100-step pilot gives47/72 correct raw poses versus frozen45,
with eight wrong estimates. The full v7 ablation/three-seed study is running in
runs/localization-v7-native-development. Its first completed encoder/pair
ablations give44/72 and45/72; the no-match ablation/repeats remain pending.
Do not present the pilot as a consistent gain or completed model study.
The model remains below100000 total parameters; current sources are pinned.

Training-source diagnosis finds only3 and23 correct local coarse proposals in
two difficult training components. A separate uniform-positive100-step pilot
uses10463 projected points across all12 training components and gives45/72
(18 strict), so it has not shown a raw-pose gain. Its full study has not run.
Both pilots pass independent234-label/72-output, checkpoint, sampling/RNG and
actual-model controls. All points/labels are simulated geometry, not human marks.

Corrected exact-in-window controls recover67/72 oracle poses; all visible
returned reference points recover72/72. All288 branches, axial projection,
rotation/window masks, physical ranges and pose-estimator replay pass. These
use query truth, are not deployed methods and are not formal performance bounds.
The frozen four-orientation bank already selects all45 correct poses available
among its own branches; reranking those outputs alone cannot add a correct case.

LoMa-R and RoMa-v2 native512px common-profile comparisons are running on
standard public ARM CPU workers with two threads each. Both passed actual
native known-transform/invalid-input controls. Runs41255920157/41256049036 are
bounded; complete coverage is not established. A fixed36+36 case-range recovery
plan preserves all72 cases, exact models/weights/settings, common geometry,
paired same-worker controls and all earlier attempts. No incomplete baseline
is a ranking or algorithmic accuracy result. No paid/larger runners are used.

Software checks:179 tests pass/four optional skips, fresh100-row/20-summary
procedural smoke with exact replay; initial native inference reproduces every
frozen selected orientation and pose decision (maximum coordinate difference
0.0000611px). Sources and evidence are in NATIVE_REFINEMENT_V7.md and reports/.
The original fresh-final gate, consistent learned gain, reliable acceptance,
uncertainty, full modern comparisons, transfer and paper targets remain unmet.
No calibration/test pixels, human review, paid compute, automatic outreach or
verified orbital-navigation claim. All earlier ray-based3D claims remain
superseded by reports/GEOMETRY_CORRECTION_V5.md; preserve frozen artifacts.

Older entries below are historical progress snapshots.

## Corrected training completed; orientation follow-up active

All four corrected v5 runs completed4000 optimizer steps (16000 total),
including the planar-data control and all predefined true3D winner seeds.
The primary/repeat raw pose results are 44/72, 45/72, 45/72; the frozen orientation
baseline is45/72. The primary gives44/72 and does not establish
a learned advantage. Every initial/selected/final and optimizer/RNG checkpoint,
failed later checkpoint and source snapshot is retained. No best-seed claim.

A stronger simple SIFT-first fallback gives53/72 correct raw poses, but also
six wrong estimates; this is not calibrated acceptance or the70% accepted-pose
target. All trained candidates receive the identical fallback comparison.
These are72 dependent development pairs from eight previously exposed
geometries/four corrected footprint components, not a fresh final test.

Independent replay verifies all288 candidate outcomes, actual weight changes,
completed budgets, checkpoint/configuration selection and every predefined seed.
Exact corrected correspondences recover26/26 poses; rounding to an8pixel grid
reduces this to3/26, demonstrating precision sensitivity under this control.
The raw-depth/source-closure audit remains separate and required.

A v6 orientation-regime follow-up is now running at
runs/localization-v6-orientation-development, with publicly pinned code from
62e7f846f84d05af55f6aa394396386bc8521929. It compares the unchanged wide3D
training control with training-only quarter-turn canonicalization plus small
residual rotation; inference, architecture, optimizer and budget stay fixed.
The100-step software pilot passed replay, retaining initial weights and45/72.
Do not edit either live v6 training file or its pinned shared sources.

All earlier ray-based LunarGeo3D metrics/labels and oracle-window ceilings
are superseded by the optical-axis depth correction. Frozen historical
artifacts remain unchanged. Corrected26-geometry footprint audit:12 train/four
selection components, no cross-split component; unknown trajectory/batch and
shared renderer dependence remain. No calibration/test pixels were loaded.
The original fresh-final data gate is still unmet.

Software:173 tests pass/four optional skips; fresh procedural smoke100 rows/
20 summaries and exact replay; actual selected-candidate quarter-turn/blank
controls and training audit are saved separately. These are software checks,
not scientific validation of navigation risk. See
[corrected learning study](reports/COARSE_TRAINING_V5.md) and
[geometry correction](reports/GEOMETRY_CORRECTION_V5.md).

Goal remains active. Consistent learned improvement, reliable acceptance,
uncertainty, modern native-pose comparisons, fresh transfer/final evidence and
the paper targets remain outstanding. No paid compute, human review or automatic
email. LuSNAR scenes7–9 remain reserved/uninspected. Scene1 acquisition timed out
with an incomplete archive prefix; individual complete metadata members were
CRC-verified, but no full-archive or transfer result is claimed.

All older sections below are historical; earlier ray-based3D interpretations
are superseded.

## Supplemental training completed — balanced sampling

All ten requested development runs completed 2,000 steps each (20,000 total),
plus a separate 100-step pilot. The second five-run cohort addresses the first
cohort's 26% positive-label fraction with exact 50/50 sampling. The selected
64,098-parameter reliability configuration recovers 27/72, 31/72 and 36/72 raw
poses for predefined seeds 20261101/02/03; the deterministic native+LM baseline
is 26/72. The primary gain is +1.39 points, with a post-selection descriptive
component interval [-11.11,+7.78]. Five pairs improve and four regress for the
primary seed. All outcomes use the same eight development geometries/four
components; they are not fresh independent tests. The 70% target is unmet.

The balanced descriptor/refinement ablations give 24/72 and 25/72. The original
five runs and failed reject-all head remain unchanged below. Balanced reliability
avoids the original reject-all failure, but its fixed point cutoff is uncalibrated
and no pair-level acceptance policy exists. Native 2px precision among co-visible
returned points is 57.64% for the balanced primary versus 48.57% for the frozen
baseline; returned/scorable counts and stricter pose failures are reported.

Both independent audits pass: the common 234 supervision cases, 720 candidate
outcomes, complete training budgets, source/weight hashes, selection and every
predefined seed replay correctly. Local tests: 147 passed/four optional skipped.
Fresh procedural smoke: 100 rows/20 summaries with exact replay. The balanced
candidate's known-shift median error is 0.353 native px; blank/default-rejection
controls pass. These are software controls, not navigation validation.
The two cohorts took 36.24 wall-clock CPU minutes including evaluation, below
the training cap. Training and model-selection groups remain separate and no
calibration/test pixels were loaded. Own initial/selected/final weights and
optimizer/RNG snapshots, supervised caches and experiment artifacts are packaged
separately with SHA256 checksums. No paid compute, human review or email outreach.

See [supplemental report](reports/LOCALIZATION_BALANCED_V3.md) and
[reproduction](LOCALIZATION_TRAINING.md). Fresh final evaluation, current native
pose baselines, transfer tests and a superiority paper remain outstanding.

## User-authorized development training completed - iteration8

After the user requested "Continue with the training", development-only fitting
resumed on the original train split. The [training report](reports/LOCALIZATION_TRAINING_V3.md)
and [commands](LOCALIZATION_TRAINING.md) document the amendment; the earlier
training stop remains historical. The fresh-data gate still governs final claims.

Five runs completed 2,000 optimizer steps each, with three predefined seeds for
the selected descriptor_3d configuration. Total model sizes including frozen
coarse:56,544 (descriptor),63,505 (pair refinement),64,098 (reliability). Training
uses18 geometries/eight components; selection uses8 geometries/four components.
No calibration/test pixels were loaded. Old weights, sources and metrics remain
unchanged. This is development fitting, not a fresh trajectory/terrain test.

Selected primary/repeat outcomes:26/72,25/72,26/72 correct raw poses. The frozen
native+LM baseline gives26/72; a consistent learned advantage is not established.
The pair-refinement ablation gives26/72. Reliability retains initial weights
(selected step0) after its trained head overrejects; later checkpoints recover
zero poses on the16-case selection bank. Both selected and final weights are
saved; no failed head or seed is omitted. Checkpoint selection uses the fixed
metadata-selected bank, never the expanded72-case readout.

All10,000 optimizer steps completed in17.46 wall-clock minutes including model
selection/evaluation on two configured CPU threads. A separately retained100-step
pilot preceded the common configuration. Actual trained-weight changes, every
selection checkpoint, source snapshots and full case outputs are saved.

Verification:145 tests passed/four optional skipped; fresh20-summary procedural
smoke; actual candidate known-shift median error0.376 native pixels, blank
handling and default rejection passed. Independent replay validates234 exact
supervision cases,360 candidate records, all completed budgets, source/weight
hashes, checkpoint/configuration selection and all three winner seeds. Research
proposals require explicit opt-in; no calibrated pair-acceptance policy exists.
No SOTA, deployment-risk or verified orbital accuracy claim, paid compute, human
review or automatic outreach is made.

## Localization diagnosis; data gate failed before training - 31 October 2026

The [accepted localization study](LOCALIZATION_RESEARCH.md) is implemented
through its conditional stop: all released metadata, development geometry and
precision experiments, deterministic native refinement, separate pose-refit
ablation, replay tools and [diagnostic report](reports/LOCALIZATION_DIAGNOSTICS_V3.md).
No new learned compact-v3 model or final-test improvement is claimed.

All 76,802 NPZ cameras across 12,974 stereo geometries contain only intrinsics
and cam2world. TychoCrater is empty. Trajectory/batch IDs are absent. Retaining
each whole released mode/folder gives three already exposed groups, zero fresh
versus twelve required. This does not prove no unused terrain exists; the
requested conservative split cannot be certified. Training and final evaluation
are deferred as explicitly required by the accepted plan.

Development: eight geometries / four selected footprint components; 72 dependent
stereo/light pairs. Oracle recovers 8/8 poses. Rounding query truth to 16 native
pixels gives 0/8 with the original estimator. Coarse CNN gives 0/72 correct poses;
deterministic native refinement gives 7/72. Separate original-inlier LM refitting
raises these to 3/72 and 26/72. Native refinement plus LM yields 12/24 same-light
and 14/48 changing-light poses, but only 1/72 at the stricter 0.5-degree/0.1%-range
criterion. Precision and pose conditioning both matter; no learned contribution
is established.

Two-original-pixel precision among co-visible points improves 4.06% to 48.57%.
Both frontends return 22,697 matches; 14,938 reference points are co-visible.
No new calibrated acceptance, speed-superiority, modern pose ranking or real
orbital accuracy claim is supported.

Validation: 137 tests passed/four optional skipped, fresh 20-summary procedural
smoke with 100-row CSV replay. Independent replay checks 312 diagnostic rows /
15 aggregates, all 76,802 camera records, and 312 separate refit poses / 15 tables.
Original diagnostic and non-numerical import-cleanup replay are both retained;
all count outcomes agree. Earlier sources, weights and frozen protocols remain
unchanged. No paid compute, human review or email sending occurred.

## Compact south-pole study completed - 31 October 2026

The bounded [compact study](COMPACT_RESEARCH.md) has completed all planned local trials, all40 final baseline shards, saved-metric replay and the research package. Distribution version: `compact-v2`. The global research objective in issue #2 remains open: these results do not establish global SOTA or verified real navigation accuracy.

The [measured report](reports/RESEARCH_ITERATION_6.md), [interpretation](reports/COMPACT_INTERPRETATION.md), [two-page brief](reports/compact_brief_v2.pdf) and downloadable release assets contain the final evidence. Public freeze commit: `392cdf1020c925cec61d3340a7720e26cc6cdce7`; protocol SHA256: `cf97903e715fbb3f2596aa06397235bb52739cae3e1f8c93aa11893499e5ab83`. Models, policies, matching/scoring sources and case definitions were not adjusted after test.

Primary model: **376/384 correct acceptances, zero incorrect outputs, 0/288 nonoverlap acceptances**. Other predefined seeds:351/384 (zero wrong),380/384 (one nonoverlap accepted). Old checkpoint with new confidence:368/384,zero wrong; old frozen gate:100/384. Both old and primary checkpoints have384/384 raw correct proposals. All284 correct old-gate rejections fail only its global-correlation condition. The primary gain over the old gate is+71.88points (exploratory grouped95% interval+68.63 to+75.00); retraining adds only+2.08points over the recalibrated old checkpoint and is not consistently better across seeds.

On the full common comparison, compact retains35/36, LoMa-B27/36 with one wrong acceptance, LoMa-R27/36 with none, RoMa-v2 fast51231/36 with none, original RoMa31/36 with one wrong, and MatchAnything32/36 with none. All methods reject every common nonoverlap pair. Compact-minus-MatchAnything accuracy gain is+8.33points with interval−2.78 to+19.44: superiority is inconclusive. Median paired speedup is101.2×; the paired lower accuracy bound meets the declared−3-point margin. The predeclared acceptance and accuracy–speed point targets are met within this specific coarse rendered benchmark.

3D remains a negative result: compact meets the secondary pose criterion in2/288 cases (142 poses estimated); original CNN meets2/288, SIFT7/288 with much lower correspondence/pose coverage. New POLAR configurations give23/40 compact nominally correct acceptances and one outside criterion, versus8/40 and zero outside for the old frozen gate, or22/40 and zero outside for the recalibrated old checkpoint. Unknown reference drift/calibration uncertainty remains; real orbital accuracy is unverified.

Data:28 selected spatial components split8/4/4/12;12 final components across three polar sectors. Training uses18 geometries/54 lighting images. Trajectory IDs and Sun angles are absent; spatial separation is not verified trajectory separation. That shortfall was documented after training, before final inference. All66 cached geometries have source-identical depths across lighting variants. Inference128px and tolerance3 input pixels equals12 original512px pixels. Calibration has only12 positives/12 negatives across four components; no1% risk guarantee follows.

Four2000-step ablations and all three predefined selected-configuration seeds completed. Expanded architecture control won the declared raw-success development criterion (93/96 versus92/96,89/96,87/96). Added complexity did not improve that criterion under this budget. Corrected-coordinate actual common pretrained controls passed for all newer methods; LoMa-B/R author profiles passed. RoMa-v2 precise author control hit600seconds (exit124); no precise-profile lunar result is inferred. CPU models/architectures vary, and every speed comparison uses its same-job compact control.

Verification:128 tests passed/four optional skipped; fresh20-summary procedural smoke; public-freeze CLI exact(8,-4) and blank rejection/null/empty-correspondence controls. Independent replay validates all saved2D/3D/sensor records,13 confidence policies and14 aggregate tables. Additional checks reproduce10 interval calculations and360 same-host control geometries/CPU medians. The two-page PDF and final figures were visually inspected. Fresh asset installation verifies98 data files and seven checkpoints.

Models/data/experiment/report assets are checksummed. The experiment capsule contains2,293 files (134,833,819bytes). All40 original final shards completed successfully; the two redundant ARM recovery jobs were cancelled after the original results finished and their partial artifacts retained. No recovery predictions were selected. No paid compute, human review or automatic email was used. Earlier measurements below remain historical.

## Validation hardening — 30 October 2026

The [reserve protocol](reports/reserve_protocol_v1.json) was published at commit `b7f2e0fe24be08dac5c0e1201e9451b6a625d59f` before either reserve was opened. SHA256: `6394d81f9714f4ef18a0eae61df761ef4b38d9c0d03e680399a1400cdbeeec20`. It binds the selected standalone CNN, strict standalone development gate, source/model bytes,64 patches,12 footprint/acquisition components,9,216 positive cases,1,728 negative cases and all four geometric families. No threshold fitting or model selection occurs on reserves. Both reserves are consumed; final evaluation/audit results are recorded in [iteration 5](reports/RESEARCH_ITERATION_5.md). The full 13-predictor ensemble is not part of this primary reserve protocol and remains development-only.

**Completed and audited:** all 10,944 cases × three methods = 32,832 outcomes passed native-pixel, analytic-geometry, frozen-gate and public-output replay. Raw CNN: 8,414/9,216 correct (91.30%), 306 wrong positive fits and 1,625/1,728 raw nonoverlap proposals. Frozen gated CNN: **4,603/9,216 correctly accepted (49.95%), zero observed incorrect outputs, zero accepted nonoverlap pairs**. It discards 3,811 otherwise correct proposals. The exploratory component-bootstrap positive-rate interval is 45.12–53.77%; this is within Tycho, not global independence or a deployment guarantee. Predeclared empirical targets were met. Both reserves stay consumed. Validation: 112 research tests passed/4 optional tests skipped; 107 minimal-core tests passed/5 skipped; fresh 20-summary smoke and actual trained-model CLI controls passed.

The deployment CLI now returns no public transform when the gate rejects. A procedural actual-model CLI control recovered the exact(8,-4) shift; a different-real-scene control returned `null`, with raw proposals hidden. A separate ensemble export enforces its existing frozen development rule, retaining1,069 public outputs and no transform on its rejected proposals; that export is not fresh validation.

Zero-shot real-photo controls completed on 42 acquisitions in the three earlier CNSFM scenes: raw CNN336/336 exact-warps correct; gated334/336,0/84 different-scene acceptances; SIFT303/336,0/84 negatives. Every input/error and the actual reverse-inference gate replay passed audit. The321 real repeat-observation cases retain unknown accuracy labels;13/45 S1,19/45 S2 and45/231 S3 are accepted, but these counts cannot establish correctness.

An external-reference experiment downloaded selected original NASA Ames POLAR sensor images, per-terrain calibration and laser clouds for four laboratory terrains. All24 cases/72method outcomes completed. The strict gate rejects all 8 fixed-camera relighting pairs, although4 raw fits agree within 3 px with nominal fixed-camera geometry. Nominal laser-reference inlier agreement and its sampling/visibility/unknown calibration uncertainty are recorded separately. This is an important negative transfer result, not lunar orbital accuracy validation.

The sections below retain historical development records. Statements that reserves were untouched or experiments were running describe the earlier stage, not the current status above.

## Development iteration record — before reserve evaluation

The [fourth iteration](reports/RESEARCH_ITERATION_4.md) trains an original 28,272-parameter dense structural descriptor on 384 training patches and 12,344 eligible physical-light pairs. Completed common development matching: initial 334/1,152 correct, no-motion training 994, motion-augmented training **1,072 (93.06%)**; all 4,896 outcomes and both full training input caches passed audits. Raw non-overlap fits are 69/72. Audited standalone image-evidence filtering retains **981 correct/3 wrong** at the declared ≤1% development budget, or 549/zero at the strict setting; both reject all 72 negatives. These are development-fitted policies, not held-out guarantees. The stronger-model-with/without-CNN pool comparison continues. Original RoMa passed its actual pretrained CPU proof (.04926 px error), and the complete 32-shard blind development run is executing. **108 tests** and a fresh 20-summary smoke pass. Both reserves remain untouched; current-SOTA superiority and real navigation accuracy remain unverified.

The [third iteration](reports/RESEARCH_ITERATION_3.md) records the controlled legacy 8/modern 4/expanded 12 comparison design, common full feature support and architecture-family voting. Full image replay and equal-grid confidence calibration still await terminal completion. That iteration prepared the dense baseline; the fourth iteration records its now-passing software proof and running full comparison. No dense lunar accuracy or fresh geographic gain is claimed.

The objective in [RESEARCH_ITERATIONS.md](RESEARCH_ITERATIONS.md) remains incomplete. LoFTR completed all 576 original Sun pairs (271 correct), with exact common pixels and independent error audit. The eight-patch development geometry comparison has now completed: the image-selected hybrid is correct on 975/1,152 positives (84.64%), versus LoFTR 704 (61.11%), structural similarity search 824, DISK/LightGlue 624 and the expanded trained adapter 616. The adapter's descriptor improvement did not translate to better aggregate registration. Hybrid success by translation/rotation/scale/combined family is 271/240/253/211 out of 288 each. Raw hybrid output fits 40/72 non-overlap negatives; confidence filtering is necessary.

Development image-evidence policies retain 641 correct hybrid fits with zero wrong, or 907 correct with four wrong at the declared ≤1% empirical accepted-error budget. A conditional structural fallback raises these to 857/zero and 916/four respectively. All these policies reject all 72 development non-overlap negatives. Single predictors receive the same confidence grid; the cascade adds an explicitly disclosed conditional stage. These are development-fitted counts, not held-out risk guarantees. Every native input/error and all 3,672 hybrid choices were exhaustively replayed, including independent reverse selection; cascade decisions and equal-grid predictor calibration were also audited.

Original-author EfficientLoFTR native/2× CPU completed a two-patch comparison and audit: 142/182 correct of 288 positives; the hybrid reaches 242 on those exact same case IDs. Full eight-patch 2×/4× MPS EfficientLoFTR and author MatchAnything experiments are running. The separate structural-refinement probe completed and passed pixel/error audit: 242→246/288 positives, four gains and no losses, with the same 9/18 raw negative fits. This is a modest development gain (+1.39 points), not a solution to rejection or a fresh geographic result. Author source/configuration/weights are pinned, optional dependencies remain isolated, and actual native/2×/4× MatchAnything inference passed controls.

The frozen data design remains 384 train, 64 dev and two untouched 32-patch geographic reserves, excluding earlier evaluated acquisitions/buffered footprints from reserves. Expanded all-light adaptation trained on 11,727 eligible pairs. **No reserve outcomes exist.** All regions are within Tycho; only 2 train, 2 dev, 4 reserve-A and 8 reserve-B acquisition/footprint components exist, so repeated patch/light pairs are not independent evidence. Validation: **102 tests passed** including actual learned-model inference, and a fresh procedural smoke produced 20 summaries. See [iteration 2 report](reports/RESEARCH_ITERATION_2.md), completed-measurement capsule and audits. No real navigation accuracy or state-of-the-art superiority is established.

## Implemented and executed
- **M0–M2:** original benchmark contract, deterministic procedural checks and classical baselines.
- **M3 software:** checksummed acquisition of 15 real LROC NAC observations, 12 repeat pairs, 3 geographically separate scenes; native nodata masks; shared-map-grid loading; split/provenance validation; offline checkpoint review and validated import. All 12 pairs were visually inspected. Fresh downloads verified all source hashes.
- **M4:** pinned pretrained DISK + LightGlue, weight checksums, real forward-pass validation, masked features and shared RANSAC geometry. Controlled validation: 396 inliers and 0.0689 px median endpoint error.
- **M5 software/pilot:** frozen evaluation on all three regions; 126 evaluations (72 repeat-observation comparisons and 54 exact-transform controls), none/CLAHE ablation, failed-fit accounting, threshold sensitivity, CSV/JSON, three figures, failure examples and a two-page PDF report. Source/input hashes and grouped summaries verified.
- **M6 software:** clean-environment installation, 33 passing tests including pretrained inference, dependency consistency check, runnable reproduction command, offline interactive viewer, automated CI, code/model/data notices, citation and pilot release.

## Real-image scientific validation outside current scope
Independent point correspondences and their uncertainty are not supplied by the published dataset. The map reference is therefore provisional. M3 independent-reference acceptance, M5 verified navigation-accuracy conclusions and the corresponding research-grade M6 acceptance remain open in [issue #1](https://github.com/ThomasdeChillaz/lunar-image-matching/issues/1). No separate terrain-model experiment or LIMA reproduction is claimed. The released pilot implements the full software workflow; it does not claim completed independent scientific validation.

The 29 October scope amendment in MILESTONES.md removes human review as a delivery requirement. The original independent-reference criteria remain necessary for any future real navigation-accuracy claims, which this project does not make. No future background execution or automatic outreach is scheduled.

## Research extension in progress — 26 October 2026

The requested five-part extension is tracked in [RESEARCH_EXTENSION.md](RESEARCH_EXTENSION.md). All 42 upstream images and 321 pairs are now frozen in a separate manifest, with crop-local Sun direction sourced from archived LROC product metadata. Bidirectional matching diagnostics, conservative correctness labels, development-only policy calibration, source-bound manual-review tooling, range-based MoonAnything acquisition and an original CNSFM Windows execution workflow are implemented. The expanded evaluation is running; physical Hapke renders and metadata have downloaded with CRC/SHA provenance. Human landmark annotations and an independent second review have not been obtained. No new scientific acceptance or improvement claim is made.

## Verified progress — 29 October 2026

- **Expanded real experiment executed:** 321 pairs × three methods × two preprocessing settings = 1,926 bidirectional evaluations. Exact source snapshots, every saved map-reference metric and every geometry diagnostic were audited. Their reference uncertainty remains unknown.
- **Physical adaptation executed:** 48 train, 16 dev and 16 test patches, one source acquisition per patch, geographically separated Tycho bands. The 16,384-parameter projection trained on 358 co-visible training descriptor pairs and selected checkpoint 300 on 126 development pairs. The complete evaluation includes 1,152 method/case measurements.
- **Held-out physical measurements:** frozen DISK/LightGlue correct on 93/128 pairs; adapter 98/128, at the same three-native-pixel threshold. Default SIFT 1/128; CLAHE learned baseline 90/128. These are 128 px synthetic/rendered cases, not independently verified real navigation results.
- **Rejection experiment:** the frozen learned policy retains 17 correct held-out fits and accepts zero incorrect fits; the adapter’s policy retains 41 correct fits but accepts three incorrect fits. Both policies reject all 16 known non-overlap negatives. Zero-error development selection did not guarantee zero held-out errors.
- **CNSFM comparison executed and audited:** original binary completed all 321 common-grid pairs independently in both directions (642 executions, ten successful Windows shards). Original-output hashes, crater IDs, pixel/mask identity and exhaustive coverage were checked. Shared-geometry scoring returned fits for 45/45 S1, 42/45 S2 and 115/231 S3 pairs. Native probes also succeeded in all three scenes. Whole-image redetection is a documented variant; no paper-percentage reproduction or independently verified accuracy is claimed.
- **Additional real areas evaluated:** 54 real LRO appearance crops, 32 acquisitions, 27 metadata-selected repeat pairs across five Tycho areas; 108 frozen/adapted method measurements. No source acquisition occurs in the physical sample, and physical train/dev terrain is excluded. Real dev/test groups and footprints are separated. Exact input bytes and every saved geometric diagnostic were audited. Outcomes vary by area and remain provisional map diagnostics.
- **Real adaptation is not an established improvement:** the unchanged adapter also ran on all 321 original real pairs. It returns only 99/231 S3 alignments versus the frozen matcher's 158/231. Reduced map disagreement among returned fits does not compensate for this loss of coverage. The frozen matcher remains the default; the adapter remains experimental. Human checkpoints are required for true-error conclusions.
- **Reference review (superseded by scope amendment):** the user initially volunteered and subsequently declined all human review. No completed human annotation has been imported. No review is required or awaited. No synthetic or detector coordinates are treated as independent real checkpoints.
- **Evidence:** [measured report](reports/EXTENSION_REPORT.md), [saved-measurement audit](reports/extension_audit.json), [CNSFM execution artifact hashes](reports/cnsfm_windows_artifacts.json) and [real Tycho audit](reports/tycho_transfer_audit.json). These cover executed geometry/splits/calibration, not full scientific completion. Independent reference scoring, real-data gate calibration and verified real adaptation improvements remain open. Known-grid landmark conversion and saved-prediction scoring tools remain available for optional future validation.

## Physical lighting coverage follow-up

- The frozen anchor-light evaluation covers only 21.09–77.94° of Sun-direction separation. Its training anchor design reaches at most 80.72° before co-visible filtering; the downloaded images permit Sun pairs over 150° apart. The existing results therefore do not establish opposite-light robustness.
- The descriptive [lighting strata analysis](reports/physical_lighting_analysis.json) recomputes exact raw non-dark support and full 3D Sun separation. At 60–90°, frozen learned matching is correct on 16/38 cases and the adapter on 18/38. With less than 25% shared non-dark support, the respective counts are 0/6 and 1/6 at I/F>.01. These are known-geometry controls, not independently verified real-image accuracy. The old light-index/rotation coupling is disclosed.
- `evaluate_crosslighting.py` completed all 36 Sun combinations per selected tile with the same known translation (6,-4) and zero rotation: 576 pairs and 2,880 method evaluations. All exact input bytes, scores, supporting-point diagnostics and unchanged gate decisions passed [pixel/geometry audit](reports/crosslighting_audit.json). Sun separations span 13.52–155.64°, including 161 pairs at or above 90°. This is follow-up on previously inspected terrain, not a fresh geographic test.
- **Controlled follow-up outcomes:** NCC correct on 344/576 pairs; SIFT 27/576; frozen DISK/LightGlue 303/576; CLAHE learned baseline 305/576; adapter 323/576. The adapter-minus-frozen difference is +3.47 percentage points, with an exploratory footprint/acquisition-grouped interval of +1.48 to +5.23. These rendered, fixed-translation results do not establish real-image gains or global superiority.
- **Transported gate failure:** the original frozen learned policy retains only 5/303 correct fits and accepts 7 incorrect fits. Its selected translation limit is .05 of image size, while the exact shift is .05634. It is left unchanged and explicitly experimental; selecting a replacement from these outcomes would contaminate evaluation. See the [controlled lighting report](reports/CROSS_LIGHTING_REPORT.md).
- No independent human landmark JSON was obtained. No generated points are substituted for independent references; annotation is now outside current scope.
- The physical uncertainty analysis now joins overlapping declared footprints as well as shared acquisitions. One overlapping held-out tile pair reduces 16 acquisitions to 15 resampling components. The unchanged +3.91-point anchor result has a revised exploratory interval of −2.50 to +10.42 percentage points; the conclusion remains inconclusive. This is a reporting correction, not a change to saved matching outcomes.

## Current scope — no human review required

On 29 October 2026, the user declined both personal and third-party landmark review. The project proceeds as an automated benchmark. Known-geometry procedural controls and physical changing-light renders support accuracy/error measurements; real lunar repeat images support fit rates, cycle consistency, match coverage and provisional map-agreement diagnostics. Human annotation and real-data correctness calibration are outside current scope. No model-generated correspondence is substituted for independent ground truth. Existing experiments and their limitations are unchanged; the current workflow does not await a review file. The original full real-accuracy objective was not achieved and is not retrospectively marked achieved.
