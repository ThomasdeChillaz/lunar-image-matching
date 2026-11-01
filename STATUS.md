# Status

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
