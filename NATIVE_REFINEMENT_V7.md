# Native correspondence precision experiment (development only)

## Completed native study and stronger engineering controls

The complete v7 cohort gives45/42/42 across the selected pair-refinement seeds
versus45/72 frozen. All10000 optimizer steps,360 candidate rows, sampling/RNG,
source/checkpoint and234 supervision cases pass replay; every selected model
passes analytic actual-inference controls. See reports/NATIVE_TRAINING_V7.md.
The100-step47/72 pilot is not substituted for this negative result.

An affine viewpoint correction gives49/72, but retains only53/72 with the same
SIFT-first fallback and adds wrong proposals. A subsequent explicit3D surface
normalization control gives54/72 (three wrong), or57/72 with SIFT (two wrong).
Both use the frozen28,272-parameter model. Strict fallback poses27→25 and the
four-component descriptive gain intervals include zero. These are engineering
controls, not learned novelty, calibrated acceptance or a new final test.
No query depth or true pose is provided to their inference interfaces.
All geometry/native-error/common-estimator replays and actual model controls pass.

```bash
python verify_native_training_v7.py --run runs/localization-v7-native-development --output runs/new-native-audit.json
python evaluate_geometry_warp_v11.py --full --output runs/new-surface-control
python verify_geometry_warp_v11.py --run runs/new-surface-control --output runs/new-surface-audit.json
```

The full v8 coverage experiment is still running and remains a distinct
controlled data ablation. Native modern jobs each stopped after64/72; partial
artifacts are verified without claiming completion. Exact36+36 assembly and
all earlier research targets remain outstanding. Earlier sections retain the
historical protocol and execution snapshots.


The corrected coarse-learning study did not establish a consistent pose gain. The next hypothesis targets the precision bottleneck directly: train the native-resolution local encoder and pair-conditioned scoring, keeping coarse proposals, four-orientation search, reference geometry and pose fitting fixed.

## Software and current pilot evidence

The new initial native inference reproduces every selected orientation and pose-success decision of the frozen baseline on all72 exposed development cases:45 correct,17 stricter poses. Maximum coordinate difference is0.0000611 native pixels, from floating-point arithmetic. This is an implementation-equivalence control, not learning.

A separate100-step joint no-match pilot selects step100 by the fixed16-case bank (11 poses, then3079 two-pixel-correct matches versus3067 initially). Its expanded development readout gives47/72 correct raw poses,16 strict poses and55 estimates (eight wrong). The frozen baseline gives45 correct,17 strict and57 estimates (12 wrong). This small pilot is not a consistent learned improvement, correctly accepted coverage, a fresh test or SOTA evidence. It does not meet the full research targets.

Independent replay checks all234 cached supervision cases,72 candidate outcomes, seeded initial and trained weights, source/data/checkpoint hashes, actual100 optimizer steps, exact50/50 sampling and final sampler RNG state. A native known-pose/blank actual-model control is separate. Do not use the broad72-case readout for checkpoint/configuration selection.

## Fixed full experiment

All variants use corrected axial-depth labels and the same existing native-patch sampler. It draws components uniformly with rejection to enforce half in-window geometric positives and half invalid proposals, incorrect locations or nonoverlapping terrain. Proposal coordinates determine where to sample; exact camera/depth projection determines truth. Brightness does not determine visibility. Full depth validity and crop bounds were checked before sampling.

1. Encoder3D: train the native descriptor encoder with exact local correspondence targets.
2. Pair refinement: add the small two-image convolutional scoring head.
3. Explicit no-match: add a joint1090-class categorical outcome (33x33 locations plus one no-match bin), training invalid local windows as no-match.

The local target distributes mass bilinearly at the exact simulated coordinate. Peak decoding refines the strongest mode locally. The no-match head predicts odds relative to total local-match probability; inference rejects a point when that bin beats every local peak. This is point rejection, not a calibrated pair-acceptance policy. Its initial bias of-7.5 mathematically preserves all initial proposals, so an untrained reject-all head cannot look like a useful result.

AdamW, learning rate0.0001, weight decay0.0001, batch8,2000 steps per run, selection every500 steps including step0. Run all three variants with seed20261101, then the bank-selected configuration with seeds20261102/03. Select by unfiltered pose count, then2px-correct correspondence count, then earliest step; retain every failure and seed. Combined frozen-coarse plus native model sizes stay below100000 parameters. Exposure, blur, noise and quantization supplement actual rendered lighting. No large model is trained from scratch.

```bash
python train_native_refinement_v7.py --output runs/new-native-initial --initial-control
python train_native_refinement_v7.py --output runs/new-native-pilot --steps 100 --interval 100 --pilot
python verify_native_training_v7.py --run runs/new-native-pilot --output runs/new-native-pilot-audit.json
python train_native_refinement_v7.py --output runs/new-native-full --steps 2000 --interval 500
```

Use the validated environment and two CPU threads. Preserve output directories and live source snapshots. No calibration/test pixels are loaded. The original fresh-final data gate remains unmet: these are18 training/eight selection geometries,12/four corrected spatial components, with unknown trajectory/batch and shared-renderer dependence.

## Modern native pose comparison

The new native baseline workflow accepts original512px source images and reports internal model resizing separately. Pinned LoMa-R and RoMa v2 must pass actual native512px pretrained known-transform and invalid-input controls before lunar inference. Every model receives the same reference geometry and3px AP3P estimator. Query depth/true pose enter only scoring. Each modern run includes an alternating, warmed compact control on identical cases and CPU allocation. Peak process memory includes both loaded models and is not an isolated per-method memory ranking.

The local validated runtime lacks supported LoMa dependencies; its failed attempt is preserved as an environment limitation, not a lunar result. Use the isolated supported Linux workflow, without changing the running training environment. A timed-out or incomplete baseline remains incomplete; no other model/configuration is substituted. Start with the common profile; author-profile execution is a separately declared run. These are development comparisons, not a new final ranking or global SOTA claim.

```bash
python native_baseline_pose_v7.py --method loma_r --source upstream --profile common --output runs/new-native-loma-r
python verify_native_baseline_v7.py --run runs/new-native-loma-r --output runs/new-native-loma-r-audit.json
```

The research goal remains the original one: a reproducible, sufficiently large and consistent camera-localization improvement with reliable acceptance, fresh validation/transfer, fair strong baselines and a defensible paper. No pilot gain, software pass or geometry correction completes it. No paid compute, human review, automatic email or orbital-navigation claim.

## Subsequent development evidence

The v6 orientation study has completed without a gain. The v7 full native
encoder and pair-refinement primary runs give44/72 and45/72 so far; the study
and predefined repeats are incomplete. Do not prefer the47/72 short pilot as
if it were a completed selected model. Its stricter score was16/72 versus17
frozen, showing a tradeoff rather than uniform dominance.

A source-coverage audit found only3 and23 correct local coarse proposals in
two difficult training components. A separate sampling ablation replaces
positive slots with10463 uniform projected points over the12 training groups,
using exact-query-centered jitter, while retaining v7 negatives, model, loss,
optimizer and sensor perturbations. Its100-step pilot gives45/72 raw correct,
18 strict and11 wrong estimates: no raw-pose gain. This changes positive
coverage/centering, not just adds copies of successful coarse matches. Full
uniform-data training has not been executed; its results are not promised.

Corrected exact-in-window diagnostics recover67/72 poses with oracle projected
correspondences. All visible returned reference points give72/72. The replay
checks projection, camera range, all quarter-turn/window masks and re-executes
the same pose estimator. This is diagnostic query-truth use, not deployable
inference or a formal pose-performance bound. The frozen branch oracle itself
recovers only45/72, all already selected by the existing geometry rule.

Native model throughput is slow enough to threaten the initial one-hour jobs.
The fixed execution recovery assigns cases0–35 to the original run and36–71
to separate bounded-range workers, preserving all72 eligible pairs. No model,
weights, numerical profile, coordinate conversion or pose estimator changes.
The range runner imports the original control/scoring functions. Complete
comparison and independent replay remain mandatory. See
reports/native_case_range_plan_v8.json. Do not infer a global ranking from
partial logs. All attempts, failures and duplicate outputs are retained.

## Full uniform-positive experiment launched

The full v8 cohort now uses2000 optimizer steps per ablation/seed, the same
500-step selection interval and the same initial-checkpoint eligibility.
The three primary variants and two bank-selected repeats require10000 steps.
The100-step pilot was only a software/early-training control, with no raw
pose gain. No calibration/test pixels or new final-test claims are introduced.
Source was public at e408c3f before launch. Both local cohorts use two threads;
concurrent training elapsed times are not an isolated inference benchmark.

The second full v7 pair-refinement seed gives42/72; the third is pending.
Use verify_native_training_v7.py, then report_native_training_v7.py once the
cohort is complete. The new fixed-range modern combiner requires the exact
assigned first36 original and last36 range cases, including paired compact
controls. Its negative tests prevent cherry-picking better duplicate outputs.
All partial and duplicate original outputs remain preserved.

The additional no-match diagnostic in reports/native_no_match_generalization_v9.json
uses fixed unaugmented cached proposals in both training and selection groups.
Its validity AUC is below chance on both samples, and all sampled points are
rejected. This is not the exact augmented sampler distribution or an
independent test. It rules out describing the failure solely as novel-terrain
transfer or merely an unfortunate categorical rejection threshold. The full
uniform-positive run is the declared data-coverage test; do not promise it will
repair the head.
