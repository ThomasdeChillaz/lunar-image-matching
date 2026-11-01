# AI coding-agent execution brief

## Objective

Current next execution: v13 geometry-normalized-supervision cohort. Preserve
its pinned source and historical v7/v8/v11 results. Require exact v11 initial
pose-status equivalence before fitting; use all72 exposed selection cases.
Training-only normalization uses true query pose; inference never does.
See SURFACE_LEARNING_V13.md. No new learned outcome yet at this snapshot.

Current execution: v7 completed and independently replayed all10000 steps/five
runs, with no consistent learned gain (selected seeds45/42/42 versus45 frozen).
Deterministic reference-surface normalization gives54/72 and the fixed SIFT
fallback57/72 with two wrong raw poses. Its intervals include zero and it is
not calibrated acceptance or new learning. Use this stronger simple control
for future learning claims. Full v8 uniform-positive fitting/repeats completed: selected seeds47/45/47;
all five runs/10000 steps pass exact replay and actual model controls. No local
training job is live. The frozen source cohort remains preserved. Same-worker
surface/RoMa CPU run41262874858 completed and passes88-record/control/metric
replay:54/72 surface and12.03x paired speedup on fixed16 cases. This does not
apply to the57/72 SIFT variant. No process remains live; do not restart terminal
handles. Geometry-normalized supervision is the next learning hypothesis.
Native LoMa-R/RoMa-v2 common comparisons now complete and replay all72
assigned cases and paired controls:42/53 correct,28/19 wrong respectively.
Original compact45 is20.6x/16.8x faster in paired CPU jobs. The surface54 method
has no same-worker latency result yet; the new bounded workflow is prepared.
Native author-precise settings and EfficientLoFTR/MOONSt3R remain incomplete.
Fresh final, reliable acceptance, transfer, uncertainty and paper targets remain
unmet. Preserve all failed heads, seeds and sources.

The active user goal is to keep testing hypotheses and improving until the
original target results and/or defensible state-of-the-art results are verified.
The axial-depth correction in reports/GEOMETRY_CORRECTION_V5.md supersedes all
earlier LunarGeo ray-based3D metrics/labels. Use the explicit v5 geometry and
audited v5 data for new fitting/scoring; preserve legacy sources for replay.
The corrected v5 cohort is complete; see reports/COARSE_TRAINING_V5.md. The v6 orientation-regime cohort is complete without a learned gain. The v7 native-refinement cohort is live: do not mutate its pinned sources. Uniform-positive pilots and native modern comparisons are separate development evidence. The full
accuracy/acceptance/uncertainty/comparison goal remains active, not completed by
software checks or development gains.

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
