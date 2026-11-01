# AI coding-agent execution brief

## Objective

Continue improving lunar camera localization under the full user-approved
research targets. Do not replace the goal with software checks or a small
exposed-development gain. Fresh final validation, reliable acceptance,
remaining required native baselines, transfer, uncertainty and the paper are
still required. No global SOTA or orbital-navigation accuracy is established.

Current fitting status: v16 is TERMINAL, all10,000 updates complete. Every
selected model retains initialization54/72; final primary checkpoints47/43/39
regress. All1,800 checkpoint rows and sampler records replay. Actual five-model
controls and exact primary72-case forward replay pass. Never restart that job
or edit its pinned M20 source. It did not produce a learned localization gain.

The measured diagnosis in reports/context_failure_diagnosis_v17.md finds target
likelihood improving while reachable proposals fall on training and selection.
All2,160 geometry records replay. Tested truth-only filters recover at most two
extra cases;252 oracle fits replay, but this is not model performance or a
universal upper bound. The next prepared core is search_window_learning_v18.py:
set labels matching native33px support, optional genuine baseline anchors,
and a trainable coarse CNN while the native fine CNN stays frozen.81,696 or
82,208 total parameters. Five numerical/gradient controls pass. NO v18 training
yet. Implement/freeze its full runner, actual72-case initial gate and measured
pilot before full fitting; do not stop at software readiness or a cost pilot.

Next immediate comparison: evaluate_native_matchanything_v17.py, native512,
all72 exposed pairs and same geometry/AP3P, versus SIFT-first surface. Actual
pretrained native controls passed. Publish and verify all sources before launch,
use two threads and no simultaneous research CPU jobs for paired timing. Save
/resume cases; no planar affine prefilter or silent profile substitution. Read
its new public commit for --source-commit. No full lunar comparison is claimed
yet. Afterward complete required MOONSt3R and author-profile comparisons.

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
