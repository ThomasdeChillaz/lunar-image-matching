# AI coding-agent execution brief

## Objective

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
