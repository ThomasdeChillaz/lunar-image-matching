# AI coding-agent execution brief

## Objective

Continue improving lunar camera localization under the full user-approved
research targets. Do not replace the goal with software checks or a small
exposed-development gain. Fresh final validation, reliable acceptance,
remaining required native baselines, transfer, uncertainty and the paper are
still required. No global SOTA or orbital-navigation accuracy is established.

Current new hypothesis is PREPARED, not fitted: dual-softmax reachable-window
mass couples row/column competition while leaving architecture/inference,
thresholds, geometry, margins and anchors unchanged. The actual v18 loss can
prefer a toy score change that reduces mutual proposals2→1; the added loss
penalizes it. This does not prove the lunar failure's cause. See
MUTUAL_ASSIGNMENT_V23.md, train_mutual_pilot_v23.py and train_mutual_cohort_v24.py.
100-update train-only pilot+independent exact replay must pass before6000full
updates/three seeds; all72exposed cases, initialization eligible.81,696total
parameters. No source was fitted yet.249tests/four skips and fresh100-row smoke
pass, including actual new-loss gradient/protection/resumption integration.
The full public source must be byte-verified before any fitting.

Current v18 fitting is TERMINAL12,000updates, all selected initialization54/72.
Six actual selected controls/exact72-case forward and a CLEAN public-checkout
replay now pass:2160rows,144nativeMatchAnythingrows and metadata proof. No
learned v18/scale upgrade is adopted. Capsules are packaged, not newly released.
Do not restart/alter any old fitting source or consumed data. Scale bank all
12/16, strict5/4/4, all48rows replay; no fullscale expansion.

The original MOONSt3R baseline is TERMINAL INCOMPLETE71/72 after a host clock/
suspension divergence. Calendar age41891s exceeded original14400s, though
completed-case inference total3239s. It was stopped using its own timeout;
original source/protocol/cases remain unchanged and no summary exists. Do not
resume/reset this expired run. The separate600-calendar-second recovery
contract fixes only its one missing case and all71existing hashes before
inference. See MOONST3R_RECOVERY_V28.md and
reports/moonst3r_recovery_contract_v28.json. Publish/verify code+contract first;
run recover_native_moonst3r_v28.py into a NEW amended directory, then audit.
Label any full result amended two-phase, never completed original protocol.
This resource amendment stays inside unchanged user72CPUwall-hour allowance.
Actual pretrained gate passed (3973matches,median0knownshiftpx), but no completed
lunar accuracy claim. No matching configuration or correctness-based row choice
is allowed in recovery. Model training overlap remains unknown.

All old queues50522/50750/51689 are TERMINAL;51689 ends ready_for_public_source_freeze.
No duplicate science job. First publish the new tested sources/contract, then
run bounded recovery+audit, then new100-update pilot+exact replay, and only if
gates/cost pass start the complete three-seed6000-update study. One two-thread
science CPU job at a time. CRESENT is metadata candidate only; fresh-data gate
remains failed. Full fresh final/acceptance/nonoverlap/transfer/remaining modern
profiles/uncertainty/paper/compact-v3 CLI requirements remain unmet. No email.

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
