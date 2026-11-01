# AI coding-agent execution brief

## Objective

Continue improving lunar camera localization under the full user-approved
research targets. Do not replace the goal with software checks or a small
exposed-development gain. Fresh final validation, reliable acceptance,
remaining required native baselines, transfer, uncertainty and the paper are
still required. No global SOTA or orbital-navigation accuracy is established.

Current fitting: train_global_context_v16.py is LIVE from public freeze
b2d749f639385f577da4a8ccbab6ba1f466238c0. Preserve its source bytes and existing
run runs/localization-v16-global-context-development. Never restart completed
runs. Use recorded live process/session evidence, not stale historical handles.
The first pointwise2,000-update ablation completed; its48/49/44/47 readouts lose
to54/72 initialization, so step0 was retained. All360 rows and sampler/optimizer
budget independently replay. Self/cross context and predefined repeats remain
in progress/pending; no complete-cohort or context model gain claim yet.

After all five runs: verify_global_cohort_v16.py audits the full saved cohort.
Optional --retrain separately records exact retrained update counts. Do not
run another scientific fitting job concurrently. report_global_cohort_v16.py
reports all configurations/seeds and the same SIFT-first fallback, never a
best-seed headline. The current partial figure was executed and inspected.
Do not change the frozen inference kernel, pose criterion or selection rule.

The v13 surface cohort is terminal:55/55/55 raw correct, same SIFT fallback
58/58/58 versus57, one wrong raw estimate. The1.39pp gain is inconclusive on
four exposed components. Release surface-supervision-development-v13 and all
four public assets are byte/CRC verified; compact-v2 remains Latest. Current
210 tests/four optional skips and fresh100-row smoke pass. The100-update
train-only context cost pilot exactly retrains; it is not a localization result.

Latest/version metadata still show the same MoonAnything sample; no new fresh
data established. The gate remains failed and LuSNAR7-9 reserve pixels are
uninspected. Fresh evaluation, calibrated acceptance, remaining required native
baselines, transfer, timing and paper remain necessary. No global SOTA or
orbital accuracy claim. No calibration/test pixels, paid compute, review or
email. The full research objective remains active and unmet.

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
