# AI coding-agent execution brief

## Objective

Continue improving lunar camera localization under the full user-approved
research targets. Do not replace the goal with software checks or a small
exposed-development gain. Fresh final validation, reliable acceptance,
remaining required native baselines, transfer, uncertainty and the paper are
still required. No global SOTA or orbital-navigation accuracy is established.

Current live job: v13 surface-supervision cohort, source freeze19a4357.
Three primary2000-step runs are complete: raw54, normalized encoder55, and
normalized pair55 correct poses out of72. The selected pair configuration's
predefined repeats are still running. Monitor the existing process; never
restart it or mutate its pinned sources. Finish the full audit, actual-model
controls, report and capsule after all five runs complete.

Completed negative control: broad12px initialization adds15 wrong poses and
zero correct ones despite unchanged3px final fits. All72 final records and
60 broad branches replay. No broad branch recovers a correct missing pose;
genuine correspondence fractions are very low. Do not adopt this recovery
rule. See reports/POSE_RECOVERY_V14.md.

Next model hypothesis: small global pair-context proposal matching. The
cross-context module has25664 parameters,53936 with the frozen backbone;
optional later combination with separate learned fine weights totals89169.
Zero context optimizer steps have run. Initial native controls and exact72-case
point/pose equality to frozen54/72 pass. Capacity-controlled forward/reverse3D supervision and boundary/rotation/negative
controls now pass. The bounded training-pilot runner is implemented but not
executed. Publish the current source, run its100-step measured pilot, then
implement and freeze the full context cohort before full fitting. GLOBAL_CONTEXT_V15.md records the unfinished requirements.

Strong compact controls remain54/72 frozen surface and57/72 SIFT+surface with
two wrong raw poses. Native common LoMa-R/RoMa-v2 scores are42/53; same-worker
surface/RoMa timing is12.03x on fixed16 cases, not the SIFT or new context model.
The v7/v8 selected learned seeds45/42/42 and47/45/47 are completed history.
All current new fitting/scoring is exposed-development evidence from eight
selection geometries/four components. The fresh-data gate still fails.

The axial-depth correction in reports/GEOMETRY_CORRECTION_V5.md supersedes
earlier LunarGeo ray-based3D metrics/labels. Use audited v5 geometry/data and
preserve legacy sources/results. No calibration/test pixels, paid compute,
human review or automatic email.205 tests/four optional skips and a fresh
100-row software smoke pass at this milestone.

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
