# Milestones

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
