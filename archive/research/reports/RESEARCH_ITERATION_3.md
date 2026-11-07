# Research iteration 3: stronger controls before fresh geography

This iteration is in progress. No geographic reserve has been evaluated. Completed measurements remain in [iteration 2](RESEARCH_ITERATION_2.md); this document records new designs and software validation, not completed performance improvements.

## Controlled prediction pools

`evaluate_modern_blend_dev.py` streams the same 1,224 development cases from five source experiments. A case is processed only after all twelve predictors have produced an outcome. It compares:

- **Legacy:** the earlier eight classical, structural and learned predictions.
- **Modern:** original EfficientLoFTR and author MatchAnything, each at internal 2×/4× resolution.
- **Expanded:** all twelve predictions.

Every pool receives the same post-alignment image features, including the complete radius 7 orientation-feature support mask. Global, balanced and consensus selectors are evaluated separately: nine declared methods, 11,016 planned outcomes. This controls the support-mask change rather than attributing it to the larger candidate pool. Pool-specific agreement is recomputed after excluding other pools' candidates. All four ELoFTR/MatchAnything variants share one architecture-family vote; resolutions and pretraining variants do not create four nominally independent votes.

The selector accepts only typed prediction identities/families/matrices and native images/masks. Known motion, correctness, endpoint error and geometry-family labels are excluded. Reported selection time covers all pools and both directions, excluding cached predictor inference; no speed improvement is claimed. Selected outputs use image-region support, not invented independent point correspondences.

The source plans and native input hashes must match exactly. Completion requires every planned outcome and all upstream terminal hashes; consumed append-log records must also equal final upstream JSON measurements. This catches a changed log even if its final JSON checksum remains valid. Missing predictors are explicit errors or incomplete work, never silent substitutions.

The full decision replay (`verify_modern_blend_dev.py`) and identical-grid development confidence calibration are waiting on terminal completion. Neither schedules a reserve experiment. Source versions used by active runs remain unchanged.

## Dense baseline in a separate runtime

RoMa is an important dense comparison beyond the semi-dense models. Its current original-author release requires PyTorch>=2.5.1, outside the validated local 2.2.2 environment. The dedicated `.github/workflows/dense-matcher-probe.yml` uses standard Linux CI and a separate pinned PyTorch 2.8.0/torchvision 0.23.0 CPU environment. It first executes a genuine pretrained known-transform test and explicit blank/invalid-input controls. No lunar result is claimed before that test passes.

The pinned original RoMa source is `77f8d68803526dcddfd9b7a46bc76125bdc25f15`. The outdoor weights and DINOv2 ViT-L/14 weights were retrieved from the official author/Meta URLs and their actual SHA256 values frozen before inference in `roma_model_lock.json`. Downloaded source/weights remain outside Git. RoMa is MIT licensed; official DINOv2 code and these model weights are Apache-2.0.

The [development input package](dense_dev_inputs_v1.zip) ([checksum](dense_dev_inputs_v1.sha256)) contains exact native reference/query/mask arrays for all 1,224 planned cases. Every input hash was checked against the completed audited common benchmark before export. It contains only development inputs and public CC BY 4.0 attribution; transform truth, evaluation points, correctness labels, predictions and reserve images are excluded. Remote inference checks every native array's hash before each forward. Local scoring later joins the frozen geometry after all planned prediction shards exist; missing/duplicate shards, mixed workflow executions or changed implementations are errors. The original pretrained proof is also independently reconstructed locally before accepting the dense experiment.

The manually dispatched workflow supports `probe` or `full`. Full inference is divided into eight blind shards and starts only after the actual pretrained software-control job succeeds. Standard Linux CI is used, with read-only repository permission and no model-weight artifacts. No successful pretrained proof, completed blind shard or scored dense lunar result is claimed yet.

The declared CPU profile retains author 560px coarse/864px upsampled resolution, symmetric matching, 5000 balanced sampled correspondences and common partial-affine geometry. It uses the author's float32 feature option, native PyTorch local correlation, and the author's KDE `half=False` option during sampling for consistent CPU density arithmetic. This arithmetic profile is explicit; no vendor source is edited, and no GPU paper timing or pose-AUC reproduction is claimed. The author's edge-coordinate output is shifted by half a pixel to satisfy this benchmark's integer pixel-center convention; the roundtrip control passes.

Relevant stronger recent models remain part of the unfinished comparison scope: [RoMa v2](https://arxiv.org/html/2511.15706v3) uses DINOv3 under a custom license; [RoMa-Ω](https://arxiv.org/html/2609.09507v1) depends on a VGGT-Ω checkpoint whose official release requires access approval. Their source/runtime requirements were inspected, but neither has been executed here. The running semi-dense experiments and prepared original-RoMa integration cannot establish superiority against these models or current lunar-specialist methods.

## Software validation and next decisions

New tests check that the legacy pool cannot use new-model predictions, excluded candidates cannot influence legacy consensus, architecture variants cannot multiply votes, missing predictions cannot silently shrink an experiment, altered/partial append logs cannot satisfy final bindings, and dense normalized coordinates map to native pixel centers. Current local validation is **102 passing tests**, including the existing actual learned-model controls, plus a fresh 20-summary procedural smoke. This does not establish a successful pretrained RoMa forward; that is the separate Linux acceptance check.

Early prefixes of a running experiment are not final results. The stronger learned variants are competitive in completed cases, and a larger pool must demonstrate additional benefit rather than inherit the earlier improvement over weaker LoFTR. If expanded selection does not help, preserve that negative result and investigate confidence/image evidence using development data only.

Next: complete and audit the current common-plan comparisons; validate the dense matcher and add its actual lunar outcomes; test useful complementary hypotheses; then freeze method, policies, source/model bytes, complete comparison plan and meaningful improvement criteria before reserve A. All four motion families, failures, negatives and acquisition/footprint-grouped uncertainty remain required. State-of-the-art superiority remains unproven.
