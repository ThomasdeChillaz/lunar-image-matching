# Geometry-normalized supervision: complete development results

Five runs completed 10000 optimizer steps. The predefined selected configuration is `normalized_pair`; its three seed results are **55/55/55 correct raw poses out of72**, versus **54/72** for frozen surface refinement. The unchanged SIFT-first baseline gives **57/72**.

| Experiment | Seed | Selected step | Correct /72 | Strict /72 | Wrong raw poses | Same SIFT fallback | Parameters, total |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_encoder | 20261101 | 2000 | 54 | 18 | 3 | 57 | 56544 |
| normalized_encoder | 20261101 | 1500 | 55 | 17 | 2 | 58 | 56544 |
| normalized_pair | 20261101 | 2000 | 55 | 18 | 2 | 58 | 63505 |
| normalized_pair | 20261102 | 2000 | 55 | 17 | 2 | 58 | 63505 |
| normalized_pair | 20261103 | 2000 | 55 | 20 | 2 | 58 | 63505 |

Each ablation receives2000 updates, eight visible examples per batch and the same learning rate, component/point/light/jitter sampling and sensor perturbation draws. Common support contains10340 exact visible integer-reference points across18 training geometries/12 components. Query truth constructs training normalization/visibility and scoring only; inference receives reference depth and an image-estimated query pose. Darkness does not create occlusion labels. Raw and normalized training differ in geometry, resampling and effective search extent.

Checkpoint selection uses all72 exposed model-selection pairs, then2px-correct point count, then earlier step. Initialization is eligible. This new selection rule was frozen before fitting; earlier16-case results are preserved. No seed is substituted for a better-looking headline. The no-match head is absent from this precision-only ablation; reliable rejection remains unfinished.

For each run, the JSON retains strict/primary counts, native correspondence support, four-component intervals, per-component results and all gained/lost cases. A paired interval containing zero remains inconclusive. These are four exposed components, not72 independent terrains. SIFT-first routing uses estimated-pose availability only, never correctness. The speed ratio measured for the frozen v11 model cannot be assigned to these new networks.

Verification:193 repository tests/four optional skips, a corrected fresh100-row software smoke, actual initial equivalence on every historical case, independent train projection/visibility labels, all checkpoint native metrics and final pose-estimator replay, selected/final/optimizer/source hashes, exact sampler RNG replay and actual selected-weight quarter-turn/blank controls. This is saved-study reproducibility, not independent retraining or geographic validation.

Earlier zero-step preparation with the feature-batch bug remains preserved. All failed checkpoints and seed outcomes are retained. Fresh-final, low-error calibrated acceptance/nonoverlap, required remaining native baselines, transfer and orbital navigation targets remain unmet. No paid compute, human review or automatic email.
