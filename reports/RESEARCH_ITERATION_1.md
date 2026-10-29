# Research iteration 1: LoFTR and structural development hypotheses

## Same-input LoFTR baseline

Unmodified pretrained outdoor LoFTR was evaluated on all 576 original fixed-translation physical Sun pairs. Every reference/query/mask byte matches the earlier methods; native resolution is 128px, known translation is (6,-4), shared partial-affine RANSAC uses 3px and six inliers, and correctness is median valid-point endpoint error <=3px. Missing fits and dark inputs stay in denominators. No resizing, lunar fine-tuning, confidence-threshold tuning or test-based rejection policy was added. Source and weight hashes are pinned separately from the original model lock.

| Method | Correct / 576 | Success |
|---|---:|---:|
| NCC | 344 | 59.72% |
| Original lunar descriptor adapter | 323 | 56.08% |
| DISK + LightGlue + CLAHE | 305 | 52.95% |
| DISK + LightGlue | 303 | 52.60% |
| LoFTR outdoor | 271 | 47.05% |
| Default SIFT | 27 | 4.69% |

LoFTR returned 384 fits; median endpoint error among returned fits was 1.223px. Median forward CPU inference was about 584ms in this run, excluding initialization/warmup. This is not a controlled cross-method speed ranking. The original terrain was already inspected; this is a baseline addition, not a fresh geographic test or a reproduction of the JPL lunar-trained model.

Evidence: [LoFTR outcomes](loftr_crosslighting_results.csv), [summary](loftr_crosslighting_summary.json), [independent pixel/geometry audit](loftr_crosslighting_audit.json). The verifier reconstructs the translation by array slicing, checks native pixel/mask identity against all original methods, and independently recomputes saved errors and failure labels.

## Structural hypotheses: development only

Eight predetermined development patches, all 36 Sun combinations, four separate seeded motion families, and 72 disjoint-terrain negatives were evaluated: 1,224 cases per method, 4,896 method outcomes. Structural representations are original small experiments inspired by medical/multimodal registration, not reproductions of the cited methods.

| Geometry | Cases | NCC correct | SIFT correct | Self-similarity correct | Unoriented gradient correct |
|---|---:|---:|---:|---:|---:|
| Translation | 288 | 179 | 25 | 262 | 227 |
| Rotation | 288 | 24 | 28 | 39 | 24 |
| Scale | 288 | 35 | 15 | 62 | 38 |
| Combined | 288 | 4 | 16 | 10 | 6 |

NCC and both structural methods currently estimate **translation only**. Rotation/scale rows are explicitly out-of-model stress tests. Their low success motivates the next geometry-search hypothesis; these rows cannot support a fair claim that their geometric model is better than SIFT. Self-similarity returned no alignments on 72 known non-overlap pairs, versus 24 raw NCC fits; this development result is not a false-acceptance guarantee.

Structural methods' support coordinates are transformed template probes, not independently detected feature correspondences. Their coverage describes the template footprint; their transform residual is zero by construction and is not accuracy or confidence evidence. The new rejection gate does not use this residual. Scores, peak separation and independently estimated reverse cycles remain measurable diagnostic hypotheses.

The first new rejection policy uses coverage, reverse-cycle error, support and optional correlation/peak margin, without the old maximum-displacement cutoff. With a zero accepted-development-error objective, it retains 170/373 correct self-similarity fits, 131/295 gradient fits, 75/84 SIFT fits and no NCC fits. These counts pool the separately labelled geometry families for policy development only. The NCC result reveals insufficient discrimination under the strict constraint; rejecting everything is not a successful useful policy. The small number of development geographic components limits confidence.

Evidence: [input/analytic-geometry audit](structural_dev_audit.json), [development outcomes](structural_dev_results.csv), [separate-family summaries](structural_dev_summary.json), [frozen empirical gates](structural_dev_gate.json). No geographic reserve has been evaluated, and no state-of-the-art improvement is claimed.

## Reproduce or audit this iteration

The [measurement capsule](research_measurements_v1.zip) ([SHA256](research_measurements_v1.sha256)) includes complete saved outcomes and frozen plans for the original common Sun-pair experiment, added LoFTR and initial structural screen. It excludes downloaded source rasters, pretrained weights and the large training descriptor cache. The [historical source snapshot](development_source_snapshot_v1.json) preserves exact inference source bytes for the initial development screen; audits do not depend on local-only commit IDs. Acquire pinned data/weights with the repository tools, then extract into `runs/`:

```sh
python fetch_moonanything.py
python moon_dataset.py --output runs/moon-manifest-rebuilt.json
python research_dataset.py --output runs/research-manifest-rebuilt.json
python -m zipfile -e reports/research_measurements_v1.zip runs
python verify_loftr.py runs/loftr-crosslighting-v1 --output runs/loftr-recheck.json
python verify_research_dev.py runs/structural-dev-v1 --output runs/structural-recheck.json
```

Expanded training can be reproduced with `python train_research_adapter.py --output runs/research-adapter-new`, followed by `python verify_research_training.py runs/research-adapter-new --output runs/training-recheck.json`. Reproduction of descriptors requires the pinned source rasters and weights; their absence is an explicit dependency, not substituted data.

## Remaining work

Expanded all-light-pair adaptation completed on 384 patches with 64 development patches: 11,727 eligible training pairs, 256 development descriptor pairs and selected checkpoint 800. Selected development descriptor loss is 2.80655 versus identity 3.51365; this is a surrogate, not registration success. See [training audit](research_training_audit.json) and [candidate adapter](research_adapter_v1.json). Structural geometry search and the learned-candidate registration comparisons are running on development only; two 32-patch fresh geographic reserves are frozen and unused. Remaining: finish development comparisons and diagnostic combinations, then freeze models/policies/case plans before a fresh reserve evaluation. Add stronger current specialist/dense baselines before any state-of-the-art claim. See [research protocol](../RESEARCH_ITERATIONS.md).
