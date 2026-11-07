# Native-window supervision results

**Complete predefined ablations and repeat seeds.**

All readouts are exposed development results:72 pairs from four terrain components. Frozen surface gives54/72 correct raw poses; the stronger same SIFT-first baseline gives57/72. Raw estimates are not calibrated accepted navigation poses.

| Variant | Seed | Selected update | Correct /72 | Strict /72 | Wrong raw | Same SIFT fallback | Total parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchored_window | 20261101 | 0 | 54 | 16 | 3 | 57 | 81696 |
| coarse_adaptation | 20261101 | 0 | 54 | 16 | 3 | 57 | 81696 |
| cross_adaptation | 20261101 | 0 | 54 | 16 | 3 | 57 | 82208 |
| window_only | 20261101 | 0 | 54 | 16 | 3 | 57 | 81696 |
| window_only | 20261102 | 0 | 54 | 16 | 3 | 57 | 81696 |
| window_only | 20261103 | 0 | 54 | 16 | 3 | 57 | 81696 |

Each primary variant receives2,000 updates and the same component/light/rotation/augmentation draws. The controls separate search-window labels, genuine-match anchors, coarse-encoder adaptation and cross-image context. The native fine encoder, reference-surface iterations and3px pose estimator stay frozen; coarse adaptation changes trainable capacity, disclosed separately. Initialization is eligible; the protocol selects on unfiltered pose count, then2px point count, then earlier update. Repeat seeds belong only to the selected primary configuration.

Intervals in the JSON describe paired terrain-component resampling after hypothesis, model and checkpoint selection on these same exposed cases. A lower endpoint at zero is inconclusive. Lighting variants are not independent terrain tests. A raw-pose gain does not establish1% accepted-error or nonoverlap risk.

The SIFT-first comparison uses available estimated-pose status only, never correctness. The separately measured historical CPU speed ratio cannot be assigned to a new model. No final-test, orbital-accuracy, novelty or SOTA claim follows from this study.

Audit: 2160 saved checkpoint rows and 12000 sampler/optimizer-update records; exact retrained updates: 0. All failed and selected outcomes are retained.
