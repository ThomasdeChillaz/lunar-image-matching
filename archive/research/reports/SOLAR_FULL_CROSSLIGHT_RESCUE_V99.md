# Cross-light correspondence pooling on the full development cohort

Pooling SIFT matches across lighting templates did not recover any additional pose. Both fresh arms recovered 114 of 192 positive pairs and returned zero poses on 192 nonoverlap pairs. The independent replay verified every paired outcome and all 768 runs. The direct arm reproduced all 384 raw outcome records from the earlier full direct benchmark.

| Outcome | Direct relighting | Cross-light rescue |
|---|---:|---:|
| Correct positive poses | 114/192 | 114/192 |
| Correct changed-light poses | 74/128 | 74/128 |
| Wrong positive poses | 18/192 | 18/192 |
| Positive pairs with no pose | 60/192 | 60/192 |
| Raw nonoverlap poses | 0/192 | 0/192 |
| Median online inference | 1.577 s | 1.576 s |

The paired raw gain was zero on every case. These cases use eight observed terrain groups and 24 catalogue lighting templates. The results precede calibration and share a simulator. Timing includes each online inference on the same two-thread CPU, separate from reference-library preparation.

## Failure diagnosis

| Stage | Positive cases | Nonoverlap cases |
|---|---:|---:|
| Direct pose already present | 132 | 0 |
| Original eighth slot retained | 0 | 2 |
| Pool contains fewer than six distinct matches | 55 | 75 |
| Pooled pose solve returns no pose | 5 | 115 |
| Pooled solve returns a pose | 0 | 0 |

Both arms retained the original nine-call pose budget. The pooling method removed duplicate and conflicting endpoints across lights before attempting its replacement pose solve. The missed positive pairs were primarily limited by correspondence supply. A solver reconfiguration alone would leave 55 of the 60 misses without enough matches for the current pose estimator.

The archived 8-bit query images still contain grayscale variation. Among the 39 positive cases with zero distinct pooled correspondences, the median image 95th-percentile grayscale code was 2 out of 255. In the 132 cases with a direct pose it was 72. The 39 cases use 25 distinct query files, and none is uniform. These image statistics describe rendered optical signal. They provide no ground-truth shadow labels or evidence of a prospective exposure correction. A blind, fixed normalization control can test whether usable correspondences can be recovered from that variation before changing the pose solver.

The full study took 1,419.69 seconds under its reserved 3,182-second bound. Calibration, unseen terrain and off-catalogue rescue performance remain open.

- [Frozen protocol](solar-full-crosslight-rescue-dispatch-v99.json)
- [Independent all-row replay](solar_full_crosslight_rescue_replay_v99.json)
- [Failure-stage audit](solar_crosslight_failure_audit_v100.json)
- [Saved query texture diagnostic](solar_crosslight_query_texture_v100.json)
- [Matched baseline and withheld-light comparison](SOLAR_OFFCATALOGUE_DIRECT_V95.md)
