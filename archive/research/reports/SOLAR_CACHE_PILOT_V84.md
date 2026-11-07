# Cached inference: faster execution, unchanged recovery

Both fresh cache pilots complete all 40 rows and their independent saved-metric
replays. All 80 paired saved outputs match the original implementation exactly:
native correspondences, camera poses, Sun estimates, hypotheses, assignments,
measurement order and metrics. Only declared execution timings, cache counters
and version metadata differ.

| Mode | Original worker time | Cached worker time | Raw correct poses at every checkpoint | Full-study projection |
|---|---:|---:|---:|---:|
| Directional SH | 471.26 s | 308.96 s | 1/4 | 70,009 s |
| Pooled | 473.33 s | 310.11 s | 1/4 | 70,844 s |

These are observed complete **40-row worker workloads**, including input setup,
scoring and output overhead. Both use the same local CPU and two threads, but
the runs were not randomized/interleaved repetitions. They establish an
engineering saving on this workload, not a general model-speed claim or a
modern-model efficiency advantage. Offline terrain preparation is unchanged
and reported separately. The outer inference timer includes final cache/input
checks; the internal raw timer does not include the last geometry guard.

The eight unique pairs still contain **four positives and four mismatches**,
repeated at five checkpoints per mode. Both changed-light positives fail;
each checkpoint recovers the same unchanged-light pair. There are no raw pose
outputs on the four mismatches. This small subset establishes neither a 1%
risk bound nor a useful accepted-pose result. It selects no checkpoint and
does not replace the full comparison or eight lunar on/off ablations.

The full-study builder again refuses both modes under the unchanged
12,000-second maximum. It retains all 384 cases and all five checkpoints.
No full protocol/output is created, and no full-study compute is charged.
After the two additional pilots/replays, the original conservative allowances
remain **15,867 training seconds and 50,700 baseline seconds**. No failures,
charges or deadlines are reset or refunded. Calibration/final terrain remains
closed. No localization improvement, superiority or novelty is claimed.

## Verification and evidence

- [Exact results, hashes and resource refusal](solar_cache_pilot_results_v84.json)
- [Directional saved-output comparison](solar_cache_sh_saved_comparison_v84.json)
- [Pooled saved-output comparison](solar_cache_pooled_saved_comparison_v84.json)
- [Directional metric replay](solar-sh-checkpoint-selection-cost-pilot-replay-v84.json)
- [Pooled metric replay](solar-pooled-checkpoint-selection-cost-pilot-replay-v84.json)
- [Original cached prediction artifacts](solar_cache_pilot_predictions_v84.tar.gz)
- [Clean public-checkout CI receipt](solar_ci_cache_v84.json)

Local verification passes 982 tests with four skips and the fresh eight-toggle
image smoke/replay. Public-checkout CI passes 980 tests with six platform skips
and its controlled benchmark. All 49 learning and 42 reference source files
retain their original bytes.

## Next accuracy experiment

The [saved failure diagnosis](SOLAR_POSE_BOOTSTRAP_DIAGNOSIS_V84.md) still applies:
poor candidate coverage and nearly duplicate hypotheses prevent a pose, which
prevents the learned response from running. More training of that execution
path cannot repair the recorded failures. Candidate coverage and bootstrap
must be tested directly, with native-resolution and beam-diversity controls
separated from any learned improvement. Sun-support flags also do not establish
correct physical Sun inference.

A pose-free counterfactual retrieval prototype and feature adapter are prepared
outside the production repository. Their toy/stub checks are not actual neural
qualification or lunar evidence. In particular, globally extracting native
features before active selection would violate the principal method's deferred
native-feature rule. Such a version can only be an explicitly charged diagnostic
control; it cannot silently become the main A/B/C matcher or claim active
compute savings. Any principal replacement must preserve that rule or obtain an
explicitly revised research contract.

The original requirements remain active. Matched scientific ablations, seed
repetitions, modern comparisons, transfer, calibration, a qualifying final
freeze/test and the final paper/release package remain unfinished.
