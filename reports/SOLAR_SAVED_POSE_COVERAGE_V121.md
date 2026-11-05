# How much could selecting among saved poses recover?

On 192 simulated lunar development pairs, an ideal selector among five
completed methods could return **120 correct poses (62.50%)**. It would
need at least **135 correct poses** to reach 70%, even if it made no
acceptance errors. This places a ceiling on re-ranking, calibration, and
fallbacks that only pick among these exact saved poses. A method that
combines correspondences or estimates a new pose can exceed that ceiling.

| Saved outputs | All positives | Same light index | Changed light index |
|---|---:|---:|---:|
| Direct with 24 lighting templates | 114/192 | 40/64 | 74/128 |
| SIFT | 41/192 | 41/64 | 0/128 |
| MatchAnything–EfficientLoFTR | 42/192 | 40/64 | 2/128 |
| Direct with 12 lighting templates | 86/192 | 40/64 | 46/128 |
| Direct with cross-light pooling | 114/192 | 40/64 | 74/128 |
| Hindsight selection across all five | **120/192** | **46/64** | **74/128** |

All six gains over 24-template Direct were same-light pairs. Four came
from one terrain group and one each from two other groups. The two
changed-light poses recovered by MatchAnything were also recovered by
Direct. Both additional Direct variants added zero cases to the union
of full-template Direct, SIFT, and MatchAnything.

## What to measure next

Among the **54 changed-light pairs missed by all five methods**, at least
one method saved six or more correct correspondences on **16** at two
native pixels and **17** at four native pixels. On the remaining **37**,
every method saved fewer than six four-pixel correspondences. The count
alone cannot establish spatial coverage, unique matches, or compatibility
with the three-pixel pose estimator. Conversely, a saved-match shortfall
does not rule out usable image information or a solution from fewer points.

Separate the next test into proposal capture and pose recovery. First
measure how often a coarse proposal puts at least six distributed true
correspondences inside the frozen native-search radius. Then measure
whether image-based refinement finds them and whether the unchanged pose
solver recovers the camera. Score height-perturbation controls separately.
Keep the complete case roster in the denominator at every stage.
[Prepared proposal study](solar-warp-v117/README.md) ·
[Native handoff and capture audit](solar-warp-handoff-v120/README.md)

## Scope and replay

Each saved row uses the common raw pose criterion of rotation error
≤2° and translation error ≤1% of median reference range. Exact case IDs
were joined against all 384 original development cells, and each of the
five method tables contains every cell exactly once. The identical Direct
repeat in v99 was checked and counted once. A public replay can use the
v99 `paired_mechanism_trace` for the case IDs and light indices.

On the 192 matched nonoverlap cases, MatchAnything returned 51 raw poses.
The other four methods returned zero. Hindsight selection uses the true
label and pose, so its positive-case ceiling supplies no empirical
acceptance or false-acceptance performance. Results reuse eight terrain
groups. This analysis does not include the separate ARM cohort, real
imagery, or reserved terrain. It performs no new inference.

To reproduce the counts, build one set of correct positive case IDs per
method from `rows`, then count their union for the whole cohort and each
light stratum. For each omitted positive case, take the maximum of
`correspondence["2"]["correct"]` and separately of
`correspondence["4"]["correct"]` across the five methods. Count maxima of
at least six. On the full cohort, the corresponding missed-case counts
are 24 and 26 of 72. The six additional same-light IDs are
`u020-l16-16-positive`, `u021-l00-00-positive`, `u022-l16-16-positive`,
`u023-l16-16-positive`, `u056-l16-16-positive`, and `u073-l16-16-positive`.

| Input | SHA-256 |
|---|---|
| [v92 Direct replay](solar_full_direct_render_replay_v92.json) | `b11cc65d41367d82c85fc770b109a83fcd4fb65dca2a7da7c56ae039725af558` |
| [v94 native replay](solar_full_native_baselines_replay_v94.json) | `75af1832538230553d3920bea1ee8bf9e7522232b319a30eb060cfe1616ae183` |
| [v95 12-template replay](solar_offcatalogue_direct_replay_v95.json) | `9e441047e928e8f05ab3ec594e00ef151c0f546e88f88114fa4a330f4dccd8b6` |
| [v99 cross-light replay](solar_full_crosslight_rescue_replay_v99.json) | `8347b60efa4d5f3721f2e118087475ec791339c595f43e2cd204b2d678312182` |
| [v80 case manifest, as bound in v94](solar_full_paired_development_v94.json) | `750919f111d69cba13b1560409955f7af2c4369ca6cc526b93bc01579993c3e2` |
