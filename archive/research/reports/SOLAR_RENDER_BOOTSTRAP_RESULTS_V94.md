# Compact relighting bootstrap pilot

A rendered reference, a rendered pose seed, and projected native search windows each failed to add a correct pose on this eight-case development pilot. Every compact variant recovered the same positive pair. Direct relighting recovered three of the four positives on the same cases.

| Method | Correct positive poses | Correct changed-light poses | Wrong positive poses | Nonoverlap poses | Median online inference |
|---|---:|---:|---:|---:|---:|
| Frozen compact historical control | 1/4 | 0/2 | 0/4 | 0/4 | 9.30 s |
| Relit reference only | 1/4 | 0/2 | 0/4 | 0/4 | 10.30 s |
| Relit reference and pose seeds | 1/4 | 0/2 | 1/4 | 0/4 | 10.37 s |
| Relit reference, seeds and windows, active | 1/4 | 0/2 | 1/4 | 0/4 | 9.85 s |
| Relit reference, seeds and windows, uniform | 1/4 | 0/2 | 0/4 | 0/4 | 7.66 s |
| Relit reference, seeds and windows, active with coarse learned response | 1/4 | 0/2 | 2/4 | 1/4 | 11.10 s |
| Direct relighting | 3/4 | 1/2 | 1/4 | 0/4 | 1.36 s |

The new arms used the same 512-pixel inputs, 128 native patch-search budget, fixed 4,096 minimal-set proposal allocation and final 3-pixel pose threshold. Direct relighting uses a different correspondence pipeline. Its online timing excludes reference-library preparation. The frozen and direct rows come from the [previous matched comparison](SOLAR_MATCHED_RELIGHTING_V91.md).

## Where the compact path lost the gain

- On `u026-l00-17`, the 128-pixel render search identified the same light that won the direct comparison, but its candidate yielded only five front-facing inliers and failed to produce a seed. The pose-seeded and full active compact arms eventually returned wrong poses. Direct relighting retained 17 correspondences within one native pixel and recovered a correct pose.
- On `u060-l16-16`, the best 128-pixel relighting candidate supplied five supported matches, too few for a usable seed. Direct relighting recovered the pose.
- On `u074-l08-01`, the best 128-pixel relighting candidate yielded one front-facing inlier. Direct relighting returned a pose but missed the translation criterion. The [full-development geometry diagnostic](solar_full_direct_pose_geometry_v92.json) measures weaker spatial support and translation observability among wrong direct poses.

The response arm changed coarse candidate scores using trial-Sun descriptor moments. All four positive traces confirm that it scored 256 supported reference anchors and 2,704 query positions. This arm did not apply learned angular re-ranking during native patch updates. It tested one unselected step-2,000 checkpoint.

## Scope and replay

The independently replayed 40 rows matched the original geometry scorer. The five-arm pilot used 402.78 seconds of its reserved 600-second ceiling. There were no infrastructure failures. The four positive and four nonoverlap cases were fixed before inference. Counts describe raw poses on those cases. The small sample leaves the rate of false poses uncertain. Calibration and final terrain remain closed.

The direct path is the stronger development control. Its full test on the development groups recovered [114 of 192 positive poses](SOLAR_FULL_DIRECT_RENDER_V92.md). The compact pilot supports investigating why native relit correspondences disappear in low-resolution initialization, and whether direct relighting gains from patch selection aimed at the least constrained translation direction.

- [Frozen mechanism protocol](solar-render-bootstrap-dispatch-v94.json)
- [Independent replay with per-case pose errors](solar_render_bootstrap_mechanism_replay_v94.json)
- [Frozen full-development SIFT and MatchAnything protocol](solar-full-native-baselines-dispatch-v94.json)
