# Direct relighting on all 384 lunar development cases

The source-frozen direct renderer recovered **114/192 correct raw poses
(59.4%)** and **74/128 (57.8%)** under changed light indices. It returned 18
wrong positive poses and zero poses on 192 nonoverlap pairs. Independent
replay recovered all 384 cases and all reported pose and correspondence
metrics. This is the known-reference, shared-simulator primary track.
[Exact results and terrain groups](solar_full_direct_render_results_v92.json) ·
[Independent replay](solar_full_direct_render_replay_v92.json) ·
[Frozen protocol](solar-full-direct-render-v92.json)

| Light indices | Correct / positive | Strict / positive | Wrong / positive | No pose / positive | Raw pose / nonoverlap | Median inference |
|---|---:|---:|---:|---:|---:|---:|
| All | 114/192 | 14/192 | 18/192 | 60/192 | 0/192 | 1.293 s |
| Same | 40/64 | 4/64 | 6/64 | 18/64 | 0/64 | 1.338 s |
| Changed | 74/128 | 10/128 | 12/128 | 42/128 | 0/128 | 1.286 s |

Correct poses meet both 2° and 1% of median reference range. Strict poses
meet 0.5° and 0.1%. The eight terrain groups range from 3/24 to 24/24 correct.
Repeated pairs within those groups remain dependent. The observed
nonoverlap count cannot establish a population error rate below 1%.
Even retaining every correct raw pose would leave this method at 59.4% on
the cohort. Meeting the 70% acceptance target would require at least
21 additional correct poses before rejection.

The two-thread job took 602.49 seconds under its original 1,256-second
allowance. Median measured input setup plus inference took 1.317 seconds.
The 768-image library was generated during earlier simulation and has a
6,247.81-second conservative preparation bound that includes the original
acquisition's query work. The inference medians exclude that preparation.

## The geometry of wrong poses

The saved-only diagnostic reproduced all reported inlier counts using
the 3-pixel criterion. It computed features from estimated poses, reference
geometry, query intrinsics, and saved correspondences, then joined saved
correctness. It used the same computations for correct and wrong outputs.

| Median | 114 correct poses | 18 wrong positive poses |
|---|---:|---:|
| Inliers | 38.5 | 10 |
| Occupied image cells on a 4×4 grid | 10 | 4 |
| Minor-axis spread of query inliers | 81.9 px | 23.4 px |
| Normalized pose Jacobian condition number | 32.18 | 92.97 |
| Local translation sensitivity per pixel | 9.92 m | 57.05 m |

The translation sensitivity assumes independent pixel noise and fixed,
exact geometry and inliers. Its roughly 5.8× difference motivates acquiring
patches that support poorly observed translation directions. It is a
descriptive development diagnostic with no fitted acceptance threshold.
[Geometry diagnostic](solar_full_direct_pose_geometry_v92.json) ·
[Diagnostic source](../diagnose_direct_pose_geometry_v92.py)

## Research comparison

In the original eight-pair comparison, direct relighting recovered 3/4
correct poses, SIFT 2/4, and MatchAnything and frozen compact 1/4 each.
The full classical and pretrained comparison has been frozen to the same
384 cases. Its performance remains pending. [Matched pilot](SOLAR_MATCHED_RELIGHTING_V91.md) ·
[Full matched baseline protocol](solar-full-native-baselines-v93.json)

Ansar and colleagues' 2026 LIMA follow-up builds illumination-matched
reference maps from archived imagery at different Sun angles. Wang and
colleagues combine shadow landmarks and online rendering with a navigation
prior, reporting mean horizontal and vertical errors of 5.7 m and 0.7 m
in their final analog flight phase. Their inputs, hardware, and error
protocols differ from the fixed reference-camera development cohort here.
[LIMA follow-up](https://arc.aiaa.org/doi/abs/10.2514/6.2026-2244) ·
[Wang et al.](https://www.sciencedirect.com/science/article/pii/S1000936126000865)

The contribution to test is whether physically relit pose hypotheses and
targeted measurements improve the absolute pose under a fixed acquisition
budget, including off-catalogue light and map uncertainty. The current
generic library includes the true rendered Sun directions. Evidence beyond
this simulator, calibrated error and coverage, and the original frozen
acceptance and efficiency gates are required before a comparative
performance claim to Ansar. [Original requirements](requirements_and_sota_v69.json)
