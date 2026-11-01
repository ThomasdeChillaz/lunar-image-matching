# Native lunar pose results including MOONSt3R

These are all72 **exposed development pairs**, generated from eight geometries in four terrain components. They do not establish fresh-test accuracy. Correct means rotation ≤2° and translation ≤1% of median reference range; strict means the already frozen stricter rule recorded in the native protocol. All methods receive the same reference geometry and final3-native-pixel AP3P pose estimator; query depth/true query pose are scoring only.

| Exact evaluated configuration | Correct raw | Strict /72 | Wrong raw | No pose |
|---|---:|---:|---:|---:|
| SIFT-first + surface pipeline | 57/72 (79.2%) | 25 | 2 | 13 |
| Frozen CNN + surface | 54/72 (75.0%) | 16 | 3 | 15 |
| RoMa v2 fast/common | 53/72 (73.6%) | 17 | 19 | 0 |
| Frozen orientation CNN | 45/72 (62.5%) | 17 | 12 | 15 |
| LoMa-R common | 42/72 (58.3%) | 14 | 28 | 2 |
| MatchAnything-EfficientLoFTR native512 | 39/72 (54.2%) | 5 | 30 | 3 |
| MOONSt3R common-native, amended | 28/72 (38.9%) | 21 | 44 | 0 |

The strongest completed simple pipeline remains **SIFT-first + reference-surface refinement:57/72**. Its advantage over the tested RoMa v2 fast profile is four pairs, which is not evidence of a general SOTA advantage. These are whole configured pipelines: the compact method uses four image rotations and reference-surface rectification, whereas the dense models use their declared single-pair settings. Lunar adaptation and unknown MOONSt3R training overlap further limit attribution. Remaining author-recommended profiles are still outstanding.

MOONSt3R recovers **28/72 raw correct poses, including21 strict**, versus frozen CNN+surface54/72 with16 strict. It is weaker at the main threshold on this dataset, but the strict result prevents a blanket claim that our correspondences are more precise. Neither result is calibrated acceptance. The compact surface pipeline still emits three wrong proposals; SIFT-first emits two. A tested model must improve the complete pose/coverage/error tradeoff before adoption.

## Resource amendment retained

The original four-calendar-hour MOONSt3R run was stopped **incomplete at71/72** after observed clock/suspension divergence:calendar age41,891seconds exceeded14,400seconds, while completed-case inference totalled3,239seconds. The exact cause is not established. A separately published600-calendar-second contract froze the71original row hashes and the single missing case before inference. The supplemental phase completed in42.48seconds and the amended72-case study independently replays. All71original rows remain byte-identical; the original expired protocol is never described as completed or reset. The amendment stays inside the user's unchanged72CPUwall-hour comparison allowance.

## What this means for the new training

The new81,696-parameter coupled-assignment study is a separate prospective experiment. Its toy loss counterexample, exact100-update optimizer replay and software checks are prerequisites, not localization gains. All three seeds and all checkpoint results must be retained; selection includes initialization. If trained models fail to beat the frozen baseline, they will not be adopted. No breakthrough is claimed from matching a limited subset of pretrained configurations.

The JSON companion binds all source summaries/audits/case hashes and reports counts by terrain and light-index equality. It includes an exact256-draw four-component descriptive bootstrap for paired raw success. Those intervals describe exposed development cases and cannot repair the missing fresh-data gate. Light-index equality is not a quantified Sun-angle measurement. No model forward, new raster input, final-test inference or metric/pose-estimator rerun occurs in this reporter.

No fair paired MOON timing or isolated-memory ranking is available. Full fresh terrain validation, calibrated acceptance/nonoverlap, transfer, remaining baseline profiles and independently characterized orbital accuracy remain unresolved.
