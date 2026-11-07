# Two pilots for improving direct lunar relighting

The first pilot tests whether extra native patches improve an existing pose. The second tests whether pooling matches across reference illuminations can rescue cases where the original solver returns no pose. Both use the same eight previously observed development cases and a frozen direct control.

| Pilot and arm | Correct positive poses | Correct changed-light poses | Wrong positive poses | Raw nonoverlap poses |
|---|---:|---:|---:|---:|
| Translation-window pilot, direct | 3/4 | 1/2 | 1/4 | 0/4 |
| Translation-window pilot, uniform | 2/4 | 1/2 | 2/4 | 0/4 |
| Translation-window pilot, translation-directed | 2/4 | 1/2 | 2/4 | 0/4 |
| Cross-light pilot, direct | 3/4 | 1/2 | 1/4 | 0/4 |
| Cross-light pilot, pooled rescue | 3/4 | 1/2 | 1/4 | 0/4 |

## Translation-directed patch acquisition

The two patch policies searched the same eligible pool with a 128-window cap and used the same native normalized cross-correlation and common pose refit. The translation-directed policy ranks candidates by the improvement it predicts in the weakest translation direction, including reference height perturbations when ranking them.

Both patch policies regressed on `u060-l16-16`. The original translation error was 15.55 m. Adding 79 reciprocal matches with uniform selection increased it to 150.33 m. Adding 70 with translation-directed selection increased it to 168.21 m. Neither policy rescued the wrong direct pose on `u074-l08-01`. These cases motivate examining the precision of the added correspondences and testing terrain-aware patch warping before scaling this implementation.

All 24 rows and acquisition traces independently replayed. The run took 87.91 seconds. Median online inference was 1.649 s for the control, 1.835 s for uniform and 1.896 s for translation-directed selection. The pilot used extra patches after a full SIFT run, so its window count measures incremental work.

## Cross-light correspondence rescue

The rescue arm uses the same nine pose solves as the direct arm. It pools distinct matches across reference lighting only if the first seven solves fail and the eighth candidate has fewer than the minimum six correspondences. Failed pooling restores the exact original correspondence and pose output.

All four positive pairs already had a direct pose, so none activated the rescue. All four nonoverlap pairs activated it. None returned a pose. This verifies execution, fallback and the fixed solve budget on those cases. Measuring positive recovery requires the full 384-case comparison, including the 60 positives where the historical direct control returned no pose.

All 16 rows and pose-call traces independently replayed. The fresh direct arm reproduced the earlier pilot's raw metrics on all eight cases. The run took 61.10 seconds. The conservative projection for both arms on all 384 cases is 3,182 seconds, within the original 12,000-second study cap. A separate published protocol is required for that full comparison.

- [Translation-window protocol](solar-direct-additive-windows-pilot-v97.json) and [independent replay](solar_direct_additive_windows_pilot_replay_v97.json)
- [Cross-light protocol](solar-crosslight-mechanism-dispatch-v98.json) and [independent replay](solar_crosslight_mechanism_results_v98.json)
- [Full paired development benchmark](SOLAR_FULL_PAIRED_DEVELOPMENT_V94.md) and [withheld-light follow-up](SOLAR_OFFCATALOGUE_DIRECT_V95.md)
