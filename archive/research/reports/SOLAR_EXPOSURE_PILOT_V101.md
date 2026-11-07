# Exposure normalization on the frozen native pose pilot

Blind exposure normalization produced two additional correct poses on 24 simulated development positives. It also produced three additional wrong poses. It recovered none of the six near-black positives where the original matcher found zero correspondences.

Both arms searched the same 24 prepared reference illuminations with native SIFT and the same nine-call AP3P budget. The exposure arm independently maps each nonuniform query and reference image's 99.5th-percentile grayscale code to white. The original method uses the original 8-bit images. There were 24 positive and 24 matched nonoverlap pairs from eight development terrain groups. No query Sun direction was supplied.

| Outcome | Original | Exposure normalization |
|---|---:|---:|
| Correct raw poses | 15/24 | 17/24 |
| Correct on changed-light positives | 11/16 | 12/16 |
| Stricter correct poses | 2/24 | 10/24 |
| Wrong positive poses | 1/24 | 4/24 |
| Positives with no pose | 8/24 | 3/24 |
| Raw nonoverlap poses | 0/24 | 0/24 |
| Median complete inference | 1.598 s | 2.266 s |
| Median query keypoints | 27 | 345.5 |

The paired raw gain was +8.33 points overall with a 95% terrain-resampling interval of 0.00 to 20.83 points. The changed-light gain was +6.25 points with an interval of 0.00 to 18.75. Both intervals include zero. The treatment is slower and the wrong-pose count rises, so the strict-pose improvement and the increased number of detected features cannot by themselves justify adopting it.

## Near-black failure

| Frozen positive | Input p99.5 code | Original best template matches | Exposure best template matches | Exposure outcome |
|---|---:|---:|---:|---|
| u025-l00-00 | 2 | 0 | 1,890 | Wrong pose |
| u027-l08-01 | 2 | 0 | 1,862 | Wrong pose |
| u059-l08-01 | 12 | 0 | 2 | No pose |
| u061-l00-17 | 10 | 0 | 6 | Wrong pose |
| u063-l16-01 | 12 | 0 | 1 | No pose |
| u075-l00-00 | 24 | 0 | 7 | No pose |

In the two cases with more than 1,800 matches, **zero** returned points were correct within two native pixels under independent geometric projection. Most of the returned points had co-visible references. About 66% of each query became pure white after normalization. These results expose a correspondence-quality failure despite the number of features and apparently viable pose hypotheses. The source of the repeated image patterns needs a separate controlled test.

The pilot used 266.14 wall seconds under its 902-second reservation. Independent replay checked all 96 method–case records and reproduced all 48 original-arm native point and pose arrays byte-for-byte. The measured full-run estimate of 4,369 seconds fits the current budget, but this pilot does not justify promoting blanket exposure normalization. Calibration and final terrain remain closed.

[Published pre-inference protocol](solar-radiometric-pilot-dispatch-v101.json) · [Independent replay and full rows](solar_radiometric_pilot_replay_v101.json)
