# Five-method lunar pose comparison

The frozen development roster contains 48 overlapping cases and their 48
matched nonoverlaps across eight terrain groups. It selects one motion and one
roll camera per group, each with one same-light and two changed-light positive
conditions. The five arms produce 480 planned method–case outcomes.

| Arm | Role |
| --- | --- |
| Frozen compact matcher | Shared-Sun and quantization-aware development control |
| SIFT | Conventional feature baseline |
| LoMa-R | Published rotation-aware pretrained baseline |
| RoMa v2 | Published dense pretrained baseline |
| Direct | Geometry-aware render-and-match control using its 24-light reference library |

All arms receive the same native 512-pixel image content, camera intrinsics and
reference geometry. The common estimator uses a three-native-pixel
reprojection limit. Query depth, query pose and true query Sun are outside the
primary inference interface. Pose scoring uses at most 2 degrees of rotation
error and 1% of median reference range for translation error. Runtime includes
cold per-case reference setup. Model loading, offline reference preparation
and shared-worker peak memory are reported separately where measurable.

The exact ordered subset, author sources, weights, settings and data bytes
were published before inference. The cap covers setup, controls and execution
on one two-thread Linux ARM worker. The 17,322-second admission estimate comes
from two completed qualification runs. The original ledger charges the full
18,000-second cap once.

The comparison measures raw pose recovery and raw wrong outputs on
model-selection terrain. Calibration and final terrain stay closed. Positive
uncertainty is resampled by terrain group. Negative sensitivity accounts for
both terrain endpoints. The five-arm result cannot by itself verify orbital
navigation accuracy or calibrated acceptance.

[Published protocol](https://github.com/ThomasdeChillaz/lunar-image-matching/blob/24046f9c34488fe2efd83464386b0ca8e46906e2/reports/solar-five-arm-v115/comparison-protocol.json)

The authorized GitHub run started at 00:31:03 UTC on 5 November 2026.
Its five-hour wall-time cap begins at that timestamp. No scores were available
at publication. A clean transport rehearsal verified both release archives,
1,730 assembled files and all 32 reference archives. The final local suite
passed 1,328 tests with four skipped. The eight fresh procedural controls
passed independent saved-array replay.
[First attempt and live status](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/41941904269)
