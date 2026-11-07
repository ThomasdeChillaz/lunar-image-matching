# SPICE-HL3 physical lab baseline and pose-reference discrepancy

The frozen SIFT baseline returned a relative pose on all 24 native stereo-camera pairs. None met the joint nominal criterion of at most 5° rotation error and 15° signed translation-direction error. Rotation passed its 5° threshold in 23 pairs. Translation direction passed in zero. All 24 saved rows independently replayed.

| Measurement | Frozen baseline |
|---|---:|
| Camera image size | 640 × 360 |
| Pairs with a pose | 24/24 |
| Rotation error at most 5° | 23/24 |
| Rotation error at most 2° | 18/24 |
| Median rotation error | 1.540° |
| Median signed translation-direction error | 40.928° |
| Median of pairwise epipolar medians, fitted from matches | 0.422 px |
| Median of pairwise epipolar medians, nominal motion-capture geometry | 21.831 px |
| Matches within 1 px, fitted / nominal motion capture | 3,678 / 36 of 4,673 |

Twenty-three direction errors lie between 38.18° and 44.48° across the two traversals. The tight direction error and gap between fitted and independent geometry warrant an audit of the reference and projection conventions before comparing matcher quality on this cohort. Small residuals under an essential matrix fitted from the same matches do not establish correspondence accuracy. The study used the released marker-to-camera calibration and timestamp binding, with no independently verified correction to those inputs. A separate camera/marker frame check and uncertainty characterization are still needed.

An independent metadata check compared the frozen camera composition with the dataset's documented −Y gravity and 25° down-tilted ZED2. Under the current composition, the camera's right axis is about 55° down in both traversals. Gravity falls predominantly along that axis instead of the camera's image-down axis. This flags an optical-frame or transform interpretation discrepancy. It does not identify which convention needs repair. The original scores remain preserved. [Dataset coordinate description](https://arxiv.org/html/2506.22956v1#S7.SS4)

The dataset represents two sun-facing traversals of one physical lunar analog. This study scores relative rotation and signed translation direction and has no absolute reference-map pose, metric translation result, or isolated relighting comparison. It ran in 8.64 wall seconds under the 120-second reservation. The output and reference uncertainty remain relevant evidence for future transfer design. [Data descriptor](https://doi.org/10.1038/s41597-026-06668-8) · [Frozen protocol](spice-hl3-c-geometry-dispatch-v101.json) · [Independent replay](spice_hl3_geometry_replay_v101.json)
