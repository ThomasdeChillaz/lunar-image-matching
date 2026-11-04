# Supplied-Sun contrast synthesis and camera pose

Two-reference synthesis and nearest-even template selection each recovered 16 of 24 positive poses. On the 16 withheld odd-light positives, both recovered 10. Synthesis recovered one case the nearest arm missed and lost one the nearest arm recovered. The median same-view photometric improvements measured earlier did not translate into a net pose gain on these changed-view pairs. [Earlier photometric control](SOLAR_REFERENCE_PHOTOMETRY_V103.md)

Both arms receive the true query Sun direction in the separate secondary track. They use the same native 512-pixel input, grayscale pipeline, SIFT matcher, reference geometry and common AP3P solver. Each supported case tests a single reference template. The nearest arm uses one even render. The synthesis arm combines two even renders where a valid angular bracket exists. On the eight exact-template positives both produce identical native outputs.

| Outcome | Nearest even | Two-reference synthesis |
|---|---:|---:|
| Correct raw poses | 16/24 | 16/24 |
| Correct on withheld odd lights | 10/16 | 10/16 |
| Stricter correct poses | 4/24 | 5/24 |
| Wrong positive poses | 3/24 | 4/24 |
| Positives with no pose | 5/24 | 4/24 |
| Raw nonoverlap poses | 0/24 | 0/24 |
| Raw nonoverlap poses, common support | 0/14 | 0/14 |
| Median complete inference | 0.197 s | 0.227 s |

The overall paired raw gain was zero, with a 95% terrain-group interval of −12.50 to +12.50 points. For the withheld-light positives, the interval was −18.75 to +18.75. The synthesis arm had no valid bracket on ten nonoverlap cases, which remain in the denominator. The common-support result separates those automatic null outcomes from nonoverlap cases where both arms actually matched images.

All 96 scheduled method–case rows independently replayed. The run used 89.29 seconds under the 542-second reservation. This evaluates a specified horizontal Lambertian component motivated by LIMA, combined with our matcher and pose solver. It is a control on template construction, with no claim to reproduce the full LIMA algorithm or to improve estimated-Sun localization. [LIMA paper](https://doi.org/10.2514/6.2025-2073)

[Published pre-inference protocol](solar-supplied-sun-pilot-dispatch-v102.json) · [Independent replay and full rows](solar_supplied_sun_pilot_replay_v102.json)
