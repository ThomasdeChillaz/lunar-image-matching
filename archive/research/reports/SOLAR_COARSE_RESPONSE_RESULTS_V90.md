# Coarse lighting response fails the matched pose pilot

The distributional coarse response did not improve camera localization in the eight-case development pilot. The first two trained checkpoints tied the frozen control at one correct pose out of four positives. The final two recovered zero, and the 2,000-step checkpoint returned poses on three positives outside tolerance and one nonoverlap pair. Every profile failed both changed-light cases. The independent saved-array and source-label replay agreed on all 48 rows. [Results and costs](solar_distributional_evaluation_results_v90.json) · [Independent replay](solar_coarse_response_paired_replay_v90.json)

| Same-case profile | Correct raw poses / 4 | Correct changed-light poses / 2 | Wrong positive poses | Raw poses on nonoverlap / 4 | Median inference |
|---|---:|---:|---:|---:|---:|
| Frozen compact control | 1 | 0 | 0 | 0 | 9.30 s |
| Coarse response, initialization | 1 | 0 | 1 | 0 | 10.91 s |
| Coarse response, 500 updates | 1 | 0 | 2 | 1 | 10.54 s |
| Coarse response, 1,000 updates | 1 | 0 | 1 | 2 | 10.35 s |
| Coarse response, 1,500 updates | 0 | 0 | 2 | 1 | 10.79 s |
| Coarse response, 2,000 updates | 0 | 0 | 3 | 1 | 11.28 s |

All profiles used the same four native-view proposal rotations, 128 paid native search windows, shared Sun, active policy, frozen native refiner, and common 3-pixel AP3P solver. Correctness requires rotation error at most 2° and absolute translation error at most 1% of median reference range. The numbers include blank cases, failed estimates, and all four nonoverlap pairs. They are raw development outcomes before acceptance calibration. [Frozen experiment](solar-coarse-response-evaluation-study2000-cost-pilot-v90.json)

The earliest missing evidence differs by case. One changed-light image pair triggered the common low-information guard. For the other, the frozen model put truth inside at least one of eight native search windows for just 9 of 225 co-visible anchors. The 2,000-step response reduced that count to 4 of 225. Its selector paid for none of those windows. The final solver returned a pose 145.7° and 4.13 km from truth. On the positive pair the frozen control recovered, training reduced the number of final matches within 2 pixels from 48 to 24 and increased translation error from 5.20 m to 18.39 m. Better teacher likelihood during training did not translate into better correspondence or pose accuracy. [Saved window and pose replay](solar_coarse_response_paired_replay_v90.json) · [Training audit](solar_distributional_completed_study_audit_v90.json)

SIFT recovered two of the same four positives in the earlier matched run, both under unchanged lighting, and returned no nonoverlap poses. The more relevant improvement target is to find enough correct candidates under changed light and then spend the limited native search budget on them. A same-geometry direct rendering control can test whether a physically relit reference provides the missing evidence before another teacher-distillation cycle. [Matched SIFT evidence](SOLAR_DIAGNOSTICS_V86.md) · [Prior-art comparison](SOLAR_DISTRIBUTIONAL_PILOT_V89.md)

The primary seed completed all 2,000 updates in 169.21 seconds and reproduced the original 100-step trace and endpoint states. The matched pose pilot took 425.34 seconds. The original full-stage estimator projected 44,342 seconds for the six profiles on all 384 development pairs against the predeclared 12,000-second cap. The controller refused the full run before reservation. Checkpoint selection and confidence calibration remain pending. [Study checkpoint hashes](../runs/solar-distributional2000-v90/study/science/summary.json) · [Pilot replay](solar_coarse_response_paired_replay_v90.json)
