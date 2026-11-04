# Matched results and current research requirements

The strongest interim result for Adnan Ansar is the controlled failure under missing lighting templates. Withholding exact odd-angle templates lowered changed-light pose recovery from 74/128 to 46/128, a 21.88-point loss with a paired terrain interval of −43.75 to −4.69 points. On those same simulated pairs, MatchAnything–EfficientLoFTR recovered 2/128 and SIFT recovered 0/128. The result measures how the reference-light catalogue affects a complete localization pipeline on eight development terrains. [Paired benchmark](SOLAR_FULL_PAIRED_DEVELOPMENT_V94.md) · [Withheld-light comparison](SOLAR_OFFCATALOGUE_DIRECT_V95.md)

## Published comparisons

| Work | Reported setup and measurement | Matched evidence still needed |
|---|---|---|
| [LIMA](https://doi.org/10.2514/6.2025-2073) and its [technical description](https://patents.justia.com/patent/20260219064) | Co-registered references with known Sun vectors are combined into a contrast template and correlated against a target under supplied illumination. | A supplied-Sun control with the same reference access, matcher and pose solver. The two-reference implementation isolates template construction. |
| [Wang et al. 2026](https://doi.org/10.1016/j.cja.2026.104149) | Coarse position and attitude, Sun ephemeris, DEM rendering and masked correlation. Reports horizontal error below 20 m, altitude error below 10 m and about 0.5 s per frame on a Jetson TX2 in a descent and analog validation setup. | Match the prior, inputs, scoring and hardware before comparing metric accuracy or speed. |
| [MOONSt3R](https://arxiv.org/html/2510.18172) | Synthetic stereo with known intrinsics. Dynamic results report 98.0% rotation and 91.7% translation direction below 2 degrees as separate quantities. | Evaluate absolute reference-map translation, joint pose success, raw nonoverlap outcomes and equal calibration conditions. |
| [WARG](https://arxiv.org/html/2606.10602#S2.SS1) | Within a supplied search area and after YuTu-2 fine-tuning, reports mean localization error of 2.93 m single view and 1.68 m with panoramic views. | Match the rover-to-orbit localization task and search prior before comparing metric error. |
| [REDI-Match](https://arxiv.org/html/2606.24330) and [RoMa-Ω](https://arxiv.org/html/2609.09507) | Recent general-purpose methods improve on earlier models on some rotational and multimodal benchmark tasks. Published efficiency uses GPUs. | Use the same lunar images, reference geometry, raw and calibrated scoring, and CPU budget. |

The completed native CPU benchmark has a limited modern comparison against MatchAnything–EfficientLoFTR. Full-library direct relighting was 37.50 raw pose points higher on the same pairs, with a paired terrain interval of 19.27 to 55.73 points. Its median online inference was 1.293 seconds versus 3.820. The reference library’s offline preparation and additional information matter for the system-level comparison. The prepared ARM pilot puts LoMa-R, RoMa v2, SIFT and the frozen compact model on one worker for eight cases. That pilot qualifies instrumentation and cost. Direct relighting and the eventual candidate still need a same-worker comparison for an efficiency claim. [Paired benchmark](SOLAR_FULL_PAIRED_DEVELOPMENT_V94.md) · [ARM preflight evidence](solar_modern_arm_preflight_receipt_v101.json)

## Evidence gates

| Original target | Current evidence |
|---|---|
| At least 70% correctly accepted | The full-library raw ceiling is 114/192. A 192-positive cohort needs at least 135 correct poses, 21 more than observed. The 12-light raw ceiling is 86/192. Rejection alone cannot increase either ceiling. |
| At least a 10-point accepted gain over the strongest matched compact control, with a positive paired terrain interval | The full cross-light comparison recovered zero additional poses. The prospective exposure and terrain-warp pilots target different parts of the matching problem. |
| At most 1% empirical accepted error and 1% nonoverlap acceptance | The raw library results returned 0/192 nonoverlap poses on repeated terrain endpoints. Even treating cases as independent for illustration gives a one-sided 95% upper bound of 1.55%. Separate calibration and final evaluation are required for the empirical targets. |
| Pose accuracy advantage or within three points at five times the speed on the same CPU | The eight-case ARM qualification is staged. It excludes direct relighting, and no stronger-model inference has run. |
| Real transfer, complete ablations and untouched reserve | The independently tracked analog geometry cohort is staged. LuSNAR, POLAR, orbital transfer, full module ablations and map-height robustness remain open. Four calibration groups and twelve final groups stay closed until the required gates. |

The follow-up experiments have distinct roles. Exposure normalization tests whether low-code texture can yield more correct matches under the same pose budget. The supplied-Sun comparison tests two-reference synthesis against the nearest even template. Its angular support differs for nonoverlap cases, so rejection must be reported both overall and on common-supported pairs. [Match-scarcity analysis](SOLAR_MATCH_SCARCITY_V102.md) · [Supplied-Sun control](SUPPLIED_SUN_CONTROL_V102.md) · [Arithmetic limits](solar_development_acceptance_limits_v102.json)
