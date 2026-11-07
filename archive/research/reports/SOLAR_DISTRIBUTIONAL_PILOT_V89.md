# Distributional lunar matcher pilot and rotation diagnosis

The first 100 updates of the 95,116-parameter candidate completed in 18.392 seconds on two CPU threads. The exact replay took 18.020 seconds. It reproduced all 100 optimizer steps, sampling decisions, loss and gradient scalars, initial and final model and optimizer states, and RNG state. Each run charged its original 300-second reservation. The training balance after replay is 14,229 seconds and the baseline balance is 49,080 seconds.

The candidate predicts a distribution over the frozen coarse descriptor. Training uses only even-light observations from the sealed 64-camera bank, with 3,933 points across 16 terrain groups. The prediction can marginalize camera viewpoint during initial matching and use a proposed viewpoint later. The pilot establishes reproducibility in the coarse 128-pixel descriptor domain. Pose evaluation, native refinement integration, and checkpoint selection remain pending.

| Evidence | Published record |
|---|---|
| Training protocol and exact input identities | [Pilot protocol](solar-distributional100-pilot-v89.json) |
| First 100 actual steps | [Pilot history](../runs/solar-distributional100-v89/pilot/science/history_100.json) |
| Reproduction protocol | [Exact replay protocol](solar-distributional100-replay-v89.json) |
| Replay outcome and original budget | [Replay result](../runs/solar-distributional100-v89/replay/replay.json), [watchdog](../runs/solar-distributional100-v89/replay/watchdog.json) |
| All eight controlled translation checks | [Saved replay](solar_translation_smoke_replay_v89.json) |
| All eight 22.5° rotation checks | [Saved replay](solar_rotation_stress_replay_v89.json), [candidate-choice arithmetic](solar_rotation_stress_acquisition_review_v89.json) |

## What the rotation failure isolates

The historical frontend passed all eight composite translation checks and failed all eight composite rotation checks. Six rotation configurations still produced raw poses, with 0.28–0.63 m translation error and 0.16–0.32° rotation error. A true correspondence fell within a reachable rotated search window for 192 of 256 anchors. Only 5–14 of the 32 selected searches could reach it, depending on configuration. The two shared-Sun plus active-selection configurations each produced four retained matches within 3 pixels and failed RANSAC. The saved-coordinate audit reproduced the ground-truth affine/pose agreement to about 1.3e−13 pixels. The most direct intervention is to improve which candidate gets a native-resolution search.

## How the research compares

| Reference | Reported benchmark | Scope for this project |
|---|---|---|
| Ansar and colleagues’ LIMA study | Fine-tuned LoFTR returned 70.6% of matches within 1 px and 85.9% within 5 px on unseen 0°-elevation lighting. | The closest illumination reference. Compare equivalent input geometry and pixel-error denominators before drawing a ranking. [LIMA, Table 1](https://doi.org/10.2514/6.2025-2073) |
| Geo-LoFTR | Known-map localization with DTM-conditioned matching and a 1 m position criterion in the Mars simulation. The model forward pass took 887 ms on a Jetson AGX Xavier. | A close geometry-aware comparison. Both the planet and hardware differ from the two-thread lunar study. [Geo-LoFTR](https://arxiv.org/html/2502.09795v3) |
| Casadesus Vila and Gao’s October 2026 lunar frontend benchmark | On PolarTraverse, SuperPoint + LightGlue achieved 93.3% relative-pose AUC at 5°, versus 37.8% for SIFT + FLANN. SIFT + FLANN scored higher on four synthetic conditions. | A strong rationale for a matched learned and classical control and a separate transfer result. The benchmark uses relative rotation and translation direction. [Lunar visual SLAM benchmark, Table II](https://arxiv.org/html/2610.30774v1#S5) |
| MoonAnything LunarGeo | Stereo pairs rendered on lunar topography with three lighting configurations, dense depth, and Moon-fixed camera poses. | A candidate external transfer dataset for descent views. The published showcase evaluates stereo reconstruction. [MoonAnything](https://arxiv.org/html/2604.00682v1#S3.SS2) |

The latest matched development pilot recovered two of four raw positive poses with SIFT and one of four with each of the three frozen compact controls. All four methods recovered zero of two changed-light poses. Those small, common-case counts are the current local accuracy reference. The next head-to-head experiment holds the candidate budget, native refinement, and absolute pose estimator fixed while changing the learned ranking. A successful candidate must then meet the frozen accuracy, false-acceptance, speed, ablation and transfer gates before a performance result is ready to share with Ansar. [Matched development pilot](SOLAR_DIAGNOSTICS_V86.md) · [Original acceptance requirements](requirements_and_sota_v69.json)
