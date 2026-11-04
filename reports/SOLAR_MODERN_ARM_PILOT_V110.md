# LoMa-R and RoMa v2 on the same lunar development cases

Four matchers ran on the same two-thread Linux ARM worker. They received the same eight native 512-pixel cases, reference geometry and final pose estimator. Four pairs overlap and four are certified nonoverlaps. Two of the positives change illumination. No method received the query Sun label. The four positives cover four model-selection terrain groups. LoMa-R and RoMa v2 passed fresh pretrained known-transform and invalid-input controls on this worker. [Frozen protocol](solar-modern-arm-v110/pilot-protocol.json) · [Completed Actions run](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/41930272861)

| Method | Correct raw poses / 4 | Changed-light correct / 2 | Wrong positive pose / 4 | Raw pose on nonoverlap / 4 | Median inference on ARM |
|---|---:|---:|---:|---:|---:|
| Frozen compact with shared Sun and active selection | 1 | 0 | 0 | 0 | 6.82 s |
| SIFT | 2 | 0 | 0 | 0 | 0.057 s |
| LoMa-R | 1 | 0 | 1 | 1 | 53.23 s |
| RoMa v2 | 1 | 0 | 2 | 3 | 52.14 s |
| Historical Direct relighting, same cases | 3 | 1 | 1 | 0 | Different CPU |

A correct pose is within 2° rotation and 1% of the median reference range in translation. The stricter criterion of 0.5° and 0.1% was met on one SIFT and one historical Direct positive. The other ARM profiles met it on zero. The same-worker timing is the median over all eight model-resident inference rows. It includes pose estimation and any online physics. Model loading and offline reference preparation are separate. All reported wrong and nonoverlap poses are raw proposals. This pilot contains no calibrated acceptance results.

The compact control was about 7.6–7.8× faster than LoMa-R and RoMa v2 with the same raw correct-pose count. SIFT was faster and recovered the other unchanged-light pose. The 24-template historical Direct relighting control recovered one of the two changed-light positives on the exact same case manifest. It uses a simulator-matched reference library, and its runtime cannot be compared across the two machines. On the larger historical development cohort, Direct recovered 114/192 raw poses, MatchAnything–EfficientLoFTR 42/192 and SIFT 41/192. [Larger common-cohort results](solar_full_paired_development_v94.json)

One positive changes the pilot recovery rate by 25 percentage points. The pilot qualifies the two modern baselines and measures their cost on the same machine. It cannot establish a general performance ranking. LoMa and RoMa v2 report results on different datasets and metrics in their papers. The current same-worker data do not support a superiority or deployment claim. [LoMa paper and results](https://www.davnords.com/loma) · [RoMa v2 paper](https://arxiv.org/abs/2511.15706)

The ARM watchdog finished in 1,000.76 seconds under its original 1,800-second reservation. Independent replay verified all 32 fresh rows, with no cached rows or source/scoring discrepancies. The conservative projection for the same four profiles on all 384 cases is 65,729 seconds, above the original 18,000-second stage cap. A future budgeted comparison needs a smaller prospectively fixed cohort, while retaining the full historical results. [Exact metrics](solar_arm_v110_pilot_results.json) · [Watchdog](solar_arm_v110_remote_watchdog.json) · [Independent replay](solar_arm_v110_independent_replay.json)

The results ZIP contains the 32 prediction arrays, row seals, actual pretrained controls and original worker summary. It excludes model weights and the worker authorization. Extract it into a clean checkout. Its `solar-arm-v110-replay-inputs.json` lists the 46 exact relative files needed from the `payload/` prefix of the earlier transport asset. Check every listed SHA-256 before running the unchanged verifier. The verifier recomputes scoring from saved arrays without running a model. [Results ZIP](solar_arm_v110_pilot_artifacts.zip) · [Transport asset](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/download/solar-arm-transport-v98/solar-arm-v98-public.zip)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python verify_solar_evaluation_v98.py runs/solar-modern-arm-v110/pilot \
    --protocol reports/solar-modern-arm-v110/pilot-protocol.json \
    --output runs/solar_arm_v110_replay_repeat.json
```
