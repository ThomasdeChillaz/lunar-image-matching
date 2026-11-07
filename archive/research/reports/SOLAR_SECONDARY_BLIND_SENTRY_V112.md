# Blind scattered-light pose-ranking pilot

The compact matcher generated seven and five blind hypotheses for the two renders of the same changed-light camera pair. No hypothesis met the pose criterion. In both runs the intersection of valid reference-map pixels across the candidate bank was empty. The direct-only and scattered-light ranking arms both fell back to the original order. This stopped the study after the six preregistered exposures.

## Measured result

| Camera pair | Renderer seed | Blind beams | Distinct poses | Shared valid pixels | Correct pose anywhere in bank |
|---|---:|---:|---:|---:|---:|
| Overlapping, changed light | 0 | 7 | 3 | 0 | No |
| Overlapping, changed light | 1 | 5 | 4 | 0 | No |
| Matched nonoverlap | 0 | 8 | 7 | 0 | Ineligible |
| Matched nonoverlap | 1 | 4 | 3 | 11 | Ineligible |

All four ordinary final AP3P calls returned `ransac_failed`. Both positive top-ranked raw beams were wrong and both negatives produced a raw beam. The atlas arms returned those same raw beams through their fallback. These were research hypotheses. The experiment performed no acceptance calibration.

The 12 positive proposals had rotation errors from 51.63° to 164.26° and translation errors from 1,417.96 m to 5,184.87 m. The correctness limits were 2° and 19.41 m. The two renderer seeds repeat one geometry and illumination condition. They do not supply independent terrain evidence.

Six of the first positive's seven beams saw zero direct-atlas pixels, while the remaining beam saw 112,264 direct pixels and 100,886 combined pixels. The second positive's best-covered beam saw 46,580 direct pixels and 463 combined pixels. Some views therefore lack even direct support, while uncertain secondary transport further reduces coverage for others. Candidate accuracy is the immediate limit. Even a perfect reranker over these banks would recover 0/2.

## Protocol and verification

Four existing model-selection terrain groups received new camera variants. Their geometry passed the 8/8 exact-correspondence pose control and all planned nonoverlaps were certified. The worker rendered six native 512 × 512, 64-sample scene-linear EXRs. The model consumed opaque image inputs and reference geometry. The ranker received the blind pre-AP3P pose and estimated-Sun hypotheses and the intersection of both arms' validity masks. The scorer verified the sealed hashes and independently recomputed common coverage before opening query truth.

Geometry generation took 22.08 s, independent geometry audit 62.57 s and the six-exposure stage 92.71 s. Reference solar-table generation took 158.65 s, four blind calls took 41.79 s total and atlas preparation plus projection took 41.30 s. The original cumulative ledger charged the full 500 s and 1,200 s caps. It has 26,675 baseline and 13,493 training seconds remaining. The last 18 exposures, supplied-Sun evaluation and full ranking study did not run. Calibration and final reserves were untouched.

## Comparison and next hypothesis

In the separate same-worker benchmark, the compact matcher, SIFT, LoMa-R and RoMa v2 all recovered 0/2 changed-light poses. Historical Direct recovered 1/2 on those same inputs using a different CPU. On the larger historical cohort, Direct recovered 74/128 changed-light poses with 24 lighting templates and 46/128 when the alternate templates were withheld. These counts describe other cases. They motivate a proposal-stage test on the fresh cameras. [Same-worker comparison](SOLAR_MODERN_ARM_PILOT_V110.md) · [Withheld-light comparison](SOLAR_OFFCATALOGUE_DIRECT_V95.md)

Ray-traced shadow localization, direct render matching and modeling secondary illumination each have published precedents. The narrow hypothesis here is an incremental top-one pose gain from terrain-scattered irradiance over direct irradiance on the same blind candidate bank and common pixel support. The pilot measured why that comparison could not run on these inputs. The next test should ask whether a reference-map feasibility constraint during minimal-set pose generation, or a separate geometry-aware candidate source, can put a correct pose into the bank under changed light. [Shadow-based localization](https://publications.ri.cmu.edu/ray-tracing-and-use-of-shadows-as-features-for-determining-location-in-lunar-polar-terrain) · [Direct render matching](https://doi.org/10.1016/j.cja.2026.104149) · [Secondary-illumination reconstruction](https://doi.org/10.1016/j.actaastro.2025.09.068)

## Replay

Extract the [replay archive](solar-secondary-sentry-artifacts-v112.zip) at the repository root. It includes the six original EXRs, the scoring cameras and all blind banks, masks, ranks and receipts used by the scorer. The archive SHA-256 is `1dc64c4a3c9a0a284087bbc9e1d5b70298280b60f263cc032ad1504cf924e492`. The command below reproduced the [recorded metrics](solar-secondary-sentry-replay-v112.json) byte for byte from a separate clean checkout. It reads saved arrays and runs no matcher or renderer.

```sh
run=runs/solar-secondary-acquisition-v112
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python replay_solar_secondary_sentry_v112.py \
    --rank-seal-sha256 "$(sha "$run/blind-sentry/ranks/COMPLETE.json")" \
    --blind-complete-sha256 "$(sha "$run/blind-sentry/COMPLETE.json")" \
    --watchdog-sha256 "$(sha "$run/watchdog-sentry.json")" \
    --output "$run/replay-check.json"
```

[Frozen cohort](solar-secondary-ranking-roster-v112.json) · [Acquisition protocol](solar-secondary-acquire-dispatch-v112.json) · [Blind evaluation protocol](solar-secondary-sentry-dispatch-v112.json)
