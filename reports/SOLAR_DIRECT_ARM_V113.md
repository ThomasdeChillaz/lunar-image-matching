# Direct relighting qualifies on the matched Linux ARM cases

Direct recovered three of four positive poses, including one of the two changed-light poses. It returned one wrong positive pose and zero poses on the four nonoverlap cases. Independent saved-array replay verified all eight fresh rows. The raw pose outcomes reproduce the earlier Direct result on these same cases. [Successful ARM run](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/41938284734) · [Independent replay](solar-direct-arm-v113-independent-replay.json)

| Method | Correct raw positive poses | Changed-light positives correct | Wrong positive poses | Raw poses on nonoverlap | Median inference |
|---|---:|---:|---:|---:|---:|
| Direct, 24 reference illuminations | 3/4 | 1/2 | 1/4 | 0/4 | 0.773 s |
| Frozen compact | 1/4 | 0/2 | 0/4 | 0/4 | 6.82 s |
| SIFT | 2/4 | 0/2 | 0/4 | 0/4 | 0.057 s |
| LoMa-R | 1/4 | 0/2 | 1/4 | 1/4 | 53.23 s |
| RoMa v2 | 1/4 | 0/2 | 2/4 | 3/4 | 52.14 s |

Direct ran on a separate GitHub worker in the same two-thread Linux ARM64 software and hardware class as the four earlier methods. The two runs used the same native images, case order, reference geometry and final pose criterion of 2° rotation and 1% of the median reference range. Direct searches a simulator-matched, pre-rendered reference library. No method received the query Sun direction. Model loading, offline reference-library preparation and orchestration are separate from median inference. Four positive cases provide a timing and execution qualification. The counts cannot establish a general ranking or calibrated reliability. [Original four-arm run](SOLAR_MODERN_ARM_PILOT_V110.md) · [Frozen Direct protocol](solar-direct-arm-v113/pilot-protocol.json)

## The larger matched experiment

The fresh Direct control passed its known translation, blank image, invalid support and query-Sun rejection checks. The parent watchdog completed in 89.72 seconds, including 62.16 seconds of setup. The slowest complete measured Direct case took 1.077 seconds. The original ledger retained the full 1,800-second reservation and has 24,695 baseline seconds remaining.

Combining measured Direct and prior modern costs gives a conservative projection of **17,322 seconds** for 96 cases per method, or 480 method–case rows. The forecast includes observed startup and control costs and applies the original 1.5 multiplier. This is the largest eligible predefined cohort within the 18,000-second comparison cap. A separate freeze and fresh same-worker execution are required for that five-arm comparison. [Measured admission forecast](solar-direct-arm-v113-measured-forecast.json) · [Fixed cohort definitions](solar_matched_arm_cohorts_v103.json)

## Replay

The [replay archive](solar-direct-arm-v113-replay-artifacts.zip) has SHA-256 `1a89758991addca5cfa151ec8a5576947d450678f2a6e5f1a3ef7740dd5dde06`. Extract it at the root of a clean checkout. Its `solar-direct-arm-v113-replay-inputs.json` identifies the Git-tracked metadata and exact `payload/` members to restore from the [public base release](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/download/solar-arm-transport-v98/solar-arm-v98-public.zip). Verify each listed digest. The [Direct library supplement](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-arm-direct-v113-transport) is needed for model execution. Saved-array replay uses the base inputs and reads no model or render library.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python verify_solar_matched_arm_v113.py runs/solar-direct-arm-v113/pilot \
    --protocol reports/solar-direct-arm-v113/pilot-protocol.json \
    --output runs/solar-direct-arm-v113-independent-replay-check.json
```
