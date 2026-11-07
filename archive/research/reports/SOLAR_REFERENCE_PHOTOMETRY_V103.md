# Can two reference renders predict a withheld Sun angle?

A two-reference synthesis improved same-view contrast prediction over the nearest even-light template in 289 of 342 scored conditions. All eight terrain groups favored synthesis on their median condition. This is a diagnostic for the separate supplied-Sun track. The original 512-pixel reference camera stays fixed, and the method receives the target Sun direction and even-light reference renders. Odd-light reference pixels serve only as the scoring target. The runs used a single simulator and the already observed development terrain.

The public LIMA description combines co-registered reference images under known lighting. Our implementation isolates the horizontal Lambertian component and compares it with nearest-reference selection under identical intensity conversion and stretch. Both compute a native contrast image. The primary metric is median absolute signed-gradient error across 25 × 25 windows whose withheld raw image has at least 5% of adjacent edges differing by at least two grayscale codes. [LIMA paper](https://doi.org/10.2514/6.2025-2073) · [Technical description](https://patents.justia.com/patent/20260219064)

| Conditions | Scheduled | Valid angular bracket | Scored on primary metric | Wins / ties / losses | Median signed-gradient error, nearest → synthesis |
|---|---:|---:|---:|---:|---:|
| All odd lights | 384 | 352 | 342 | 289 / 1 / 52 | 1.091 → 1.010 codes |
| Lights 1 and 17 used in the pose pilot | 64 | 64 | 56 | 53 / 0 / 3 | 1.125 → 1.039 codes |

At lights 1 and 17, median whole-image contrast error fell from 6.07 to 3.02 codes. Median linear-domain absolute error fell from 0.00308 to 0.00099. The median paired reduction in local gradient error was 0.096 codes. Terrain-specific paired medians ranged from 0.034 to 0.771 codes. The 32 conditions at light 23 had no valid angular bracket. Ten other conditions lacked informative raw windows, including eight at lights 1 and 17. Views, patches and lighting variants share terrain and should not be counted as independent tests.

In the 6,023 low-signal windows at lights 1 and 17, stretching pushed 4,196 measured target windows and 4,196 nearest-template windows over the texture threshold. Synthesis pushed 4,533 over it. Additional contrast can amplify faint structure, and these counts cannot distinguish real features from artifacts. The diagnostic opened only sealed reference-role images. It estimates neither geometric correspondence quality nor pose accuracy.

## Implications for the next pose experiments

The 48-case supplied-Sun pilot remains the direct check of whether improved templates produce correct poses under viewpoint change. Its two arms have the same matcher and pose-solve budgets. Results will retain all scheduled cases and separately report common angular support. The main estimated-Sun track remains separate.

The exposure pilot addresses a different mechanism. Its existing 24 positive cases include eight distinct conditions with fewer than six saved matches across the entire reference library. Six were near-black, with a query 95th-percentile grayscale code of at most three and zero saved matches. Four of the eight retained at least 230 geometrically available patch centres. A net gain of three correct positives would be the first achievable increase of at least ten percentage points on this pilot. Such a result would still require the larger paired evaluation.

Across the 192 full-library positives, the 70% target requires at least 135 correct poses, 21 above the 114 already recovered before rejection. If a method helped only the low-signal failures where both images were near-black and changed no other outcomes, it would need to recover at least 21 of those 33 scheduled failures. Those 33 rows correspond to 21 distinct conditions on four terrain groups. Separately, the 192 nonoverlap rows repeat eight endpoint pairs, with at most four terrain-disjoint pairs. Zero raw nonoverlap poses in that design does not establish a 1% error bound.

## Reproduction

The source and metadata-only protocol were hashed before scoring. The output includes all 384 case rows and the grouped summaries. Independent arithmetic replay reproduced the result counts and median paired differences without decoding the images again.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python check_reference_photometry_v103.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python diagnose_reference_photometry_v103.py evaluate \
  --freeze reports/reference_photometry_protocol_v103.json \
  --freeze-sha256 f5dc31b425d5d4c8a0f10b76c38bcfaf9a6d6728cffbbeff6260089f8fe2912e \
  --output /tmp/reference_photometry_v103-replay.json
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python audit_solar_recovery_opportunity_v103.py \
  --output /tmp/solar_recovery_opportunity_v103-replay.json
```

[Reference protocol](reference_photometry_protocol_v103.json) · [Full photometry rows](reference_photometry_v103.json) · [Saved exposure and endpoint audit](solar_recovery_opportunity_v103.json) · [Supplied-Sun pilot context](SUPPLIED_SUN_CONTROL_V102.md)
