# Five-method lunar pose comparison on one ARM worker

Reference-side render-and-match recovered **30 of 48** camera poses and
**19 of 32** under changed light indices. LoMa-R and RoMa v2 each recovered
15 of 48. The compact shared-Sun matcher recovered six, all on same-light
pairs. The geometry-aware control leads on this cohort. The frozen compact
control had zero changed-light successes. This comparison does not
measure the learned response head or establish an ablation gain.

| Matcher | Correct pose | Changed light | Strict pose | Wrong on overlap | Raw pose on nonoverlap | Median inference |
|---|---:|---:|---:|---:|---:|---:|
| Frozen compact, shared Sun and active patches | 6/48 | 0/32 | 0/48 | 2/48 | 6/48 | 6.731 s |
| SIFT | 11/48 | 0/32 | 1/48 | 0/48 | 0/48 | 0.0535 s |
| LoMa-R | 15/48 | 3/32 | 2/48 | 12/48 | 9/48 | 53.972 s |
| RoMa v2 | 15/48 | 5/32 | 4/48 | 29/48 | 43/48 | 51.918 s |
| Direct reference-side render-and-match | **30/48** | **19/32** | **6/48** | 2/48 | 0/48 | 0.7707 s |

The pose criterion is rotation error ≤2° and translation error ≤1% of
median reference range. The strict criterion is ≤0.5° and ≤0.1%. Every
eligible positive and its matched nonoverlap is in the denominator. There
were no operational failures. The nonoverlap counts are raw pose proposals
before any acceptance calibration.

Direct's correct-pose gain was **31.25 percentage points** over each
pretrained comparator. The descriptive paired 95% terrain-bootstrap
intervals were [12.50, 47.92] points over LoMa-R and [8.33, 52.08] over RoMa
v2. On changed light, the gains were 50.00 and 43.75 points, respectively.
These intervals resample eight development terrain groups and retain the
lighting variants with their camera. The 48 nonoverlaps reuse both terrain
endpoints, so zero observed raw outputs cannot establish a 1% error bound.
No calibration or final terrain was accessed.

All methods received the same native 512-pixel image content, camera
intrinsics, reference geometry for the same three-pixel AP3P solve, and
scoring rules. Query depth, pose and true query Sun were withheld from
inference. Direct uses a frozen library of 24 physical illuminations
rendered from the reference terrain. The catalogue covers the tested
lighting grid. A separate historical withheld-template experiment found
86/192 correct with 12 templates versus 114/192 with all 24. This sensitivity
matters for deployment on lighting outside a precomputed grid.
[Comparison protocol](SOLAR_FIVE_ARM_V115_PROTOCOL.md) ·
[Withheld-template study](SOLAR_OFFCATALOGUE_DIRECT_V95.md)

Direct's median model-resident inference was about **70× faster** than
LoMa-R and **67× faster** than RoMa v2 on the same two-thread ARM worker.
SIFT was about 14× faster than Direct. Including the input setup actually
incurred by the reused loader, the medians were 6.738, 0.0621, 53.986,
51.927 and 0.7823 seconds in table order. Direct also required 2,026.58
seconds of offline reference preparation for 16 reference cameras. Report
that cost separately when reference preparation cannot be amortized. The
worker's 3.80 GiB lifetime peak excludes child processes and cannot rank
model-specific memory.

## What this supports

The strongest measured result is that physical reference rendering makes
more correct pose proposals under changed illumination on these simulated
cases. Direct still recovered only 62.5% of positives before rejection and
12.5% at the strict threshold. It missed the 70% correctly accepted target
even under perfect rejection of its wrong poses. Real orbital accuracy,
calibrated risk, and an improvement attributable to the three integrated
compact modules are open. The full comparative study should keep the
frozen Direct control and separately measure whether new coarse
proposals bring enough accurate, distributed matches within reach.
[Saved-pose coverage study](SOLAR_SAVED_POSE_COVERAGE_V121.md)

## Verification

The original GitHub run sealed 480 of 480 method-case cells. Its official
Actions ZIP has 13,164,199 bytes and SHA-256
`813f2c9e6c5986ae711ab885d3a6c25a146cb87fead8261f99984e689f5e538b`.
The directory-mirror replay was rejected because a manifest symlink left
the audit directory. Using the exact public base ZIP resolved that staging
issue. The strict local x86 verifier then found a one-ULP difference in
one projected LoMa-R coordinate, (1.42\times10^{-14}) native pixels.
Independent model-free arithmetic reproduced all correspondence threshold
decisions and all pose classifications. The original artifacts and both
local replay failures were preserved. The unchanged verifier then passed
all 480 cells on ARM with the original wheel versions. That replay performed
no new matching, training, or model loading.
[Portable arithmetic receipt](solar-five-arm-results-v123/portability_recheck_v123.json) ·
[Replay source](solar-five-arm-results-v123/recheck_saved_rows.py) ·
[Selected input hashes](solar-five-arm-results-v123/replay_input_hashes_v115.json)
[Original benchmark](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/41941904269) ·
[Exact ARM replay](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/41964097603) ·
[Exact replay rows](solar-five-arm-results-v123/independent_replay_v115.json) ·
[Full paired table](solar-five-arm-results-v123/matched_comparison_v115.json) ·
[Replay data and commands](solar-five-arm-results-v123/README.md)
