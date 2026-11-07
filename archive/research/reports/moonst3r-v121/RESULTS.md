# MOONSt3R on eight prospectively selected development cases

MOONSt3R recovered two of four positive camera poses. Both had the same
lighting index. It returned two wrong poses on overlapping terrain and
three raw poses on four matched nonoverlap pairs. Direct recovered all
four positive poses, including both changed-light cases, and returned
zero raw nonoverlap poses.

| Matcher | Correct pose | Changed light | Strict pose | Wrong on overlap | Raw pose on nonoverlap |
|---|---:|---:|---:|---:|---:|
| Frozen compact | 1/4 | 0/2 | 0/4 | 0/4 | 0/4 |
| SIFT | 2/4 | 0/2 | 0/4 | 0/4 | 0/4 |
| LoMa-R | 2/4 | 0/2 | 0/4 | 1/4 | 1/4 |
| RoMa v2 | 2/4 | 0/2 | 1/4 | 2/4 | 3/4 |
| Direct | 4/4 | 2/2 | 0/4 | 0/4 | 0/4 |
| MOONSt3R | 2/4 | 0/2 | 0/4 | 2/4 | 3/4 |

The first five rows come from the exact ARM replay of the same case IDs.
The MOONSt3R row uses the official lunar-trained descriptors with the
shared native reference geometry and three-pixel AP3P estimator. Query
depth, pose, and Sun were absent from inference. The raw pose criterion
was ≤2° and ≤1% of median reference range. The strict criterion was ≤0.5°
and ≤0.1%. All methods completed every selected case. This is a small
qualification pilot on four development terrains. It does not rank the
methods, estimate accepted-pose risk, or establish generalization.
[Full parent study](../SOLAR_FIVE_ARM_RESULTS_V123.md) ·
[Fixed roster and training overlap disclosure](README.md)

The fresh 512-pixel translation control returned 3,973 inliers with a
zero median native-pixel error. Blank and all-invalid inputs returned no
matches. The two-thread x86 run took 486.47 wall seconds under the
one-use 1,800-second reservation. Its median local inference was 54.41
seconds and lifetime worker peak was 5.72 GiB. The other five profiles
ran on Linux ARM. The timing and memory measurements cannot compare
hardware or rank model efficiency.
[Exact case results and receipt](same-case-results.json)

All eight saved cells replayed independently. Download
`moonst3r-v121-replay-data.zip` from the development release and verify
its SHA-256 before extracting it in a replay checkout. The unchanged
`verify_solar_evaluation_v98.py` checks the saved arrays against the
original input and label digests. Use `comparison-protocol.json` from
this directory and the pinned source and input paths. The v121 original
admission and watchdog records remain separate from the replay archive.
[Verified development release](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-five-arm-v115-results) ·
[Prospective protocol](comparison-protocol.json)
