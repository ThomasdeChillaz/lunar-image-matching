# Low-light render-and-match proposal results

The exact projection change made the frozen experiment fit its budget.
On the generated timing control, the cold pass fell from 171.29 ms to
130.59 ms. All five saved cell files, scores and rank orders matched
byte for byte. The conservative full forecast fell from 193.48 s to
178.30 s. The archived study took 70.52 wall seconds through completion
and independently replayed all 32 cells. It retained all 16 render-seed
pairings, both detector scales, the original camera bank and the
original controls.
[Method and parity tests](METHOD.md) ·
[Timing receipt](timing-receipt-v125.json) ·
[Sealed archive completion](archive-complete-v125.json)

## A good image-space initialization can still be a poor camera pose

At the primary 32-pixel detector scale, the full bank contained a
supported proposal within 10° and 10% of reference range on every seed
pair of the faint case and bright control. None of the five rankers put
such a proposal in its top eight. At the 16-pixel detector scale the
coarse result was also zero. The original advancement gate failed.

| Development condition | 3D top-eight coarse basin | Plane top-eight coarse basin | Six distributed anchors in one 3D proposal at ±16 px | Same anchor check with plane |
|---|---:|---:|---:|---:|
| Faint u061 | 0/4 | 0/4 | 4/4 | 4/4 |
| Bright u013 | 0/4 | 0/4 | 4/4 | 4/4 |
| Uniform u025 | 0/4 | 0/4 | 0/4 | 0/4 |
| Weak u027 | 0/4 | 0/4 | 0/4 | 0/4 |

The radiance gradient and reference-order controls reached zero
distributed-anchor cases at ±16 px. A separate support-only ranking,
using the weakest cell count and total support, also reached zero at
±16 px and zero coarse basin cases across the full archive. The uniform
observation in u025 produced no eligible ranked candidates at either
detector scale. [Support-only projection check](support-only-native-capture-v125.json)

For the faint and bright conditions, 3D and plane selected the same
first camera for all four seed pairs at the primary scale. That camera's
errors were 14.54° and 24.43% of reference range in the faint case, and
25.79° and 47.26% in the bright control. Yet its projections brought at
least six distributed anchors inside the nominal search window in each
pair. The score can therefore prefer a useful image-space initialization
while choosing a camera outside the coarse pose tolerance. The
projection check uses sealed scoring geometry and does not measure
whether an actual full patch is supported, distinctive or matched
accurately. The four seed pairs reuse one camera pair per condition.

The archive used matched reference illumination. The result supplies no
3D-specific advantage over the fitted plane, native pose recovery,
estimated-Sun gain, or calibrated acceptance. The archived native
matching follow-up was not admitted by the original gate. Calibration
and final terrain stayed closed.
[All coarse rows and the original gate](archive-independent-replay-v125.json) ·
[Capture definition](../solar-warp-handoff-v120/README.md)

## What the modern comparison says about the next change

Across the five same-worker methods, even a hindsight selector of
correct saved poses recovers 32/48 positives, short of the 34 needed for
70%. The historical MatchAnything–EfficientLoFTR results on the same
pinned cases recovered 9/48 and 0/32 changed-light positives. They add
zero successes to the union. That join compares raw accuracy across
separate workers. It cannot compare latency.

Of the 12 changed-light misses, nine had fewer than six four-pixel
correspondences from any method. Three merit a separate check of spatial
coverage and camera conditioning. In one of them, RoMa v2 supplied 759
two-pixel correspondences but its translation error was 16.26 m against
the 13.72 m limit. The evidence points to both missing observations and
difficult pose estimation. A distinct next experiment would use the plane
initializer to place search windows, then select patches at different
terrain heights to distinguish competing camera poses. Compare that
selection against the same number of evenly distributed patches and
require the common estimator to recover the pose from measured native
correspondences. This is a separate hypothesis and requires its own freeze.
[Saved-pose diagnostic](../solar-five-arm-results-v123/saved-pose-diagnostics-v125.json) ·
[Five-method benchmark](../SOLAR_FIVE_ARM_RESULTS_V123.md)

## Reproduce the archive analysis

The development replay ZIP has **83,858,605 bytes** and SHA-256
`e2a5d610f591301348701c3483ed82d3ad87c8b6f3a4b467ad347738e453688f`.
It includes the original completion and prediction seals, score and
replay JSON, all 32 detector-cell arrays, and the two post-seal
diagnostics. Input geometry comes from the v109 research release. The
replay does not load the EXR pixels or run the matcher.
[Development replay archive](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-warp-v125-development) ·
[Original v109 input release](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-raw-radiance-v109)

From a clean repository checkout, verify and extract both archives into
their repository-relative locations. Use the pinned geometry paths in the
two original rosters. In the validated Python environment, from the repo root:

```sh
archive=runs/solar-warp-v125/archive
input=reports/solar-warp-v117
out=$(mktemp -d)
digest() { shasum -a 256 "$1" | cut -d ' ' -f1; }
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD" \
  python "$input/verify_warp_archive_v117.py" --repo "$PWD" \
  --inputs "$input/inference-inputs.json" --inputs-sha256 "$(digest "$input/inference-inputs.json")" \
  --labels "$input/scoring-inputs.json" --labels-sha256 "$(digest "$input/scoring-inputs.json")" \
  --predictions "$archive/predictions.json" --predictions-sha256 "$(digest "$archive/predictions.json")" \
  --scores "$archive/scores.json" --scores-sha256 "$(digest "$archive/scores.json")" \
  --output "$out/independent-replay.json"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD/reports/solar-warp-v117:$PWD" \
  python reports/solar-warp-v125/support_audit.py "$archive" \
  --predictions-sha256 "$(digest "$archive/predictions.json")" \
  --replay-sha256 "$(digest "$archive/independent-replay.json")" \
  --output "$out/after-seal-support.json"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD" \
  python reports/solar-warp-handoff-v120/capture_audit.py \
  --repo "$PWD" --archive "$archive" --protocol reports/solar-warp-v125/archive-dispatch.json \
  --inputs "$input/inference-inputs.json" --labels "$input/scoring-inputs.json" \
  --completion-sha256 "$(digest "$archive/COMPLETE.json")" \
  --output "$out/after-seal-native-capture.json"
```
