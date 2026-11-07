# From a coarse pose proposal to measured native image matches

The original 3D radiance experiment scores a near-correct candidate within
10° and 10% of range. That is a useful coarse diagnostic. At the native focal
length of about 443 pixels, pure 10° yaw moves an image-centre point about
78 pixels. A sideways error of 10% of range moves it about 44 pixels. A
candidate that passes the coarse gate can still sit beyond the current
16-pixel native search.

## Mechanism and controls

Use the first eight seeds from the original 32-pixel 3D ranking. For each
seed, project the reference camera's visible radiance and native 3D surface
into the hypothesized detector. The projection rejects native depth
discontinuities, keeps the closest supported surface and exposes holes.
The matched plane control uses the same fitted normal and anchor-centred
planes. Both models share exact detector support, anchor centres and
reference 3D points. The plane control can lose support at boundaries
between local planes, so those regions are excluded from both arms.

Each seed has 16 fixed spatial anchors. Match a 64-pixel scene-linear window
with reciprocal normalized correlation, initially at an 8-pixel stride,
then at a 1-pixel stride. Reject weak, ambiguous, boundary or nonreciprocal
peaks. Compare wide 3D search at ±128 pixels, narrow 3D search at ±16 pixels,
and wide fitted-plane search. The wide and narrow searches evaluate at most
2,628 and 500 positions per anchor respectively, counting both directions.
The seed and warp supply only a search centre and a template. The observed
query location comes from the patch search. Require at least six distinct
measurements across at least three regions of both images before invoking
the original 3-pixel AP3P estimator. The same spatial check applies to the
resulting inliers.

## Verification and scientific scope

The native 512-pixel procedural control uses a planar scene and an exact
40-pixel translation. Both wide arms recover 16 of 16 measured offsets and
the camera translation. The narrow arm returns zero measured offsets and
no pose. This isolates the effect of the search radius. A planar fixture
cannot show an advantage for the 3D template. The separate projection tests
check the 3D versus plane intervention, missing data, discontinuities,
occlusion and out-of-frame anchors.

Before any archived handoff, the original 32-cell proposal archive must
pass its own independent replay and gate. The geometry-only capture audit
then measures 16-, 32-, 64- and 128-pixel residual coverage after predictions
are sealed. Report visibility and missing observations separately and
require at least six anchors in three regions from one actual candidate.
A union across candidates cannot qualify. Query truth scores reachability
after the fact and is outside all matching inputs.

If reachability and measured timing permit a bounded handoff run, retain
all renderer-seed pairings and report raw pose recovery, wrong and missing
poses, measured pixel errors, coverage and complete runtime for the three
arms. The archived feasibility test supplies matching illumination. A
separate equal-information estimated-Sun run with the fixed clean16
nonoverlap cases is required before interpreting relighting or rejection.
No archived handoff was executed here.

Map rendering with masked correlation, prior-guided terrain matching,
and multiscale feature alignment are established. The narrower hypothesis
is that faint lunar radiance can produce enough independently observed,
distributed native offsets to recover a pose after ordinary matching
runs out of points.
[Lunar render-and-match](https://doi.org/10.1016/j.cja.2026.104149) ·
[Geo-LoFTR](https://arxiv.org/html/2502.09795v3#S3.SS4) ·
[PixLoc](https://arxiv.org/abs/2103.09213) ·
[Archived u061 radiance evidence](../SOLAR_RAW_CROSSVIEW_V110.md)

From the repository root in the validated environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD" \
  python -m pytest -q reports/solar-warp-handoff-v120
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD" \
  python reports/solar-warp-handoff-v120/smoke_handoff_v120.py
```
