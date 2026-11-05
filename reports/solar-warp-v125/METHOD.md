# Faster reference projection for the low-light pose proposal study

The five-method benchmark recovered 32/48 positives when hindsight selected
any saved correct pose, short of the 34 needed for 70%. Among 12 changed-light
pairs missed by all five methods, nine had fewer than six correct native
correspondences within four pixels in every method. The selected faint
`u061-l00-17-positive` condition had zero. Three other changed-light failures
did have six or more two-pixel correspondences in at least one method.
They warrant separate checks of spatial support and pose conditioning.
[Exact five-method replay and saved arrays](../solar-five-arm-results-v123/README.md)

The archived low-light experiment asks whether shared detector-cell radiance
can prioritize a camera near the true pose when keypoint proposals run short.
It isolates viewpoint by supplying the known reference illumination. Published
terrain-render matching and ray-traced shadow matching already use physical
image predictions. This experiment compares 3D, a fitted reference plane and
a broad radiance trend on the same supported subbins.
[Original study](../solar-warp-v117/ARCHIVE_GATE.md) ·
[Lunar render-and-match](https://doi.org/10.1016/j.cja.2026.104149) ·
[Lunar shadow localization](https://publications.ri.cmu.edu/ray-tracing-and-use-of-shadows-as-features-for-determining-location-in-lunar-polar-terrain)

## Projection implementation

For a source point, the four detector bins in its footprint depend only on its
integer footprint origin. Points sharing that origin therefore have the same
affected bins. First keep the closest axial depth and the earliest original
source index on a tie at each origin. Then apply the same minimum to the four
adjacent origin winners for each output bin. An empty origin cannot win.
This preserves occlusion, clipping, depth precision and tie order while
avoiding four copies of each point and a global sort per detector map.
[Implementation](fast_projection.py) · [Parity controls](test_fast_projection.py)

The original pose bank, reference sampling, photometric scores and ranking
code are reused. The timing gate keeps the 60-second procedural allowance,
the 180-second archive cap, all 16 renderer-seed pairs, both native detector
scales, and up to 2,048 poses per pair. The earlier 193.48-second refusal is
historical evidence. [Original timing result](../solar-warp-v117/TIMING_RESULTS_V123.md)

## Attribution after replay

The additional support control ranks eligible proposals by the count in the
weakest detector cell, then total count, then reference order. It ignores
radiance scores. Compare its top-eight basin recall with the 3D, plane,
gradient and reference-order arms. The audit builds the support ranking
before opening scoring geometry, and checks the independent replay and
the sealed cell digests. It supplements the original stage gate.
[Support control](support_audit.py)

Known-light coarse capture on these exposed development cases cannot
establish native correspondence accuracy, estimated-Sun robustness or a
learned pose gain. Any subsequent handoff test requires separately measured
image correspondences and the shared three-pixel pose estimator.
