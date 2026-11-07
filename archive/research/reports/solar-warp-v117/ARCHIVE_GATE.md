# Can coarse radiance propose a useful camera pose?

Direct relighting missed the six-match minimum in 38 of 39 distinct no-pose
development conditions. The later blind pose banks contained no correct
candidate for reranking. One extremely dim condition, u061, still showed
repeatable same-light structure at 16 and 32 pixels across the two cameras.
The proposed method uses that structure to search cameras before requiring
repeatable keypoints.

Project scene-linear reference radiance and native axial depths from the
reference camera into a fixed reference-derived camera bank. Score the
predicted and measured detector-cell averages on common fixed support. Run
the same camera list and score for a robust fitted reference-plane control and a fitted
affine brightness-trend control. The first stage uses a reference rendered at
the known query light to isolate view recovery. In the primary future matcher
the Sun will have to be estimated from a common reference library.

The input rosters were prepared from the sealed v110 protocol, with all 16
native EXRs and relevant geometry bytes hash-checked. Only the reference
geometry, image bytes and query intrinsics enter the proposal worker. Query
pose geometry is retained in a separate scoring roster and opens only after
the full prediction file is sealed. Opaque exposure IDs associate the two
rosters. No archived images have been decoded for this proposal.

## Admission and decisive outcome

1. Finish and archive the original live five-method comparison. Keep the
   original compute ledger and apply a separate two-thread reservation only
   if the new experiment qualifies.
2. Run a 32-camera procedural timing pilot. Freeze its actual measured cost,
   the exact source and two roster digests, and a single bounded wall-time
   cap before archived matching. The initial candidate cap is 180 seconds
   with no rendering and no new training. Refuse admission if the pilot
   does not cover all 16 pairings, both detector scales and all controls
   within that bound.
3. In the archive check, keep every combination of the two reference and
   two query seeds for u061, the bright u013 control and the two weak-signal
   u025/u027 controls. Require a proposal within 10° and 10% of median
   reference range among the top eight on all four u061 pairings at the primary 32-pixel
   scale. Report the 16-pixel results separately. Report whether the full bank
   and supported subset contain such a camera. Include a fixed prior-only
   order among eligible candidates, and require the bright
   control to recover on all four pairings. Report both alternate arms
   at exactly the same operating point and stop if they explain the effect.
   These loose limits measure coarse pose proximity. They do not guarantee
   that projected anchors fall within the 16-pixel native refinement radius.
   The principal native pose threshold remains 2° and 1%.
4. Stop on the bright control failure, a missing coarse basin, dependence
   on one renderer seed, or no advantage over both controls. If the stage
   qualifies, freeze the eight-case clean16 matched follow-up, including its
   four nonoverlaps and a fixed even-indexed reference lighting library.
   Require independently measured native correspondences and the unchanged
   3-pixel AP3P solve for any final pose. Template projections do not count
   as their own measurements. A new nonoverlap output or loss of the effect
   on clean16 disqualifies the proposal for broader testing.

These cases are exposed development data. The first stage tests whether an
idealized, known-light reference can put the camera search near the correct
solution. It does not establish sensor detectability, generalization, robust
relighting or a compact model improvement. The two render seeds per view
measure repeatability within a condition, with no independent geographic
replication.

Terrain-render localization and using ray-traced shadows as navigation
observations have published precedents. The incremental hypothesis is that
coarse aggregate radiance can create usable pose candidates on inputs where
point matchers produce too few correspondences, and that the effect requires
3D geometry beyond a plane or a broad lighting gradient.
[Terrain-render comparison](https://doi.org/10.1016/j.cja.2026.104149) ·
[Shadow-localization precedent](https://publications.ri.cmu.edu/ray-tracing-and-use-of-shadows-as-features-for-determining-location-in-lunar-polar-terrain)
