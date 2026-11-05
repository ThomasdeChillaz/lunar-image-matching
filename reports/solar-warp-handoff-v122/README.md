# Prospective patch reachability probe

The v120 geometry audit measures centre displacement. The patch matcher additionally requires its full 64-pixel crop, fixed valid template pixels, and feasible forward and reverse search grids. This probe estimates an upper bound on the anchors that could satisfy those requirements. It uses the actual fractional centre and the shared 3D and plane detector mask. It reuses the pinned v120 layout and grid code.

Call `prepare_support` with one hypothetical camera's projected centres, shared anchor visibility, common reference mask, query detector mask, and original reference anchor coordinates. The query detector mask may describe observed bad pixels or assumed complete detector support. Never build it from query geometry. Record `support_digest` alongside the study's original prediction seal before opening scoring geometry. Then call `score_after_seal` with that expected digest and a loader that returns exact native truth coordinates and co-visible flags. The function verifies the support digest before opening truth. A future experiment wrapper must also authenticate the original prediction seal and the independent scoring assets.

Each witness uses one stride-eight seed and one stride-one offset in both directions. It verifies full native crops, every selected fixed sample, supported cardinal neighbours, available rivals, and the original one-pixel reciprocal tolerance. Forward offsets must land within three native pixels of the independent truth. The three-pixel comparison measures point reachability and does not certify a camera pose. The report checks whether six anchors from one supplied candidate could cover at least three quadrants in each view. On the query side it uses the union of in-frame lattice quadrants within each reachable anchor's truth envelope. That union can count one anchor in multiple quadrants or include offsets without a feasible reciprocal path. Truth-centre quadrant counts are reported separately. Combining witnesses across candidate cameras is outside the criterion.

The probe receives no radiance and cannot test correlation, texture, ambiguity margins, or whether matching actually chooses a witness. The reported geometry remains an upper bound even when six anchors qualify. This is a separate prospective check. It does not change the v117 or v120 gates, matcher, or results.

Run the focused generated fixtures from the repository root in the validated Python environment.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest -q reports/solar-warp-handoff-v122/test_capture_patch_probe_v122.py
```
