# Reference-only terrain-scattered light atlas

This prototype predicts low-frequency secondary radiance from a finite, coarse reference terrain. The output is a relative Lambertian prediction, with no measured sensor response or camera-image input. The implementation is untrained and accepts up to 1,024 reference patches.

## Reference inputs and output

- Construct TerrainPatches with local world centres and outward normals, positive square-metre surface areas, diffuse albedo, a Boolean mapped mask and a Boolean transport_coverage mask. Set transport_coverage only when the declared finite scene supports the relevant one-bounce contributors for that receiver. A missing exterior cannot silently become black space.
- Supply build_atlas with a proposed Sun direction and frozen reference-ray states. The encoding agrees with the v63 solar states. Unknown is 0, clear is 1 and blocked is 2. The mutual matrix must be symmetric with a blocked diagonal. For two separately cast segment directions, a certified blocker is sufficient for blocked and both directions must certify clear for clear. Otherwise use unknown.
- classify_segment checks an already self-offset open segment against a first-hit callback and a distance to the first unsupported interval. A known blocker before the gap is sufficient. No hit across a gap remains unknown.
- Direct irradiance uses solar incidence and certified visibility. The one-bounce transport integrates approximate emitter radiance with emitter area, receiver and emitter cosines, inverse-square falloff and clear mutual visibility. It converts received irradiance to Lambertian outgoing radiance. Known partial contributions remain inspectable. Close facing patches outside the differential approximation and implausible form-factor totals remain uncertified.
- Call project_atlas with a candidate camera-to-world pose and native intrinsics. It intersects native integer-centred detector rays with tangent squares. The nearest patch owns a pixel even when its radiance is uncertified. A depth tie keeps the lower patch index. Pass an optional reference-only camera_covered mask when additional ray coverage information is available. Result arrays include native radiance, explicit validity, closest patch IDs and known versus projected pixel counts per 32-pixel block.
- The v111 ranker consumes np.stack([item.radiance for item in projections]) and np.stack([item.valid for item in projections]). It computes candidate-common support from these native arrays. Do not substitute each candidate's independently averaged blocks or use query truth to compute coverage.

## Reproduce the procedural checks

Run from this directory using Python with NumPy installed. The second command requires the project checkout with published commit 375bf6afe0d0416c6e8ea0c584e37dcddef882f3 available as an ancestor of its local origin/main tracking ref. It loads the ranker directly from that immutable commit, then combines both components without reading project imagery, camera files, calibration terrain or final terrain. An explicit full commit can also be supplied with --ranker-commit.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m unittest test_solar_secondary_atlas_v111 -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python check_atlas_ranker_v111.py --repo-root ../.. > procedural_composition_replay.json
    cmp procedural_composition.json procedural_composition_replay.json

Eight unit fixtures cover a direct-blocked illuminated reflector, zero solar input, unresolved near exchange, off-frame projection, ray coverage gaps, unknown transport inputs, depth ties and unknown foreground occlusion. The composition fixture builds a 65-patch scene and independently synthesizes a 128 × 128 observation of its one-bounce field. Two candidate poses differ by 0.75 metres, with the wrong pose first in the original order. Both published ranking modes place the correct pose first using all 16 common blocks.

The tangent-square surface and point-to-patch transport are coarse approximations. Tessellation gaps remain unsupported. The candidate camera's external visibility depends on supplied reference coverage. Real radiometric validity, lunar pose performance, terrain-resolution sensitivity and runtime at the maximum patch count require separate experiments.
