# Structured reference-mesh adapter for the secondary-light atlas

This prototype prepares a reference-side atlas input from the same structured triangle layout used by the project's v59 camera scene. The terrain arrays in a v59 mesh are already local east, north and up metres. The stored body origin and ENU basis identify that frame. Candidate cameras and proposed Sun vectors must use it. There is no coordinate transformation or access to query pose, query depth, true query Sun, or any camera image in the adapter.

## Interface

- Pass the ordered vertices and faces to prepare_reference_mesh along with a first-hit callback over those exact full-resolution triangles. The callback returns a positive hit distance or None within the supplied ray bound. Inside the established Blender runtime, prepare_blender_reference constructs the existing mathutils BVH directly from the same arrays.
- The adapter validates topology and conservative interior coverage with the existing v63 grid-envelope routine. It partitions structured cells by deterministic row and column cuts. It sums original triangle areas and oriented area normals. Each representative point is an actual triangle centroid nearest the area-weighted horizontal centre. Self-offset rays use that original triangle's normal. An incidence disagreement at the selected facet and coarse patch is marked unknown.
- The candidate-independent mutual table is symmetric. It uses the v111 segment classifier with the reference map's coverage and preserves unknowns. A blocker seen before coverage loss certifies blocked. Clear requires both directions to certify the segment. Nonfacing patch pairs have zero exchange weight and receive no visibility label. The receiver–emitter table is reused across candidate Sun directions.
- Call binding.bind_sun with each proposed above-horizon Sun vector. It calls the existing v63 solar-ray classifier and returns the v111 Atlas, ray-state counts and the number of attempted reference solar rays. Feed this atlas to project_atlas with a candidate camera. Neither the adapter nor the atlas derives hypotheses from query truth.

## Coverage contract and deterministic bounds

The default has an open exterior. The observed map maximum is never treated as a ceiling for the unobserved scene. The adapter can record in-map terrain hits and supported mutual paths. It keeps secondary transport uncertified while outside contributors are unresolved.

A prospective finite simulator control can declare FiniteScene(mesh_array_sha256(vertices, faces), maximum_height_m) only after the protocol establishes that these exact triangles are the entire modeled scene, including all emitting and occluding terrain, with vacuum beyond the mesh. The declaration pins the node and face arrays and rejects an invalid ceiling. For proposed Sun rays, the adapter then expands the supported vacuum to the bounded casting region. A cropped map must never self-certify as the entire scene. The Blender convenience binding guarantees that the callback is built from the same arrays. A separately supplied callback is the caller's responsibility.

The default and maximum are 16 tile groups per axis, at most 256 patches. This permits at most 65,280 directed mutual ray attempts once per mesh and at most 256 solar rays per proposed Sun. All rays are bounded to at most 50,000 metres. Candidate count and wall-time admission belong to the prospective evaluation protocol.

## Procedural verification

Run from this directory with the repository and published v111 atlas on PYTHONPATH.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=../..:../solar-secondary-atlas-v111 \
      python -m unittest test_solar_secondary_mesh_v112 -v

The tests check exact representative placement and area on a nonuniform grid, the unknown-exterior default, the explicit finite-scene path, observed ridge shadows, mutual coverage and ray caps, and certified secondary radiance on directly dark valley patches. They use a tiny test-only triangle intersector. No real lunar file, renderer or scientific inference is used.

Coarse patches approximate original geometry and albedo, and the v111 atlas marks close facing pairs unresolved when differential transport is unreliable. Patch normal aggregation and tangent-square rendering can reduce supported detector coverage. Actual coarse coverage, numeric robustness of the pinned Blender BVH, runtime and pose performance require a separate admitted lunar development pilot.
