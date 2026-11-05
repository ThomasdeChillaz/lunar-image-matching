# Reference radiance warp proposal prototype

This prototype ranks a fixed bank of hypothetical query cameras using a reference radiance image and its native axial depth. The caller chooses a reference image under the known illumination for this ceiling test. The only query inputs are scene-linear luminance and intrinsics. The code imports the audited axial projection, local ENU look-at and roll helpers. It does not read images, select an archived case, score against pose truth, or infer the Sun.

## Proposal and controls

1. Build at most 2,048 poses with `reference_pose_bank`. The original reference camera comes first. A fixed six-axis Halton sequence draws camera translations, look targets and roll relative to that camera. The bounds follow the simulated camera family. They do not represent a verified spacecraft attitude or position prior. The bank uses NumPy only.
2. Build at most 4,096 points with `build_reference_cloud`. Keep the native integer anchor centres and depth-consistent scene-linear tile means. The flat control uses the same rays and radiance, intersected with a robust elevation plane fitted from the reference map. Depth with unknown photometry still enters the occlusion calculation.
3. Call `rank_pose_bank` on the same poses at 32 native pixels per cell, then at 16 as a separate sensitivity. Project points into a fixed stride-four detector raster with a bounded 2 × 2 subbin footprint and keep the nearest positive axial depth in each subbin. Within each cell, the observation, 3D prediction and fitted-plane prediction are averaged over the exact same subbins. All arms require the same minimum intersecting support in every fixed cell. The default region is the central half of the detector in each axis, with at least two required cells in each of at least three quadrants. An explicit fixed mask can be supplied before looking at the query.
4. Compare 3D terrain, the fitted reference plane and the affine detector trend fitted to the 3D prediction. The score fits a scene-wide positive gain and offset and weights each represented quadrant equally. Pose indices resolve exact ties in bank order. Uniform observations, invalid fixed cells and inadequate common support produce no proposal. A small numerical contrast floor is expressed in relative scene-linear units. It does not characterize physical detector sensitivity.

This is a sparse reference-view warp. Its z-buffer handles competing samples and preserves unknown foreground occlusion among sampled points. It cannot certify terrain missing behind the reference view, and a coarse source tile can approximate rather than exactly integrate the corresponding query footprint. The result is a list of raw candidate indices and losses. It gives no acceptance decision, 6-DoF pose accuracy, or unobserved correspondences. In particular, a candidate projection cannot supply its own final AP3P observations.

## Procedural verification

With the existing NumPy/OpenCV environment active, run from the repository root.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  PYTHONPATH="$PWD" python -m pytest -q reports/solar-warp-v117
```

The eight projection checks cover deterministic proper poses, axial depth and flat reprojection, detector bin occlusion and boundary handling, a faint translated pattern at both scales, different displacements at two surface depths, common-subbin fairness, exact cache reuse and fail-closed query and map support. The synthetic tests use images no larger than 128 × 128. They read no archived imagery or truth.

## Pending scientific work

The published preparation binds the two archived rosters. The controller requires official v115 terminal status and preservation of its available artifacts before any reservation. The exact timing and archive dispatches are frozen and published separately. A short two-thread timing check must precede any archived inference. The known-Sun ceiling still needs to test candidate basin coverage, both renderer seeds and the fixed planar/trend controls, with query camera truth joined only after blind outputs are sealed. A later estimated-Sun and matched-nonoverlap study requires its own protocol. This published preparation has no execution authority and contains no measured native timing or archived pose recovery.

Once the original benchmark is terminal and its artifacts are verified, use
the original controller from the repository root. Freezing writes a new JSON
dispatch. Publish that exact dispatch before running its phase.

```sh
PYTHONPATH="$PWD" python reports/solar-warp-v117/control_warp_archive_v117.py \
  freeze timing --v115-audit PATH_TO_ORIGINAL_ARTIFACT_AUDIT
PYTHONPATH="$PWD" python reports/solar-warp-v117/control_warp_archive_v117.py \
  run timing --public-commit EXACT_PUBLIC_COMMIT --v115-audit PATH_TO_ORIGINAL_ARTIFACT_AUDIT
```

The archive phase follows the same freeze and publication sequence after the
timing receipt meets its bound. A completed archive still needs the independent
gate result before a primary estimated-Sun trial can qualify.
