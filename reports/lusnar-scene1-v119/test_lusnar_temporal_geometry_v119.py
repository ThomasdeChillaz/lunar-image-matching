import numpy as np

from lunar_geometry_v5 import project_z, world_points_z
from lusnar_temporal_geometry_v119 import (
    CONVENTIONS, Convention, half_intrinsics, interpret_depth, intrinsics, left_camera,
    predicted_alignment, samples,
)


def test_published_axes_and_left_camera_centre():
    convention = Convention("as_logged", "rover", -20, 511.5)
    camera = left_camera({"position": [0., 0., 0.], "quaternion_wxyz": [1., 0., 0., 0.]}, convention)
    np.testing.assert_allclose(camera[:3, 3], [1., -.155, -1.5])
    np.testing.assert_allclose(camera[:3, 0], [0., 1., 0.])
    np.testing.assert_allclose(camera[:3, 2], [np.cos(np.deg2rad(20)), 0., np.sin(np.deg2rad(20))])
    np.testing.assert_allclose(np.linalg.det(camera[:3, :3]), 1.)
    assert len(CONVENTIONS) == 24


def test_crop_free_half_pixel_roundtrip_and_metric_ray_depth():
    k = intrinsics(511.5)
    z = np.full((3, 4), 2.)
    yy, xx = np.indices(z.shape)
    norm = np.sqrt(1. + ((xx-511.5)/k[0, 0])**2 + ((yy-511.5)/k[1, 1])**2)
    storage = (z * norm * 100)[::-1]
    result = interpret_depth(storage, {"row_order": "bottom_to_top_storage", "depth": "ray", "scale_m": .01}, k)
    np.testing.assert_allclose(result, z)
    pixels = np.array([[100., 180.], [640., 520.]])
    p = world_points_z(pixels, [2., 3.], k, np.eye(4))
    mapped, _, _ = project_z(p, half_intrinsics(k), np.eye(4))
    np.testing.assert_allclose(mapped, (pixels+.5)/2-.5)


def test_bilinear_depth_requires_terrain_and_one_depth_surface():
    depth = np.ones((5, 5))
    mask = np.ones_like(depth, bool)
    depth[1, 1] = 4.
    mask[3, 3] = False
    result = samples(depth, mask, [[.2, .4], [2.2, 2.3], [3.2, 0.1], [4, 2]])
    np.testing.assert_array_equal(np.isfinite(result), [False, False, True, False])
    assert result[2] == 1.


def test_optical_alignment_scores_later_independent_depth_and_keeps_failures():
    k = intrinsics(511.5)
    xy = np.array([[250., 350.], [560., 730.], [700., 500.]])
    depth = np.array([2., 3., 5.])
    target = np.eye(4)
    target[0, 3] = .3
    world = world_points_z(xy, depth, k, np.eye(4))
    expected_xy, expected_z, _ = project_z(world, k, target)
    good = predicted_alignment(xy, expected_xy, depth, expected_z, k, np.eye(4), target)
    assert good["at_2_native_pixels"] == 3 and good["median_absolute_log_depth_error"] == 0
    target[0, 3] = 100.
    wrong = predicted_alignment(xy, expected_xy, depth, expected_z, k, np.eye(4), target)
    assert wrong["valid_projection_count"] == 0 and wrong["points"] == 3
    assert wrong["median_pixel_error"] == 32.


def test_exact_geometry_oracle_uses_independent_later_depth_and_half_pixels():
    from lusnar_temporal_geometry_v119 import exact_geometry_pose

    k = np.array([[110., 0., 95.5], [0., 110., 95.5], [0., 0., 1.]])
    reference_depth = np.full((192, 192), 10.)
    query_depth = reference_depth.copy()
    terrain = np.ones_like(reference_depth, bool)
    q = np.eye(4)
    q[0, 3] = .2
    result = exact_geometry_pose(reference_depth, query_depth, terrain, terrain, np.eye(4), q, k)
    assert result["eligible"] and result["passed"]
    assert result["translation_m"] < 1e-6
    q_depth_wrong = query_depth * 2
    invalid = exact_geometry_pose(reference_depth, q_depth_wrong, terrain, terrain, np.eye(4), q, k)
    assert not invalid["eligible"] and invalid["correspondences"] == 0
