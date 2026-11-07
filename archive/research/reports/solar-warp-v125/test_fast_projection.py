import numpy as np
import pytest

from fast_projection import FastProjectionBank, detector_winners, original, rank_pose_bank
from north_render_plan_v31 import camera_intrinsics, camera_to_world


def points(xy, depth):
    depth = np.asarray(depth, float)
    return np.column_stack((np.asarray(xy, float) * depth[:, None], depth))


def assert_parity(world, k, pose, shape, stride, footprint):
    expected = original.detector_winners(world, k, pose, shape, stride, footprint=footprint)
    actual = detector_winners(world, k, pose, shape, stride, footprint=footprint)
    assert actual.dtype == expected.dtype == np.int16
    np.testing.assert_array_equal(actual, expected)
    return actual


@pytest.mark.parametrize("footprint", [False, True])
def test_boundaries_clipped_footprints_and_unknown_front_winners(footprint):
    upper = np.nextafter(15.5, -np.inf)
    lower = np.nextafter(-.5, -np.inf)
    world = points([(4, 4), (4, 4), (4, 4), (-.5, -.5), (lower, 0), (upper, upper),
                    (15.5, 7), (6, 6), (0, 0), (9, 9), (np.nan, 0)],
                   [4, 2, 2, 1, 1, 1, 1, 2, -1, 0, 1])
    actual = assert_parity(world, np.eye(3), np.eye(4), (16, 16), 4, footprint)
    assert np.any(actual == 1) and not np.any(actual == 2)
    assert actual[0, 0] == 3 and actual[-1, -1] == 5
    assert not np.any(np.isin(actual, [4, 6, 8, 9, 10]))
    # Unknown foreground photometry still suppresses every farther sample.
    radiance = np.arange(len(world), dtype=float)
    radiance[1] = np.nan
    raster = original.rasterize(actual, radiance)
    assert np.isnan(raster[actual == 1]).all()


def test_equal_depth_across_origins_uses_earlier_source_index():
    world = points([(9.5, 5), (5.5, 5), (8, 5), (5.5, 5)], [10, 10, 15, 8])
    result = assert_parity(world, np.eye(3), np.eye(4), (16, 16), 4, True)
    assert result[1, 2] == 3  # nearest wins regardless of origin or index
    result = assert_parity(world[:3], np.eye(3), np.eye(4), (16, 16), 4, True)
    assert result[1, 2] == 0  # equal depth ties across origins go to first source


def test_native_randomized_terrain_and_oblique_camera_bank_parity():
    rng = np.random.default_rng(20261105)
    pose = np.asarray(camera_to_world([20., -31., 1250.], [-44., 40., 0.]))
    k = np.asarray(camera_intrinsics(512))
    k[0, 1] = .017
    world = rng.uniform([-1500., -1500., -400.], [1500., 1500., 1500.], (4096, 3))
    world[1:8] = world[:1]
    world[50:61] = np.nan
    for candidate in original.reference_pose_bank(pose, count=6):
        assert_parity(world, k, candidate, (512, 512), 4, True)
        assert_parity(world, k, candidate, (512, 512), 4, False)
    assert_parity(world[:0], k, pose, (64, 96), 4, True)
    assert_parity(world[:0], k, pose, (64, 96), 4, False)


def test_full_rank_evidence_both_scales_and_seed_cache_are_exact():
    shape = 128, 128
    pose = np.asarray(camera_to_world([0., 0., 1250.], [0., 0., 0.]))
    k = np.asarray(camera_intrinsics(128))
    rng = np.random.default_rng(42)
    luminance = (1e-7 + 1e-6*rng.random((8, 8))).repeat(16, axis=0).repeat(16, axis=1)
    depth = np.full(shape, 1250.)
    depth[:64] = 1000.
    cloud = original.build_reference_cloud(luminance, depth, k, pose)
    variant = original.build_reference_cloud(luminance * 1.01, depth, k, pose)
    bank = np.repeat(pose[None], 4, axis=0)
    bank[:, 0, 3] = np.array([-32., 0., 32., 32.]) * 1250 / k[0, 0]
    query = np.full(shape, 1e-7)
    query[:64, :88] = luminance[:64, 40:]
    query[64:, :96] = luminance[64:, 32:]
    old, new = (cls(cloud, k, bank, shape, 4) for cls in (original.ProjectionBank, FastProjectionBank))
    for reference in (cloud, variant):
        for scale in (32, 16):
            original_evidence, fast_evidence = {}, {}
            expected = original.rank_pose_bank(reference, query, k, bank, cell_pixels=scale,
                                               min_cells_per_quadrant=1, evidence=original_evidence,
                                               projection_bank=old)
            actual = rank_pose_bank(reference, query, k, bank, cell_pixels=scale,
                                    min_cells_per_quadrant=1, evidence=fast_evidence, projection_bank=new)
            assert actual == expected
            assert actual["orders"]["geometry_3d"][:2] == [2, 3]
            assert fast_evidence.keys() == original_evidence.keys()
            for name in fast_evidence:
                np.testing.assert_array_equal(fast_evidence[name], original_evidence[name], strict=True)
    assert tuple(new.indices) == tuple(old.indices) == (0, 1, 2, 3)
    assert new.get(0) is new.indices[0]
    for index in new.indices:
        for real, expected in zip(new.get(index), old.get(index), strict=True):
            np.testing.assert_array_equal(real, expected, strict=True)
    implicit = rank_pose_bank(cloud, query, k, bank, cell_pixels=16, min_cells_per_quadrant=1)
    assert implicit == original.rank_pose_bank(cloud, query, k, bank, cell_pixels=16, min_cells_per_quadrant=1)
    with pytest.raises(ValueError, match="same reference geometry and cameras"):
        new.validate(cloud, k, bank[::-1], shape, 4)
