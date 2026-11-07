import numpy as np
import pytest
pytest.importorskip('torch')
from geometry_warp_native_v11 import map_refined, surface_warp


def test_surface_rectification_uses_axial_depth_and_estimated_pose_native_centers():
    y, x = np.mgrid[:160, :160]; image = (x+y).astype(np.uint8)
    k = np.array([[100., 0, 79.5], [0, 100., 79.5], [0, 0, 1.]])
    camera = np.eye(4); camera[0, 3] = -.8; camera[1, 3] = .4
    warped, supported, coordinates = surface_warp(image, np.full(image.shape, 10.), k, np.eye(4), camera, k)
    assert supported[80, 80] and not supported[10, 80]
    np.testing.assert_allclose(coordinates[80, 80], [88, 76], atol=1e-5)
    np.testing.assert_allclose(map_refined([[80.25, 80.5]], coordinates), [[88.25, 76.5]], atol=1e-5)
    assert warped[80, 80] == 164


def test_invalid_geometry_and_borders_are_masked_without_brightness_labels():
    black = np.zeros((160, 160), np.uint8); depth = np.full(black.shape, 10.)
    k = np.array([[100., 0, 79.5], [0, 100., 79.5], [0, 0, 1.]])
    _, support, _ = surface_warp(black, depth, k, np.eye(4), np.eye(4), k)
    assert support[80, 80]
    depth[80, 80] = np.nan
    _, support, _ = surface_warp(black, depth, k, np.eye(4), np.eye(4), k)
    assert not support[80, 80] and not support[60, 60]
    assert map_refined([], np.zeros((160, 160, 2), np.float32)).shape == (0, 2)
