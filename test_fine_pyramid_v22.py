"""Coordinate, candidate-boundary and actual shared-encoder controls."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from fine_pyramid_v22 import at_scale, common_support, encode, refine
from localization_diagnostics_v3 import refine_native


def test_resize_pixel_centres_and_common_candidate_border():
    np.testing.assert_array_equal(at_scale(np.array([1.5, 5.5, 9.5]), 4), [0., 1., 2.])
    p = np.array([[80, 80], [55, 80], [80, 80], [80, 80]], np.float32)
    q = np.array([[72, 72], [72, 72], [71, 72], [184, 72]], np.float32)
    np.testing.assert_array_equal(common_support(p, q, (256, 256)), [True, False, False, False])
    with pytest.raises(ValueError):common_support([[np.nan, 1]], [[1, 1]], (256, 256))


def test_native_only_exactly_replays_existing_decoder_on_common_support():
    torch.manual_seed(19)
    dense = torch.nn.functional.normalize(torch.randn(1, 7, 192, 192), dim=1)
    points = np.array([[80, 80], [20, 20], [100, 110]], np.float32)
    expected = refine_native(points[[0, 2]], points[[0, 2]], dense, dense)
    actual = refine(points, points, {1: dense}, {1: dense}, (1,))
    for a, b in zip(actual[:2], expected[:2]):np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[2], [0, 2])


def test_shared_encoder_multiscale_translation_and_empty_support():
    import cv2
    from contrastive_terrain import TerrainDescriptor
    from synthetic import texture
    torch.set_num_threads(2); torch.manual_seed(20261101)
    image = texture(823, 192)
    other = cv2.warpAffine(image, np.array([[1., 0, 8], [0, 1., -4]]), (192, 192))
    encoder = TerrainDescriptor().eval()
    a, b = encode(encoder, image), encode(encoder, other)
    p = np.array([[80, 80], [96, 96], [112, 112]], np.float32)
    for profile in ((1, 2), (1, 2, 4)):
        pp, qq, indices = refine(p, p, a, b, profile)
        np.testing.assert_array_equal(pp, p)
        np.testing.assert_array_equal(indices, np.arange(3))
        assert np.max(np.linalg.norm(qq-p-[8, -4], axis=1)) < 1
    pp, qq, indices = refine([[1, 1]], [[1, 1]], a, b)
    assert pp.shape == qq.shape == (0, 2) and not len(indices)
    with pytest.raises(ValueError):refine(p, p, a, b, (4,))
