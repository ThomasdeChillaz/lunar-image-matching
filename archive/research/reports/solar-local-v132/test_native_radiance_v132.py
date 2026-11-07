import cv2
import numpy as np
import pytest

from native_radiance import (correlation_surface, direction, measure, prepare,
                             quadratic_offset, search)
from smoke_v126 import generated


def texture():
    rng = np.random.default_rng(20261132)
    return 1e-9 * (1. + cv2.GaussianBlur(rng.random((512, 512)), (0, 0), .7))


def test_fractional_native_measurement_and_independent_reverse():
    cv2.setNumThreads(1)
    image = texture()
    motion = np.array([3.3, -2.65])
    query = cv2.warpAffine(image, np.c_[np.eye(2), motion], (512, 512))
    a, av = prepare(image)
    b, bv = prepare(query)
    center = np.array([256., 256.])
    raw = search(a, b, av, bv, center, center, subpixel=True)
    integer = search(a, b, av, bv, center, center, subpixel=False)
    assert raw["reason"] is None and raw["reverse"] is not None
    assert np.linalg.norm(np.asarray(raw["query_xy"])-center-motion) < .1
    assert raw["round_trip_error_px"] < .1
    assert np.linalg.norm(np.asarray(raw["query_xy"])-center-motion) < np.linalg.norm(
        np.asarray(integer["query_xy"])-center-motion)


def test_missing_support_never_becomes_a_brightness_observation():
    image = texture()
    valid = np.ones(image.shape, bool)
    valid[210:305, 210:305] = False
    a, mask = prepare(image, valid)
    assert not mask[204:311, 204:311].any()
    b, bv = prepare(image)
    center = np.array([256., 256.])
    assert search(a, b, mask, bv, center, center, subpixel=True)["query_xy"] is None


def test_reverse_measurement_uses_the_known_reference_footprint():
    image = texture()
    valid = np.ones(image.shape, bool)
    valid[240:248, 240:248] = False
    a, av = prepare(image, valid)
    motion = np.array([3.3, -2.65])
    query = cv2.warpAffine(image, np.c_[np.eye(2), motion], (512, 512))
    b, bv = prepare(query)
    center = np.array([256., 256.])
    result = search(a, b, av, bv, center, center, subpixel=True)
    assert result["reason"] is None
    assert np.linalg.norm(np.asarray(result["query_xy"])-center-motion) < .15
    assert result["round_trip_error_px"] < .2


def test_blank_and_independent_noise_cannot_supply_matches():
    image = texture()
    a, av = prepare(np.ones_like(image))
    b, bv = prepare(image)
    center = np.array([256., 256.])
    assert search(a, b, av, bv, center, center, subpixel=True)["query_xy"] is None
    other = 1e-9 * np.random.default_rng(42).random(image.shape)
    c, cv = prepare(other)
    assert search(b, c, bv, cv, center, center, subpixel=True)["query_xy"] is None


def test_common_support_masked_ncc_matches_a_direct_calculation():
    rng = np.random.default_rng(72)
    template = rng.random((64, 64)).astype(np.float32)
    mask = rng.random((64, 64)) > .2
    target = rng.random((68, 68)).astype(np.float32)
    available = rng.random(target.shape) > .1
    scores, reason = correlation_surface(template, mask, target, available)
    assert reason is None
    for y, x in ((0, 0), (2, 3), (4, 4)):
        common = mask & available[y:y+64, x:x+64]
        expected = np.corrcoef(template[common], target[y:y+64, x:x+64][common])[0, 1]
        assert abs(scores[y, x]-expected) < 2e-6


def test_subpixel_peak_is_local_and_rejects_saddles():
    y, x = np.indices((9, 9))
    scores = -((x-4.3)**2 + 2*(y-3.8)**2)
    np.testing.assert_allclose(quadratic_offset(scores, 4, 4), [.3, -.2], atol=1e-12)
    assert quadratic_offset((x-4.)**2-(y-4.)**2, 4, 4) is None


@pytest.mark.parametrize("multidepth", [False, True])
def test_generated_geometry_and_common_pose_estimator(multidepth):
    cv2.setNumThreads(1)
    args, valid, truth = generated(multidepth)
    reference, depth, k, pose, query, qk, _ = args
    result = measure(reference, query, depth, k, pose, qk, truth[None],
                     reference_valid=np.ones_like(valid))
    for name, row in result.items():
        assert row["pose"] is not None, (name, row["failures"])
        assert np.linalg.norm(np.asarray(row["pose"])[:3, 3]-truth[:3, 3]) < 2.
        assert row["maximum_patch_budget"] == 36 and row["attempted_patches"] <= 36


def test_input_contract_requires_native_boolean_support():
    with pytest.raises(ValueError):
        prepare(np.zeros((256, 256)))
    with pytest.raises(ValueError):
        prepare(np.zeros((512, 512)), np.ones((512, 512)))
