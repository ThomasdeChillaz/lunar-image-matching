import json

import cv2
import numpy as np
import pytest

from likelihood import (GEOMETRY_SIGMA, KAPPA, REACH, independent_peak, native,
                        optimize_camera, run, sample_fields, selected_peak)
from smoke_v126 import generated


def test_batched_interpolation_preserves_each_anchor_field():
    y, x = np.indices((49, 49))
    values = np.asarray([x+2*y, 100+3*x-y], float)
    np.testing.assert_allclose(sample_fields(values, np.array([[3.25, 4.5], [5.5, 6.25]])), [12.25, 110.25])
    with pytest.raises(ValueError):
        sample_fields(values, np.array([[1., 2.]]))


def potential_fixture():
    k = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    y, x = np.mgrid[80:441:72, 80:441:72]
    anchors = np.c_[x.ravel(), y.ravel()].astype(float)
    depth = np.full((512, 512), 1100.)
    world, _ = native.control.pipeline.map_points_z(anchors, depth, k, np.eye(4))
    truth = np.eye(4); truth[0, 3] = 8.
    query, _, _ = native.control.project_z(world, k, truth)
    yy, xx = np.indices((49, 49))
    grid = np.stack([xx-REACH, yy-REACH], axis=-1)
    rng = np.random.default_rng(20261135)
    scores = []
    for index, offset in enumerate(query-anchors):
        correct = .9*np.exp(-((grid-offset)**2).sum(axis=-1)/(2*2.**2))
        if index % 3 == 0:
            distractor = rng.uniform(-12., 12., 2)
            correct = np.maximum(correct, .99*np.exp(-((grid-distractor)**2).sum(axis=-1)/(2*1.3**2)))
        scores.append(correct)
    score = np.asarray(scores)
    values = np.exp(KAPPA*(score-native.MIN_CORRELATION))
    blurred = np.asarray([cv2.GaussianBlur(a, (13, 13), GEOMETRY_SIGMA, borderType=cv2.BORDER_CONSTANT) for a in values])
    records = [dict(anchor_index=i, center=p, local_world=w) for i, (p, w) in enumerate(zip(anchors, world, strict=True))]
    return dict(records=records, scores=score, support=np.ones_like(score, bool), blurred=blurred,
                initial_pose=np.eye(4)), k, truth


def test_shared_camera_resolves_incoherent_stronger_patch_peaks():
    fields, k, truth = potential_fixture()
    pose, fit = optimize_camera(fields, np.eye(4), k)
    assert fit["energy_improved"] and fit["final_energy"] < fit["initial_energy"]
    assert np.linalg.norm(pose[:3, 3]-truth[:3, 3]) < 2.
    assert fit["calibrated_probability"] is False
    assert fit["iterations"] <= 40


def test_unknown_support_cannot_supply_a_measured_correspondence():
    scores = np.ones((49, 49))
    support = np.zeros_like(scores, bool)
    assert selected_peak(scores, support, [0., 0.])["query_offset"] is None
    support[24, 24] = True
    result = selected_peak(scores, support, [0., 0.])
    assert result["reason"] == "insufficient_peak_comparison"
    json.dumps(result, allow_nan=False)


def test_hard_peak_matches_frozen_native_curve_decision():
    rng = np.random.default_rng(135)
    image = (1+rng.random((512, 512))).astype(np.float32)
    query = cv2.warpAffine(image, np.array([[1., 0., 3.3], [0., 1., -2.7]]), (512, 512))
    a, av = native.prepare(image); b, bv = native.prepare(query)
    center = np.array([256., 256.])
    template, mask = native.sample(a, av, center, 64)
    observed, valid = native.sample(b, bv, center, 112)
    values, _ = native.correlation_surface(template, mask, observed, valid)
    ours = independent_peak(np.where(np.isfinite(values), values, 0.), np.isfinite(values), center)
    old = native.direction(a, av, b, bv, center, center, subpixel=True)
    for key in ("query_xy", "reason", "score", "margin", "integer_offset"):
        assert ours[key] == old[key]


@pytest.mark.parametrize("multidepth", [False, True])
def test_generated_native_matching_and_common_ap3p(multidepth):
    cv2.setNumThreads(1)
    args, valid, truth = generated(multidepth)
    ref, depth, k, pose, query, qk, _ = args
    result = run(ref, query, depth, k, pose, qk, pose, reference_valid=np.ones_like(valid))
    for name in ("hard", "joint"):
        row = result["methods"][name]
        assert row["pose"] is not None, (name, row["failures"])
        assert np.linalg.norm(np.asarray(row["pose"])[:3, 3]-truth[:3, 3]) < 2.
        assert row["attempted_patches"] <= 36
        assert row["independent_landmark_validation"] is False
    assert result["methods"]["joint"]["conditional_correspondences"]
    # Exclude transient ndarray fields, which are saved separately with masks.
    json.dumps({k: v for k, v in result.items() if k != "fields"}, allow_nan=False)


def test_blank_and_absent_camera_do_not_fabricate_a_pose():
    args, valid, _ = generated(False)
    ref, depth, k, pose, query, qk, _ = args
    for initial in (None, pose):
        result = run(np.ones_like(ref), np.ones_like(query), depth, k, pose, qk, initial,
                     reference_valid=np.ones_like(valid))
        assert all(row["pose"] is None for row in result["methods"].values())
