"""Verify the frozen frontend and solver seams with constructed inputs only."""

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import solar_blind_beams_v113 as v113
from solar_blind_beams_v112 import localize_with_blind_beams
from solar_frontend_v63 import ReferenceGeometry
import solar_inference_cache_v84 as frozen
import solar_pose_proposals_v73 as original


POINTS = np.column_stack([60.+np.arange(6)*16., np.array([68., 73., 80., 86., 97., 105.])])
K = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
SUN = np.array([[0., 0., 1.]])


def pose(off_nadir, tag=0):
    vector = np.array([math.pi - math.radians(off_nadir), 0., 0.])
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(vector)[0].T
    result[0, 3] = tag
    return result


def inputs():
    world = np.zeros((256, 3))
    world[:, 2] = 100.
    world[:6, :2] = 100*(POINTS-K[:2, 2])/440.
    geometry = ReferenceGeometry(
        np.full((512, 512), 100.), K.copy(), np.eye(4), np.tile(SUN, (256, 1)), SUN.copy(),
        np.ones((256, 1), np.uint8), np.ones((256, 1, 7), np.uint8), SUN[0].copy(),
        np.repeat(world[:, None], 7, axis=1),
    )
    matcher = object.__new__(frozen.SolarMatcherV84)
    matcher.pose_initialization = "quantization_aware"
    matcher.inference_revision = frozen.CACHE_REVISIONS[matcher.pose_initialization]
    matcher.coarse, matcher.fine, matcher.proposal_angles = None, None, (0.,)
    matcher.total_parameters = 0
    matcher.native_refinement_encoder = "procedural"
    checker = ((np.indices((512, 512)).sum(axis=0) % 2)*100).astype(np.uint8)
    return matcher, checker, checker.copy(), geometry, K.copy()


def test_actual_seams_preserve_native_and_final_ap3p_and_count_bypass(monkeypatch):
    initial_calls, final_calls = [], []
    n = len(POINTS)

    def initial_beams(*_):
        return [dict(tag=i, score=0., pose=None, sun_index=0,
                     assignment=dict(keep=np.arange(n), candidate=np.zeros(n, int))) for i in range(2)]

    def assignments(_graph, _sun, _mode, trial, *_args, **_kwargs):
        return dict(total=float(trial[0, 3]), keep=np.arange(n), candidate=np.zeros(n, int),
                    native_inlier_keep=np.arange(n), solar_supported=True)

    def matched(_graph, beam, _measurements):
        return POINTS.copy(), POINTS.copy(), np.full((n, 3), beam["tag"]), None

    def native(world, *_args):
        i = int(world[0, 0])
        return pose(80 if i == 0 else 25, i), {}

    def refined(native_pose, *_args):
        i = int(native_pose[0, 3])
        return pose(10 if i == 0 else 100, i)

    def original_root(_world, _pixels, _k, iterations):
        initial_calls.append(("original", iterations))
        return pose(85, (len(initial_calls)-1) % 2), {}

    def cone_root(_world, _pixels, _k, iterations):
        initial_calls.append(("cone", iterations))
        return pose(10, (len(initial_calls)-1) % 2), dict(
            iterations_requested=iterations, minimal_sets_executed=iterations,
            roots_evaluated=iterations*2, roots_rejected_off_nadir=iterations,
            lm_refits_rejected_off_nadir=1,
        )

    def final(p, q):
        final_calls.append((p.copy(), q.copy()))
        return pose(110, 99), dict(reason=None, iterations_requested=2000)

    def coarse(*_):
        points = np.concatenate([POINTS, np.zeros((256-n, 2))])
        return points, points[:, None].copy(), np.ones((256, 1)), np.zeros((256, 1), int)

    def reference_support(*_):
        return np.empty((n, 3)), np.arange(256) < n

    def rotation_bank(query, _angles):
        return SimpleNamespace(images=[query], native_support=lambda *_: True, transform=lambda x, *_: x)

    def refiner(_model):
        return SimpleNamespace(forward_calls=0, refine=lambda _a, _b, _p, q: (q, {}))

    for name, value in dict(initial_beams=initial_beams, assignments=assignments, _matched=matched,
                            estimate=native, _lm_update=refined, candidates=coarse, map_points_z=reference_support,
                            RotationBank=rotation_bank, SparseNativeRefiner=refiner).items():
        monkeypatch.setattr(frozen, name, value)
    monkeypatch.setattr(original, "propose", original_root)
    monkeypatch.setattr(v113.cone, "propose_with_simulator_cone", cone_root)
    original_solver, original_method = frozen.solve_v84, frozen.SolarMatcherV84.localize

    plain, original_bank = localize_with_blind_beams(*inputs(), budget=32, final_estimator=final)
    observed, bank, diag = v113.localize_with_blind_cone_v113(*inputs(), budget=32, final_estimator=final)

    assert initial_calls == [("original", 512)]*2 + [("cone", 512)]*2
    assert len(final_calls) == 2
    np.testing.assert_array_equal(final_calls[0], final_calls[1])
    for key in ("pose", "reference_xy", "query_xy", "proposal_samples_requested", "final_estimator_iterations"):
        np.testing.assert_equal(plain[key], observed[key])
    assert plain["proposal_samples_requested"] == 4096
    assert original_bank.to_dict() == bank.to_dict()
    assert diag["initial_v73"] == dict(calls=2, returned_poses=2, iterations_requested=1024,
                                     minimal_sets_executed=1024, roots_evaluated=2048,
                                     roots_rejected_off_nadir=1024, lm_refits_rejected_off_nadir=2)
    assert diag["native_v84"] == {
        "native_ap3p": dict(calls=6, returned_poses=6, off_nadir_poses=3, invalid_poses=0),
        "native_lm": dict(calls=6, returned_poses=6, off_nadir_poses=3, invalid_poses=0),
    }
    assert diag["pre_final_beam_poses"] == 2
    assert diag["pre_final_off_nadir_poses"] == 1
    assert diag["ordinary_final_off_nadir"] is True
    assert diag["ordinary_final_iterations_reported"] == 2000
    assert frozen.solve_v84 is original_solver and frozen.SolarMatcherV84.localize is original_method


def test_blank_and_truth_boundary():
    matcher, reference, query, geometry, k = inputs()
    reference[:] = query[:] = 0
    raw, bank, diag = v113.localize_with_blind_cone_v113(matcher, reference, query, geometry, k)
    assert raw["reason"] == bank.reason == "blank_or_uninformative_image"
    assert not bank.capture_reached and not bank.candidates
    assert diag["initial_v73"]["calls"] == diag["native_v84"]["native_ap3p"]["calls"] == 0
    assert diag["ordinary_final_off_nadir"] is None
    with pytest.raises(TypeError, match="known_sun"):
        v113.localize_with_blind_cone_v113(matcher, reference, query, geometry, k, known_sun=SUN[0])
    matcher.pose_initialization = "strict3"
    with pytest.raises(ValueError, match="exact quantization-aware"):
        v113.localize_with_blind_cone_v113(matcher, reference, query, geometry, k)
