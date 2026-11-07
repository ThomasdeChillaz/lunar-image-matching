"""Procedural checks for the frozen v84 observer. No model or dataset is run."""

from copy import deepcopy
import json
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import solar_inference_cache_v84 as frozen
from solar_blind_beams_v112 import _observed_program, _snapshot, localize_with_blind_beams
from solar_frontend_v63 import ReferenceGeometry


SUNS = np.array([[0., 0., 1.], [.6, 0., .8], [0., .6, .8]])


def _pose(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def test_captures_only_post_filter_beams_before_the_same_final_estimator():
    class FixtureGraph(frozen.CandidateGraph):
        def search_indices(self, centre, round_index):
            return np.array([0, 1, 2]) if round_index == 2 else np.array([0])

    points = np.column_stack([50. + np.arange(6)*16., np.full(6, 50.)])
    graph = FixtureGraph(
        points, np.ones((6, 3)), points[:, None], np.ones((6, 1)), np.zeros((6, 1)),
        np.zeros((6, 3), np.uint8), SUNS, np.ones((6, 7, 3)),
        np.ones((6, 3, 7), np.uint8), np.array([[0., 0.], [1.25, 0.], [2.5, 0.]]),
        np.array([True, False, False]), np.array([True, False, False]),
    )
    original_function = frozen.solve_v84
    original_method = frozen.SolarMatcherV84.localize
    ordinary_module_globals = tuple(vars(frozen))
    n = len(points)

    def one_run(observed):
        log = []
        captured = []
        proposals = iter(range(4))

        def initial(*_):
            return [dict(tag=i, assignment=dict(keep=np.arange(n), candidate=np.zeros(n, int)),
                         score=0., pose=None, sun_index=0) for i in range(4)]

        def assign(_graph, sun_index, _mode, pose, *_args, **_kwargs):
            index = int(pose[0, 3])
            preferred_sun = (2, 2, 1, 2)[index]
            total = (90., 10., 10., 80.)[index] - int(sun_index != preferred_sun)
            return dict(total=total, keep=np.arange(n), candidate=np.zeros(n, int),
                        native_inlier_keep=np.arange(n), solar_supported=True)

        def matched(_graph, beam, _measurements):
            i = beam["tag"]
            support = n if i < 3 else n-1
            return points[:support], points[:support], np.full((support, 3), i), None

        def pose_proposal(*_):
            i = next(proposals)
            log.append(("proposal", i))
            return _pose(i), {}

        def native_pose(world, *_):
            i = int(world[0, 0])
            return (_pose(i), {}) if i else (None, {})

        def lookup(vectors):
            log.append(("physical_lookup", vectors.tolist()))
            return dict(solar_states=np.ones((n, len(vectors)), np.uint8),
                        solar_variants=np.ones((n, len(vectors), 7), np.uint8))

        def measure(i, _c, _pose, sun):
            log.append(("patch", i, sun))
            return dict(query_xy=points[i])

        def final(p, q):
            log.append(("final_ap3p", len(p), len(q)))
            return _pose(99), dict(reason=None, iterations_requested=2000)

        def observe(current_graph, beams, candidates, best):
            log.append(("observer", tuple(b["tag"] for b in candidates)))
            captured.append(_snapshot(current_graph, beams, candidates, best))

        namespace = dict(vars(frozen), CandidateGraph=FixtureGraph, initial_beams=initial, assignments=assign,
                         _matched=matched, estimate=native_pose, _lm_update=lambda pose, *_: pose,
                         _capture_v112=observe)
        if observed:
            exec(_observed_program(frozen.__file__), namespace)
            solve = namespace["solve_v84"]
        else:
            solve = FunctionType(original_function.__code__, namespace, "solve_v84", original_function.__defaults__)
        with patch("solar_pose_proposals_v73.propose", side_effect=pose_proposal):
            result = solve(graph, np.eye(3), measure, budget=32, final_estimator=final,
                           solar_lookup=lookup, pose_initialization="quantization_aware")
        return result, captured, log

    result, _, plain_log = one_run(False)
    observed, captures, log = one_run(True)
    assert log[-2:] == [("observer", (1, 2)), ("final_ap3p", 6, 6)]
    assert [event for event in log if event[0] != "observer"] == plain_log
    assert ("physical_lookup", SUNS[1:].tolist()) in plain_log
    np.testing.assert_equal(observed, result)
    assert len(captures) == 1 and captures[0].capture_reached
    assert [beam.source_beam_index for beam in captures[0].candidates] == [1, 2]
    assert [beam.score for beam in captures[0].candidates] == [10., 10.]
    assert [beam.sun_index for beam in captures[0].candidates] == [2, 1]
    np.testing.assert_array_equal([b.sun_reference_enu for b in captures[0].candidates], SUNS[[2, 1]])
    np.testing.assert_array_equal(captures[0].candidates[0].pose, _pose(1))
    np.testing.assert_array_equal(observed["pose"], _pose(99))
    assert tuple(vars(frozen)) == ordinary_module_globals
    assert frozen.solve_v84 is original_function and frozen.SolarMatcherV84.localize is original_method


def test_final_rescue_empty_and_snapshot_is_immutable():
    graph = SimpleNamespace(sun_vectors=SUNS.copy(), solar_materialized_mask=np.ones(3, bool))
    poses = [dict(pose=_pose(x), sun_index=x, score=float(x)) for x in range(3)]
    absent = _snapshot(graph, poses, [], poses[-1])
    assert absent.capture_reached and absent.candidates == ()
    assert absent.reason == "no_post_filter_pose_candidates"
    observed = _snapshot(graph, poses, [poses[0], poses[2]], poses[2])
    before = deepcopy(observed.to_dict())
    poses[2]["pose"][0, 3] = 42
    graph.sun_vectors[2, 2] = .7
    assert observed.to_dict() == before
    json.dumps(before, allow_nan=False)
    graph.solar_materialized_mask[0] = False
    with pytest.raises(ValueError, match="uncomputed final Sun state"):
        _snapshot(graph, poses, [poses[0]], poses[0])


def test_opt_in_frontend_preserves_blank_behavior_and_rejects_query_truth():
    geometry = ReferenceGeometry(
        np.ones((512, 512)), np.eye(3), np.eye(4), np.tile([0., 0., 1.], (256, 1)),
        SUNS, np.ones((256, 3), np.uint8), np.ones((256, 3, 7), np.uint8), SUNS[0], np.ones((256, 7, 3)),
    )
    image = np.zeros((512, 512), np.uint8)
    matcher = object.__new__(frozen.SolarMatcherV84)
    ordinary = matcher.localize(image, image, geometry, np.eye(3))
    observed, bank = localize_with_blind_beams(matcher, image, image, geometry, np.eye(3))
    for result in (ordinary, observed):
        result.pop("elapsed_seconds")
    np.testing.assert_equal(observed, ordinary)
    assert bank.candidates == () and not bank.capture_reached
    assert bank.reason == "blank_or_uninformative_image"
    for name in ("known_sun", "query_depth", "query_pose", "query_true_sun"):
        with pytest.raises(TypeError, match=name):
            localize_with_blind_beams(matcher, image, image, geometry, np.eye(3), **{name: SUNS[0]})
    with pytest.raises(ValueError, match="exact frozen"):
        localize_with_blind_beams(object(), image, image, geometry, np.eye(3))
