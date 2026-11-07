"""Procedural-only draft tests; do not run during the live scientific pilot."""
from copy import deepcopy

import numpy as np
import pytest

from solar_acquisition_cache_v84 import (AcquisitionProjectionCacheV84,
                                       acquisition_predictions_v84)
from solar_active_v63 import choose_action
from solar_assignment_v63 import CandidateGraph, acquisition_predictions


def fixture():
    rng = np.random.default_rng(84)
    n, k, s = 8, 8, 3
    reference = np.array([[64., 64.], [192., 64.], [320., 64.], [448., 64.],
                          [64., 320.], [192., 320.], [320., 448.], [448., 448.]])
    base = np.column_stack([np.linspace(-2., 2., n), np.linspace(-1.5, 1.5, n),
                            np.full(n, 10.)])
    world = np.repeat(base[:, None, :], 7, axis=1)
    world[:, :, 2] += np.array([0., -.5, .5, -1., 1., -2., 2.])
    query = np.repeat(reference[:, None, :], k, axis=1)
    query += rng.normal(0., 3., query.shape)
    suns = np.array([[1., 0., 1.], [0., 1., 1.], [-1., 0., 1.]])
    suns /= np.linalg.norm(suns, axis=1, keepdims=True)
    graph = CandidateGraph(reference, base, query, rng.random((n, k)),
                           rng.random((n, k)), rng.integers(0, 3, (n, s), dtype=np.uint8),
                           suns, world, rng.integers(0, 3, (n, s, 7), dtype=np.uint8))
    p0, p1 = np.eye(4), np.eye(4)
    p1[:3, 3] = [.1, -.2, .15]
    beams = [dict(pose=p0, sun_index=0, score=-.5,
                  assignment=dict(independent_sun_indices=None)),
             dict(pose=p1, sun_index=1, score=.1,
                  assignment=dict(independent_sun_indices=None)),
             dict(pose=None, sun_index=2, score=-2.,
                  assignment=dict(independent_sun_indices=None))]
    actions = np.array([(i, c) for i in range(n) for c in range(k)], int)
    intrinsics = np.array([[100., 0., 255.5], [0., 100., 255.5], [0., 0., 1.]])
    return graph, beams, actions, intrinsics


def exact_result(actual, expected):
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('mode', ['shared', 'off', 'independent'])
def test_predictions_and_all_policy_actions_equal_frozen(mode):
    graph, beams, actions, intrinsics = fixture()
    if mode == 'independent':
        for j, beam in enumerate(beams):
            beam['assignment']['independent_sun_indices'] = (np.arange(8)+j) % 3
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    expected = acquisition_predictions(graph, beams, actions, intrinsics, mode)
    actual = acquisition_predictions_v84(graph, beams, actions, intrinsics, mode, cache)
    exact_result(actual, expected)
    exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics, mode, cache), expected)
    points = graph.reference_xy[actions[:, 0]]
    used = np.zeros(len(actions), bool)
    used[[0, 2, 8, 33]] = True
    for policy in ('active', 'geometry', 'uniform', 'appearance'):
        a = choose_action(actual[0], actual[1], points, used, policy,
                          graph.appearance.reshape(-1), actual[2])
        b = choose_action(expected[0], expected[1], points, used, policy,
                          graph.appearance.reshape(-1), expected[2])
        assert a == b
    assert cache.stats['projection_computations'] == 1
    assert cache.stats['projection_cache_hits'] == 1


def test_measurement_weights_sun_and_solar_states_refresh_on_projection_hit():
    graph, beams, actions, intrinsics = fixture()
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    before = acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    graph.query_xy[0, 0] += [5., -3.]
    beams[0]['score'] = 5.
    beams[0]['sun_index'] = 2
    graph.solar_variants[:, 2] = 2
    graph.solar_variants[0, 2, 0] = 0
    actual = acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    expected = acquisition_predictions(graph, beams, actions, intrinsics)
    exact_result(actual, expected)
    assert not np.array_equal(before[0][0, 0, :, :2], actual[0][0, 0, :, :2])
    assert not np.array_equal(before[1], actual[1])
    assert not np.array_equal(before[0][..., 2], actual[0][..., 2], equal_nan=True)
    assert cache.stats['projection_computations'] == 1
    assert cache.stats['projection_cache_hits'] == 1


def test_independent_per_anchor_suns_refresh_after_every_measurement():
    graph, beams, actions, intrinsics = fixture()
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    acquisition_predictions_v84(graph, beams, actions, intrinsics, 'independent', cache)
    for j, beam in enumerate(beams):
        beam['assignment']['independent_sun_indices'] = (np.arange(8)+j) % 3
    exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics, 'independent', cache),
                 acquisition_predictions(graph, beams, actions, intrinsics, 'independent'))
    assert cache.stats['projection_computations'] == 1


def test_none_pose_and_live_identity_are_distinct_cache_keys():
    graph, beams, actions, intrinsics = fixture()
    beams = [beams[2]]
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    absent = acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    assert not absent[2].any()
    beams[0]['pose'] = np.eye(4)
    live = acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    exact_result(live, acquisition_predictions(graph, beams, actions, intrinsics))
    assert live[2].any() and cache.stats['projection_computations'] == 2


def test_pose_change_and_dtype_keep_original_arithmetic():
    graph, beams, actions, intrinsics = fixture()
    beams = [beams[0]]
    beams[0]['pose'] = beams[0]['pose'].astype(np.float32)
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache),
                 acquisition_predictions(graph, beams, actions, intrinsics))
    beams[0]['pose'] = beams[0]['pose'].astype(np.float64)
    exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache),
                 acquisition_predictions(graph, beams, actions, intrinsics))
    beams[0]['pose'][:3, 3] += [.03, -.02, .01]
    exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache),
                 acquisition_predictions(graph, beams, actions, intrinsics))
    assert cache.stats['projection_computations'] == 3


@pytest.mark.parametrize('changed', ['world', 'actions', 'intrinsics', 'case'])
def test_immutable_case_inputs_fail_instead_of_reusing_stale_projection(changed):
    graph, beams, actions, intrinsics = fixture()
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    if changed == 'world': graph.world_variants[0, 0, 0] += .1
    elif changed == 'actions': actions[0] = [1, 2]
    elif changed == 'intrinsics': intrinsics[0, 0] += 1.
    else: graph = deepcopy(graph)
    with pytest.raises(ValueError):
        acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    assert cache.stats['projection_computations'] == 1


def test_materialization_guard_remains_live_on_projection_cache_hit():
    graph, beams, actions, intrinsics = fixture()
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    graph.solar_materialized_mask[0] = False
    with pytest.raises(ValueError, match='computed rays'):
        acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    with pytest.raises(ValueError, match='computed rays'):
        acquisition_predictions(graph, beams, actions, intrinsics)
    exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics, 'off', cache),
                 acquisition_predictions(graph, beams, actions, intrinsics, 'off'))


def test_bounded_cache_eviction_changes_no_prediction_and_protects_values():
    graph, beams, actions, intrinsics = fixture()
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics, capacity=1)
    original = beams[0]['pose'].copy()
    for dx in (0., .1, 0.):
        beams[0]['pose'] = original.copy()
        beams[0]['pose'][0, 3] = dx
        exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache),
                     acquisition_predictions(graph, beams, actions, intrinsics))
    assert cache.stats['entries'] == 1 and cache.stats['evictions'] == 2
    xy, front = cache.projection(graph, beams, actions, intrinsics)
    assert not xy.flags.writeable and not front.flags.writeable
    with pytest.raises(ValueError): xy.flat[0] = 0.
    assert cache.finish()['projection_computations'] == 3


def test_no_cache_option_matches_source_without_global_installation():
    graph, beams, actions, intrinsics = fixture()
    exact_result(acquisition_predictions_v84(graph, beams, actions, intrinsics),
                 acquisition_predictions(graph, beams, actions, intrinsics))
    assert acquisition_predictions.__module__ == 'solar_assignment_v63'


@pytest.mark.parametrize('changed', ['actions', 'intrinsics'])
def test_finish_checks_original_mutable_input_bindings(changed):
    graph, beams, actions, intrinsics = fixture()
    cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    acquisition_predictions_v84(graph, beams, actions, intrinsics, cache=cache)
    if changed == 'actions': actions[0] = [1, 2]
    else: intrinsics[0, 0] += 1.
    with pytest.raises(ValueError): cache.finish()
