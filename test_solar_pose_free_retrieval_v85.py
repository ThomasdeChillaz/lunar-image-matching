"""Toy feature arithmetic only. Not executed while parent QA/science is live."""
import inspect

import numpy as np
import pytest

from solar_pose_free_retrieval_v85 import (Limits, admit_workload,
                                         extract_reference_and_query_features,
                                         retrieve_candidates)
from solar_physics_v63 import solar_log_evidence


def feature_fixture():
    n, m = 2, 8
    bank = np.eye(32, dtype=np.float32)[:m].copy()
    params = np.array([[0., 0.], [5., 0.]])
    return dict(reference_response_features=bank[:n].copy(),
                frozen_reference_descriptors=bank[:n].copy(),
                known_reference_sun=np.array([0., 0., 1.]),
                reference_normals=np.tile([0., 0., 1.], (n, 1)),
                reference_views=np.tile([0., 0., 1.], (n, 1)),
                query_descriptors=bank,
                query_native_xy=np.column_stack([20.+20*np.arange(m), np.full(m, 40.)]),
                query_rotation_indices=np.zeros(m, np.int64),
                query_valid=np.ones(m, bool),
                sun_vectors=np.array([[1., 0., 0.], [0., 1., 0.]]),
                sun_parameters=params, coarse_sun_mask=np.ones(2, bool))


def test_one_global_sun_cannot_take_each_anchors_best_sun():
    inp = feature_fixture()
    calls = []

    def predictor(base, condition, sun, *, pooled):
        calls.append((base.copy(), condition.copy(), sun.copy(), pooled))
        out = np.zeros((2, 32), np.float32)
        if np.array_equal(sun, inp['sun_vectors'][0]):
            out[0, 0] = 1.
            out[1, [1, 9]] = [.8, .6]
        else:
            out[0, [0, 9]] = [.6, .8]
            out[1, 1] = 1.
        return out

    result = retrieve_candidates(**inp, arm='sh', response_predictor=predictor)
    assert result['shared_sun_index'] == 0
    assert result['global_score']['score'] == pytest.approx(1.8/256.)
    assert result['global_score']['score'] < 2./256.  # forbidden per-anchor oracle
    assert result['global_score']['mutual_count'] == 2
    assert result['candidates']['query_bank_indices'].shape == (2, 8)
    assert result['candidates']['compatibility'][1, 0] == pytest.approx(.8)
    assert len(calls) == result['diagnostics']['unique_scored_suns'] == 2
    assert result['diagnostics']['logical_sun_evaluations'] > len(calls)
    for base, condition, sun, pooled in calls:
        np.testing.assert_array_equal(base, inp['reference_response_features'])
        np.testing.assert_array_equal(condition[:, :3], np.tile(inp['known_reference_sun'], (2, 1)))
        np.testing.assert_array_equal(condition[:, 3:6], inp['reference_normals'])
        np.testing.assert_array_equal(condition[:, 6:], inp['reference_views'])
        assert sun.shape == (3,) and not pooled


def test_native_and_pooled_use_same_bank_budget_and_constant_appearance_cache():
    inp = feature_fixture()
    calls = []

    def predictor(base, condition, sun, *, pooled):
        calls.append(pooled)
        return inp['frozen_reference_descriptors'].copy()

    native = retrieve_candidates(**inp, arm='native')
    pooled = retrieve_candidates(**inp, arm='pooled', response_predictor=predictor)
    assert calls == [True]
    for key in ('query_bank_indices', 'query_native_xy', 'rotation_indices', 'compatibility'):
        np.testing.assert_array_equal(native['candidates'][key], pooled['candidates'][key])
    assert native['shared_sun_index'] == pooled['shared_sun_index']
    assert native['diagnostics']['descriptor_macs'] == pooled['diagnostics']['descriptor_macs'] == 2*8*32
    assert native['diagnostics']['logical_sun_evaluations'] == pooled['diagnostics']['logical_sun_evaluations']
    assert all(r['score'] == native['sun_score_curve'][0]['score'] for r in native['sun_score_curve'])
    assert native['scoring_track'] == 'appearance_only'
    assert native['diagnostics']['sun_identifiability'].startswith('Not established')


def test_optional_physical_term_is_frozen_formula_and_unknown_is_zero():
    inp = feature_fixture()
    native = retrieve_candidates(**inp, arm='native')
    p = np.linspace(.1, .9, 8)
    unknown = retrieve_candidates(**inp, arm='native', solar_states=np.zeros((2, 2), np.uint8),
                                  query_soft_shadow=p)
    for a, b in zip(native['sun_score_curve'], unknown['sun_score_curve']):
        assert a['score'] == b['score'] and b['physical'] == 0.
    states = np.array([[1, 2], [1, 2]], np.uint8)
    physical = retrieve_candidates(**inp, arm='native', solar_states=states,
                                   query_soft_shadow=np.full(8, .95))
    assert physical['shared_sun_index'] == 1
    expected = 2.*(1.+float(solar_log_evidence(np.uint8(2), np.array(.95))))/256.
    assert physical['global_score']['score'] == pytest.approx(expected)
    assert physical['scoring_track'] == 'appearance_plus_physical'


@pytest.mark.parametrize('which', ['states', 'measurements'])
def test_half_physical_evidence_is_rejected(which):
    inp = feature_fixture()
    args = {'solar_states':np.zeros((2, 2), np.uint8)} if which == 'states' else {'query_soft_shadow':np.full(8, .5)}
    with pytest.raises(ValueError, match='together'):
        retrieve_candidates(**inp, arm='native', **args)


def test_mutual_global_support_and_duplicate_native_location_rule():
    inp = feature_fixture()
    inp['frozen_reference_descriptors'][1] = inp['frozen_reference_descriptors'][0]
    result = retrieve_candidates(**inp, arm='native')
    assert result['global_score']['unique_anchors'] == 1
    assert result['global_score']['mutual_count'] == 1
    assert result['global_score']['score'] == pytest.approx(1./256.)


def test_cross_rotation_spatial_suppression_and_deterministic_ties():
    inp = feature_fixture()
    inp['query_descriptors'] = np.vstack([inp['query_descriptors'], inp['query_descriptors'][0:1], inp['query_descriptors'][0:1]])
    inp['query_native_xy'] = np.vstack([inp['query_native_xy'], [20., 40.], [28., 40.]])
    inp['query_rotation_indices'] = np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    inp['query_valid'] = np.ones(10, bool)
    a = retrieve_candidates(**inp, arm='native')
    b = retrieve_candidates(**inp, arm='native')
    np.testing.assert_array_equal(a['candidates']['query_bank_indices'], b['candidates']['query_bank_indices'])
    assert a['candidates']['query_bank_indices'][0, 0] == 8  # native y/x then rotation0 before rotation3
    assert a['candidates']['suppressed_before_eighth'][0] == 2
    for row in a['candidates']['query_native_xy']:
        distances = np.sum((row[:, None]-row[None])**2, -1)
        assert (distances[np.triu_indices(8, 1)] > 64.).all()


def test_invalid_bank_entries_never_become_candidates():
    inp = feature_fixture()
    inp['query_descriptors'] = np.vstack([inp['query_descriptors'], np.full((1, 32), np.nan, np.float32)])
    inp['query_native_xy'] = np.vstack([inp['query_native_xy'], [30., 40.]])
    inp['query_rotation_indices'] = np.zeros(9, np.int64)
    inp['query_valid'] = np.array([True]*8+[False])
    result = retrieve_candidates(**inp, arm='native')
    assert not (result['candidates']['query_bank_indices'] == 8).any()
    assert result['diagnostics']['valid_query_entries'] == 8


def test_operation_memory_and_logical_sun_limits_reject_before_provider():
    inp = feature_fixture()
    calls = []

    def predictor(*args, **kwargs):
        calls.append(1)
        return inp['frozen_reference_descriptors'].copy()

    for limits, text in [(Limits(max_descriptor_macs=1), 'multiply'),
                         (Limits(max_array_bytes=1), 'memory'),
                         (Limits(max_scored_suns=1), 'Sun')]:
        with pytest.raises(ValueError, match=text):
            retrieve_candidates(**inp, response_predictor=predictor, limits=limits)
    assert calls == []
    bound = admit_workload(256, 10404, 1885, 259)
    assert bound['logical_sun_bound'] == 292
    assert bound['descriptor_macs_bound'] == 24887033856
    assert bound['estimated_peak_array_bytes'] >= 256*10404*8  # includes NxM score arrays
    for size in [(257, 8, 2, 2), (2, 10405, 2, 2), (2, 8, 1886, 2), (2, 8, 300, 260)]:
        with pytest.raises(ValueError): admit_workload(*size)


def test_no_true_query_geometry_or_sun_argument_and_no_fake_extraction():
    parameters = inspect.signature(retrieve_candidates).parameters
    for forbidden in ('query_pose', 'query_depth', 'query_sun', 'true_query_sun', 'labels'):
        assert forbidden not in parameters
        with pytest.raises(TypeError): retrieve_candidates(**feature_fixture(), **{forbidden:None})
    with pytest.raises(NotImplementedError, match='not implemented'):
        extract_reference_and_query_features()


def test_missing_provider_bad_catalogue_and_spatial_shortfall_fail_honestly():
    inp = feature_fixture()
    with pytest.raises(ValueError, match='provider'):
        retrieve_candidates(**inp)
    inp['sun_parameters'][1] = inp['sun_parameters'][0]
    with pytest.raises(ValueError, match='Unique'):
        retrieve_candidates(**inp, arm='native')
    inp = feature_fixture()
    inp['query_native_xy'][:] = [20., 40.]
    with pytest.raises(ValueError, match='eight spatially distinct'):
        retrieve_candidates(**inp, arm='native')
