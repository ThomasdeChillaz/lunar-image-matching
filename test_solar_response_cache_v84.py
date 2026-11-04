"""Procedural exact-cache controls only; no lunar images or trained artifacts."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from compact_lunar import CompactDescriptor
from lunar_geometry_v5 import world_points_z
from solar_frontend_v63 import PairResponseCache, ReferenceGeometry, reference_anchors
from solar_native_v63 import canonical_patch
from solar_response_cache_v84 import PairResponseCacheV84
from solar_response_v63 import LightingResponseModel


@pytest.fixture
def fixture():
    # No global RNG change and no actual learned checkpoint dependency.
    with torch.random.fork_rng():
        torch.manual_seed(20261104)
        model = LightingResponseModel(CompactDescriptor(False)).eval()
        with torch.no_grad():
            model.coefficients.weight.add_(.02*torch.randn_like(model.coefficients.weight))
            model.coefficients.bias.add_(.02*torch.randn_like(model.coefficients.bias))
            model.condition[-1].weight.add_(.01*torch.randn_like(model.condition[-1].weight))
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    depth = np.full((512, 512), 100.)
    suns = np.array([[0., 0., -1.], [.2, .1, -1.], [-.3, 0., -1.]])
    suns /= np.linalg.norm(suns, axis=1, keepdims=True)
    world = world_points_z(reference_anchors(), np.full(256, 100.), k, np.eye(4))
    g = ReferenceGeometry(depth, k, np.eye(4), np.tile([0., 0., -1.], (256, 1)), suns,
        np.ones((256, 3), np.uint8), np.ones((256, 3, 7), np.uint8), suns[0].copy(),
        np.repeat(world[:, None], 7, axis=1))
    rng = np.random.default_rng(20261104)
    a, b = [rng.integers(0, 256, (512, 512), dtype=np.uint8) for _ in range(2)]
    return dict(model=model, reference=a, query=b, geometry=g, k_query=k.copy(),
        anchors=np.array([100, 101]), graph_sun_ids=np.array([2, 0, 1]))


def make(f, cls=PairResponseCacheV84, **kwargs):
    return cls(**f, **kwargs)


def point(f, i=0): return reference_anchors()[f['anchors'][i]].astype(float)


@pytest.mark.parametrize('kind', ['identity', 'roll_translation', 'query_boundary', 'missing_centre', 'missing_surface', 'pre_recentering_outside', 'behind_camera'])
def test_canonical_maps_patches_and_unavailable_reasons_are_exact(fixture, kind):
    f = fixture; pose = np.eye(4); centre = point(f)+[1., -2.]
    if kind == 'roll_translation':
        angle = np.deg2rad(1.5); c, s = np.cos(angle), np.sin(angle)
        pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; pose[:3, 3] = [.2, -.1, .05]
    elif kind == 'query_boundary': centre = np.array([1., 1.])
    elif kind.startswith('missing'):
        p = np.rint(point(f)).astype(int); dx = 0 if kind == 'missing_centre' else 10
        f['geometry'].depth[p[1], p[0]+dx] = np.nan
    elif kind == 'pre_recentering_outside': pose[0, 3] = 1000.
    elif kind == 'behind_camera': pose[2, 3] = 200.
    expected, before = canonical_patch(f['reference'], f['query'], point(f), f['geometry'].depth,
        f['geometry'].intrinsics, f['geometry'].cam2world, f['k_query'], pose, query_center=centre)
    cache = make(f); actual, after = cache.canonical_pair(0, centre, pose)
    assert (expected is None) == (actual is None) and before['reason'] == after['reason']
    if expected is not None:
        for a, b in zip(expected, actual): np.testing.assert_array_equal(a, b)
        for name in ('reference_xy', 'query_xy'): np.testing.assert_array_equal(before[name], after[name])
        assert before['mapping'] == after['mapping']
    cache.finish()


@pytest.mark.parametrize('pooled', [False, True])
def test_scores_and_original_two_patch_batches_match_across_pose_and_sun_changes(fixture, pooled, monkeypatch):
    f = fixture; model = f['model']; batches = []; original = model.patch_features
    def observed(value): batches.append(tuple(value.shape)); return original(value)
    monkeypatch.setattr(model, 'patch_features', observed)
    old, new = make(f, PairResponseCache, pooled=pooled), make(f, pooled=pooled)
    poses = [np.eye(4), np.eye(4)]; poses[1][:3, 3] = [.2, -.1, 0.]
    requests = [(0, 0, point(f), poses[0], [0, 1]),
        (0, 0, point(f), poses[0], [2]),
        (0, 0, point(f), poses[0], [1, 0]),
        (0, 0, point(f), poses[0], [0, 1]),
        (0, 0, point(f)+[1, -1], poses[1], [0, 1]),
        (1, 2, point(f, 1), poses[0], [2, 0, 1])]
    for args in requests:
        expected, actual = old(*args), new(*args)
        np.testing.assert_array_equal(expected, actual)
    assert batches and all(shape == (2, 1, 33, 33) for shape in batches)
    assert new.cost['canonicalizations'] < old.cost['canonicalizations']
    assert new.cost['geometric_pair_cache_hits'] >= 2
    assert new.cost['reference_surface_builds'] == 2
    assert new.cost['reference_prediction_cache_hits'] >= 2
    assert new.model is model and sum(p.numel() for p in model.parameters()) == 95808
    new.finish()


def test_lru_score_and_pair_eviction_preserves_results_and_returns_copies(fixture):
    f = fixture; old = make(f, PairResponseCache, capacity=1); new = make(f, capacity=1)
    for shift, suns in ((0., [0, 1]), (1., [2]), (0., [0, 1]), (0., [1, 0])):
        args = (0, 0, point(f)+[shift, 0.], np.eye(4), suns)
        expected, actual = old(*args), new(*args)
        np.testing.assert_array_equal(expected, actual)
        actual[:] = -100.  # Caller-owned output cannot poison stored scores.
        np.testing.assert_array_equal(new(*args), expected)
        assert len(new.cache) <= 1 and len(new.pairs) <= 1
        assert len(new.reference_predictions) <= 1 and len(new.reference_self_predictions) <= 1
    assert new.cost['patch_encoder_calls'] <= old.cost['patch_encoder_calls']
    new.finish()


def test_none_invalid_and_nonfinite_inputs_do_not_acquire_features(fixture):
    f = fixture; old, new = make(f, PairResponseCache), make(f)
    assert old(0, 0, point(f), None, [0]) is None and new(0, 0, point(f), None, [0]) is None
    for centre, pose in (([1., 1.], np.eye(4)), (point(f), np.diag([1., 1., 1., np.nan]))):
        args = (0, 0, centre, pose, [0])
        if np.isfinite(pose).all(): assert old(*args) is None and new(*args) is None
        else:
            with pytest.raises(ValueError, match='Finite calibrated'): old(*args)
            with pytest.raises(ValueError, match='Finite calibrated'): new(*args)
    assert new.cost['patch_encoder_calls'] == 0 and new.cost['patches_encoded'] == 0
    new.finish()


def test_invalid_sun_columns_preserve_pre_forward_validation(fixture):
    f = fixture; old, new = make(f, PairResponseCache), make(f)
    for cache in (old, new):
        with pytest.raises(ValueError, match='valid target Sun'): cache(0, 0, point(f), np.eye(4), [])
        assert cache.cost['patch_encoder_calls'] == 0
    new.finish()


@pytest.mark.parametrize('mutation', ['copy', 'load', 'mode', 'replace', 'data'])
def test_model_mutation_fails_closed_and_data_writes_need_boundary_digest(fixture, mutation):
    f = fixture; model = f['model']; cache = make(f)
    cache(0, 0, point(f), np.eye(4), [0])
    if mutation == 'copy':
        with torch.no_grad(): next(model.parameters()).add_(.1)
    elif mutation == 'load': model.load_state_dict(deepcopy(model.state_dict()))
    elif mutation == 'mode': model.train()
    elif mutation == 'replace': model.coefficients.bias = torch.nn.Parameter(model.coefficients.bias.detach().clone())
    else: model.coefficients.weight.data.add_(.1)  # Deliberately bypass normal version counters.
    with pytest.raises(RuntimeError, match='mutated'):
        cache.finish() if mutation == 'data' else cache.cache_token()


@pytest.mark.parametrize('changed', ['reference', 'query', 'k_query', 'depth', 'reference_sun', 'normal'])
def test_input_snapshot_is_private_and_external_mutation_blocks_final_output(fixture, changed):
    f = fixture; cache = make(f)
    if changed in ('reference', 'query'): f[changed][0, 0] ^= 1
    elif changed == 'k_query': f['k_query'][0, 0] += 1.
    elif changed == 'depth': f['geometry'].depth[0, 0] += 1.
    elif changed == 'reference_sun': f['geometry'].reference_sun[0] += .1
    else: f['geometry'].normals[100, 0] += .1
    with pytest.raises(RuntimeError, match='input mutated|geometry attribute replaced or mutated'): cache.finish()


@pytest.mark.parametrize('attribute', ['depth', 'intrinsics', 'cam2world', 'normals', 'sun_vectors', 'reference_sun', 'world_variants'])
@pytest.mark.parametrize('equal_bytes', [False, True])
def test_live_geometry_attribute_replacement_cannot_escape_boundary_check(fixture, attribute, equal_bytes):
    f = fixture; cache = make(f); geometry = f['geometry']
    original = getattr(geometry, attribute); replacement = original.copy()
    if not equal_bytes: replacement.flat[0] += .125
    setattr(geometry, attribute, replacement)
    assert getattr(geometry, attribute) is not original
    with pytest.raises(RuntimeError, match='geometry attribute replaced or mutated: '+attribute): cache.finish()


def test_finish_is_required_and_closes_case_with_no_cross_case_warm_state(fixture):
    first = make(fixture); first(0, 0, point(fixture), np.eye(4), [0]); cost = first.finish()
    assert cost['immutable_boundary_checks'] == 1
    with pytest.raises(RuntimeError, match='closed'): first(0, 0, point(fixture), np.eye(4), [0])
    second = make(fixture)
    assert not second.cache and not second.pairs and not second.features and not second.surfaces
    second.finish()


def test_nonfinite_model_response_is_preserved_until_original_caller_validation(fixture):
    f = fixture
    with torch.no_grad(): f['model'].coefficients.bias.fill_(float('nan'))
    old, new = make(f, PairResponseCache), make(f)
    args = (0, 0, point(f), np.eye(4), [0, 1])
    expected, actual = old(*args), new(*args)
    np.testing.assert_array_equal(expected, actual)
    assert np.isnan(actual).all()
    new.finish()
