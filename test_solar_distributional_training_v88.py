"""DRAFT procedural tests. Not executed during source-only implementation.

No lunar image/checkpoint/bank is used. The actual 100-update pilot and replay
remain separately budgeted scientific work; these tests are software controls.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import solar_distributional_training_v88 as t


def arrays(n=8):
    rng = np.random.default_rng(22)
    native = np.broadcast_to(np.column_stack([np.arange(n) * 8 + 129.5,
                                              np.arange(n) * 8 + 129.5])[:, None], (n, 2, 2)).copy()
    sun = np.tile(np.array([0., 0., 1.], np.float32), (24, 1))
    condition = np.zeros((n, 12, 10), np.float32)
    condition[:, :, 2] = condition[:, :, 5] = condition[:, :, 8] = 1.
    views = np.broadcast_to(np.array([1., 0., 0., 1., 0., 1., 0.], np.float32), (n, 2, 7)).copy()
    teacher = np.full((n, 2, 24, 32), np.nan, np.float32)
    features = rng.normal(size=(n, 2, 12, 32)).astype(np.float32)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    teacher[:, :, ::2] = features
    available = np.zeros((n, 2, 24), bool)
    available[:, :, ::2] = True
    return dict(reference_images=rng.integers(0, 256, (12, 128, 128), dtype=np.uint8),
                reference_normal_xy=np.zeros((2, 128, 128), np.float32),
                reference_geometry_valid=np.ones((128, 128), bool), native_xy=native,
                coarse_xy=((native + .5) / 4 - .5).astype(np.float32),
                world=np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]).astype(np.float64),
                point_label_ids=np.arange(n, dtype=np.int64) + 1000,
                observed_conditions=condition, target_suns=sun, target_view_basis=views,
                teacher=teacher, available=available, correspondence_visible=np.ones((n, 2), bool),
                coarse_support=np.ones((n, 2), bool), reference_input_available=np.ones(n, bool))


class ToyBank:
    groups = ('trainingA', 'trainingB')
    by_group = {'trainingA': (0, 1), 'trainingB': (2,)}
    manifest_sha256 = 'a' * 64

    def __init__(self, a=None):
        self.a = a if a is not None else arrays()
        self.requested = []

    def load(self, unit_index):
        self.requested.append(unit_index)
        return self.a

    def preflight(self):
        return {str(i): t.validate_unit(self.a, len(self.a['world'])) for i in range(3)}

    def validate_model(self, model, expected_coarse_digest):
        from solar_native_v63 import state_digest
        assert state_digest(model.coarse) == expected_coarse_digest


def test_fixed_hyperparameters_cannot_be_silently_swept():
    assert t.PilotSettings().validate().updates == 100
    for setting in (replace(t.PilotSettings(), seed=42), replace(t.PilotSettings(), batch_points=16),
                    replace(t.PilotSettings(), marginal_weight=0.), replace(t.PilotSettings(), updates=2000)):
        with pytest.raises(ValueError, match='fixed'):
            setting.validate()


def test_both_masks_and_common_target_sun_are_applied_without_row_relabel():
    a = arrays()
    a['reference_input_available'][0] = False
    a['coarse_support'][0, 0] = False
    a['available'][0, 0] = False
    a['teacher'][0, 0] = np.nan
    # Point1 remains a valid reference input but has no target in either role.
    a['available'][1] = False
    a['teacher'][1] = np.nan
    bank = ToyBank(a)
    rng = np.random.default_rng(20261101)
    for _ in range(30):
        b = t.sample_training_batch(bank, rng)
        ids = b['trace']['point_indices']
        assert 0 not in ids and 1 not in ids and len(ids) == len(set(ids)) == 6
        assert b['trace']['observed_reference_light'] in t.EVEN
        assert b['trace']['target_light'] in t.EVEN
        np.testing.assert_array_equal(b['target_sun'], np.broadcast_to(b['target_sun'][0], (6, 3)))
        role, light = b['trace']['target_role'], b['trace']['target_light']
        np.testing.assert_array_equal(b['target_teacher'], a['teacher'][ids, role, light])
        assert b['trace']['point_label_ids'] == a['point_label_ids'][ids].tolist()


def test_sampler_is_balanced_by_group_then_camera_not_point_count():
    bank = ToyBank()
    a = np.random.default_rng(20261101)
    b = np.random.default_rng(20261101)
    traces_a = [t.sample_training_batch(bank, a)['trace'] for _ in range(400)]
    traces_b = [t.sample_training_batch(bank, b)['trace'] for _ in range(400)]
    assert traces_a == traces_b
    counts = {i: sum(v['unit_index'] == i for v in traces_a) for i in range(3)}
    assert 140 < counts[2] < 260
    assert all(50 < counts[i] < 150 for i in (0, 1))
    assert abs(counts[0] - counts[1]) < 70


@pytest.mark.parametrize('kind', ['odd', 'unavailable_finite', 'duplicate_id', 'wrong_centres',
                                  'query_as_observed_view', 'target_shortfall', 'bad_normal'])
def test_bank_schema_fails_closed_for_supervision_corruption(kind):
    a = arrays()
    if kind == 'odd':
        a['available'][0, 0, 1] = True
        a['teacher'][0, 0, 1] = a['teacher'][0, 0, 0]
    elif kind == 'unavailable_finite':
        a['available'][0, 0, 0] = False
    elif kind == 'duplicate_id':
        a['point_label_ids'][1] = a['point_label_ids'][0]
    elif kind == 'wrong_centres':
        a['coarse_xy'][0, 0, 0] += .01
    elif kind == 'query_as_observed_view':
        a['observed_conditions'][:, :, 6:9] = [1., 0., 0.]
    elif kind == 'target_shortfall':
        a['available'][1:, 1, 2] = False
        a['teacher'][1:, 1, 2] = np.nan
    else:
        a['reference_normal_xy'][0] = 2.
    with pytest.raises(ValueError):
        t.validate_unit(a, len(a['world']))


def test_valid_schema_keeps_odd_nan_all_rows_and_explicit_support():
    a = arrays()
    counts = t.validate_unit(a, 8)
    assert len(counts) == 24 and set(counts.values()) == {8}
    assert all(not value.flags.writeable for value in a.values())
    assert np.isnan(a['teacher'][:, :, 1::2]).all()


def test_unsealed_bank_is_rejected_before_any_array_decode(tmp_path, monkeypatch):
    import sys
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    calls = []
    def blocked(manifest):
        calls.append(manifest)
        raise ValueError('actual full-bank watchdog missing')
    monkeypatch.setitem(sys.modules, 'solar_coarse_bank_v88',
                        SimpleNamespace(validate_completed_bank=blocked))
    monkeypatch.setattr(t.np, 'load', lambda *a, **k: pytest.fail('NPZ decoded before completed-bank gate'))
    with pytest.raises(ValueError, match='watchdog'):
        t.CoarseTrainingBank(tmp_path, 'manifest.json', t.sha(path),
            allowed_training_groups=['trainingA'], eligible_unit_indices=[0], expected_teacher_sha256='0' * 64)
    assert calls == ['manifest.json']


def test_old_canonical_teacher_format_cannot_be_relabelled_coarse128(tmp_path, monkeypatch):
    import sys
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'format': 'solar-canonical33-bank-v79'}))
    monkeypatch.setitem(sys.modules, 'solar_coarse_bank_v88', SimpleNamespace(
        validate_completed_bank=lambda p: {'manifest_sha256': t.sha(path)}))
    monkeypatch.setattr(t.np, 'load', lambda *a, **k: pytest.fail('Wrong-domain teacher decoded'))
    with pytest.raises(ValueError, match='coarse128'):
        t.CoarseTrainingBank(tmp_path, 'manifest.json', t.sha(path),
            allowed_training_groups=['trainingA'], eligible_unit_indices=[0], expected_teacher_sha256='0' * 64)


def model_fixture():
    """Actual compact factory on procedural parameters; no lunar checkpoint."""
    import torch
    from compact_lunar import CompactDescriptor
    from solar_distributional_response_v86 import TrainingViewPrior
    prior = TrainingViewPrior.from_training_views(
        np.array([[0., 0., 1.], [.2, 0., np.sqrt(.96)]]),
        np.array([[0., 1., 0.], [0., 1., 0.]]), ['trainingA', 'trainingB'],
        allowed_training_groups=['trainingA', 'trainingB'], source_sha256={'procedural': '0' * 64})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(33)
        encoder = CompactDescriptor(False).eval().requires_grad_(False)
    return t.initialize(encoder, prior)


def test_actual_factory_frozen_foundation_and_all_heads_receive_finite_gradients():
    import torch
    model, optimizer, rng, coarse_digest = model_fixture()
    assert model.parameter_report['actual_parameters'] == 95116
    before = {k: v.clone() for k, v in model.coarse.state_dict().items()}
    batch = t.sample_training_batch(ToyBank(), rng)
    loss, metrics = t.pilot_objective(model, batch)
    loss.backward()
    assert metrics['primary_matching_view_mode'] == 'training_prior_marginal'
    assert metrics['covariance_mode'] == 'full'
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.coarse.parameters())
    for layer in (model.reference_fine, model.observed_film, model.mean_factors, model.noise_factors):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    assert model.mean_angular.grad.abs().sum() > 0
    assert model.noise_angular.grad.abs().sum() > 0
    assert all(torch.equal(v, before[k]) for k, v in model.coarse.state_dict().items())


def test_marginal_branch_never_receives_true_target_view():
    model, _, rng, _ = model_fixture()
    b = t.sample_training_batch(ToyBank(), rng)
    original = model.marginal_distribution
    calls = []
    def spy(*args, **kwargs):
        calls.append(kwargs)
        assert 'target_view_basis' not in kwargs
        return original(*args, **kwargs)
    model.marginal_distribution = spy
    t.pilot_objective(model, b)
    assert len(calls) == 1 and calls[0] == {'diagonal_only': False}


def test_checkpoint_snapshot_has_no_mutable_alias_and_exact_rng_state():
    import torch
    model, optimizer, rng, _ = model_fixture()
    original = t.snapshot(model, optimizer, rng, 0)
    assert t.identical(original, t.snapshot(model, optimizer, rng, 0))
    rng.integers(10)
    assert not t.identical(original, t.snapshot(model, optimizer, rng, 0))
    with torch.no_grad():
        next(model.reference_fine.parameters()).add_(.1)
    assert not torch.equal(original['model']['reference_fine.layers.0.weight'],
                           model.state_dict()['reference_fine.layers.0.weight'])


def test_original_encoder_or_training_prior_mutation_is_not_accepted():
    import torch
    model, optimizer, rng, digest = model_fixture()
    seal = t.prior_digest(model)
    with torch.no_grad():
        model.view_mean[1] += .1
    with pytest.raises(ValueError, match='prior'):
        t.fitting_step(model, optimizer, ToyBank(), rng, expected_coarse_digest=digest,
                       expected_prior_digest=seal)
    model, optimizer, rng, digest = model_fixture()
    with torch.no_grad():
        next(model.coarse.parameters()).add_(.1)
    with pytest.raises(ValueError, match='coarse'):
        t.fitting_step(model, optimizer, ToyBank(), rng, expected_coarse_digest=digest)


def test_complete100_update_trace_and_checkpoint_replay_on_procedural_only_model(tmp_path):
    model, optimizer, rng, digest = model_fixture()
    root = tmp_path / 'pilot'
    summary = t.train_pilot(model, optimizer, rng, ToyBank(), root,
                            expected_coarse_digest=digest, provenance={'split': 'training', 'toy': True})
    assert summary['updates'] == 100 and not summary['model_promoted']
    rows = json.loads((root / 'history_100.json').read_text())
    jsonl = [json.loads(s) for s in (root / 'trace.jsonl').read_text().splitlines()]
    assert rows == jsonl and len(rows) == 100 and [r['step'] for r in rows] == list(range(1, 101))
    assert (root / 'step_0.pt').is_file() and (root / 'step_100.pt').is_file()
    model, optimizer, rng, digest = model_fixture()
    replay = t.replay_pilot(model, optimizer, rng, ToyBank(), root, expected_coarse_digest=digest)
    assert replay['passed'] and replay['exact_model_optimizer_rng_sampling_losses_gradients']
    # Altering the recorded sampler history is a hard evidence failure.
    rows[0]['sampling']['target_light'] = 23
    (root / 'history_100.json').write_text(json.dumps(rows))
    model, optimizer, rng, digest = model_fixture()
    with pytest.raises(ValueError, match='evidence'):
        t.replay_pilot(model, optimizer, rng, ToyBank(), root, expected_coarse_digest=digest)


def test_midpilot_deadline_preserves_failure_and_partial_trace_without_success(tmp_path):
    model, optimizer, rng, digest = model_fixture()
    calls = [0]
    def clock():
        calls[0] += 1
        if calls[0] > 8:
            raise TimeoutError('procedural external-controller deadline')
    root = tmp_path / 'failed'
    with pytest.raises(TimeoutError):
        t.train_pilot(model, optimizer, rng, ToyBank(), root,
                       expected_coarse_digest=digest, provenance={'split': 'training'}, clock=clock)
    failed = json.loads((root / 'FAILED.json').read_text())
    assert failed['all_partial_artifacts_preserved'] and 0 < failed['completed_updates'] < 100
    assert (root / 'step_0.pt').is_file() and (root / 'trace.jsonl').is_file()
    assert not (root / 'summary.json').exists() and not (root / 'step_100.pt').exists()


def test_no_import_time_torch_job_or_controller_authority():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(t.__file__).read_text())
    imports = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    assert 'torch' not in imports
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in ('reserve', 'Popen', 'run') for n in tree.body)
