"""New-objective integration and exact continuation; no downloaded data."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from compact_lunar import CompactDescriptor
from context_supervision_v15 import proposal_grid
import train_mutual_pilot_v23 as pilot
import train_mutual_cohort_v24 as cohort
import train_search_window_v18 as engine
from dataset import sha256


def test_objective_binding_is_local_to_scope_and_restored_after_failure(monkeypatch):
    previous_step, previous_sources = engine.fitting_step, engine.SOURCES
    monkeypatch.setattr(cohort, 'fit_step', lambda *args: 'new-objective')
    with pytest.raises(RuntimeError, match='controlled interruption'):
        with cohort.objective_binding():
            assert engine.SOURCES is cohort.SOURCES
            assert engine.fitting_step(None, None, None, None, None, 1, 'coarse_adaptation') == 'new-objective'
            with pytest.raises(ValueError):engine.fitting_step(None, None, None, None, None, 1, 'window_only')
            raise RuntimeError('controlled interruption')
    assert engine.fitting_step is previous_step and engine.SOURCES is previous_sources


def test_resume_rejects_changed_clock_and_enlarged_budget_even_with_a_new_seal(tmp_path):
    expected = dict(publication_commit='a'*40, objective='dual_window_mass_v23', steps=2000, interval=500,
        pilot=False, variants=['coarse_adaptation'], seeds=[20261101, 20261102, 20261103],
        cohort_wall_cap_seconds=21600, per_invocation_wall_cap_seconds=21600, temperature=.1,
        assignment_correction_weight=1., created_utc='2026-11-01T00:00:00+00:00')
    def save(protocol, seal=True):
        (tmp_path/'protocol.json').write_text(json.dumps(protocol))
        if seal:(tmp_path/'PROTOCOL_SEAL.json').write_text(json.dumps(dict(protocol_sha256=sha256(tmp_path/'protocol.json'))))
    save(expected); assert cohort.frozen_protocol(tmp_path, 'a'*40) == expected
    save(dict(expected, created_utc='2026-11-02T00:00:00+00:00'), False)
    with pytest.raises(ValueError, match='seal'):cohort.frozen_protocol(tmp_path, 'a'*40)
    save(dict(expected, cohort_wall_cap_seconds=999999))
    with pytest.raises(ValueError, match='configuration'):cohort.frozen_protocol(tmp_path, 'a'*40)


def sample(tiles, prepared, rng, negative_pair=False):
    images = torch.from_numpy(rng.random((2, 1, 128, 128), dtype=np.float32))
    grid = torch.from_numpy(proposal_grid()); visible = torch.full((676,), not negative_pair)
    return images, dict(forward=grid, reverse=grid, forward_visible=visible, reverse_visible=visible), dict(verified_nonoverlap=negative_pair)


def test_new_objective_changes_coarse_weights_preserves_fine_and_resumes_exactly(monkeypatch):
    torch.set_num_threads(1); torch.manual_seed(123)
    coarse = SimpleNamespace(model=CompactDescriptor(False).eval(), metadata=dict(variant='expanded'), device='cpu')
    model = engine.make_model(coarse, 'coarse_adaptation'); opt = engine.make_optimizer(model); rng = np.random.default_rng(22)
    initial = copy.deepcopy(model.state_dict()); monkeypatch.setattr(pilot, 'sample_training', sample)
    first = pilot.fit_step(model, opt, {}, None, rng, 1)
    assert first['labels']['mutual_forward_correction'] > 0
    saved = copy.deepcopy(engine.checkpoint(model, opt, rng, 1, [first], 0))
    expected = pilot.fit_step(model, opt, {}, None, rng, 2); weights = copy.deepcopy(model.state_dict())
    engine.verify_frozen(model, initial)
    assert all(p.grad is None for p in model.fine.parameters())
    assert any(not torch.equal(v, initial['coarse.'+k]) for k, v in model.coarse.state_dict().items())
    restored = engine.make_model(coarse, 'coarse_adaptation'); other = engine.make_optimizer(restored); generator = np.random.default_rng(999)
    engine.restore(saved, restored, other, generator)
    assert pilot.fit_step(restored, other, {}, None, generator, 2) == expected
    assert all(torch.equal(v, weights[k]) for k, v in restored.state_dict().items())
