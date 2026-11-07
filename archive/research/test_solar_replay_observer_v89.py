"""Pure mocked replay-evidence tests; no Torch/model/NN/data construction."""
import json
import sys
from types import SimpleNamespace

import pytest

import solar_distributional_training_v88 as t


def fixture(tmp_path, monkeypatch, wrong_step=None):
    root = tmp_path / 'pilot'
    root.mkdir()
    rows = [dict(step=i, loss=i / 100, sampling={'unit_index': i % 3}) for i in range(1, 101)]
    (root / 'history_100.json').write_text(json.dumps(rows))
    for n in ('step_0.pt', 'step_100.pt'):
        (root / n).write_bytes(b'fake metadata snapshot; never decoded')
    (root / 'summary.json').write_text(json.dumps(dict(complete=True, format=t.PILOT_FORMAT,
        bank_manifest_sha256='a' * 64, updates=100,
        artifacts={n: t.sha(root / n) for n in ('step_0.pt', 'step_100.pt', 'history_100.json')})))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        load=lambda path, **kw: {'step': 0 if path.name == 'step_0.pt' else 100}))
    monkeypatch.setattr(t, 'snapshot', lambda model, opt, rng, step, settings: {'step': step})
    monkeypatch.setattr(t, 'prior_digest', lambda model: 'prior')
    monkeypatch.setattr(t, 'identical', lambda a, b: a == b)
    counter = [0]
    def actual_step(*a, **kw):
        counter[0] += 1
        i = counter[0]
        return dict(loss=(-1 if i == wrong_step else i / 100), sampling={'unit_index': i % 3})
    monkeypatch.setattr(t, 'fitting_step', actual_step)
    bank = SimpleNamespace(manifest_sha256='a' * 64, validate_model=lambda *a: None, preflight=lambda: None)
    return root, bank, rows


def test_observer_gets_exact100_new_rows_and_cannot_modify_comparison(tmp_path, monkeypatch):
    root, bank, expected = fixture(tmp_path, monkeypatch)
    saved = []
    def observer(row):
        saved.append(json.loads(json.dumps(row)))
        row['sampling']['unit_index'] = 'observer mutation must be isolated'
    receipt = t.replay_pilot(None, None, None, bank, root,
                            expected_coarse_digest='coarse', record_callback=observer)
    assert receipt['passed'] and saved == expected


def test_mismatch_row_is_observed_before_replay_fails(tmp_path, monkeypatch):
    root, bank, _ = fixture(tmp_path, monkeypatch, wrong_step=3)
    saved = []
    with pytest.raises(ValueError, match='step 3'):
        t.replay_pilot(None, None, None, bank, root,
                      expected_coarse_digest='coarse', record_callback=saved.append)
    assert [row['step'] for row in saved] == [1, 2, 3]
    assert saved[-1]['loss'] == -1


def test_observer_deadline_is_part_of_replay_work(tmp_path, monkeypatch):
    root, bank, _ = fixture(tmp_path, monkeypatch)
    saved = []
    def clock():
        if saved:
            raise TimeoutError('original work deadline reached during evidence write')
    with pytest.raises(TimeoutError, match='evidence'):
        t.replay_pilot(None, None, None, bank, root,
                      expected_coarse_digest='coarse', record_callback=saved.append, clock=clock)
    assert len(saved) == 1
