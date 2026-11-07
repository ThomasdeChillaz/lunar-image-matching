"""Pure metadata/fake-process qualification. No Torch/model/data/rays/jobs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import solar_distributional_pilot_v89 as q


def settings(mode='pilot'):
    created = datetime(2026, 11, 4, 11, tzinfo=timezone.utc)
    c = dict(training_pilot_cap_seconds=300, training_replay_cap_seconds=300,
             compute_ledger='ledger.json', compute_ledger_sha256='l' * 64,
             initializer='original-teacher', initializer_sha256={'selected.pt': 'w' * 64},
             expected_frozen_descriptor_sha256='t' * 64)
    p = dict(prepared_only=False, purpose='distributional_coarse100', mode=mode,
             category='training', cpu_threads=2, fixed_settings=deepcopy(q.FIXED),
             calibration_or_final_access=False, max_calendar_seconds=300,
             reservation_journal=q.CANONICAL, compute_ledger=c['compute_ledger'],
             compute_ledger_sha256=c['compute_ledger_sha256'], teacher_directory=c['initializer'],
             teacher_sha256=c['initializer_sha256'], expected_frozen_descriptor_sha256=c['expected_frozen_descriptor_sha256'],
             created_utc=created.isoformat(), publication_dispatch_lease_seconds=1800,
             latest_start_utc=(created + timedelta(seconds=1800)).isoformat(),
             absolute_deadline_utc=(created + timedelta(seconds=2100)).isoformat(),
             output='runs/' + mode, bank_manifest_sha256='a' * 64, bank_prior_sha256='b' * 64)
    return p, c, created


@pytest.mark.parametrize('mode', ['pilot', 'replay'])
def test_each_new_mode_has_own_fresh_startup_clock_and_original300_limit(mode):
    p, c, created = settings(mode)
    start = created + timedelta(seconds=20)
    assert q.validate_settings(p, c, start) == start + timedelta(seconds=300)
    with pytest.raises(TimeoutError):
        q.validate_settings(p, c, q.utc(p['latest_start_utc']) + timedelta(microseconds=1))
    with pytest.raises(TimeoutError):
        q.validate_settings(p, c, created - timedelta(microseconds=1))


@pytest.mark.parametrize('change', [
    {'prepared_only': True}, {'category': 'baseline'}, {'cpu_threads': 4},
    {'max_calendar_seconds': 301}, {'max_calendar_seconds': True},
    {'reservation_journal': 'runs/fresh-free-journal.json'}, {'output': '../elsewhere'},
    {'output': '/tmp/elsewhere'}, {'calibration_or_final_access': True},
    {'teacher_directory': 'different-teacher'}, {'publication_dispatch_lease_seconds': 1801},
])
def test_fixed_math_original_caps_and_canonical_identity_cannot_change(change):
    p, c, created = settings()
    p.update(change)
    with pytest.raises(ValueError):
        q.validate_settings(p, c, created)


def test_settings_do_not_allow_loss_change_or_deadline_extension():
    p, c, created = settings()
    p['fixed_settings']['marginal_weight'] = 0.
    with pytest.raises(ValueError):
        q.validate_settings(p, c, created)
    p, c, created = settings()
    p['absolute_deadline_utc'] = (q.utc(p['absolute_deadline_utc']) + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError):
        q.validate_settings(p, c, created)


def test_reservation_identity_is_protocol_bound_and_mode_specific(tmp_path):
    path = tmp_path / 'protocol.json'
    path.write_text('frozen protocol bytes')
    assert q.job_id(path, {'mode': 'pilot'}) == q.sha(path) + '/distributional100_pilot'
    assert q.job_id(path, {'mode': 'replay'}) == q.sha(path) + '/distributional100_replay'


def publication_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, 'solar_learning_pipeline_v73', SimpleNamespace(
        local_source_closure=lambda: {'base.py': q.sha('base.py')}))
    for name in ('base.py', *q.EXTRA_SOURCES):
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fake source, never imported\n')
    p = dict(mode='pilot', learning_contract='contract.json', bank_manifest='bank.json',
             bank_prior='prior.json', bank_protocol='bank-protocol.json', bank_run='bank-run')
    for name in q.metadata_paths(p):
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    p['source_sha256'] = {n: q.sha(n) for n in ('base.py', *q.EXTRA_SOURCES)}
    p['metadata_sha256'] = {n: q.sha(n) for n in q.metadata_paths(p)}
    Path('job.json').write_text(json.dumps(p))
    return p


def test_source_preflight_uses_remaining_original_time_not_fresh_perfile_cap(tmp_path, monkeypatch):
    p = publication_fixture(tmp_path, monkeypatch)
    started = datetime(2026, 11, 4, 11, tzinfo=timezone.utc)
    current = [started]
    monkeypatch.setattr(q, 'now', lambda: current[0])
    timeouts = []
    def public(args, timeout):
        timeouts.append(timeout)
        current[0] += timedelta(seconds=1)
        return Path(args[-1].split(':', 1)[1]).read_bytes()
    monkeypatch.setattr(q.subprocess, 'check_output', public)
    with pytest.raises(TimeoutError):
        q.verify_sources('job.json', p, 'a' * 40, started + timedelta(seconds=2))
    assert timeouts == [2., 1.]


def pilot_fixture(tmp_path):
    p, _, created = settings()
    path = tmp_path / 'pilot-protocol.json'
    path.write_text(json.dumps(p))
    root = tmp_path / 'pilot'
    science = root / 'science'
    science.mkdir(parents=True)
    rows = [dict(step=i, loss=i / 100, sampling={'unit_index': i % 4}) for i in range(1, 101)]
    (science / 'history_100.json').write_text(json.dumps(rows))
    (science / 'trace.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in rows))
    for name in ('step_0.pt', 'step_100.pt'):
        (science / name).write_bytes(b'fake checkpoint; never loaded')
    (science / 'execution.json').write_text('{}')
    summary = dict(complete=True, format='distributional-coarse128-pilot-v88', updates=100,
                   seed=20261101, bank_manifest_sha256=p['bank_manifest_sha256'],
                   model_selection_performed=False, calibration_or_final_access=False, model_promoted=False,
                   artifacts={name: q.sha(science / name) for name in
                              ('step_0.pt', 'step_100.pt', 'history_100.json', 'trace.jsonl', 'execution.json')})
    (science / 'summary.json').write_text(json.dumps(summary))
    a = dict(complete=True, mode='pilot', protocol_sha256=q.sha(path), updates=100,
             bank_manifest_sha256=p['bank_manifest_sha256'], bank_prior_sha256=p['bank_prior_sha256'],
             science_summary_sha256=q.sha(science / 'summary.json'))
    (root / 'worker_summary.json').write_text(json.dumps(a))
    start = created + timedelta(seconds=2)
    w = dict(complete=True, protocol_sha256=q.sha(path), calendar_seconds=10., cap_seconds=300,
             child_returncode=0, failure=None, started_utc=start.isoformat(),
             ended_utc=(start + timedelta(seconds=10)).isoformat(),
             hard_deadline_utc=(start + timedelta(seconds=300)).isoformat())
    (root / 'watchdog.json').write_text(json.dumps(w))
    seal = dict(external_watchdog_passed=True, protocol_sha256=q.sha(path),
                worker_summary_sha256=q.sha(root / 'worker_summary.json'), watchdog_sha256=q.sha(root / 'watchdog.json'))
    (root / 'COMPLETE.json').write_text(json.dumps(seal))
    return path, p, root, a, rows


def test_completed_pilot_remains_valid_long_after_its_old_dispatch_lease(tmp_path, monkeypatch):
    path, p, root, _, _ = pilot_fixture(tmp_path)
    monkeypatch.setattr(q, 'now', lambda: datetime(2030, 1, 1, tzinfo=timezone.utc))
    original, _, _ = q.completed_pilot(path, root)
    assert original == p


def test_completed_pilot_cannot_inherit_extended_old_deadline(tmp_path):
    path, _, root, _, _ = pilot_fixture(tmp_path)
    w = q.read(root / 'watchdog.json')
    w['hard_deadline_utc'] = (q.utc(w['hard_deadline_utc']) + timedelta(seconds=1)).isoformat()
    (root / 'watchdog.json').write_text(json.dumps(w))
    seal = q.read(root / 'COMPLETE.json')
    seal['watchdog_sha256'] = q.sha(root / 'watchdog.json')
    (root / 'COMPLETE.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='original startup'):
        q.completed_pilot(path, root)


@pytest.mark.parametrize('corruption', ['missing_checkpoint_hash', 'wrong_format', 'wrong_bank', 'short_trace'])
def test_pilot_requires_all_five_actual_artifacts_and100_trace_rows(tmp_path, corruption):
    _, p, root, a, _ = pilot_fixture(tmp_path)
    science = root / 'science'
    summary = q.read(science / 'summary.json')
    if corruption == 'missing_checkpoint_hash':
        summary['artifacts'].pop('step_0.pt')
    elif corruption == 'wrong_format':
        summary['format'] = 'old-canonical-study'
    elif corruption == 'wrong_bank':
        summary['bank_manifest_sha256'] = 'z' * 64
    else:
        (science / 'trace.jsonl').write_text((science / 'trace.jsonl').read_text().splitlines()[0] + '\n')
        summary['artifacts']['trace.jsonl'] = q.sha(science / 'trace.jsonl')
    (science / 'summary.json').write_text(json.dumps(summary))
    a['science_summary_sha256'] = q.sha(science / 'summary.json')
    with pytest.raises(ValueError):
        q.verify_worker_evidence(p, root, a)


def test_replay_requires_actual_saved100_rows_not_only_a_passed_flag(tmp_path):
    _, p, original, _, rows = pilot_fixture(tmp_path)
    p.update(mode='replay', original_pilot_run=str(original))
    root = tmp_path / 'replay'
    root.mkdir()
    receipt = dict(passed=True, updates=100, exact_model_optimizer_rng_sampling_losses_gradients=True,
                   original_summary_sha256=q.sha(original / 'science' / 'summary.json'))
    (root / 'replay.json').write_text(json.dumps(receipt))
    (root / 'replay_trace.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    a = dict(complete=True, mode='replay', updates=100, bank_manifest_sha256=p['bank_manifest_sha256'],
             bank_prior_sha256=p['bank_prior_sha256'], replay_sha256=q.sha(root / 'replay.json'),
             replay_trace_sha256=q.sha(root / 'replay_trace.jsonl'))
    q.verify_worker_evidence(p, root, a)
    (root / 'replay_trace.jsonl').write_text('')
    a['replay_trace_sha256'] = q.sha(root / 'replay_trace.jsonl')
    with pytest.raises(ValueError, match='trace'):
        q.verify_worker_evidence(p, root, a)


def test_controller_import_has_no_scientific_imports_or_reservations():
    import ast
    tree = ast.parse(Path(q.__file__).read_text())
    imports = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
    assert not set(imports) & {'torch', 'numpy', 'cv2', 'PIL'}
