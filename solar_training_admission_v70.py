"""Verify measured training evidence before constructing a model or replaying it."""
import hashlib
import json
import math
import subprocess
from pathlib import Path
from datetime import datetime

SOURCES = (
    'train_solar_response_v63.py', 'solar_training_admission_v70.py',
    'run_solar_job_v70.py', 'solar_training_v63.py', 'solar_response_v63.py',
    'compact_lunar.py', 'contrastive_terrain.py', 'solar_compute_budget_v67.py',
    'dataset.py', 'matchers.py', 'geometry.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pinned(path, digest):
    if sha(path) != digest:
        raise ValueError('Frozen training evidence changed: ' + str(path))
    return read(path)


def validate_pilot_evidence(p):
    """A pair of protocol booleans is not measured reproducibility evidence."""
    pilot = pinned(p['pilot_protocol'], p['pilot_protocol_sha256'])
    summary = pinned(p['pilot_summary'], p['pilot_summary_sha256'])
    replay = pinned(p['pilot_replay_receipt'], p['pilot_replay_receipt_sha256'])
    run_receipt = pinned(p['pilot_watchdog_receipt'], p['pilot_watchdog_receipt_sha256'])
    replay_receipt = pinned(p['replay_watchdog_receipt'], p['replay_watchdog_receipt_sha256'])
    if pilot['updates'] != 100 or pilot['seed'] != 20261101:
        raise ValueError('The predefined primary 100-update pilot is required')
    for key in ('bank_sha256', 'source_sha256', 'initializer_sha256', 'pooled', 'cpu_threads'):
        if pilot[key] != p[key]:
            raise ValueError('Pilot and study configuration differ: ' + key)
    if not summary['complete'] or summary['updates'] != 100 or summary['protocol_sha256'] != sha(p['pilot_protocol']):
        raise ValueError('Complete source-bound pilot summary required')
    if not replay['passed'] or replay['updates'] != 100 or not replay['exact_model_optimizer_rng_loss_gradient_sampler']:
        raise ValueError('Actual exact optimizer/model/sampler replay required')
    if replay['protocol_sha256'] != sha(p['pilot_protocol']) or replay['pilot_summary_sha256'] != sha(p['pilot_summary']):
        raise ValueError('Exact replay belongs to another pilot')
    for receipt, mode in ((run_receipt, 'run'), (replay_receipt, 'replay')):
        if not receipt['complete'] or receipt['mode'] != mode or receipt['protocol_sha256'] != sha(p['pilot_protocol']):
            raise ValueError('Pilot/replay must finish inside their external watchdogs')
        if receipt['program']!='train_solar_response_v63.py' or receipt['child_returncode']!=0 or receipt['source_sha256']!=pilot['source_sha256']['run_solar_job_v70.py']:
            raise ValueError('Executed pilot watchdog/source/program differs')
        duration=(datetime.fromisoformat(receipt['absolute_deadline_utc'])-datetime.fromisoformat(receipt['started_utc'])).total_seconds()
        elapsed=receipt['calendar_seconds']
        if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or not 0<elapsed<=min(receipt['cap_seconds'],duration):
            raise ValueError('Pilot/replay exceeded its measured original deadline')
    if set(summary['checkpoint_sha256'])!={'step_0.pt','step_100.pt'}:
        raise ValueError('Both actual 0/100 pilot checkpoints are required')
    for name, digest in summary['checkpoint_sha256'].items():
        if sha(Path(p['pilot_summary']).parent / name) != digest:
            raise ValueError('Pilot checkpoint changed')
    if not all(isinstance(v,(int,float))and not isinstance(v,bool)and math.isfinite(v)for v in (summary['calendar_seconds'],p['measured_training_projection_seconds'],p['max_calendar_seconds'])):
        raise ValueError('Finite measured cost and fixed allowance required')
    projected = 2 * summary['calendar_seconds'] * 2000 / 100
    if projected <= 0 or abs(projected - p['measured_training_projection_seconds']) > 1e-6 or projected > p['max_calendar_seconds']:
        raise ValueError('Measured factor-two training cost does not fit the fixed allowance')


def validate_training_contract(protocol_path, commit, require_study=True):
    p = read(protocol_path)
    if len(commit or '') != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Exact source publication commit required for training and replay')
    if p.get('prepared_only', False) or p['updates'] not in (100, 2000) or p['seed'] not in (20261101, 20261102, 20261103) or p['cpu_threads'] != 2:
        raise ValueError('Fixed seed, update and CPU training contract required')
    if subprocess.check_output(['git', 'show', commit + ':' + str(protocol_path)]) != Path(protocol_path).read_bytes():
        raise ValueError('Training protocol was not published before execution')
    if not set(SOURCES).issubset(p['source_sha256']):
        raise ValueError('Training/replay dependency hashes are incomplete')
    for name, digest in p['source_sha256'].items():
        if sha(name) != digest or subprocess.check_output(['git', 'show', commit + ':' + name]) != Path(name).read_bytes():
            raise ValueError('Executed training source differs from freeze: ' + name)
    ledger = pinned(p['compute_ledger'], p['compute_ledger_sha256'])
    if not ledger['accounting_complete'] or ledger['remaining_training_wall_seconds'] < p['max_calendar_seconds']:
        raise ValueError('Original cumulative training budget is not established')
    bank = pinned(p['bank'], p['bank_sha256'])
    producer = pinned(bank['producer_completion_path'], p['bank_producer_completion_sha256'])
    if not producer['complete'] or producer['manifest_sha256'] != p['bank_sha256']:
        raise ValueError('A completed, sealed teacher bank is required')
    geometry = pinned(bank['geometry_audit_path'], bank['geometry_audit_sha256'])
    if not geometry['passed'] or not geometry['complete_dataset'] or geometry['completed_units'] != 96 or not .95 <= geometry['oracle_success'] <= 1:
        raise ValueError('Complete 96-unit geometry gate required before learning')
    if set(p['initializer_sha256']) != {'selected.pt', 'training.json', 'COMPLETE.json'}:
        raise ValueError('All frozen initializer artifacts must be pinned')
    for name, digest in p['initializer_sha256'].items():
        if sha(Path(p['initializer']) / name) != digest:
            raise ValueError('Frozen proposal initializer changed')
    if p['updates'] == 2000 and require_study:
        validate_pilot_evidence(p)
    return p
