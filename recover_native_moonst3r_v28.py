"""Explicit supplemental phase for a calendar-expired common-native comparison.

Never resumes or rewrites the expired run. A public contract freezes the
existing case hashes and only missing identities before supplemental inference.
This is an amended development comparison, not an uninterrupted original run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from dataset import sha256
from evaluate_native_moonst3r_v19 import SOURCES as ORIGINAL_SOURCES
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import propose_pose, score
from native_moonst3r_v19 import NativeMOONSt3R

SOURCES = list(dict.fromkeys(ORIGINAL_SOURCES+['recover_native_moonst3r_v28.py', 'verify_moonst3r_recovery_v28.py']))


def inventory(root):
    root = Path(root); protocol = json.loads((root/'protocol.json').read_text())
    rows = {}; files = {}
    for path in sorted((root/'cases').glob('*.json')):
        row = json.loads(path.read_text())
        if row['pair_id'] not in protocol['case_ids'] or row['pair_id'] in rows:raise ValueError('Unexpected/duplicate original case')
        rows[row['pair_id']] = row; files[path.name] = sha256(path)
    missing = [i for i in protocol['case_ids'] if i not in rows]
    return protocol, rows, files, missing


def freeze_contract(original, output):
    root = Path(original); protocol, rows, files, missing = inventory(root)
    elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(protocol['created_utc'])).total_seconds()
    observation = root/'CALENDAR_DEADLINE_OBSERVATION.json'
    if len(rows) != 71 or len(missing) != 1 or len(protocol['case_ids']) != 72 or (root/'summary.json').exists():
        raise ValueError('This amendment permits only the recorded71-case interrupted comparison and its single missing identity')
    if elapsed <= protocol['wall_cap_seconds'] or not observation.exists():raise ValueError('Recorded calendar-expired run required')
    if root.is_absolute() or '..' in root.parts:raise ValueError('Use a project-relative original run')
    value = dict(scope='Explicit amended two-phase exposed-development comparison. Original four-calendar-hour run remains incomplete; supplemental phase may complete only its single missing case.',
        original_run=str(root), original_protocol_sha256=sha256(root/'protocol.json'), original_case_sha256=files,
        original_declared_wall_cap_seconds=protocol['wall_cap_seconds'], deadline_observation_sha256=sha256(observation),
        original_complete_cases=71, missing_case_ids=missing, all_case_ids=protocol['case_ids'],
        original_recorded_case_inference_seconds=sum(r['elapsed_seconds'] for r in rows.values()),
        additional_calendar_budget_seconds=600, inference_configuration_changes_allowed=False,
        reason='Host clock/suspension discontinuity: calendar deadline exceeded while monotonic alarm remained active. Original was stopped with its own timeout handler. Additional fixed-case phase is declared separately; no deadline reset or accuracy-based duplicate selection.',
        source_sha256={p: sha256(p) for p in SOURCES}, frozen_utc=datetime.now(timezone.utc).isoformat())
    write_new(output, value); return value


def check_deadline(created, limit, now=None):
    now = now or datetime.now(timezone.utc)
    if (now-created).total_seconds() > limit:raise TimeoutError('Supplemental calendar deadline expired; preserve attempt, never silently restart')


def run(output, contract_path, commit, dependency_dir):
    root = Path(output)
    if root.exists():raise FileExistsError('Do not overwrite or resume supplemental attempts')
    if not re.fullmatch('[0-9a-f]{40}', commit):raise ValueError('Full public source freeze required')
    contract = json.loads(Path(contract_path).read_text()); original = Path(contract['original_run'])
    old, old_rows, old_files, missing = inventory(original)
    if old_files != contract['original_case_sha256'] or missing != contract['missing_case_ids']:
        raise ValueError('Frozen original rows/missing identities changed')
    if (sha256(original/'protocol.json') != contract['original_protocol_sha256']
            or sha256(original/'CALENDAR_DEADLINE_OBSERVATION.json') != contract['deadline_observation_sha256']
            or contract['additional_calendar_budget_seconds'] != 600 or len(missing) != 1):raise ValueError('Recovery contract changed')
    for name in SOURCES:
        if sha256(name) != contract['source_sha256'][name] or subprocess.check_output(['git', 'show', commit+':'+name]) != Path(name).read_bytes():
            raise ValueError('Public supplemental source differs: '+name)
    for name, want in old['source_sha256'].items():
        if sha256(name) != want:raise ValueError('Original matching implementation changed')
    created = datetime.now(timezone.utc); started = perf_counter(); root.mkdir(parents=True); (root/'cases').mkdir(); (root/'source').mkdir()
    protocol = dict(old, source_commit=commit, source_sha256={n: sha256(n) for n in SOURCES},
        scope=contract['scope'], created_utc=created.isoformat(), wall_cap_seconds=600,
        resource_policy='600 additional calendar seconds for only the frozen missing case; UTC checked before/after expensive operations, plus alarm. Original experiment remains expired/incomplete.',
        recovery_contract_path=str(contract_path), recovery_contract_sha256=sha256(contract_path),
        original_protocol_sha256=contract['original_protocol_sha256'], original_run=str(original),
        original_complete_cases=71, supplemental_case_ids=missing)
    write_new(root/'protocol.json', protocol)
    for name in SOURCES:shutil.copy2(name, root/'source'/name)
    for name in old_files:shutil.copy2(original/'cases'/name, root/'cases'/name)
    def timeout(*unused):raise TimeoutError('Supplemental alarm expired')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    try:
        torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
        check_deadline(created, 600)
        manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
        if sha256(manifest_path) != protocol['manifest_sha256']:raise ValueError('Common manifest changed')
        for name, want in protocol['input_sha256'].items():
            if sha256(name) != want:raise ValueError('Common native input changed')
        model = NativeMOONSt3R(dependency_dir=dependency_dir, threads=2)
        if model.provenance != json.loads((original/'provenance.json').read_text()):raise ValueError('Supplemental pretrained configuration differs')
        write_new(root/'provenance.json', model.provenance); check_deadline(created, 600)
        cases = {c['pair_id']: c for c in manifest['cases'] if c['split'] == 'model_selection'}
        tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
        for identity in missing:
            case = cases[identity]; tile = tiles[case['tile_id']]
            with np.load(tile['native_cache'], allow_pickle=False) as z:data = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
            a, b = tile['cameras']; failure = None; check_deadline(created, 600); tick = perf_counter()
            try:
                p, q = model.points(data['images'][2*case['light_a']], data['images'][2*case['light_b']+1])
                check_deadline(created, 600)
                pose, info = propose_pose(p, q, data['depth_a'], a['intrinsics'], a['cam2world'], b['intrinsics'])
            except (RuntimeError, cv2.error) as exc:
                p = q = np.empty((0, 2)); pose = None; info = dict(inliers=0, reason='inference_error')
                failure = dict(type=type(exc).__name__, message=str(exc))
            elapsed = perf_counter()-tick; check_deadline(created, 600)
            row = dict(case, method='moonst3r_native', elapsed_seconds=elapsed, inference_error=failure, pose_info=info,
                **score(p, q, pose, data, tile['cameras']))
            check_deadline(created, 600)
            write_new(root/'cases'/(hashlib.sha256(identity.encode()).hexdigest()[:20]+'.json'), row)
        rows = [json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
        if len(rows) != 72 or {r['pair_id'] for r in rows} != set(protocol['case_ids']):raise ValueError('Amended denominator incomplete')
        check_deadline(created, 600)
        summary = dict(scope=contract['scope'], cases=72, correct=sum(r['pose_correct'] for r in rows),
            strict=sum(r['strict_pose_correct'] for r in rows), wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows),
            inference_errors=sum(r['inference_error'] is not None for r in rows), median_inference_seconds=float(np.median([r['elapsed_seconds'] for r in rows])),
            elapsed_this_invocation_seconds=perf_counter()-started, supplemental_calendar_seconds=(datetime.now(timezone.utc)-created).total_seconds(),
            protocol_sha256=sha256(root/'protocol.json'), case_sha256={p.name: sha256(p) for p in sorted((root/'cases').glob('*.json'))},
            original_cases_preserved=71, supplemental_cases=1, original_calendar_protocol_completed=False,
            calibration_or_test_pixels_loaded=0)
        write_new(root/'summary.json', summary); print(json.dumps(summary, indent=2)); return summary
    except Exception as exc:
        write_new(root/'FAILURE.json', dict(error_type=type(exc).__name__, error=str(exc), scope='Supplemental attempt failed/incomplete; preserve original expired run and all completed rows.'))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--freeze-contract'); p.add_argument('--contract-output')
    p.add_argument('--output'); p.add_argument('--contract'); p.add_argument('--source-commit'); p.add_argument('--dependency-dir'); a = p.parse_args()
    if a.freeze_contract:freeze_contract(a.freeze_contract, a.contract_output)
    else:run(a.output, a.contract, a.source_commit, a.dependency_dir)
