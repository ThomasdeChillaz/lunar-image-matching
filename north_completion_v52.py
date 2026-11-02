"""Separate completion-only amendment after the original six-hour timeout.

Never resumes fitting or rewrites the original attempt. All existing files are
copied byte for byte to a new root. Only missing endpoint/selected-negative
inference runs, with the unchanged v39 evaluation function and a separately
published30minute allowance. The original protocol/deadline remain preserved.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

ORIGINAL = Path('runs/north-v39-expanded-control')
INVENTORY = 'reports/north_original_timeout_inventory_v52.json'
CAP = 1800


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_tree(root):
    root = Path(root)
    paths = sorted(p for p in root.rglob('*') if p.is_file() or p.is_symlink())
    if any(p.is_symlink() for p in paths): raise ValueError('Do not follow linked experiment files')
    return {str(p.relative_to(root)): sha(p) for p in paths}


def copy_original(source, destination, expected):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('New completion root must be separate from the original')
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in expected.items():
        original = source / name; target = destination / name
        if sha(original) != digest: raise ValueError('Original experiment changed during copy')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != digest: raise ValueError('Existing copied evidence differs; preserve and inspect')
            continue
        temporary = Path(str(target) + '.copying')
        if temporary.exists(): raise FileExistsError('Preserve interrupted copy before explicit repair')
        shutil.copy2(original, temporary)
        if sha(temporary) != digest: raise ValueError('Copy checksum mismatch')
        temporary.replace(target)
        if original.stat().st_ino == target.stat().st_ino and original.stat().st_dev == target.stat().st_dev:
            raise ValueError('Original and recovery files must not be hard-linked')


def verify_copies(source, destination, expected):
    for name, digest in expected.items():
        if sha(Path(source) / name) != digest or sha(Path(destination) / name) != digest:
            raise ValueError('An original or previously completed output was altered')


def completed_training(state):
    if (state['step'] != 2000 or len(state['history']) != 2000
            or [r['step'] for r in state['history']] != list(range(1, 2001))
            or {int(s['step']) for s in state['optimizer']['state'].values()} != {2000}):
        raise ValueError('All original fitting/Adam updates must already be complete')


def missing_cases(folder, cases):
    folder = Path(folder); expected = {hashlib.sha256(c['pair_id'].encode()).hexdigest() + '.json': c for c in cases}
    files = {p.name: p for p in folder.glob('*.json') if p.name != 'protocol.json'}
    if set(files) - set(expected): raise ValueError('Unexpected case in original partial cache')
    for name, path in files.items():
        row = read(path)
        if any(row[k] != v for k, v in expected[name].items()): raise ValueError('Cached case identity differs')
    return [c for c in cases if hashlib.sha256(c['pair_id'].encode()).hexdigest() + '.json' not in files]


def completion_budget(original, root, started):
    # This new object is used ONLY by the unchanged inference evaluator.
    # It is explicitly recorded; original protocol.json is never rewritten.
    return dict(original, root=str(root), created_utc=started.isoformat(),
        absolute_deadline_utc=(started + timedelta(seconds=CAP)).isoformat(),
        max_calendar_seconds=CAP, completion_only_amendment=True, additional_fitting_updates_allowed=0,
        original_absolute_deadline_utc=original['absolute_deadline_utc'], original_time_protocol_completed=False)


def original_ended(inventory):
    for path, digest in inventory['terminal_state_sha256'].items():
        if sha(path) != digest: raise ValueError('Original terminal state changed')
        state = read(path)
        if state['state'] != 'failed': raise ValueError('Original timeout/follow-up failure must stay recorded')
        for field in ('pid', 'child_pid'):
            if state.get(field) and subprocess.run(['ps', '-p', str(state[field]), '-o', 'command='],
                    capture_output=True, text=True).stdout.strip():
                raise ValueError('Original PID is still active or reused; inspect before completion')


def run(output, commit, resume=False):
    started = datetime.now(timezone.utc); tick = perf_counter(); inventory = read(INVENTORY)
    original_ended(inventory)
    software_path = 'reports/north_completion_software_checks_v52.json'; software = read(software_path)
    required = {'north_completion_v52.py', 'test_north_completion_v52.py', 'verify_north_completion_v52.py'}
    if (not software['passed'] or not software['full_repository_regression'] or not software['fresh_smoke']
            or not required.issubset(software['source_sha256'])):
        raise ValueError('Current source-bound regression and fresh smoke required')
    for name, digest in software['source_sha256'].items():
        if sha(name) != digest: raise ValueError('Completion source changed after checks')
    import torch
    from compact_lunar import CompactMatcher
    from evaluate_geometry_warp_v11 import control
    from north_training_bundle_v38 import load_bundle
    from search_window_learning_v18 import SearchWindowMatcher
    from train_north_control_v39 import (SOURCES, FineFeatureCache, counts, evaluate,
        read_rows, save_rows, selection_key, record_once, totals)
    from train_search_window_v18 import configure, make_model, verify_frozen
    configure(); sources = list(dict.fromkeys(SOURCES + sorted(required) + ['north_modern_comparison_v49.py',
        INVENTORY, 'reports/north_completion_contract_v52.json']))
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('Full public source revision required')
    for name in sources:
        if subprocess.check_output(['git', 'show', commit + ':' + name]) != Path(name).read_bytes():
            raise ValueError('Completion source differs from public freeze')
    if sha(ORIGINAL / 'protocol.json') != inventory['original_protocol_sha256']:
        raise ValueError('Original protocol changed')
    original_protocol = read(ORIGINAL / 'protocol.json'); root = Path(output)
    resolved_source, resolved_root = ORIGINAL.resolve(), root.resolve()
    if resolved_source == resolved_root or resolved_source in resolved_root.parents or resolved_root in resolved_source.parents:
        raise ValueError('Completion root must be separate before creating any output')
    contract = read('reports/north_completion_contract_v52.json')
    if (contract['original_inventory_sha256'] != sha(INVENTORY)
            or contract['original_protocol_sha256'] != sha(ORIGINAL / 'protocol.json')
            or snapshot_tree(ORIGINAL) != contract['original_file_sha256']):
        raise ValueError('Original evidence differs from public completion freeze')
    if root.exists():
        if not resume: raise FileExistsError('Preserve existing completion; explicit same-deadline resume required')
        recovery = read(root / 'recovery_protocol.json'); expected = recovery['original_file_sha256']
        if (recovery['publication_commit'] != commit or recovery['source_sha256'] != {n: sha(n) for n in sources}
                or recovery['software_check_sha256'] != sha(software_path)):
            raise ValueError('Recovery source/checks changed on resume')
        started = datetime.fromisoformat(recovery['created_utc'])
    else:
        expected = contract['original_file_sha256']; root.mkdir(parents=True)
        recovery = dict(created_utc=started.isoformat(), absolute_deadline_utc=(started + timedelta(seconds=CAP)).isoformat(),
            max_calendar_seconds=CAP, publication_commit=commit, source_sha256={n: sha(n) for n in sources},
            original_file_sha256=expected, original_inventory_sha256=sha(INVENTORY), software_check_sha256=sha(software_path),
            additional_fitting_updates=0, original_time_protocol_completed=False,
            scope='Separate completion-only inference allowance; original six-hour timeout and all original files stay unchanged. No new fitting, case/threshold/selection change or fresh-test claim.')
        record_once(root / 'recovery_protocol.json', recovery)
    copy_original(ORIGINAL, root, expected)
    if (root / 'COMPLETION_COMPLETE.json').exists():
        verify_copies(ORIGINAL, root, expected)
        for name, digest in read(root / 'COMPLETION_COMPLETE.json')['sha256'].items():
            if sha(root / name) != digest: raise ValueError('Sealed completion changed')
        return read(root / 'summary.json')
    budget = completion_budget(original_protocol, root, started)
    def clock():
        now = datetime.now(timezone.utc)
        if now < started or now >= datetime.fromisoformat(budget['absolute_deadline_utc']) or perf_counter() - tick >= CAP:
            raise TimeoutError('Separate original30minute completion allowance exhausted; do not reset')
    if sha('reports/north_training_data_v38.json') != original_protocol['manifest_sha256']:
        raise ValueError('Original training/selection manifest changed')
    clock(); bundle = load_bundle('reports/north_training_data_v38.json', selection_only=True)
    for seed in (20261101, 20261102, 20261103):
        original = ORIGINAL / f'coarse_adaptation-{seed}'
        state = torch.load(original / 'latest.pt', map_location='cpu', weights_only=True); completed_training(state)
        checkpoint = torch.load(original / 'checkpoint_step_2000.pt', map_location='cpu', weights_only=True)
        if checkpoint.keys() != state['model'].keys() or any(not torch.equal(v, checkpoint[k]) for k, v in state['model'].items()):
            raise ValueError('Original completed optimizer/checkpoint boundary differs')
    folder = root / 'coarse_adaptation-20261103'
    missing = missing_cases(folder / 'partial-selection-2000', bundle[0]['cases'])
    if not {c['pair_id'] for c in missing}.issubset(contract['missing_endpoint_case_ids']):
        raise ValueError('Missing cases differ from the published completion contract')
    if [c['pair_id'] for c in bundle[0]['negative_cases']] != contract['selected_negative_case_ids']:
        raise ValueError('Negative-case contract changed')
    record_once(root / 'completion_case_contract.json', dict(
        original_missing_endpoint_case_ids=contract['missing_endpoint_case_ids'],
        selected_negative_case_ids=[c['pair_id'] for c in bundle[0]['negative_cases']],
        unchanged_original_case_denominators=True))
    coarse = CompactMatcher(bundle[0]['coarse_model']); coarse.model.eval().requires_grad_(False)
    torch.manual_seed(20261103); model = make_model(coarse, 'coarse_adaptation')
    initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(latest['model'], strict=True); verify_frozen(model, initial); features = FineFeatureCache()
    try:
        clock(); endpoint_path = folder / 'selection_step_2000.json.gz'
        if not endpoint_path.exists():
            rows = evaluate(SearchWindowMatcher(coarse, model), bundle, bundle[0]['cases'],
                folder / 'partial-selection-2000', features, budget, tick)
            save_rows(endpoint_path, rows)
        else: rows = read_rows(endpoint_path)
        record_once(folder / 'validation_step_2000.json', dict(step=2000, selection=counts(rows), results_sha256=sha(endpoint_path)))
        validations = [read(folder / f'validation_step_{step}.json') for step in (0, 500, 1000, 1500, 2000)]
        winner = max(validations, key=lambda r: selection_key(r['selection'], r['step']))
        for name, step in (('final.pt', 2000), ('selected.pt', winner['step'])):
            source = folder / f'checkpoint_step_{step}.pt'; target = folder / name
            if not target.exists(): shutil.copy2(source, target)
            if sha(source) != sha(target): raise ValueError('Final/selected weights must copy an existing checkpoint')
        model.load_state_dict(torch.load(folder / 'selected.pt', map_location='cpu', weights_only=True), strict=True)
        verify_frozen(model, initial); clock()
        selected_control = control(SearchWindowMatcher(coarse, model)); record_once(folder / 'selected_controls.json', selected_control)
        negatives = evaluate(SearchWindowMatcher(coarse, model), bundle, bundle[0]['negative_cases'],
            folder / 'partial-selected-negatives', features, budget, tick)
        negative_path = folder / 'selected_negatives.json.gz'
        if not negative_path.exists(): save_rows(negative_path, negatives)
        elif read_rows(negative_path) != negatives: raise ValueError('Saved negative rows differ')
        meta = dict(seed=20261103, completed_steps=2000, selected_step=winner['step'], selection=winner['selection'],
            validation=validations, fitting_seconds=latest['fitting_seconds'], history=latest['history'],
            selected_negative_counts=totals(negatives), selected_checkpoint_sha256=sha(folder / 'selected.pt'),
            fine_encoder_unchanged=True, actual_selected_control_passed=bool(selected_control['passed']),
            eligible_for_adoption=bool(selected_control['passed']), calibrated_acceptance_evaluated=False)
        record_once(folder / 'summary.json', meta)
        record_once(folder / 'COMPLETE.json', dict(sha256={p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}))
        all_seeds = [read(root / f'coarse_adaptation-{seed}/summary.json') for seed in (20261101, 20261102, 20261103)]
        keys = ('seed', 'completed_steps', 'selected_step', 'selection', 'selected_negative_counts',
                'selected_checkpoint_sha256', 'fitting_seconds', 'actual_selected_control_passed', 'eligible_for_adoption')
        clock(); verify_copies(ORIGINAL, root, expected)
        summary = dict(complete=True, total_updates=6000, protocol_sha256=sha(root / 'protocol.json'),
            seeds=[{k: r[k] for k in keys} for r in all_seeds], independent_full_optimizer_retraining=False,
            exact_pilot_optimizer_updates=100, scope=recovery['scope'], calibrated_acceptance_evaluated=False,
            calibration_or_test_pixels_loaded=0, original_backbone_unchanged=True,
            original_time_protocol_completed=False, completion_only_amendment=True, additional_fitting_updates=0,
            recovery_protocol_sha256=sha(root / 'recovery_protocol.json'), recovery_calendar_seconds=(datetime.now(timezone.utc) - started).total_seconds())
        record_once(root / 'summary.json', summary)
        record_once(root / 'COMPLETION_COMPLETE.json', dict(sha256={str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'COMPLETION_COMPLETE.json'}))
        print(json.dumps(summary, indent=2), flush=True); return summary
    except BaseException as exc:
        record_once(root / ('completion-failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'),
            dict(error_type=type(exc).__name__, error=str(exc), original_time_protocol_completed=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default='runs/north-v52-completion-only')
    p.add_argument('--publication-commit', required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); run(a.output, a.publication_commit, a.resume)
