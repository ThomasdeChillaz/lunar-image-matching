"""Bounded single-seed northern cross-image context study.

Existing attention architecture, not novelty. Same full training data, sampler,
loss, coarse/fine initialization and optimizer rates; context capacity +512.
All96positive selection pairs at five checkpoints and96selected negatives.
No confidence calibration or final inference. Never reset the original deadline.
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from benchmark_north_control_v38 import inference_inputs, score_case, totals
from compact_lunar import CompactMatcher
from evaluate_geometry_warp_v11 import control
from geometry_warp_native_v11 import infer
from lunargeo_dataset import write_new
from north_training_bundle_v38 import load_bundle, prepare_training, sha
from search_window_learning_v18 import SearchWindowMatcher
from train_balanced_localization_v3 import save_torch
from train_north_context_v61 import SOURCES as PILOT_SOURCES, fitting_step
from train_search_window_v18 import (configure, make_model, make_optimizer,
    record_once, selection_key, verify_frozen, weights_once)

SEEDS = (20261101,)
STEPS = 2000
INTERVAL = 500
MANIFEST = 'reports/north_training_data_v38.json'
PILOT = 'runs/north-v61-context-cost-pilot'
PILOT_AUDIT = 'reports/north_context_pilot_audit_v61.json'
BANK = 'runs/north-v38-baseline-cost-bank'
BASELINES = 'runs/north-v39-full-baselines'
BASELINE_AUDIT = 'reports/north_full_baseline_audit_v39.json'
SOURCES = list(dict.fromkeys(PILOT_SOURCES + [
    'benchmark_north_control_v38.py', 'train_north_control_v39.py',
    'verify_north_results_v39.py','train_north_context_study_v61.py',
    'verify_north_context_study_v61.py','test_north_context_study_v61.py']))
CAP_SECONDS = 3 * 3600


class FineFeatureCache(OrderedDict):
    """Bound the native maps to ~2.2GB (64 maps of 32x512x512 float32)."""
    def __init__(self, capacity=64):
        super().__init__(); self.capacity=capacity

    def __getitem__(self,key):
        value=super().__getitem__(key);self.move_to_end(key);return value

    def __setitem__(self,key,value):
        super().__setitem__(key,value);self.move_to_end(key)
        while len(self)>self.capacity:self.popitem(last=False)


def read(path):
    return json.loads(Path(path).read_text())


def public_sources(commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Exact public source revision required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', commit + ':' + name]) != Path(name).read_bytes():
            raise ValueError('Cohort source differs from public freeze: ' + name)


def cost_admission(fitting_seconds,case_seconds):
    estimate=2*(fitting_seconds/100*STEPS+case_seconds*576+60)
    if (not np.isfinite([fitting_seconds,case_seconds,estimate]).all()
            or min(fitting_seconds,case_seconds)<=0 or estimate>=CAP_SECONDS):
        raise ValueError('Measured cross-context pilot exceeds fixed three-hour allowance')
    return float(estimate)


def guards():
    """Return immutable gate hashes and costs; a partial bank cannot pass."""
    audit = read(PILOT_AUDIT)
    pilot = read(Path(PILOT) / 'summary.json')
    if (not audit['passed'] or audit['completed_steps'] != 100
            or not audit['exact_full_optimizer_retraining']
            or not audit['exact_model_optimizer_losses_gradients_rng']
            or audit['pilot_summary_sha256'] != sha(Path(PILOT) / 'summary.json')
            or audit['pilot_protocol_sha256'] != sha(Path(PILOT) / 'protocol.json')
            or pilot['protocol_sha256'] != sha(Path(PILOT) / 'protocol.json')
            or not pilot['complete'] or pilot['model_selection_evaluated']):
        raise ValueError('Actual independently replayed train-only pilot required')
    bank = read(Path(BANK) / 'summary.json')
    if not bank['complete'] or bank['full'] or bank['eligible_cases_per_method'] != 16:
        raise ValueError('Complete fixed 16-case cost bank required')
    if bank['results_sha256'] != sha(Path(BANK) / 'results.json.gz'):
        raise ValueError('Measured cost bank changed')
    baseline = read(Path(BASELINES) / 'summary.json')
    if (not baseline['complete'] or not baseline['full']
            or baseline['eligible_cases_per_method'] != 192
            or baseline['results_sha256'] != sha(Path(BASELINES) / 'results.json.gz')):
        raise ValueError('Both full 96-positive/96-negative common baselines required')
    replay = read(BASELINE_AUDIT)
    if (not replay['passed'] or replay['replayed_rows'] != 384
            or replay['summary_sha256'] != sha(Path(BASELINES) / 'summary.json')
            or replay['results_sha256'] != baseline['results_sha256']
            or replay['source_sha256'] != sha('verify_north_results_v39.py')):
        raise ValueError('Full common baseline metrics must independently replay')
    for path in (Path(PILOT) / 'protocol.json', Path(BANK) / 'protocol.json',
                 Path(BASELINES) / 'protocol.json'):
        protocol = read(path)
        if protocol['manifest_sha256'] != sha(MANIFEST):
            raise ValueError('Pilot/baseline used a different dataset')
        for name, digest in protocol['source_sha256'].items():
            if sha(name) != digest:
                raise ValueError('Gate source changed: ' + name)
    # Measured fitting cost, plus all checkpoint and selected-negative poses.
    # This is only a conservative admission estimate, never a runtime result.
    measured = pilot['median_native_case_seconds']
    if pilot['train_cost_bank_sha256']!=sha(Path(PILOT)/'train_cost_bank.json') or pilot['fixed_train_cost_cases']!=4:
        raise ValueError('Measured cross-context native training cost bank required')
    estimate = cost_admission(pilot['fitting_seconds'],measured)
    if not np.isfinite(estimate) or estimate > CAP_SECONDS:
        raise ValueError('Measured projected cohort exceeds fixed three-hour allowance')
    return dict(pilot_audit_sha256=sha(PILOT_AUDIT),
        pilot_summary_sha256=sha(Path(PILOT) / 'summary.json'),
        cost_bank_sha256=sha(Path(BANK) / 'summary.json'),
        full_baseline_sha256=sha(Path(BASELINES) / 'summary.json'),
        full_baseline_audit_sha256=sha(BASELINE_AUDIT),
        conservative_projected_seconds=float(estimate))


def counts(rows):
    result = totals(rows)
    return dict(result, cases=len(rows), correct_poses=result['correct_raw_poses'],
        strict_poses=result['strict_raw_poses'],
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
        returned=sum(r['returned'] for r in rows),
        scorable=sum(r['scorable'] for r in rows))


def clock_check(protocol, invocation_tick):
    now = datetime.now(timezone.utc)
    started = datetime.fromisoformat(protocol['created_utc'])
    deadline = datetime.fromisoformat(protocol['absolute_deadline_utc'])
    if now < started or now >= deadline or perf_counter() - invocation_tick >= CAP_SECONDS:
        raise TimeoutError('Original cohort deadline reached; retain partial evidence, do not reset clock')


def check_seal(protocol):
    if sha(MANIFEST) != protocol['manifest_sha256']:
        raise ValueError('Dataset changed after cohort freeze')
    for name, digest in protocol['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Frozen fitting/scoring source changed: ' + name)


def save_rows(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never replace a completed evaluation')
    temporary = Path(str(path) + '.temporary')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(rows, stream, allow_nan=False)
    temporary.replace(path)


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def evaluate(matcher, bundle, cases, folder, features, protocol, tick):
    """Save each case once; an interrupted evaluation resumes that checkpoint."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    checkpoint_cases = [c['pair_id'] for c in cases]
    record_once(folder / 'protocol.json', dict(case_ids=checkpoint_cases,
        cohort_protocol_sha256=sha(Path(protocol['root']) / 'protocol.json')))
    rows = []
    for case in cases:
        clock_check(protocol, tick)
        name = hashlib.sha256(case['pair_id'].encode()).hexdigest() + '.json'
        path = folder / name
        if path.exists():
            row = read(path)
            if any(row[k] != v for k, v in case.items()):
                raise ValueError('Saved evaluation case changed')
        else:
            inputs = inference_inputs(bundle, case)
            ids = ((case['tile_id'], 2 * case['light_a']),
                   (case.get('query_tile_id', case['tile_id']), 2 * case['light_b'] + 1))
            before = perf_counter()
            error = None
            try:
                chosen, branches = infer(matcher, *inputs, features, ids)
                branch = chosen if chosen is not None else branches[0]
                p, q = branch['p'], branch['q']
                pose = None if chosen is None else branch['pose']
            except (RuntimeError, ValueError, cv2.error) as exc:
                p = q = np.empty((0, 2)); pose = None; branch = {}
                error = dict(type=type(exc).__name__, message=str(exc))
            elapsed_ms = (perf_counter() - before) * 1000
            row = dict(case, method='cross_image_context_control',
                **score_case(bundle, case, p, q, pose),
                elapsed_ms=elapsed_ms, timing_scope='Frozen fine-map cache reused; not isolated latency.',
                inference_error=error, quarter_turn=branch.get('quarter_turn'),
                pose_info=branch.get('info', {}), geometry_history=branch.get('geometry_history', []))
            write_new(path, row)
        rows.append(row)
    if len(rows) != len(cases) or len({r['pair_id'] for r in rows}) != len(cases):
        raise ValueError('Incomplete or duplicate selection denominator')
    return rows


def initial_equivalence(rows, baseline_rows):
    frozen = {r['pair_id']: r for r in baseline_rows if r['method'] == 'frozen_surface'}
    fields = ('reference_points_native', 'query_points_native', 'estimated_cam2world',
              'pose_correct', 'strict_pose_correct')
    for row in rows:
        old = frozen[row['pair_id']]
        if any(row[key] != old[key] for key in fields):
            raise ValueError('Expanded control initialization differs from frozen baseline: ' + row['pair_id'])


def snapshot(model, optimizer, rng, step, history, fitting_seconds):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        step=step, history=history, fitting_seconds=fitting_seconds)


def one_seed(root, seed, coarse, train_bundle, selection_bundle, prepared, protocol, tick, baseline):
    folder = root / ('cross_adaptation-' + str(seed))
    if (folder / 'COMPLETE.json').exists():
        for name, digest in read(folder / 'COMPLETE.json')['sha256'].items():
            if sha(folder / name) != digest:
                raise ValueError('Completed seed evidence changed: ' + name)
        return read(folder / 'summary.json')
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = make_model(coarse, 'cross_adaptation'); optimizer = make_optimizer(model)
    if sum(p.numel() for p in model.parameters()) != 82208:
        raise ValueError('Control architecture changed')
    config = dict(seed=seed, protocol_sha256=sha(root / 'protocol.json'))
    if not folder.exists():
        folder.mkdir()
        write_new(folder / 'config.json', config)
        save_torch(model.state_dict(), folder / 'initial.pt')
        save_torch(snapshot(model, optimizer, rng, 0, [], 0.), folder / 'latest.pt')
    elif read(folder / 'config.json') != config:
        raise ValueError('Different seed/protocol at existing output')
    initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_rng']; torch.set_rng_state(state['torch_rng'])
    step = state['step']; history = state['history']; fitting_seconds = state['fitting_seconds']
    if len(history) != step:
        raise ValueError('Checkpoint/history boundary differs')
    features = FineFeatureCache(); validations = []
    cases = selection_bundle[0]['cases']
    if len(cases) != 96:
        raise ValueError('All96 positive selection cases required')
    # Complete or reuse each saved checkpoint evaluation before more updates.
    while True:
        clock_check(protocol, tick); check_seal(protocol); verify_frozen(model, initial)
        if step % INTERVAL == 0:
            weights_once(folder / f'checkpoint_step_{step}.pt', model.state_dict())
            path = folder / f'selection_step_{step}.json.gz'
            if path.exists():
                rows = read_rows(path)
                if [r['pair_id'] for r in rows] != [c['pair_id'] for c in cases]:
                    raise ValueError('Saved complete selection case order differs')
            else:
                rows = evaluate(SearchWindowMatcher(coarse, model), selection_bundle, cases,
                    folder / f'partial-selection-{step}', features, protocol, tick)
                if step == 0:
                    initial_equivalence(rows, baseline)
                save_rows(path, rows)
            record = dict(step=step, selection=counts(rows), results_sha256=sha(path))
            record_once(folder / f'validation_step_{step}.json', record)
            print('NORTH FULL SEED', seed, 'STEP', step, record['selection'], flush=True)
        if step == STEPS:
            break
        step += 1
        clock_check(protocol, tick)
        before = perf_counter()
        history.append(fitting_step(model, optimizer, train_bundle[1], prepared, rng, step))
        fitting_seconds += perf_counter() - before
        verify_frozen(model, initial)
        # Atomic state every update: interruption never silently refits updates.
        save_torch(snapshot(model, optimizer, rng, step, history, fitting_seconds), folder / 'latest.pt')
        if step % 100 == 0:
            print('NORTH FIT', seed, step, 'loss', history[-1]['loss'], flush=True)
    for checkpoint in range(0, STEPS + 1, INTERVAL):
        validations.append(read(folder / f'validation_step_{checkpoint}.json'))
    winner = max(validations, key=lambda r: selection_key(r['selection'], r['step']))
    weights_once(folder / 'final.pt', model.state_dict())
    model.load_state_dict(torch.load(folder / f'checkpoint_step_{winner["step"]}.pt',
        map_location='cpu', weights_only=True), strict=True)
    weights_once(folder / 'selected.pt', model.state_dict())
    verify_frozen(model, initial)
    selected_control = control(SearchWindowMatcher(coarse, model))
    record_once(folder / 'selected_controls.json', selected_control)
    negative_rows = evaluate(SearchWindowMatcher(coarse, model), selection_bundle,
        selection_bundle[0]['negative_cases'], folder / 'partial-selected-negatives',
        features, protocol, tick)
    neg_path = folder / 'selected_negatives.json.gz'
    if not neg_path.exists():
        save_rows(neg_path, negative_rows)
    elif read_rows(neg_path) != negative_rows:
        raise ValueError('Selected negative evidence differs')
    if ({h['sample']['light_a'] for h in history} != set(range(6))
            or {h['sample']['light_b'] for h in history} != set(range(6))):
        raise ValueError('Not all six lighting settings used')
    result = dict(seed=seed, completed_steps=STEPS, selected_step=winner['step'],
        selection=winner['selection'], validation=validations, fitting_seconds=fitting_seconds,
        history=history, selected_negative_counts=totals(negative_rows),
        selected_checkpoint_sha256=sha(folder / 'selected.pt'), fine_encoder_unchanged=True,
        actual_selected_control_passed=bool(selected_control['passed']),
        eligible_for_adoption=bool(selected_control['passed']), calibrated_acceptance_evaluated=False)
    record_once(folder / 'summary.json', result)
    write_new(folder / 'COMPLETE.json', dict(sha256={p.name:sha(p) for p in folder.iterdir()
        if p.is_file() and p.name != 'COMPLETE.json'}))
    return result


def run(output, commit):
    public_sources(commit); configure(); gate = guards(); root = Path(output)
    qa=read('reports/north_context_software_v61.json')
    if not qa['passed']or not qa['fresh_smoke']:raise ValueError('Full fresh software checks required')
    for n,h in qa['source_sha256'].items():
        if sha(n)!=h:raise ValueError('Source changed since QA:'+n)
    train_bundle = load_bundle(MANIFEST, train_only=True)
    selection_bundle = load_bundle(MANIFEST, selection_only=True)
    if len(train_bundle[1]) != 48 or len(selection_bundle[1]) != 24:
        raise ValueError('Full sixteen/eight development component split required')
    prepared = prepare_training(train_bundle[1], train_bundle[2])
    coarse = CompactMatcher(train_bundle[0]['coarse_model'])
    coarse.model.eval().requires_grad_(False)
    frozen = {k:v.clone() for k,v in coarse.model.state_dict().items()}
    if root.exists():
        protocol = read(root / 'protocol.json')
        if protocol['publication_commit'] != commit or protocol['gate_hashes'] != gate:
            raise ValueError('Resume requires identical public source, gate and budget')
        check_seal(protocol)
    else:
        root.mkdir(parents=True); now = datetime.now(timezone.utc)
        protocol = dict(publication_commit=commit, root=str(root),
            scope='Northern cross-image context control; raw development poses only, not architectural novelty or held-out final results.',
            source_sha256={p:sha(p) for p in SOURCES}, manifest_sha256=sha(MANIFEST), gate_hashes=gate,
            created_utc=now.isoformat(), absolute_deadline_utc=(now+timedelta(seconds=CAP_SECONDS)).isoformat(),
            max_calendar_seconds=CAP_SECONDS, seeds=list(SEEDS), steps=STEPS, interval=INTERVAL,
            total_parameters=82208, trainable_parameters=53936, cpu_threads=2,
            optimizer='Unchangedv18AdamW, context1e-4/coarse1e-5,wd1e-4,clip5',
            objective='Unchangedv18bidirectional reachable-window/negative loss plus .5 genuine-anchor term',
            selection='All96positive cases/eightterrain groups; correct pose count,2px correspondence count,earlier checkpoint. Initialization eligible.',
            negatives='Everyfourth fitting draw: distinct nonoverlapping training components. All96selection negatives evaluated only for each selected checkpoint; no confidence calibration.',
            inference='Same reference axial depth/pose,query intrinsics and fixed3nativepx AP3P; querydepth/truepose only for scoring.',
            timing='Fine encoder feature cache reused for selection; do not use these times for speed comparisons.',
            fine_feature_cache_maximum_maps=64,
            fixed_relative_camera_templates=3,absolute_camera_views_per_terrain=4,
            parameter_difference_from_pointwise_control=512,geometric_occlusion_training_supported=False, unseen_relative_pose_generalization_established=False,
            original_released_LunarGeo_fresh_gate_passed=False, calibration_or_test_pixels_loaded=0,
            final_inference=False, environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__))
        write_new(root / 'protocol.json', protocol)
    tick = perf_counter(); clock_check(protocol, tick)
    baseline = read_rows(Path(BASELINES) / 'results.json.gz')
    results = []
    try:
        for seed in SEEDS:
            results.append(one_seed(root,seed,coarse,train_bundle,selection_bundle,prepared,protocol,tick,baseline))
        if any(not torch.equal(v, coarse.model.state_dict()[k]) for k,v in frozen.items()):
            raise ValueError('Frozen original model changed')
        check_seal(protocol); clock_check(protocol,tick)
        summary = dict(complete=True, total_updates=STEPS*len(SEEDS), protocol_sha256=sha(root/'protocol.json'),
            seeds=[{k:r[k] for k in ('seed','completed_steps','selected_step','selection',
                'selected_negative_counts','selected_checkpoint_sha256','fitting_seconds',
                'actual_selected_control_passed','eligible_for_adoption')} for r in results],
            independent_full_optimizer_retraining=False, exact_pilot_optimizer_updates=100,
            scope=protocol['scope'], calibrated_acceptance_evaluated=False,
            calibration_or_test_pixels_loaded=0, original_backbone_unchanged=True)
        record_once(root/'summary.json',summary); print(json.dumps(summary,indent=2),flush=True)
        return summary
    except BaseException as exc:
        # Immutable event records allow explicit same-deadline resumption.
        event=root/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        write_new(event,dict(error_type=type(exc).__name__,error=str(exc),completed_seeds=len(results)))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/north-v61-context-study')
    parser.add_argument('--publication-commit',required=True)
    args=parser.parse_args(); run(args.output,args.publication_commit)
