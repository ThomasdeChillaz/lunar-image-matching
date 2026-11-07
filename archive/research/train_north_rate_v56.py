"""Matched full-development camera-pose test of the coarse1e-3 learning rate.

One seed, original2000updates and every original positive/nonoverlap case.
Only the coarse AdamW rate changes. No final/calibration data or promotion.
"""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib,json,re,subprocess
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch

from benchmark_north_control_v38 import inference_inputs,score_case,totals
from compact_lunar import CompactMatcher
from evaluate_geometry_warp_v11 import control
from geometry_warp_native_v11 import infer
from lunargeo_dataset import write_new
from north_training_bundle_v38 import load_bundle,prepare_training,sha
from north_optimization_rate_v55 import rate_optimizer,check_optimizer
from north_coarse_reachability_v53 import verify_ended
from north_completion_v52 import original_ended,INVENTORY
from search_window_learning_v18 import SearchWindowMatcher
from train_balanced_localization_v3 import save_torch
from train_north_pilot_v38 import fitting_step
from train_north_control_v39 import (SOURCES as FIT_SOURCES,FineFeatureCache,
    counts,read,read_rows,save_rows,initial_equivalence,snapshot,check_seal,guards)
from train_search_window_v18 import configure,make_model,record_once,selection_key,verify_frozen,weights_once

SEED=20261101
STEPS=2000
INTERVAL=500
RATE=1e-3
CAP_SECONDS=3600
MANIFEST='reports/north_training_data_v38.json'
CONTROL=Path('runs/north-v52-completion-only/coarse_adaptation-20261101')
SOURCES=list(dict.fromkeys(FIT_SOURCES+['north_optimization_rate_v55.py',
    'north_coarse_reachability_v53.py','north_completion_v52.py',
    'train_north_rate_v56.py','verify_north_rate_v56.py','test_north_rate_v56.py',
    'verify_north_optimization_rate_v55.py']))


def clock_check(protocol,tick):
    now=datetime.now(timezone.utc)
    if (now<datetime.fromisoformat(protocol['created_utc'])
            or now>=datetime.fromisoformat(protocol['absolute_deadline_utc'])
            or perf_counter()-tick>=CAP_SECONDS):
        raise TimeoutError('Original one-hour rate-development allowance exhausted; do not reset')


def cost_admission(pilot_seconds,case_seconds):
    estimate=2*(pilot_seconds/100*STEPS+case_seconds*(96*5+96)+30)
    if not np.isfinite(estimate) or min(pilot_seconds,case_seconds)<=0 or estimate>=CAP_SECONDS:
        raise ValueError('Measured fitting/full-case cost exceeds fixed one-hour allowance')
    return float(estimate)


def gate():
    verify_ended('runs/north-v55-rate-state.json');original_ended(read(INVENTORY))
    amended=read('reports/north_amended_completion_audit_v52.json')
    if not amended['passed']or amended['recorded_original_fitting_updates']!=6000:
        raise ValueError('Full matched original cohort audit required')
    for tag in ('1e-4','1e-3'):
        a=read('reports/north_rate_'+tag+'_audit_v55.json')
        if not a['passed']or a['neural_snapshot_forwards_replayed']!=20 or not a['only_coarse_rate_changed']:
            raise ValueError('Both exploratory fixed-pair rate controls must have replayed')
    for name,digest in read(CONTROL/'COMPLETE.json')['sha256'].items():
        if sha(CONTROL/name)!=digest:raise ValueError('Original same-seed evidence changed')
    inherited=guards()
    pilot=read('runs/north-v38-training-cost-pilot/summary.json')
    bank=read('runs/north-v38-baseline-cost-bank/summary.json')
    estimate=cost_admission(pilot['fitting_seconds'],bank['results']['frozen_surface']['median_measured_inference_ms']/1000)
    return dict(original_gate_hashes=inherited,projected_seconds_with_factor2=estimate,
        matched_original_summary_sha256=sha(CONTROL/'summary.json'),
        rate_audit_sha256={tag:sha('reports/north_rate_'+tag+'_audit_v55.json')for tag in ('1e-4','1e-3')})


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
            row = dict(case, method='coarse_learning_rate_control',
                **score_case(bundle, case, p, q, pose),
                elapsed_ms=elapsed_ms, timing_scope='Frozen fine-map cache reused; not isolated latency.',
                inference_error=error, quarter_turn=branch.get('quarter_turn'),
                pose_info=branch.get('info', {}), geometry_history=branch.get('geometry_history', []))
            write_new(path, row)
        rows.append(row)
    if len(rows) != len(cases) or len({r['pair_id'] for r in rows}) != len(cases):
        raise ValueError('Incomplete or duplicate selection denominator')
    return rows


def one_seed(root, seed, coarse, train_bundle, selection_bundle, prepared, protocol, tick, baseline):
    folder = root / ('coarse_adaptation-' + str(seed))
    if (folder / 'COMPLETE.json').exists():
        for name, digest in read(folder / 'COMPLETE.json')['sha256'].items():
            if sha(folder / name) != digest:
                raise ValueError('Completed seed evidence changed: ' + name)
        return read(folder / 'summary.json')
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = make_model(coarse, 'coarse_adaptation'); optimizer = rate_optimizer(model, RATE)
    if sum(p.numel() for p in model.parameters()) != 81696:
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
    original_initial = torch.load(CONTROL / 'initial.pt', map_location='cpu', weights_only=True)
    if initial.keys() != original_initial.keys() or any(not torch.equal(v, original_initial[k]) for k,v in initial.items()):
        raise ValueError('Matched original initialization differs')
    state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    check_optimizer(model, optimizer, RATE)
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


def run(output,commit,resume=False):
    started=datetime.now(timezone.utc);tick=perf_counter();root=Path(output)
    if root.exists() and not resume:raise FileExistsError('Preserve old attempts; explicit same-deadline resume required')
    if resume and not root.exists():raise FileNotFoundError('No rate-development attempt to resume')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact public source commit required')
    for name in SOURCES:
        if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Source differs from public freeze:'+name)
    qa_path='reports/north_rate_development_software_v56.json';qa=read(qa_path)
    if not qa['passed']or not qa['full_repository_regression']or not qa['fresh_smoke']:
        raise ValueError('Current full source-bound regression and fresh smoke required')
    for name,digest in qa['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Source changed after checks:'+name)
    admission=gate();configure()
    static=dict(root=str(root),publication_commit=commit,source_sha256={n:sha(n)for n in SOURCES},
        manifest_sha256=sha(MANIFEST),seed=SEED,steps=STEPS,interval=INTERVAL,
        coarse_learning_rate=RATE,context_learning_rate=1e-4,original_coarse_learning_rate=1e-5,
        max_calendar_seconds=CAP_SECONDS,cpu_threads=2,total_parameters=81696,
        gate=admission,software_sha256=sha(qa_path),model_promoted=False,
        case_scope='All96positive/96nonoverlap northern model-selection cases; five fresh full checkpoint evaluations.',
        scope='Single-seed one-factor coarse-rate DEVELOPMENT pose experiment. Original architecture/initialization/sampler/loss/fine-refinement/selection. Original failed cohort time protocol stays failed; separate one-hour experiment, not a restart. No calibrated acceptance, final test, camera-template independence or SOTA claim.')
    if root.exists():
        protocol=read(root/'protocol.json')
        if any(protocol[k]!=v for k,v in static.items()):raise ValueError('Changed experiment on resume')
        if(root/'COMPLETE.json').exists():
            for name,digest in read(root/'COMPLETE.json')['sha256'].items():
                if sha(root/name)!=digest:raise ValueError('Sealed development evidence changed')
            return read(root/'summary.json')
    else:
        root.mkdir(parents=True)
        protocol=dict(static,created_utc=started.isoformat(),absolute_deadline_utc=(started+timedelta(seconds=CAP_SECONDS)).isoformat())
        write_new(root/'protocol.json',protocol)
    try:
        clock_check(protocol,tick)
        train=load_bundle(MANIFEST,train_only=True);selection=load_bundle(MANIFEST,selection_only=True)
        if len(train[1])!=48 or len(selection[1])!=24:raise ValueError('Original full development split required')
        prepared=prepare_training(train[1],train[2])
        coarse=CompactMatcher(train[0]['coarse_model']);coarse.model.eval().requires_grad_(False)
        torch.manual_seed(SEED);initial_model=make_model(coarse,'coarse_adaptation')
        controls=control(SearchWindowMatcher(coarse,initial_model));record_once(root/'initial_controls.json',controls)
        if not controls['passed']:raise ValueError('Fresh actual initial known-transform/blank control failed')
        base=read_rows('runs/north-v39-full-baselines/results.json.gz')
        meta=one_seed(root,SEED,coarse,train,selection,prepared,protocol,tick,base)
        clock_check(protocol,tick);check_seal(protocol)
        keys=('seed','completed_steps','selected_step','selection','selected_negative_counts',
            'selected_checkpoint_sha256','fitting_seconds','actual_selected_control_passed','eligible_for_adoption')
        result=dict(complete=True,total_updates=STEPS,seed=SEED,coarse_learning_rate=RATE,
            seed_result={k:meta[k]for k in keys},protocol_sha256=sha(root/'protocol.json'),
            model_promoted=False,calibration_or_test_pixels_loaded=0,scope=static['scope'],
            calendar_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(protocol['created_utc'])).total_seconds())
        record_once(root/'summary.json',result)
        write_new(root/'COMPLETE.json',dict(sha256={str(p.relative_to(root)):sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p!=root/'COMPLETE.json'}))
        print(json.dumps(result,indent=2),flush=True);return result
    except BaseException as exc:
        record_once(root/('FAILED-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
            dict(error_type=type(exc).__name__,error=str(exc),original_deadline_utc=protocol['absolute_deadline_utc']))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/north-v56-rate-development')
    p.add_argument('--publication-commit',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.output,a.publication_commit,a.resume)
