"""Search-window supervision cohort with frozen native refinement on exposed development data."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import prepare_supervision, sample_training
from dataset import sha256
from evaluate_geometry_warp_v11 import SOURCES as SURFACE_SOURCES, control
from geometry_warp_native_v11 import infer
from search_window_learning_v18 import SearchWindowModel, SearchWindowMatcher, objective as window_objective
from train_context_pilot_v15 import load_training
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score
from train_balanced_localization_v3 import load_bundle, save_torch
from train_context_pilot_v15 import SOURCES as PILOT_SOURCES

MANIFEST = 'reports/localization_training_data_v5.json'
SEEDS = (20261101, 20261102, 20261103)
VARIANTS = ('window_only', 'anchored_window', 'coarse_adaptation', 'cross_adaptation')
SETTINGS = {name: dict(variant='cross_context' if name=='cross_adaptation' else 'pointwise',
    train_coarse=name in ('coarse_adaptation','cross_adaptation'),
    anchor_weight=0. if name=='window_only' else .5) for name in VARIANTS}
SOURCES = list(dict.fromkeys(SURFACE_SOURCES+PILOT_SOURCES+['train_global_context_v16.py',
    'search_window_learning_v18.py','train_search_window_v18.py']))


def make_model(coarse, configuration):
    cfg=SETTINGS[configuration]
    return SearchWindowModel(coarse,variant=cfg['variant'],train_coarse=cfg['train_coarse'])


def make_optimizer(model):
    groups=[dict(params=list(model.context.parameters()),lr=.0001)]
    if model.train_coarse:groups.append(dict(params=list(model.coarse.parameters()),lr=.00001))
    return torch.optim.AdamW(groups,weight_decay=.0001)


def verify_frozen(model,initial):
    for name,value in model.state_dict().items():
        if (name.startswith('fine.') or (name.startswith('coarse.') and not model.train_coarse)) and not torch.equal(value,initial[name]):
            raise ValueError('Protected encoder changed: '+name)


def fitting_step(model,optimizer,tiles,prepared,rng,step,configuration):
    model.train();images,targets,sample=sample_training(tiles,prepared,rng,negative_pair=step%4==0)
    optimizer.zero_grad(set_to_none=True);scores,baseline=model(images)
    loss,labels=window_objective(scores,targets,baseline,SETTINGS[configuration]['anchor_weight'])
    if not torch.isfinite(loss):raise ValueError('Nonfinite window loss')
    loss.backward();gradient=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
    if not torch.isfinite(gradient):raise ValueError('Nonfinite window gradient')
    optimizer.step()
    return dict(step=step,loss=float(loss),gradient_norm=float(gradient),sample=sample,labels=labels)



def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def save_rows(path, rows):
    if Path(path).exists():
        raise FileExistsError('Never overwrite saved evaluation rows')
    temporary = Path(str(path)+'.temporary')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(rows, stream, allow_nan=False)
    temporary.replace(path)


def record_once(path, value):
    """Allow interrupted finalization to resume without replacing saved evidence."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing evidence differs: '+str(path))
    else:
        write_new(path, value)


def weights_once(path, weights):
    path = Path(path)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.keys() != weights.keys() or any(not torch.equal(saved[k], v) for k, v in weights.items()):
            raise ValueError('Existing checkpoint differs: '+str(path))
    else:
        save_torch(weights, path)


def counts(rows):
    return dict(cases=len(rows), correct_poses=sum(r['pose_correct'] for r in rows),
        strict_poses=sum(r['strict_pose_correct'] for r in rows),
        estimated_poses=sum(r['estimated_cam2world'] is not None for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
        returned=sum(r['returned'] for r in rows), scorable=sum(r['scorable'] for r in rows))


def selection_key(summary, step):
    # Include failed poses in the denominator. Never select on estimated-only accuracy.
    return (summary['correct_poses'], summary['correct_at_two_px'], -step)


def check_sources(protocol):
    if sha256(MANIFEST) != protocol['manifest_sha256']:
        raise ValueError('Development manifest changed')
    for source, digest in protocol['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Frozen cohort source changed: '+source)


def checkpoint(model, optimizer, rng, step, history, fitting_seconds):
    return dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        history=history, fitting_seconds=fitting_seconds)


def restore(state, model, optimizer, rng):
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_rng']; torch.set_rng_state(state['torch_rng'])


def evaluate(model, coarse, bundle, features, pilot=False):
    manifest, tiles, data = bundle; rows = []; model.eval()
    matcher = SearchWindowMatcher(coarse, model)
    for case in manifest['cases']:
        if case['split'] != 'model_selection' or (pilot and case['pair_id'] not in manifest['selection_bank']):
            continue
        tile = tiles[case['tile_id']]; z = data[case['tile_id']]; a, b = tile['cameras']
        ia, ib = 2*case['light_a'], 2*case['light_b']+1
        selected, branches = infer(matcher, z['images'][ia], z['images'][ib], z['depth_a'],
            a['intrinsics'], a['cam2world'], b['intrinsics'], features,
            ((case['tile_id'], ia), (case['tile_id'], ib)))
        result = selected if selected is not None else branches[0]
        # Query depth and true query pose enter only this scoring function.
        rows.append(dict(case, **score(result['p'], result['q'],
            None if selected is None else result['pose'], z, tile['cameras']),
            quarter_turn=result['quarter_turn'], pose_info=result['info'],
            geometry_history=result.get('geometry_history', [])))
    return counts(rows), rows


def validate_initial(rows):
    old = {r['pair_id']: r for r in read_rows('runs/localization-v11-surface-warp-development/results.json.gz')}
    for row in rows:
        for name in ('pose_correct', 'strict_pose_correct', 'reference_points_native',
                     'query_points_native', 'estimated_cam2world'):
            if row[name] != old[row['pair_id']][name]:
                raise ValueError('Actual initial native/pose equivalence failed: '+row['pair_id']+' '+name)


def train(root, variant, seed, bundle, coarse, prepared, protocol, features):
    folder = root/f'{variant}-{seed}'
    if (folder/'COMPLETE.json').exists():
        proof = json.loads((folder/'COMPLETE.json').read_text())
        for path, digest in proof['sha256'].items():
            if sha256(folder/path) != digest:
                raise ValueError('Completed run changed: '+path)
        return json.loads((folder/'training.json').read_text())
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = make_model(coarse, variant); optimizer = make_optimizer(model)
    parameters = sum(p.numel() for p in model.parameters())
    config = dict(protocol, variant=variant, seed=seed, ablation_settings=SETTINGS[variant],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), total_parameters=parameters)
    if not folder.exists():
        folder.mkdir(); (folder/'source').mkdir()
        write_new(folder/'frozen_config.json', config)
        save_torch(model.state_dict(), folder/'initial.pt')
        for source in SOURCES:
            shutil.copy2(source, folder/'source'/source)
        save_torch(checkpoint(model, optimizer, rng, 0, [], 0.), folder/'latest.pt')
    elif json.loads((folder/'frozen_config.json').read_text()) != config:
        raise ValueError('Cannot resume a different experiment')
    state = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
    restore(state, model, optimizer, rng)
    first = int(state['step']); history = state['history']; fitting_seconds = state['fitting_seconds']
    started = perf_counter(); completed = first
    for step in range(first, protocol['steps']+1):
        if step != first:
            tick = perf_counter()
            history.append(fitting_step(model,optimizer,bundle[1],prepared,rng,step,variant))
            completed=step; fitting_seconds+=perf_counter()-tick
        # Persist before evaluation so an interruption resumes this exact checkpoint.
        if step % 100 == 0 or step == protocol['steps']:
            save_torch(checkpoint(model, optimizer, rng, step, history, fitting_seconds), folder/'latest.pt')
        if step % protocol['interval'] == 0 or step == protocol['steps']:
            check_sources(protocol); path = folder/f'selection_step_{step}.json.gz'
            if path.exists():
                rows = read_rows(path); summary = counts(rows)
            else:
                summary, rows = evaluate(model, coarse, bundle, features, protocol['pilot'])
                if step == 0:
                    validate_initial(rows)
                save_rows(path, rows)
            expected = 16 if protocol['pilot'] else 72
            if summary['cases'] != expected:
                raise ValueError('Incomplete selection denominator')
            record = dict(step=step, selection=summary, results_sha256=sha256(path))
            record_path = folder/f'validation_step_{step}.json'
            if record_path.exists():
                if json.loads(record_path.read_text()) != record:
                    raise ValueError('Saved selection proof differs')
            else:
                write_new(record_path, record)
            weights_once(folder/f'checkpoint_step_{step}.pt', model.state_dict())
            print(variant, seed, 'selection', step, summary, 'fitting_seconds', round(fitting_seconds, 2), flush=True)
        if step and step % 100 == 0:
            print(variant, seed, 'update', step, 'loss', round(history[-1]['loss'], 4), flush=True)
        total_wall=(datetime.now(timezone.utc)-datetime.fromisoformat(protocol['created_utc'])).total_seconds() if 'created_utc' in protocol else 0
        if total_wall>protocol.get('cohort_wall_cap_seconds',float('inf')) or perf_counter()-started > protocol['per_invocation_wall_cap_seconds']:
            # latest.pt is the last complete100-step boundary; preserve partial attempt metadata.
            save_torch(checkpoint(model, optimizer, rng, completed, history, fitting_seconds), folder/'latest.pt')
            raise TimeoutError('Wall cap reached; preserve partial evidence. Resume only within the original cohort deadline.')
    validations = [json.loads(p.read_text()) for p in folder.glob('validation_step_*.json')]
    winner = max(validations, key=lambda r: selection_key(r['selection'], r['step']))
    selected = winner['step']; selected_path = folder/f'checkpoint_step_{selected}.pt'
    weights_once(folder/'selected.pt', torch.load(selected_path, map_location='cpu', weights_only=True))
    weights_once(folder/'final.pt', model.state_dict())
    initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=True)
    verify_frozen(model,initial)
    change = float(sum((v-initial[k]).square().sum() for k, v in model.state_dict().items()).sqrt())
    result = dict(config, completed_steps=protocol['steps'], selected_step=selected,
        selection_key=list(selection_key(winner['selection'], selected)), evaluation=winner['selection'],
        selected_rows_sha256=winner['results_sha256'], final_weight_change_l2=change,
        fitting_seconds=fitting_seconds, history=history, negative_pairs=sum(h['sample']['verified_nonoverlap'] for h in history),
        validation=sorted(validations, key=lambda r: r['step']))
    record_once(folder/'training.json', result)
    paths = ['frozen_config.json', 'initial.pt', 'selected.pt', 'final.pt', 'latest.pt', 'training.json']
    paths += [p.name for p in folder.glob('selection_step_*.json.gz')]
    paths += [p.name for p in folder.glob('validation_step_*.json')]
    paths += [p.name for p in folder.glob('checkpoint_step_*.pt')]
    write_new(folder/'COMPLETE.json', dict(sha256={p: sha256(folder/p) for p in paths}))
    return result


def run(output, commit, steps=2000, interval=500, pilot=False):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Published source commit required')
    for source in SOURCES:
        if subprocess.check_output(['git', 'show', commit+':'+source]) != Path(source).read_bytes():
            raise ValueError('Source differs from published freeze: '+source)
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    bundle = load_bundle(MANIFEST); manifest, tiles, data = bundle
    if manifest['schema_version'] != 5 or not manifest['train_selection_disjoint']:
        raise ValueError('Corrected disjoint development manifest required')
    if sha256('lunar_geometry_v5.py') != manifest['geometry_source_sha256'] or sha256(manifest['split_audit']) != manifest['split_audit_sha256']:
        raise ValueError('Geometry or split audit changed')
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    if sha256(Path(manifest['coarse_model'])/'selected.pt') != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen backbone changed')
    frozen = {k: v.detach().clone() for k, v in coarse.model.state_dict().items()}
    prepared, support = prepare_supervision(tiles, data)
    gate_path = Path('reports/native_window_pilot_audit_v18.json'); gate = json.loads(gate_path.read_text())
    if not gate['passed'] or not gate['exact_full_retraining'] or gate['completed_steps']!=100:
        raise ValueError('Measured pilot must independently replay before full training')
    for source,digest in gate['fitting_source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Pilot replay source differs')
    if sha256(Path(gate['pilot_root'])/'summary.json')!=gate['pilot_summary_sha256']:raise ValueError('Audited pilot changed')
    initial_gate=json.loads(Path('runs/localization-v18-window-initial-control/summary.json').read_text())
    if not initial_gate['passed'] or initial_gate['cases_per_variant']!=72:raise ValueError('Actual full initial gate required')
    for source,digest in initial_gate['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Initial-control source differs')
    root = Path(output)
    if root.exists():
        protocol = json.loads((root/'protocol.json').read_text()); check_sources(protocol)
        if (commit, steps, interval, pilot) != (protocol['publication_commit'], protocol['steps'], protocol['interval'], protocol['pilot']):
            raise ValueError('Resume must preserve source commit and fitting protocol')
    else:
        root.mkdir(parents=True)
        protocol = dict(scope='Exposed development-only native-window supervision and coarse-adaptation cohort; no calibrated acceptance, fresh final or SOTA claim.',
            publication_commit=commit, source_sha256={p: sha256(p) for p in SOURCES},
            manifest_sha256=sha256(MANIFEST), input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
            coarse_checkpoint_sha256=manifest['coarse_checkpoint_sha256'], training_support=support,
            steps=steps, interval=interval, pilot=pilot, variants=list(VARIANTS), seeds=list(SEEDS),
            context_learning_rate=.0001, coarse_learning_rate=.00001, settings=SETTINGS, weight_decay=.0001, optimizer='AdamW', gradient_clip=5., pairs_per_update=1,
            cpu_threads=2, negative_pair_every=4, tokens_per_image=676,
            selection='All72 exposed pairs; unfiltered correct pose count, then2px correct correspondence count, then earlier checkpoint. Initialization eligible. Pilot fixed16 only.',
            winner='Primary seed winner by same key; declared variant order resolves exact ties. Repeat selected configuration at20261102/03 from fresh seeded initialization.',
            supervision='Bidirectional projected3D search-window set likelihood, score margins and optional genuine frozen-proposal anchors; same train-only draws. Native fine encoder stays frozen, coarse may adapt.',
            inference='Separately frozen native fine CNN refinement, four orientation proposals, two reference-surface updates, unchanged3nativepx AP3P. No query truth in inference.',
            feature_cache='Reuse only frozen native image CNN maps across checkpoints and variants; context outputs, warped queries, correspondences and poses recomputed. No isolated inference timing claim.',
            calibration_or_test_pixels_loaded=0, fresh_data_gate_passed=False, per_invocation_wall_cap_seconds=6*3600,
            cohort_wall_cap_seconds=6*3600,
            pilot_audit_sha256=sha256(gate_path), initial_gate_sha256=sha256('runs/localization-v18-window-initial-control/summary.json'), environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__),
            created_utc=datetime.now(timezone.utc).isoformat())
        write_new(root/'protocol.json', protocol)
    if support != protocol['training_support']:
        raise ValueError('Training support changed')
    results = []; features = {}
    for variant in VARIANTS:
        results.append(train(root, variant, SEEDS[0], bundle, coarse, prepared, protocol, features))
    primary = max(results, key=lambda r: tuple(r['selection_key'])); winner = primary['variant']
    if not pilot:
        for seed in SEEDS[1:]:
            results.append(train(root, winner, seed, bundle, coarse, prepared, protocol, features))
    if any(not torch.equal(v, coarse.model.state_dict()[k]) for k, v in frozen.items()):
        raise ValueError('Frozen backbone changed')
    check_sources(protocol)
    summary = dict(scope=protocol['scope'], winner=winner, completed_optimizer_steps=sum(r['completed_steps'] for r in results),
        frozen_backbone_unchanged=True, results=[{k: r[k] for k in ('variant','seed','completed_steps','selected_step','evaluation','selection_key','fitting_seconds')} for r in results])
    if not (root/'summary.json').exists():
        write_new(root/'summary.json', summary)
    elif json.loads((root/'summary.json').read_text()) != summary:
        raise ValueError('Completed cohort differs')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def published(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Published source commit required')
    for source in SOURCES:
        if subprocess.check_output(['git','show',commit+':'+source])!=Path(source).read_bytes():
            raise ValueError('Fitting source differs from public freeze: '+source)


def configure():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)


def initial_control(output,commit):
    published(commit);configure();root=Path(output)
    if root.exists():raise FileExistsError('Preserve all control attempts')
    root.mkdir(parents=True);bundle=load_bundle(MANIFEST);coarse=CompactMatcher(bundle[0]['coarse_model'])
    coarse.model.eval().requires_grad_(False);records=[];features={}
    if sha256(Path(bundle[0]['coarse_model'])/'selected.pt')!=bundle[0]['coarse_checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    # Only two distinct inference graphs at initialization. Anchor and coarse
    # trainability change training, not an eval-mode initialized forward pass.
    for configuration in ('window_only','cross_adaptation'):
        torch.manual_seed(SEEDS[0]);model=make_model(coarse,configuration).eval()
        actual=control(SearchWindowMatcher(coarse,model))
        from synthetic import texture
        import numpy as np
        low=texture(994,128);matcher=SearchWindowMatcher(coarse,model)
        invalid=matcher.points(low,low,np.zeros_like(low,bool),np.ones_like(low,bool))
        if not actual['passed'] or any(len(p) for p in invalid):raise ValueError('Actual matcher control failed')
        summary,rows=evaluate(model,coarse,bundle,features,False);validate_initial(rows)
        if summary['cases']!=72:raise ValueError('Incomplete actual initial gate')
        save_rows(root/(configuration+'.json.gz'),rows);save_torch(model.state_dict(),root/(configuration+'.pt'))
        records.append(dict(configuration=configuration,controls=actual,invalid_mask_empty=True,evaluation=summary,
            rows_sha256=sha256(root/(configuration+'.json.gz')),checkpoint_sha256=sha256(root/(configuration+'.pt'))))
        print('window actual initial',configuration,summary,flush=True)
    result=dict(passed=True,cases_per_variant=72,exact_native_and_pose_equivalence=True,records=records,
        source_sha256={p:sha256(p) for p in SOURCES},publication_commit=commit,manifest_sha256=sha256(MANIFEST),
        scope='Actual initialized two-graph equivalence on all72 exposed selection pairs and software controls, not learned gain.')
    write_new(root/'summary.json',result);return result


def pilot(output,commit,steps=100):
    published(commit);configure();root=Path(output)
    if root.exists():raise FileExistsError('Preserve all pilot attempts')
    gate_path=Path('runs/localization-v18-window-initial-control/summary.json');gate=json.loads(gate_path.read_text())
    if not gate['passed'] or gate['cases_per_variant']!=72:raise ValueError('Full actual initialization required')
    for source,digest in gate['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Initial-control source changed')
    bundle=load_training(MANIFEST);manifest,tiles,data=bundle;prepared,support=prepare_supervision(tiles,data)
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
    if sha256(Path(manifest['coarse_model'])/'selected.pt')!=manifest['coarse_checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    torch.manual_seed(SEEDS[0]);rng=np.random.default_rng(SEEDS[0]);configuration='coarse_adaptation'
    model=make_model(coarse,configuration);optimizer=make_optimizer(model)
    initial={k:v.clone() for k,v in model.state_dict().items()};root.mkdir(parents=True);(root/'source').mkdir()
    protocol=dict(scope='Train-only100-update cost and exact-gradient pilot, no localization result.',
        publication_commit=commit,source_sha256={p:sha256(p) for p in SOURCES},manifest_sha256=sha256(MANIFEST),
        input_sha256={t['native_cache']:t['native_cache_sha256'] for t in tiles.values()},
        variant=configuration,settings=SETTINGS[configuration],seed=SEEDS[0],steps=steps,training_support=support,
        initial_gate_sha256=sha256(gate_path),selection_inference=0,selection_pixels_loaded=0,calibration_or_test_pixels_loaded=0,
        cpu_threads=2,total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        started_utc=datetime.now(timezone.utc).isoformat())
    write_new(root/'protocol.json',protocol);save_torch(initial,root/'initial.pt')
    for source in SOURCES:shutil.copy2(source,root/'source'/source)
    history=[];start=perf_counter()
    for step in range(1,steps+1):
        history.append(fitting_step(model,optimizer,tiles,prepared,rng,step,configuration))
        if step%10==0:
            save_torch(checkpoint(model,optimizer,rng,step,history,perf_counter()-start),root/'latest.pt')
            print('window pilot',step,history[-1]['loss'],'seconds',perf_counter()-start,flush=True)
        if perf_counter()-start>600:raise TimeoutError('Bounded pilot exhausted; preserve partial evidence')
    verify_frozen(model,initial);save_torch(model.state_dict(),root/'final.pt')
    for source,digest in protocol['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Pilot source changed')
    change=float(sum((v-initial[k]).square().sum() for k,v in model.state_dict().items()).sqrt())
    result=dict(completed_steps=steps,fitting_seconds=perf_counter()-start,history=history,
        fine_encoder_unchanged=True,final_weight_change_l2=change,selection_pixels_loaded=0,calibration_or_test_pixels_loaded=0)
    write_new(root/'summary.json',result)
    write_new(root/'COMPLETE.json',dict(sha256={p:sha256(root/p) for p in ('protocol.json','initial.pt','final.pt','latest.pt','summary.json')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--mode',choices=['initial','pilot','cohort'],default='cohort')
    args=parser.parse_args()
    if args.mode=='initial':initial_control(args.output,args.source_commit)
    elif args.mode=='pilot':pilot(args.output,args.source_commit)
    else:run(args.output,args.source_commit)
