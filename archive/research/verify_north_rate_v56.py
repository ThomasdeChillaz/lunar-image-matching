"""Saved full-development metric/sampler replay, not optimizer retraining."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
from compact_lunar import CompactMatcher
from north_training_bundle_v38 import load_bundle,prepare_training,sample_training,sha
from train_north_rate_v56 import (SEED,STEPS,INTERVAL,RATE,CAP_SECONDS,MANIFEST,CONTROL,read,read_rows,counts,selection_key,check_seal)
from train_search_window_v18 import configure,make_model,verify_frozen
from verify_north_results_v39 import replay_row,check_cases,initial_equivalence
from benchmark_north_control_v38 import totals
from verify_north_optimization_rate_v55 import only_rate_changed


def audit(run,output):
    root=Path(run);out=Path(output)
    if out.exists():raise FileExistsError('Preserve each independent replay')
    protocol=read(root/'protocol.json');summary=read(root/'summary.json')
    if (not summary['complete']or summary['total_updates']!=STEPS or summary['seed']!=SEED
            or summary['coarse_learning_rate']!=RATE or summary['model_promoted']
            or summary['protocol_sha256']!=sha(root/'protocol.json')
            or not 0<=summary['calendar_seconds']<=CAP_SECONDS):
        raise ValueError('Complete bounded single-seed development experiment required')
    if (protocol['steps']!=STEPS or protocol['seed']!=SEED or protocol['interval']!=INTERVAL
            or protocol['coarse_learning_rate']!=RATE or protocol['context_learning_rate']!=1e-4
            or protocol['max_calendar_seconds']!=CAP_SECONDS):
        raise ValueError('Declared one-factor protocol differs')
    sealed=read(root/'COMPLETE.json')['sha256']
    actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'COMPLETE.json'}
    if actual!=set(sealed):raise ValueError('Complete experiment inventory differs')
    for name,digest in sealed.items():
        if sha(root/name)!=digest:raise ValueError('Sealed development output changed')
    check_seal(protocol)
    for name in protocol['source_sha256']:
        if subprocess.check_output(['git','show',protocol['publication_commit']+':'+name])!=Path(name).read_bytes():
            raise ValueError('Executed/public source differs')
    if protocol['gate']['matched_original_summary_sha256']!=sha(CONTROL/'summary.json'):
        raise ValueError('Original same-seed control changed')
    if not read(root/'initial_controls.json')['passed']:raise ValueError('Actual initial controls missing')
    configure();train=load_bundle(MANIFEST,train_only=True);selection=load_bundle(MANIFEST,selection_only=True)
    prepared=prepare_training(train[1],train[2]);coarse=CompactMatcher(train[0]['coarse_model'])
    coarse.model.eval().requires_grad_(False)
    base=read_rows('runs/north-v39-full-baselines/results.json.gz')
    seed=SEED;brief=summary['seed_result'];replayed=updates=0
    folder=root/('coarse_adaptation-'+str(seed));meta=read(folder/'summary.json')
    for name,digest in read(folder/'COMPLETE.json')['sha256'].items():
        if sha(folder/name) != digest:raise ValueError('Completed seed evidence changed')
    if meta['seed'] != seed or brief['seed'] != seed or meta['completed_steps'] != STEPS:
        raise ValueError('Missing predefined seed/updates')
    torch.manual_seed(seed);model=make_model(coarse,'coarse_adaptation')
    load=lambda name:torch.load(folder/name,map_location='cpu',weights_only=True)
    initial=load('initial.pt');final=load('final.pt');latest=load('latest.pt');selected=load('selected.pt')
    original_latest=torch.load(CONTROL/'latest.pt',map_location='cpu',weights_only=True)
    only_rate_changed(original_latest['optimizer']['param_groups'],latest['optimizer']['param_groups'],RATE)
    original_initial=torch.load(CONTROL/'initial.pt',map_location='cpu',weights_only=True)
    if initial.keys()!=original_initial.keys()or any(not torch.equal(v,original_initial[k])for k,v in initial.items()):
        raise ValueError('Matched same-seed initialization differs')
    if any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()):
        raise ValueError('Seeded architecture initialization differs')
    if latest['step'] != STEPS or len(latest['history']) != STEPS or latest['history'] != meta['history']:
        raise ValueError('Complete optimizer/history required')
    if any(not torch.equal(v,latest['model'][k]) for k,v in final.items()):
        raise ValueError('Final model differs from optimizer boundary')
    if {int(s['step']) for s in latest['optimizer']['state'].values()} != {STEPS}:
        raise ValueError('AdamW update counts differ')
    if len(meta['validation']) != 5:
        raise ValueError('Missing checkpoint evaluations')
    for step,validation in zip(range(0,STEPS+1,INTERVAL),meta['validation']):
        path=folder/f'selection_step_{step}.json.gz';rows=read_rows(path)
        check_cases(rows,selection[0]['cases'])
        if (validation['step'] != step or validation['results_sha256'] != sha(path)
                or counts(rows) != validation['selection']):
            raise ValueError('Checkpoint aggregate/hash differs')
        for row in rows:replay_row(selection,row)
        if step==0:initial_equivalence(rows,base)
        model.load_state_dict(load(f'checkpoint_step_{step}.pt'),strict=True);verify_frozen(model,initial)
        replayed+=len(rows)
    winner=max(meta['validation'],key=lambda r:selection_key(r['selection'],r['step']))
    if winner['step'] != meta['selected_step'] or winner['selection'] != meta['selection']:
        raise ValueError('Checkpoint selection differs')
    chosen=load(f'checkpoint_step_{winner["step"]}.pt')
    if any(not torch.equal(v,chosen[k]) for k,v in selected.items()):
        raise ValueError('Selected weights differ')
    negative=read_rows(folder/'selected_negatives.json.gz');check_cases(negative,selection[0]['negative_cases'])
    for row in negative:replay_row(selection,row)
    if totals(negative) != meta['selected_negative_counts']:
        raise ValueError('Selected nonoverlap totals differ')
    replayed+=len(negative)
    rng=np.random.default_rng(seed)
    for step,saved in enumerate(meta['history'],1):
        _,_,sample=sample_training(train[1],prepared,rng,negative_pair=step%4==0)
        if saved['step'] != step or sample != saved['sample'] or not np.isfinite([saved['loss'],saved['gradient_norm']]).all():
            raise ValueError('Sampler/update history differs')
        if sample!=original_latest['history'][step-1]['sample']or saved['labels']!=original_latest['history'][step-1]['labels']:
            raise ValueError('Matched original sampler or geometry labels differ')
        updates+=1
    if rng.bit_generator.state != latest['numpy_rng']:
        raise ValueError('Final data augmentation RNG differs')
    for key,value in brief.items():
        if meta[key] != value:raise ValueError('Cohort summary differs: '+key)
    print('REPLAYED NORTH SEED',seed,'selected',winner['step'],flush=True)

    if replayed!=576 or updates!=2000:raise ValueError('All five positive checkpoints and selected nonoverlap cases required')
    old=read(CONTROL/'summary.json')
    first=latest['history'][0];control_first=original_latest['history'][0]
    if any(first[k]!=control_first[k]for k in ('loss','gradient_norm','labels','sample')):
        raise ValueError('Original first loss/gradient differs before rate intervention')
    result=dict(passed=True,replayed_rows=replayed,recorded_updates=updates,
        exact_original_sampler_and_label_replay=True,exact_final_sampler_rng_replay=True,
        only_coarse_rate_changed=True,matched_initialization=True,matched_first_gradient=True,
        coarse_learning_rate=RATE,context_learning_rate=1e-4,
        selected_step=meta['selected_step'],candidate_correct=meta['selection']['correct_raw_poses'],
        original_selected_step=old['selected_step'],original_correct=old['selection']['correct_raw_poses'],
        candidate_wrong=meta['selection']['wrong_raw_poses'],original_wrong=old['selection']['wrong_raw_poses'],
        candidate_nonoverlap=meta['selected_negative_counts']['raw_nonoverlap_proposals'],
        original_nonoverlap=old['selected_negative_counts']['raw_nonoverlap_proposals'],
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),
        neural_forward_repeated=False,full_optimizer_retraining=False,
        calibration_or_test_pixels_loaded=0,model_promoted=False,
        scope='One matched2000update development seed; all576saved native correspondence/pose metrics, five checkpoints, selected negatives, original sampler/labels and optimizer settings replayed. No full optimizer retraining, calibrated acceptance, final-test or SOTA claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/north-v56-rate-development')
    p.add_argument('--output',default='reports/north_rate_development_audit_v56.json')
    a=p.parse_args();audit(a.run,a.output)
