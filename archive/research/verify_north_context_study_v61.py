"""Independent saved-metric replay for expanded northern controls/cohort.

This recomputes scoring from saved native correspondences and camera poses.
It is not a neural inference replay or a full optimizer retraining claim.
The separate v37 BVH/oracle audit establishes the geometry-label controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_north_control_v38 import score_case, totals
from compact_lunar import CompactMatcher
from lunargeo_dataset import write_new
from north_training_bundle_v38 import load_bundle, prepare_training, sample_training, sha
from train_north_context_study_v61 import (MANIFEST, SEEDS, STEPS, INTERVAL, counts,
    initial_equivalence, public_sources, read_rows, read, selection_key)
from train_search_window_v18 import configure, make_model, verify_frozen


def replay_row(bundle, row):
    p = np.asarray(row['reference_points_native'], float).reshape(-1, 2)
    q = np.asarray(row['query_points_native'], float).reshape(-1, 2)
    saved = row['estimated_cam2world']
    pose = None if saved is None else np.asarray(saved, float)
    # Saved points are the inputs, never the labels. Ground truth is recomputed
    # from the checked reference/query geometry after reloading the dataset.
    actual = score_case(bundle, row, p, q, pose)
    for name, value in actual.items():
        if row[name] != value:
            raise ValueError('Saved native metric replay differs: ' + row['pair_id'] + '/' + name)


def check_cases(rows, cases):
    if [r['pair_id'] for r in rows] != [c['pair_id'] for c in cases]:
        raise ValueError('Complete fixed case list/order required')
    for row, case in zip(rows, cases):
        if any(row[k] != v for k, v in case.items()):
            raise ValueError('Saved case metadata changed')


def baseline(run, output):
    root = Path(run); protocol = read(root/'protocol.json'); summary = read(root/'summary.json')
    if (not summary['complete'] or not summary['full']
            or summary['eligible_cases_per_method'] != 192
            or summary['protocol_sha256'] != sha(root/'protocol.json')
            or summary['results_sha256'] != sha(root/'results.json.gz')
            or protocol['manifest_sha256'] != sha(MANIFEST)):
        raise ValueError('Complete sealed common baselines required')
    for name, digest in protocol['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Baseline source changed: '+name)
    configure(); bundle = load_bundle(MANIFEST, selection_only=True)
    rows = read_rows(root/'results.json.gz'); cases = bundle[0]['cases'] + bundle[0]['negative_cases']
    for method in ('frozen_surface','sift_first_surface'):
        part = [r for r in rows if r['method'] == method]
        check_cases(part,cases)
        for row in part:
            replay_row(bundle,row)
        if totals(part) != summary['results'][method]:
            raise ValueError('Common baseline totals differ')
    if len(rows) != 384:
        raise ValueError('Unexpected/missing baseline method rows')
    result = dict(passed=True,replayed_rows=len(rows),results_sha256=sha(root/'results.json.gz'),
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),
        neural_forward_repeated=False,calibration_or_test_pixels_loaded=0,
        scope='Saved native-pixel correspondence/pose metrics and all common baseline denominators replayed.')
    write_new(output,result);print(json.dumps(result,indent=2),flush=True);return result


def cohort(run, output):
    root=Path(run);protocol=read(root/'protocol.json');summary=read(root/'summary.json')
    public_sources(protocol['publication_commit']);configure()
    if (not summary['complete'] or summary['total_updates'] != 2000
            or summary['protocol_sha256'] != sha(root/'protocol.json')
            or protocol['manifest_sha256'] != sha(MANIFEST)
            or protocol['seeds'] != list(SEEDS) or protocol['steps'] != STEPS):
        raise ValueError('Complete frozen single-seed control required')
    train = load_bundle(MANIFEST,train_only=True); selection=load_bundle(MANIFEST,selection_only=True)
    prepared=prepare_training(train[1],train[2]);coarse=CompactMatcher(train[0]['coarse_model'])
    coarse.model.eval().requires_grad_(False)
    base=read_rows('runs/north-v39-full-baselines/results.json.gz')
    replayed=updates=0
    for seed, brief in zip(SEEDS,summary['seeds']):
        folder=root/('cross_adaptation-'+str(seed));meta=read(folder/'summary.json')
        for name,digest in read(folder/'COMPLETE.json')['sha256'].items():
            if sha(folder/name) != digest:raise ValueError('Completed seed evidence changed')
        if meta['seed'] != seed or brief['seed'] != seed or meta['completed_steps'] != STEPS:
            raise ValueError('Missing predefined seed/updates')
        torch.manual_seed(seed);model=make_model(coarse,'cross_adaptation')
        load=lambda name:torch.load(folder/name,map_location='cpu',weights_only=True)
        initial=load('initial.pt');final=load('final.pt');latest=load('latest.pt');selected=load('selected.pt')
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
            updates+=1
        if rng.bit_generator.state != latest['numpy_rng']:
            raise ValueError('Final data augmentation RNG differs')
        for key,value in brief.items():
            if meta[key] != value:raise ValueError('Cohort summary differs: '+key)
        print('REPLAYED NORTH SEED',seed,'selected',winner['step'],flush=True)
    if len(summary['seeds']) != 1 or replayed != 576 or updates != 2000:
        raise ValueError('Full cohort denominator differs')
    result=dict(passed=True,replayed_rows=replayed,recorded_updates=updates,
        exact_sampler_rng_replay=True,independent_full_optimizer_retraining=False,
        exact_optimizer_pilot_updates=100,neural_forward_repeated=False,
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),
        calibration_or_test_pixels_loaded=0,
        scope='All saved checkpoint/selected-negative metrics, selection, frozen fine encoder, update counts and sampler states replayed. No full optimizer retraining or accepted-pose superiority claim.')
    write_new(output,result);print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['baseline','cohort'])
    p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    (baseline if a.mode=='baseline' else cohort)(a.run,a.output)
