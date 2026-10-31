"""Replay corrected coarse-learning checkpoints and metrics without inference GT."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from verify_axial_results_v5 import projected


def aggregate(rows, bank=False):
    return dict(cases=len(rows),correct_poses=sum(r['pose_correct'] for r in rows),
        strict_poses=sum(r['strict_pose_correct'] for r in rows),
        estimated_poses=sum(r['estimated_cam2world'] is not None for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
        returned=sum(r['returned'] for r in rows),scorable=sum(r['scorable'] for r in rows),bank_only=bank)


def replay_row(row,tile,data):
    p,q=[np.asarray(row[k],float).reshape(-1,2) for k in ('reference_points_native','query_points_native')]
    target,visible=projected(p,data['depth_a'],data['depth_b'],*tile['cameras'])
    saved=np.asarray(row['target_points_native'],float).reshape(-1,2)
    if visible.tolist()!=row['visible'] or not np.allclose(target[visible],saved,atol=1e-5):
        raise ValueError('Independent axial projection/visibility differs')
    errors=np.linalg.norm(q[visible]-target[visible],axis=1)
    if row['correct_at_native_px']!={str(v):int((errors<=v).sum()) for v in (1,2,4,8,12)}:
        raise ValueError('Correspondence precision differs')
    if row['returned']!=len(p) or row['scorable']!=int(visible.sum()):raise ValueError('Point denominator differs')
    d=data['depth_a'];y,x=np.mgrid[:d.shape[0],:d.shape[1]]
    rays=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])@np.linalg.inv(np.array(tile['cameras'][0]['intrinsics'])).T
    use=np.isfinite(d.ravel())&(d.ravel()>0)
    distance=float(np.median(d.ravel()[use]*np.linalg.norm(rays[use],axis=1)))
    if not np.isclose(distance,row['reference_range_m']):raise ValueError('Physical reference range differs')
    correct=strict=False
    if row['estimated_cam2world'] is not None:
        pose=np.array(row['estimated_cam2world']);truth=np.array(tile['cameras'][1]['cam2world'])
        angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
        translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]))
        if not np.allclose([angle,translation],[row['pose_error']['rotation_deg'],row['pose_error']['translation_m']],atol=1e-6):
            raise ValueError('Camera-to-world pose error differs')
        correct=angle<=2 and translation<=.01*distance;strict=angle<=.5 and translation<=.001*distance
    if (bool(correct),bool(strict))!=(row['pose_correct'],row['strict_pose_correct']):raise ValueError('Pose criterion differs')
    if row['acceptance_policy'] is not None:raise ValueError('Raw pose results relabelled as accepted poses')


def audit(root,output):
    import torch
    from compact_lunar import CompactMatcher
    from coarse_training_v5 import build_supervision
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    manifest_path=Path('reports/localization_training_data_v5.json');manifest=json.loads(manifest_path.read_text())
    if sha256(manifest_path)!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    if protocol['depth_convention']!='optical_axis_z' or protocol['calibration_or_test_pixels_loaded']!=0:
        raise ValueError('Geometry or authorized evaluation scope differs')
    for p,d in protocol['source_sha256'].items():
        if sha256(p)!=d:raise ValueError('Cohort source changed: '+p)
    tiles={t['tile_id']:t for t in manifest['tiles']};data={}
    train={t['component'] for t in tiles.values() if t['split']=='train'}
    selection={t['component'] for t in tiles.values() if t['split']=='model_selection'}
    if train&selection or any(t['split'] not in ('train','model_selection') for t in tiles.values()):raise ValueError('Split leakage')
    for key,t in tiles.items():
        if sha256(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Native cache changed')
        with np.load(t['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in z.files}
    supervision,stats=build_supervision(manifest,tiles,data)
    if stats!=protocol['training_supervision']:raise ValueError('Uniform supervision counts differ')
    geometric_points=0
    for key,z in supervision.items():
        native=(z['p']+.5)*4-.5
        target,visible=projected(native,data[key]['depth_a'],data[key]['depth_b'],*tiles[key]['cameras'])
        masks=[cv2.resize((np.isfinite(d)&(d>0)).astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)>=.9999
               for d in (data[key]['depth_a'],data[key]['depth_b'])]
        support=[cv2.erode(m.astype(np.uint8),np.ones((25,25),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool) for m in masks]
        q=np.zeros_like(z['p']);q[visible]=(target[visible]+.5)/4-.5
        safe=np.rint(q).astype(int).clip(0,127);p=z['p'].astype(int)
        visible&=support[0][p[:,1],p[:,0]]&support[1][safe[:,1],safe[:,0]]
        if not np.array_equal(visible,z['visible']) or not np.allclose(q,z['q'],atol=1e-5):raise ValueError('Uniform 3D labels differ')
        geometric_points+=int(visible.sum())
    paths=sorted(root.glob('*/training.json'))
    if len(paths)!=(1 if protocol['pilot'] else 4):raise ValueError('Missing declared ablation or seed')
    expected={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'}
    records=[];replayed=0
    for path in paths:
        folder=path.parent;meta=json.loads(path.read_text());complete=json.loads((folder/'COMPLETE.json').read_text())
        for name,d in [('training.json',complete['training_sha256']),('selected.pt',complete['checkpoint_sha256']),
                       ('final.pt',complete['final_sha256']),('development_results.json.gz',complete['results_sha256'])]:
            if sha256(folder/name)!=d:raise ValueError('Completed artifact changed: '+str(folder/name))
        for p,d in protocol['source_sha256'].items():
            if meta['source_sha256'][p]!=d or sha256(folder/'source'/p)!=d:raise ValueError('Source snapshot differs')
        if meta['completed_steps']!=protocol['steps'] or meta['calibration_or_test_pixels_loaded']!=0:raise ValueError('Budget/scope differs')
        history=meta['history'];steps=[0]+list(range(protocol['interval'],protocol['steps']+1,protocol['interval']))
        if steps[-1]!=protocol['steps']:steps.append(protocol['steps'])
        if [h['step'] for h in history]!=steps:raise ValueError('Missing selection checkpoints')
        best=max((h['selection']['correct_poses'],h['selection']['correct_at_two_px'],-h['step']) for h in history)
        if list(best)!=meta['selection_key'] or -best[2]!=meta['selected_step']:raise ValueError('Checkpoint selection differs')
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
        final=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
        selected=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
        latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if latest['step']!=protocol['steps'] or any(not torch.equal(v,latest['model'][k]) for k,v in final.items()):raise ValueError('Optimizer did not finish')
        change=float(sum((final[k]-v).square().sum() for k,v in initial.items()).sqrt())
        if change<=0 or not np.isclose(change,meta['final_weight_change_l2']):raise ValueError('Actual weight change differs')
        if meta['selected_step']==0 and any(not torch.equal(v,selected[k]) for k,v in initial.items()):raise ValueError('Initial checkpoint differs')
        matcher=CompactMatcher(manifest['coarse_model'])
        if any(not torch.equal(v,initial[k]) for k,v in matcher.model.state_dict().items()):raise ValueError('Frozen initialization differs')
        matcher.model.load_state_dict(selected)
        n=sum(p.numel() for p in matcher.model.parameters())
        if n!=meta['trainable_parameters'] or n+28272!=meta['total_parameters'] or meta['total_parameters']>=100000:
            raise ValueError('Parameter budget differs')
        if sum(meta['sampled_tiles'].values())!=protocol['steps'] or not set(meta['sampled_tiles'])<=set(supervision):raise ValueError('Sampling scope/count differs')
        with gzip.open(folder/'development_results.json.gz','rt') as f:rows=json.load(f)
        if len(rows)!=len(expected) or {r['pair_id'] for r in rows}!=expected:raise ValueError('Eligible cases missing or repeated')
        for row in rows:
            if row['component']!=tiles[row['tile_id']]['component']:raise ValueError('Terrain group differs')
            replay_row(row,tiles[row['tile_id']],data[row['tile_id']]);replayed+=1
        if aggregate(rows)!=meta['evaluation']:raise ValueError('Full development aggregate differs')
        bank=[r for r in rows if r['pair_id'] in manifest['selection_bank']]
        selected_history=next(h for h in history if h['step']==meta['selected_step'])
        if aggregate(bank,True)!=selected_history['selection']:raise ValueError('Saved selected-bank readout differs')
        records.append(dict(variant=meta['variant'],seed=meta['seed'],selection_key=meta['selection_key'],evaluation=meta['evaluation'],selected_step=meta['selected_step']))
    order={v:i for i,v in enumerate(protocol['variants'])}
    primary=[r for r in records if r['seed']==20261101]
    winner=max(primary,key=lambda r:(tuple(r['selection_key']),-order[r['variant']]))['variant']
    expected_runs={(v,20261101) for v in protocol['variants']}
    if not protocol['pilot']:expected_runs|={(winner,s) for s in (20261102,20261103)}
    if {(r['variant'],r['seed']) for r in records}!=expected_runs or winner!=summary['winner']:raise ValueError('Winner or predefined seed cohort differs')
    for r in records:
        saved=next(s for s in summary['results'] if (s['variant'],s['seed'])==(r['variant'],r['seed']))
        if any(saved[k]!=r[k] for k in ('selection_key','selected_step','evaluation')):raise ValueError('Cohort summary differs')
    result=dict(passed=True,scope='Corrected axial-depth development training; no final or acceptance claim.',
        runs=records,trained_runs=len(records),optimizer_steps=sum(protocol['steps'] for _ in records),
        development_rows=replayed,uniform_supported_training_points=geometric_points,train_components=len(train),selection_components=len(selection),
        selected_variant=winner,calibration_or_test_pixels_loaded=0,source_sha256=sha256(__file__),
        protocol_sha256=sha256(root/'protocol.json'),summary_sha256=sha256(root/'summary.json'))
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/localization-v5-coarse-development')
    p.add_argument('--output',default='reports/coarse_training_audit_v5.json');a=p.parse_args();audit(a.run,a.output)
