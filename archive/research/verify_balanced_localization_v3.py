"""Independently replay saved development-training geometry, selection and metrics."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def projected_truth(points, da, db, a, b):
    p=np.asarray(points,float).reshape(-1,2)
    ka,ca=np.asarray(a['intrinsics']),np.asarray(a['cam2world'])
    kb,cb=np.asarray(b['intrinsics']),np.asarray(b['cam2world'])
    index=np.rint(p).astype(int); valid=((index>=0)&(index<np.array(da.shape[::-1]))).all(1)
    index=index.clip(0,np.array(da.shape[::-1])-1); ranges=da[index[:,1],index[:,0]]
    valid &= np.isfinite(ranges)&(ranges>0)
    rays=np.column_stack([p,np.ones(len(p))])@np.linalg.inv(ka).T
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    world=(rays*np.where(valid,ranges,1)[:,None])@ca[:3,:3].T+ca[:3,3]
    camera=(world-cb[:3,3])@cb[:3,:3]; homogeneous=camera@kb.T
    with np.errstate(divide='ignore',invalid='ignore'):target=homogeneous[:,:2]/homogeneous[:,2,None]
    valid &= (camera[:,2]>0)&np.isfinite(target).all(1)
    valid &= ((target>=0)&(target<np.array(db.shape[::-1])-1)).all(1)
    if len(p):
        safe=np.where(np.isfinite(target),target,-1).astype(np.float32)
        depth=cv2.remap(np.where(np.isfinite(db),db,0).astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR).ravel()
        support=cv2.remap((np.isfinite(db)&(db>0)).astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR).ravel()
        distance=np.linalg.norm(camera,axis=1)
        valid &= (support>=.9999)&(np.abs(depth-distance)<=.05+.01*distance)
    return target, valid


def audit(root, output):
    import torch
    from localization_refiner_v3 import LocalRefiner
    root=Path(root); protocol=json.loads((root/'protocol.json').read_text()); summary=json.loads((root/'summary.json').read_text())
    if protocol['final_test_enabled'] or protocol['fresh_data_gate_passed']:
        raise ValueError('Not the authorized development-only experiment')
    for p,digest in protocol['sources'].items():
        if sha256(p)!=digest:raise ValueError('Executed cohort source differs: '+p)
    paths=sorted(root.glob('*/training.json')); configs=[json.loads(p.read_text()) for p in paths]
    if len(configs)!=(1 if summary['pilot'] else len(protocol['variants'])+2):raise ValueError('Missing declared ablation/seed')
    manifest_path=configs[0]['manifest_path']; manifest=json.loads(Path(manifest_path).read_text())
    if sha256(manifest_path)!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    tiles={t['tile_id']:t for t in manifest['tiles']}; train={t['component'] for t in tiles.values() if t['split']=='train'}
    dev={t['component'] for t in tiles.values() if t['split']=='model_selection'}
    if train&dev or any(t['split'] not in ('train','model_selection') for t in tiles.values()):raise ValueError('Pixel/split leakage')
    cache={}; geometric_cases=0
    for tile_id,t in tiles.items():
        if sha256(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Supervised native cache differs')
        with np.load(t['native_cache'],allow_pickle=False) as z:cache[tile_id]={k:z[k].copy() for k in z.files}
        z=cache[tile_id]
        for case in [c for c in manifest['cases'] if c['tile_id']==tile_id]:
            if case['split']!=t['split']:raise ValueError('Case split differs')
            key=case['key']; target,visible=projected_truth(z[f'p{key}'],z['depth_a'],z['depth_b'],*t['cameras'])
            if visible.tolist()!=z[f'visible{key}'].tolist() or not np.allclose(target[visible],z[f'target{key}'][visible],atol=1e-5):
                raise ValueError('Independent training supervision differs')
            geometric_cases+=1
    replayed=0; run_results=[]
    expected_ids={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'}
    for path,meta in zip(paths,configs):
        folder=path.parent; complete=json.loads((folder/'COMPLETE.json').read_text())
        for name,digest in [('training.json',complete['training_sha256']),('selected.pt',complete['checkpoint_sha256']),
                            ('final.pt',complete['final_sha256']),('development_results.json.gz',complete['results_sha256'])]:
            if sha256(folder/name)!=digest:raise ValueError('Completed training artifact differs')
        if meta['calibration_or_test_pixels_loaded']!=0 or meta['final_test_enabled']:raise ValueError('Forbidden evaluation scope')
        if meta['completed_steps']!=protocol['steps'] or not meta['completed_requested_steps']:raise ValueError('Training budget incomplete')
        if meta['sampled_examples']!=protocol['steps']*protocol['batch_size'] or 2*meta['positive_local_labels']!=meta['sampled_examples']:
            raise ValueError('Balanced training-label count differs')
        for p,digest in meta['source_sha256'].items():
            if sha256(folder/'source'/p)!=digest:raise ValueError('Executed source snapshot differs')
        choices=[(h['selection']['correct_poses'],h['selection']['correct_correspondences_at_native_px']['2'],-h['step']) for h in meta['history']]
        best=max(choices)
        if list(best)!=meta['selection_key'] or -best[2]!=meta['selected_step']:raise ValueError('Checkpoint-selection replay differs')
        model=LocalRefiner(meta['variant'])
        model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True))
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
        final=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
        selected=model.state_dict()
        selected_change=float(sum((selected[k]-v).square().sum() for k,v in initial.items()).sqrt())
        if (meta['selected_step']==0)!=(selected_change==0):
            raise ValueError('Selected checkpoint initial/trained status differs')
        weight_change=float(sum((final[k]-v).square().sum() for k,v in initial.items()).sqrt())
        if not np.isclose(weight_change,meta['final_weight_change_l2']) or weight_change<=0:
            raise ValueError('Actual learned-weight change differs or is absent')
        if 28272+sum(p.numel() for p in model.parameters())!=meta['total_parameters'] or meta['total_parameters']>=100000:
            raise ValueError('Model parameter count differs')
        with gzip.open(folder/'development_results.json.gz','rt') as f:rows=json.load(f)
        if len(rows)!=len(expected_ids) or {r['pair_id'] for r in rows}!=expected_ids:raise ValueError('Missing development cases')
        counts=Counter(); pixels=Counter()
        for row in rows:
            t=tiles[row['tile_id']]; z=cache[row['tile_id']]; key=row['key']
            p,q=[np.asarray(row[k],float).reshape(-1,2) for k in ('reference_points_native','query_points_native')]
            indices=np.asarray(row['source_proposal_indices'],int)
            if not np.array_equal(p,z[f'p{key}'][indices]):raise ValueError('Reference coordinates changed')
            truth,visible=projected_truth(p,z['depth_a'],z['depth_b'],*t['cameras'])
            if visible.tolist()!=row['visible'] or not np.allclose(truth[visible],np.asarray(row['target_points_native']).reshape(-1,2),atol=1e-5):
                raise ValueError('Saved visibility/target differs')
            error=np.linalg.norm(q[visible]-truth[visible],axis=1)
            point_counts={str(v):int((error<=v).sum()) for v in (1,2,4,8,12)}
            if point_counts!=row['correct_at_native_px'] or row['returned']!=len(p) or row['scorable']!=int(visible.sum()):raise ValueError('Point metrics differ')
            finite=z['depth_a'][np.isfinite(z['depth_a'])&(z['depth_a']>0)]; distance=float(np.median(finite))
            if not np.isclose(distance,row['reference_range_m']):raise ValueError('Reference range differs')
            correct=strict=False; pose=row['estimated_cam2world']
            if pose is not None:
                pose=np.asarray(pose); gt=np.asarray(t['cameras'][1]['cam2world'])
                rotation=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@gt[:3,:3])-1)/2,-1,1))))
                translation=float(np.linalg.norm(pose[:3,3]-gt[:3,3]))
                if not np.allclose([rotation,translation],[row['pose_error']['rotation_deg'],row['pose_error']['translation_m']],atol=1e-6):raise ValueError('Pose error differs')
                correct=rotation<=2 and translation<=.01*distance; strict=rotation<=.5 and translation<=.001*distance
            if correct!=row['pose_correct'] or strict!=row['strict_pose_correct']:raise ValueError('Pose classification differs')
            counts.update(dict(cases=1,correct_poses=int(correct),strict_correct_poses=int(strict),estimated_poses=int(pose is not None),returned_matches=len(p),scorable_matches=int(visible.sum())))
            pixels.update(point_counts); replayed+=1
        for k,v in counts.items():
            if meta['evaluation'][k]!=v:raise ValueError('Training aggregate differs: '+k)
        if dict(pixels)!=meta['evaluation']['correct_correspondences_at_native_px']:raise ValueError('Point aggregate differs')
        run_results.append(dict(variant=meta['variant'],seed=meta['seed'],selection_key=meta['selection_key'],cases=len(rows)))
    primary=[r for r in run_results if r['seed']==20261101]
    # Preserve the predefined ablation-order tie break rather than filesystem order.
    order={v:i for i,v in enumerate(protocol['variants'])}
    winner=max(primary,key=lambda r:(tuple(r['selection_key']),-order[r['variant']]))['variant']
    if winner!=summary['winner']:raise ValueError('Ablation-selection replay differs')
    if not summary['pilot'] and {r['seed'] for r in run_results if r['variant']==winner}!={20261101,20261102,20261103}:
        raise ValueError('Missing predefined winner seeds')
    result=dict(passed=True,scope='Development-only; no new final-test or generalization claim.',
                training_geometry_cases=geometric_cases,trained_runs=len(configs),development_rows=replayed,
                selected_variant=winner,calibration_or_test_pixels_loaded=0,
                manifest_sha256=sha256(manifest_path),source_sha256=sha256(__file__),runs=run_results)
    write_new(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/localization-v3-balanced-development')
    p.add_argument('--output',default='reports/localization_balanced_audit_v3.json')
    a=p.parse_args();print(json.dumps(audit(a.run,a.output),indent=2))
