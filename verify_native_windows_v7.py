"""Independent axial/rotation/window/pose replay of oracle diagnostics."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import cv2

from dataset import sha256
from lunargeo_dataset import write_new
from verify_axial_results_v5 import projected
from localization_robust_pose_v4 import CONFIGS,estimate


def audit(root,output):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads(Path('reports/localization_training_data_v5.json').read_text())
    if sha256('reports/localization_training_data_v5.json')!=protocol['manifest_sha256'] or sha256(root/'results.json.gz')!=summary['results_sha256']:
        raise ValueError('Frozen window-control inputs/results differ')
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'};data={};distances={}
    cv2.setNumThreads(1);config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify')
    for key,t in tiles.items():
        if sha256(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Native window cache changed')
        with np.load(t['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in ('depth_a','depth_b')}
        d=data[key]['depth_a'];y,x=np.mgrid[:d.shape[0],:d.shape[1]]
        rays=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])@np.linalg.inv(np.array(t['cameras'][0]['intrinsics'])).T
        valid=np.isfinite(d.ravel())&(d.ravel()>0);distances[key]=float(np.median(d.ravel()[valid]*np.linalg.norm(rays[valid],axis=1)))
    with gzip.open(root/'results.json.gz','rt') as f:rows=json.load(f)
    for row in rows:
        t=tiles[row['tile_id']];a,b=t['cameras'];z=data[row['tile_id']]
        p=np.array(row['reference_points_native']).reshape(-1,2);q=np.array(row['coarse_query_points_rotated_native']).reshape(-1,2)
        target,visible=projected(p,z['depth_a'],z['depth_b'],a,b);gt=target[visible]
        if not np.isclose(row['reference_range_m'],distances[row['tile_id']]):raise ValueError('Physical oracle range differs')
        if visible.tolist()!=row['visible'] or not np.allclose(gt,np.array(row['target_visible_points_native']).reshape(-1,2),atol=1e-6,rtol=0):raise ValueError('Independent exact window truth differs')
        k=row['quarter_turn'];x,y=gt.T
        rotated=[gt,np.column_stack([y,511-x]),np.column_stack([511-x,511-y]),np.column_stack([511-y,x])][k]
        center=np.rint(q[visible]);support=((p[visible]>=12)&(p[visible]<500)).all(1)&((center-16>=12)&(center+16<500)).all(1)
        inside=(np.abs(rotated-center)<=16).all(1)&support
        if inside.tolist()!=row['inside_window']:raise ValueError('Native rotation/window mask differs')
        indices=np.rint(p[visible]).astype(int);depth=z['depth_a'][indices[:,1],indices[:,0]]
        ray=np.column_stack([p[visible],np.ones(visible.sum())])@np.linalg.inv(np.array(a['intrinsics'])).T
        ca=np.array(a['cam2world']);world=(ray*depth[:,None])@ca[:3,:3].T+ca[:3,3]
        for name,mask in [('all_visible_reference_points',np.ones(len(gt),bool)),('exact_inside_local_window',inside)]:
            control=row['controls'][name]
            if control['points']!=int(mask.sum()):raise ValueError('Oracle point denominator differs')
            replayed_pose,_=estimate(world[mask],gt[mask],b['intrinsics'],config)
            if (replayed_pose is None)!=(control['pose'] is None) or (replayed_pose is not None and not np.allclose(replayed_pose,control['pose'],atol=.001,rtol=0)):
                raise ValueError('Oracle pose-estimator replay differs')
            correct=False
            if control['pose'] is not None:
                pose=np.array(control['pose']);truth=np.array(b['cam2world']);angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
                translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]))
                if not np.allclose([angle,translation],[control['pose_error']['rotation_deg'],control['pose_error']['translation_m']],atol=1e-6):raise ValueError('Oracle pose error differs')
                correct=angle<=2 and translation<=.01*row['reference_range_m']
            if bool(correct)!=control['pose_correct']:raise ValueError('Oracle pose criterion differs')
    expected={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'}
    if len(rows)!=len(expected)*4 or {(r['pair_id'],r['quarter_turn']) for r in rows}!={(key,k) for key in expected for k in range(4)}:
        raise ValueError('Missing/repeated oracle branch')
    for name in ('all_visible_reference_points','exact_inside_local_window'):
        result=dict(cases=len(expected),pairs_with_any_correct_oracle_branch=sum(any(r['controls'][name]['pose_correct'] for r in rows if r['pair_id']==key) for key in expected))
        if result!=summary['results'][name]:raise ValueError('Oracle aggregate differs')
    result=dict(passed=True,scope='Corrected exact-in-window diagnostic replay, not a deployed method, fresh test or formal pose bound.',
        branches=len(rows),eligible_pairs=len(expected),results=summary['results'],results_sha256=summary['results_sha256'],source_sha256=sha256(__file__))
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.run,a.output)
