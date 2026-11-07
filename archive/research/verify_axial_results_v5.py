"""Independent axial projection/visibility/pose-error replay and input preservation."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def projected(points,da,db,a,b):
    p=np.asarray(points,float).reshape(-1,2);ka,kb=np.array(a['intrinsics']),np.array(b['intrinsics']);ca,cb=np.array(a['cam2world']),np.array(b['cam2world'])
    if not len(p):return np.empty((0,2)),np.zeros(0,bool)
    index=np.rint(p).astype(int);valid=((index>=0)&(index<np.array(da.shape[::-1]))).all(1)
    index=index.clip(0,np.array(da.shape[::-1])-1);z=da[index[:,1],index[:,0]];valid&=np.isfinite(z)&(z>0)
    ray=np.column_stack([p,np.ones(len(p))])@np.linalg.inv(ka).T
    world=(ray*np.where(valid,z,0)[:,None])@ca[:3,:3].T+ca[:3,3]
    camera=(world-cb[:3,3])@cb[:3,:3];homogeneous=camera@kb.T
    with np.errstate(divide='ignore',invalid='ignore'):q=homogeneous[:,:2]/homogeneous[:,2,None]
    valid&=(camera[:,2]>0)&np.isfinite(q).all(1)&((q>=0)&(q<np.array(db.shape[::-1])-1)).all(1)
    safe=np.where(np.isfinite(q),q,-1).astype(np.float32)
    observed=cv2.remap(np.where(np.isfinite(db),db,0).astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR).ravel()
    support=cv2.remap((np.isfinite(db)&(db>0)).astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR).ravel()
    valid&=(support>=.9999)&(np.abs(observed-camera[:,2])<=.05+.01*camera[:,2])
    return q,valid


def audit(root,output):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    for p,d in {**protocol['inputs'],**protocol['sources']}.items():
        if sha256(p)!=d:raise ValueError('Frozen source/input differs')
    if sha256(root/'results.json.gz')!=summary['results_sha256']:raise ValueError('Results changed')
    manifest=json.loads(Path('reports/localization_training_data_v5.json').read_text());tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'}
    data={};distances={};legacy_scales={}
    for key,t in tiles.items():
        if sha256(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Corrected cache changed')
        with np.load(t['native_cache']) as z:data[key]={k:z[k].copy() for k in ('depth_a','depth_b')}
        d=data[key]['depth_a'];y,x=np.mgrid[:d.shape[0],:d.shape[1]];ray=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])@np.linalg.inv(np.array(t['cameras'][0]['intrinsics'])).T
        valid=np.isfinite(d.ravel())&(d.ravel()>0);distances[key]=float(np.median(d.ravel()[valid]*np.linalg.norm(ray[valid],axis=1)))
        legacy_scales[key]=float(np.median(d.ravel()[valid]))
    sources={}
    for path in protocol['inputs']:
        for r in read(path):
            if 'experiment' in r:
                if r['experiment'] not in ('coarse','deterministic_native'):continue
                method='frozen_coarse' if r['experiment']=='coarse' else 'frozen_native'
            elif 'quarter_turn' in r:method='quarter_turn_bank'
            elif 'native-sift' in path:method=r['method']
            else:
                seed=path.split('reliability-')[1].split('/')[0];method='legacy_target_reliability_'+seed
            pair=r.get('pair_id',r['tile_id']+f'/lights-{r["light_a"]}-{r["light_b"]}')
            sources[(method,pair,r.get('quarter_turn',0))]=r
    rows=read(root/'results.json.gz');groups=defaultdict(list)
    for r in rows:
        source=sources[(r['method'],r['pair_id'],r.get('quarter_turn',0))]
        p,q=[np.asarray(r[k],float).reshape(-1,2) for k in ('reference_points_native','query_points_native')]
        if not np.array_equal(p,np.asarray(source['reference_points_native']).reshape(-1,2)) or not np.array_equal(q,np.asarray(source['query_points_native']).reshape(-1,2)):
            raise ValueError('Image matches changed during geometry correction')
        t=tiles[r['tile_id']];z=data[r['tile_id']];target,visible=projected(p,z['depth_a'],z['depth_b'],*t['cameras'])
        if visible.tolist()!=r['visible'] or not np.allclose(target[visible],np.asarray(r['target_points_native']).reshape(-1,2),atol=1e-5):raise ValueError('Independent axial visibility/projection differs')
        err=np.linalg.norm(q[visible]-target[visible],axis=1)
        if r['returned']!=len(p) or r['scorable']!=int(visible.sum()) or r['correct_at_native_px']!={str(v):int((err<=v).sum()) for v in (1,2,4,8,12)}:raise ValueError('Native point metrics differ')
        if not np.isclose(r['reference_range_m'],distances[r['tile_id']]):raise ValueError('Physical reference range differs')
        correct=strict=legacy=False
        if r['pose'] is not None:
            pose=np.array(r['pose']);truth=np.array(t['cameras'][1]['cam2world'])
            angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
            translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]))
            if not np.allclose([angle,translation],[r['pose_error']['rotation_deg'],r['pose_error']['translation_m']],atol=1e-6):raise ValueError('Pose error differs')
            correct=angle<=2 and translation<=.01*distances[r['tile_id']];strict=angle<=.5 and translation<=.001*distances[r['tile_id']]
            legacy=angle<=2 and translation<=.01*legacy_scales[r['tile_id']]
        if (bool(correct),bool(strict),bool(legacy))!=(r['pose_correct'],r['strict_pose_correct'],r['pose_correct_with_old_depth_median_scale']):raise ValueError('Pose criteria differ')
        groups[(r['method'],r['solver'])].append(r)
    expected={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'}
    for (method,solver),part in groups.items():
        if method=='quarter_turn_bank':
            chosen=[]
            for pair in sorted(expected):
                candidates=[r for r in part if r['pair_id']==pair]
                if {r['quarter_turn'] for r in candidates}!={0,1,2,3}:raise ValueError('Rotation branch missing')
                valid=[r for r in candidates if r['pose'] is not None]
                winner=max(valid,key=lambda r:(r['info']['inliers'],-r['median_inlier_residual'],-r['quarter_turn'])) if valid else None
                if any(r['selected']!=(r is winner) for r in candidates):raise ValueError('Truth-free branch selection differs')
                if winner:chosen.append(winner)
                else:chosen.append(next(r for r in candidates if r['quarter_turn']==0))
            part=chosen
        if len(part)!=72 or {r['pair_id'] for r in part}!=expected:raise ValueError('Eligible denominator differs')
        values=dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),strict=sum(r['strict_pose_correct'] for r in part),estimated=sum(r['pose'] is not None for r in part),
            correct_with_old_depth_median_scale=sum(r['pose_correct_with_old_depth_median_scale'] for r in part),returned=sum(r['returned'] for r in part),scorable=sum(r['scorable'] for r in part),correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in part))
        if values!=summary['results'][method+'/'+solver]:raise ValueError('Corrected aggregate differs')
    result=dict(passed=True,scope='Corrected development-metric replay; not a fresh-test or learned-model gain.',
        rows=len(rows),aggregates=len(groups),image_coordinates_preserved=True,query_truth_excluded_from_branch_selection=True,
        source_sha256=sha256(__file__),results_sha256=summary['results_sha256'])
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/localization-v5-axial-correction')
    p.add_argument('--output',default='reports/localization_v5_rescore_audit.json');a=p.parse_args();audit(a.run,a.output)
