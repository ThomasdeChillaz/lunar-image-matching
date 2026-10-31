"""Frozen image matches, corrected axial reference geometry, unchanged pose criteria."""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from dataset import sha256
from localization_diagnostics_v3 import point_metrics
from localization_robust_pose_v4 import CONFIGS,estimate,residuals
from lunar_geometry_v2 import pose_error
from lunar_geometry_v5 import map_points_z,correspondences_z,median_reference_range_z
from lunargeo_dataset import write_new
from rotation_bank_pose_v4 import choose_branch
from train_balanced_localization_v3 import load_bundle


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def score(row,tile,z,config,distance):
    a,b=tile['cameras'];p,q=[np.asarray(row[k],float).reshape(-1,2) for k in ('reference_points_native','query_points_native')]
    world,valid=map_points_z(p,z['depth_a'],a['intrinsics'],a['cam2world'])
    pose,info=estimate(world,q[valid],b['intrinsics'],config)
    median=None
    if pose is not None:
        r=pose[:3,:3].T;t=-r@pose[:3,3]
        err=residuals(world,q[valid],np.asarray(b['intrinsics']),cv2.Rodrigues(r)[0],t)
        median=float(np.median(err[err<=3])) if (err<=3).any() else float('inf')
    # Query truth enters only after inference, as in the historical pipeline.
    error=None if pose is None else pose_error(pose,np.asarray(b['cam2world']))
    _,target,_,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
    da=z['depth_a'];legacy_scale=float(np.median(da[np.isfinite(da)&(da>0)]))
    return dict(pair_id=row.get('pair_id',row['tile_id']+f'/lights-{row["light_a"]}-{row["light_b"]}'),
        tile_id=row['tile_id'],component=tile['component'],light_a=row['light_a'],light_b=row['light_b'],
        **point_metrics(q,target,visible),pose=None if pose is None else pose.tolist(),pose_error=error,
        pose_correct=bool(error and error['rotation_deg']<=2 and error['translation_m']<=.01*distance),
        strict_pose_correct=bool(error and error['rotation_deg']<=.5 and error['translation_m']<=.001*distance),
        pose_correct_with_old_depth_median_scale=bool(error and error['rotation_deg']<=2 and error['translation_m']<=.01*legacy_scale),
        reference_range_m=distance,reference_axial_depth_median_m=legacy_scale,
        reference_points_native=p.tolist(),query_points_native=q.tolist(),target_points_native=target.tolist(),visible=visible.tolist(),
        info=info,median_inlier_residual=median,solver=config['name'],depth_convention='optical_axis_z')


def summarize(rows):
    return dict(cases=len(rows),correct=sum(r['pose_correct'] for r in rows),strict=sum(r['strict_pose_correct'] for r in rows),
        estimated=sum(r['pose'] is not None for r in rows),
        correct_with_old_depth_median_scale=sum(r['pose_correct_with_old_depth_median_scale'] for r in rows),
        returned=sum(r['returned'] for r in rows),scorable=sum(r['scorable'] for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows))


def run(output):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);cv2.setNumThreads(1)
    manifest,tiles,data=load_bundle('reports/localization_training_data_v3.json')
    original='runs/localization-diagnostics-v3-validated/results.json.gz'
    inputs={'frozen_coarse':[r for r in read(original) if r['experiment']=='coarse'],
            'frozen_native':[r for r in read(original) if r['experiment']=='deterministic_native']}
    paths=[original]
    for seed in (20261101,20261102,20261103):
        p=f'runs/localization-v3-balanced-development/reliability-{seed}/development_results.json.gz'
        inputs[f'legacy_target_reliability_{seed}']=read(p);paths.append(p)
    sift='runs/localization-v4-native-sift/results.json.gz';sift_rows=read(sift);paths.append(sift)
    for name in ('sift_native','sift_clahe'):inputs[name]=[r for r in sift_rows if r['method']==name]
    rotation='runs/localization-v4-rotation-bank/results.json.gz';paths.append(rotation)
    protocol=dict(scope='Corrected development geometry; fixed prior image matches and weights. Not a learned-model improvement or fresh test.',
        inputs={p:sha256(p) for p in paths},manifest_sha256=sha256('reports/localization_training_data_v3.json'),
        sources={p:sha256(p) for p in ('rescore_axial_geometry_v5.py','lunar_geometry_v5.py','localization_robust_pose_v4.py')},
        convention_basis='Raw EXR/cache equality, independent training-geometry depth closure and upstream DUSt3R axial unprojection; chosen before pose evaluation.',
        intrinsics='Unchanged stored source intrinsics. Exploratory half-pixel shifts are not adopted.',
        threshold='Rotation<=2deg and translation<=1% of actual median Euclidean reference range; also replay old axial-depth-median scale.',
        visibility='Preserve historical0.05m+1% depth tolerance for this isolated convention correction.',
        legacy_warning='All previous ray-based LunarGeo pose/correspondence metrics and learned3D target labels are superseded; footprints require reaudit.',
        calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json',protocol);distances={key:median_reference_range_z(z['depth_a'],tiles[key]['cameras'][0]['intrinsics']) for key,z in data.items() if tiles[key]['split']=='model_selection'}
    records=[];summary={};configs=[CONFIGS[0],next(c for c in CONFIGS if c['name']=='ap3p_reclassify')]
    for name,source in inputs.items():
        for config in configs:
            rows=[dict(score(r,tiles[r['tile_id']],data[r['tile_id']],config,distances[r['tile_id']]),method=name) for r in source]
            if len(rows)!=72:raise ValueError('Incomplete common case list')
            records.extend(rows);summary[name+'/'+config['name']]=summarize(rows)
            print(name,config['name'],summary[name+'/'+config['name']],flush=True)
    all_rot=read(rotation);selected=[];rot_records=[];config=configs[1]
    for pair in dict.fromkeys(r['pair_id'] for r in all_rot):
        candidates=[]
        for source in [r for r in all_rot if r['pair_id']==pair]:
            r=dict(score(source,tiles[source['tile_id']],data[source['tile_id']],config,distances[source['tile_id']]),method='quarter_turn_bank',quarter_turn=source['quarter_turn'])
            if r['pose'] is not None:r['pose']=np.asarray(r['pose'])
            candidates.append(r)
        winner=choose_branch(candidates)
        for r in candidates:
            r['selected']=r is winner
            if r['pose'] is not None:r['pose']=r['pose'].tolist()
            rot_records.append(r)
        if winner is None:
            empty=dict(candidates[0]);empty.update(pose=None,pose_error=None,pose_correct=False,strict_pose_correct=False,pose_correct_with_old_depth_median_scale=False)
            selected.append(empty)
        else:selected.append(winner)
    records.extend(rot_records);summary['quarter_turn_bank/'+config['name']]=summarize(selected)
    with gzip.open(root/'results.json.gz','wt') as f:json.dump(records,f,allow_nan=False)
    result=dict(scope=protocol['scope'],results=summary,records=len(records),results_sha256=sha256(root/'results.json.gz'))
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
