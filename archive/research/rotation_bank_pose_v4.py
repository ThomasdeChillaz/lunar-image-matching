"""Image-only quarter-turn hypotheses with reference-geometry branch selection."""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from localization_diagnostics_v3 import map_points,native_points,point_metrics,refine_native
from localization_robust_pose_v4 import CONFIGS,estimate,residuals
from lunar_geometry_v2 import correspondences,pose_error
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle


def quarter_turn(points,k,size,inverse=False):
    p=np.asarray(points,float).reshape(-1,2);k=(-k if inverse else k)%4
    if k==0:return p.copy()
    x,y=p.T
    if k==1:return np.column_stack([y,size-1-x])
    if k==2:return np.column_stack([size-1-x,size-1-y])
    return np.column_stack([size-1-y,x])


def branch(matcher,reference,query,k,reference_dense,query_dense):
    image=np.rot90(query,k).copy()
    low=[cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA) for im in (reference,image)]
    p,q=matcher.points(*low);p,q=[native_points(v,reference.shape) for v in (p,q)]
    with torch.inference_mode():p,q,_=refine_native(p,q,reference_dense,query_dense)
    return p,quarter_turn(q,k,query.shape[0],inverse=True)


def choose_branch(candidates):
    """No true query pose/depth: rank by geometric inliers, then residual, then k."""
    valid=[r for r in candidates if r['pose'] is not None]
    return max(valid,key=lambda r:(r['info']['inliers'],-r['median_inlier_residual'],-r['quarter_turn'])) if valid else None


def run(output):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest,tiles,data=load_bundle('reports/localization_training_data_v3.json')
    matcher=CompactMatcher(manifest['coarse_model']);features={};rows=[]
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify')
    protocol=dict(scope='Development-only four-orientation engineering baseline, not a newly trained method.',
        sources={p:sha256(p) for p in ('rotation_bank_pose_v4.py','localization_robust_pose_v4.py','compact_lunar.py','localization_diagnostics_v3.py')},
        manifest_sha256=sha256('reports/localization_training_data_v3.json'),
        checkpoint_sha256=sha256(Path(manifest['coarse_model'])/'selected.pt'),
        total_parameters=28272,quarter_turns=[0,1,2,3],pose_config=config,
        selection='Geometric inlier count; then median inlier reprojection residual; then smaller quarter-turn.',
        selection_bank=manifest['selection_bank'],calibration_or_test_pixels_loaded=0,
        inference_inputs='Both images, reference depth/pose and query intrinsics; query truth enters scoring only.',
        timing='Sequential shared-CPU experiment with cached descriptor maps. Not a matched end-to-end speed comparison.')
    write_new(root/'protocol.json',protocol)
    for c in [c for c in manifest['cases'] if c['split']=='model_selection']:
        z=data[c['tile_id']];t=tiles[c['tile_id']];a,b=t['cameras'];ia,ib=2*c['light_a'],2*c['light_b']+1
        def features_for(index,k):
            key=(c['tile_id'],index,k)
            if key not in features:
                image=np.rot90(z['images'][index],k).copy()
                with torch.inference_mode():features[key]=matcher.model(torch.from_numpy(image).float()[None,None]/255)[0]
            return features[key]
        candidates=[];tick=perf_counter()
        for k in range(4):
            p,q=branch(matcher,z['images'][ia],z['images'][ib],k,features_for(ia,0),features_for(ib,k))
            world,use=map_points(p,z['depth_a'],a['intrinsics'],a['cam2world']);pose,info=estimate(world,q[use],b['intrinsics'],config)
            median=None
            if pose is not None:
                pose_array=np.asarray(pose);r=pose_array[:3,:3].T;tvec=-r@pose_array[:3,3]
                errors=residuals(world,q[use],np.asarray(b['intrinsics']),cv2.Rodrigues(r)[0],tvec)
                median=float(np.median(errors[errors<=3])) if (errors<=3).any() else float('inf')
            candidates.append(dict(quarter_turn=k,p=p,q=q,pose=pose,info=info,median_inlier_residual=median))
        chosen=choose_branch(candidates);elapsed=(perf_counter()-tick)*1000
        # Scoring only: none of this information influences branch selection.
        depth=z['depth_a'];distance=float(np.median(depth[np.isfinite(depth)&(depth>0)]))
        case_rows=[]
        for candidate in candidates:
            p,q=candidate['p'],candidate['q'];pose=candidate['pose']
            err=None if pose is None else pose_error(pose,np.asarray(b['cam2world']))
            _,truth,_,visible=correspondences(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
            record=dict(c,**point_metrics(q,truth,visible),quarter_turn=candidate['quarter_turn'],selected=candidate is chosen,
                selected_without_query_truth=True,pose_correct=bool(err and err['rotation_deg']<=2 and err['translation_m']<=.01*distance),
                strict_pose_correct=bool(err and err['rotation_deg']<=.5 and err['translation_m']<=.001*distance),
                pose_error=err,pose=None if pose is None else pose.tolist(),reference_range_m=distance,
                geometric_inliers=candidate['info']['inliers'],median_inlier_residual=candidate['median_inlier_residual'],
                reference_points_native=p.tolist(),query_points_native=q.tolist(),target_points_native=truth.tolist(),visible=visible.tolist(),
                case_elapsed_ms=elapsed)
            rows.append(record);case_rows.append(record)
        print(c['pair_id'],'chosen',None if chosen is None else chosen['quarter_turn'],'correct',any(r['selected'] and r['pose_correct'] for r in case_rows),flush=True)
    bank=set(manifest['selection_bank']);summary={}
    for stage in ('selection_bank','expanded_development'):
        part=[r for r in rows if stage=='expanded_development' or r['pair_id'] in bank]
        ids={r['pair_id'] for r in part};selected=[r for r in part if r['selected']]
        summary[stage]=dict(cases=len(ids),correct=sum(r['pose_correct'] for r in selected),
            strict=sum(r['strict_pose_correct'] for r in selected),estimated=len(selected),
            unrotated_correct=sum(r['pose_correct'] for r in part if r['quarter_turn']==0),
            oracle_best_rotation_correct=sum(any(r['pose_correct'] for r in part if r['pair_id']==key) for key in ids),
            note='Oracle-best rotation is diagnostic only; model result uses the image/reference-geometry selection.')
    with gzip.open(root/'results.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
    result=dict(scope=protocol['scope'],results=summary,results_sha256=sha256(root/'results.json.gz'))
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
