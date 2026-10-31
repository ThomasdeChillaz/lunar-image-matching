"""Image-derived coarse candidate coverage; GT is used only for oracle diagnosis."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from contrastive_terrain import sample_descriptors
from dataset import sha256
from localization_diagnostics_v3 import map_points,native_points
from localization_robust_pose_v4 import CONFIGS,estimate
from lunar_geometry_v2 import pose_error
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle

K_VALUES=(1,4,8,16)


def rank_coverage(similarity,candidates,target,visible,radius,topk):
    """True correspondence within radius of an image-ranked candidate center."""
    order=np.argsort(-similarity,axis=1,kind='stable')[:,:topk]
    centers=np.rint(candidates[order])
    available=(np.abs(centers-target[:,None])<=radius).all(2).any(1)&visible
    return available


def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    path='reports/localization_training_data_v3.json'
    manifest,tiles,data=load_bundle(path);coarse=CompactMatcher(manifest['coarse_model'])
    bank=set(manifest['selection_bank'])
    cases=[c for c in manifest['cases'] if c['pair_id'] in bank]
    protocol=dict(scope='Metadata-selected development bank oracle coverage; no candidate accuracy or fresh validation.',
        started_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha256(path),
        coarse_checkpoint_sha256=sha256(Path(manifest['coarse_model'])/'selected.pt'),
        sources={p:sha256(p) for p in ('diagnose_proposal_coverage_v4.py','localization_robust_pose_v4.py')},
        image_sizes=[128,256],top_k=list(K_VALUES),radius_native=16,existing_radii=[16,32,64,128,256],
        selection_bank=sorted(bank),calibration_or_test_pixels_loaded=0,
        distinction='Coarse rankings use image descriptors only. Oracle labels and poses use query truth, exclusively as labelled diagnostics.')
    write_new(output/'protocol.json',protocol);rows=[];cache={};started=perf_counter()
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify')
    for case in cases:
        t=tiles[case['tile_id']];z=data[case['tile_id']];key=case['key']
        p=z[f'p{key}'];truth=z[f'target{key}'];visible=z[f'visible{key}']
        a,b=t['cameras'];world,valid=map_points(p,z['depth_a'],a['intrinsics'],a['cam2world'])
        def record(name,available):
            selected=available[valid];pose,info=estimate(world[selected],truth[valid][selected],b['intrinsics'],config)
            err=None if pose is None else pose_error(pose,np.asarray(b['cam2world']))
            ranges=z['depth_a'];distance=float(np.median(ranges[np.isfinite(ranges)&(ranges>0)]))
            rows.append(dict(case,method=name,available_points=int(available.sum()),visible_points=int(visible.sum()),
                oracle_pose_correct=bool(err and err['rotation_deg']<=2 and err['translation_m']<=.01*distance),
                oracle_pose=None if pose is None else pose.tolist(),oracle_pose_error=err,oracle_info=info))
        for radius in protocol['existing_radii']:
            record(f'existing_radius_{radius}',visible&(np.abs(truth-np.rint(z[f'q{key}']))<=radius).all(1))
        for size in protocol['image_sizes']:
            features=[]
            for i in (2*case['light_a'],2*case['light_b']+1):
                cache_key=(case['tile_id'],i,size)
                if cache_key not in cache:
                    im=cv2.resize(z['images'][i],(size,size),interpolation=cv2.INTER_AREA)
                    with torch.inference_mode():cache[cache_key]=coarse.model(torch.from_numpy(im.copy()).float()[None,None]/255)[0]
                features.append(cache[cache_key])
            yy,xx=np.mgrid[12:size-12:4,12:size-12:4]
            grid=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
            ref=(p+.5)*size/512-.5
            with torch.inference_mode():
                rd=sample_descriptors(features[0],torch.from_numpy(ref.astype(np.float32))[None])[0]
                qd=sample_descriptors(features[1],torch.from_numpy(grid)[None])[0]
                similarity=(rd@qd.T).numpy()
            candidates=native_points(grid,(512,512),(size,size))
            for topk in K_VALUES:
                record(f'global_{size}_top{topk}',rank_coverage(similarity,candidates,truth,visible,16,topk))
        print(case['pair_id'],'coverage computed',flush=True)
    with gzip.open(output/'results.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
    names=list(dict.fromkeys(r['method'] for r in rows));summary={}
    for name in names:
        subset=[r for r in rows if r['method']==name]
        summary[name]=dict(cases=len(subset),oracle_correct=sum(r['oracle_pose_correct'] for r in subset),
                           available_points=sum(r['available_points'] for r in subset),
                           fewer_than_six=sum(r['available_points']<6 for r in subset))
    result=dict(scope=protocol['scope'],results=summary,elapsed_seconds=perf_counter()-started,results_sha256=sha256(output/'results.json.gz'))
    write_new(output/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
