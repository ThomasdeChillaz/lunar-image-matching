"""Corrected exact, quantized and noisy correspondence pose controls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from localization_robust_pose_v4 import estimate,CONFIGS
from lunar_geometry_v5 import correspondences_z,median_reference_range_z
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle


def run(output):
    torch_threads=2
    import torch
    torch.set_num_threads(torch_threads);cv2.setNumThreads(1)
    manifest,tiles,data=load_bundle('reports/localization_training_data_v5.json')
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify');rows=[];groups=defaultdict(list)
    for key,t in tiles.items():
        z=data[key];a,b=t['cameras'];p,q,world,_=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],stride=16)
        indices=np.linspace(0,len(p)-1,min(256,len(p))).astype(int);p,q,world=p[indices],q[indices],world[indices]
        distance=median_reference_range_z(z['depth_a'],a['intrinsics']);truth=np.array(b['cam2world'])
        seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed)
        controls=[('exact',q.copy())]+[(f'rounded_{step}px',np.rint(q/step)*step) for step in (2,4,8,16,32)]
        for noise in (1,2,4,8):
            noisy=q+rng.normal(0,noise,q.shape)
            for fraction in (0.,.1,.3):
                pixels=noisy.copy();bad=rng.choice(len(q),int(len(q)*fraction),replace=False)
                pixels[bad]=rng.uniform(0,512,(len(bad),2))
                controls.append((f'noise_{noise}px_outliers_{fraction:g}',pixels))
        for name,pixels in controls:
            pose,info=estimate(world,pixels,b['intrinsics'],config);angle=translation=None;correct=False
            if pose is not None:
                angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
                translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]));correct=angle<=2 and translation<=.01*distance
            row=dict(tile_id=key,split=t['split'],component=t['component'],control=name,points=len(p),
                reference_points_native=p.tolist(),query_points_native=pixels.tolist(),world_points=world.tolist(),
                pose=None if pose is None else pose.tolist(),rotation_error_deg=angle,translation_error_m=translation,
                reference_range_m=distance,pose_correct=bool(correct),inlier_info=info)
            rows.append(row);groups[name].append(row)
    results={name:dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),estimated=sum(r['pose'] is not None for r in part)) for name,part in groups.items()}
    exact=results['exact'];passed=exact['correct']/exact['cases']>=.95
    result=dict(passed=passed,scope='Corrected development geometry controls, not matcher performance or new terrain validation.',
        source_sha256=sha256(__file__),manifest_sha256=sha256('reports/localization_training_data_v5.json'),
        caution='Exact projected correspondences are a self-consistency control. Independent raw-depth/source closure audit is required separately and already passed.',
        timing='Executed after v5 fitting started; must not be described as its preregistered diagnostic gate.',
        results=results,rows=rows,calibration_or_test_pixels_loaded=0)
    write_new(output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not passed:raise ValueError('Exact correspondence diagnostic gate failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/localization-v5-axial-pose-controls.json')
    run(p.parse_args().output)
