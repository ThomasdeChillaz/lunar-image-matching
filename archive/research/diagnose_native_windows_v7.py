"""Corrected exact-in-window controls; query truth is diagnostic only."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from localization_diagnostics_v3 import native_points
from localization_robust_pose_v4 import CONFIGS,estimate
from lunar_geometry_v5 import correspondences_z,median_reference_range_z
from lunargeo_dataset import write_new
from rotation_bank_pose_v4 import quarter_turn
from train_balanced_localization_v3 import load_bundle


def run(output):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest,tiles,data=load_bundle('reports/localization_training_data_v5.json');model=CompactMatcher(manifest['coarse_model'])
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify');rows=[]
    protocol=dict(scope='Corrected exact-in-window development diagnostic, not a matcher, fresh test or formal pose-performance bound.',
        source_sha256=sha256(__file__),manifest_sha256=sha256('reports/localization_training_data_v5.json'),
        frozen_checkpoint_sha256=manifest['coarse_checkpoint_sha256'],window_radius_native=16,support_radius_native=12,
        note='All four branches are examined using query truth after image-only proposals. Never use oracle choices as deployed predictions.',
        calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json',protocol)
    for c in [c for c in manifest['cases'] if c['split']=='model_selection']:
        z=data[c['tile_id']];t=tiles[c['tile_id']];a,b=t['cameras'];ref,query=z['images'][2*c['light_a']],z['images'][2*c['light_b']+1]
        distance=median_reference_range_z(z['depth_a'],a['intrinsics'])
        for k in range(4):
            p,q=model.points(*[cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA) for im in (ref,np.rot90(query,k).copy())])
            p,q=[native_points(v,ref.shape) for v in (p,q)]
            # Query geometry is first used after the image-only proposals.
            _,gt,world,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
            center=np.rint(q[visible]);target_rotated=quarter_turn(gt,k,512)
            support=((p[visible]>=12)&(p[visible]<500)).all(1)&((center-16>=12)&(center+16<500)).all(1)
            in_window=(np.abs(target_rotated-center)<=16).all(1)&support
            controls={}
            for name,mask in [('all_visible_reference_points',np.ones(len(gt),bool)),('exact_inside_local_window',in_window)]:
                pose,info=estimate(world[mask],gt[mask],b['intrinsics'],config);error=None;correct=False
                if pose is not None:
                    truth=np.array(b['cam2world']);angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
                    translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]));error=dict(rotation_deg=angle,translation_m=translation)
                    correct=angle<=2 and translation<=.01*distance
                controls[name]=dict(points=int(mask.sum()),pose=None if pose is None else pose.tolist(),pose_error=error,pose_correct=bool(correct),inlier_info=info)
            rows.append(dict(c,quarter_turn=k,reference_points_native=p.tolist(),coarse_query_points_rotated_native=q.tolist(),
                visible=visible.tolist(),target_visible_points_native=gt.tolist(),inside_window=in_window.tolist(),reference_range_m=distance,controls=controls))
    with gzip.open(root/'results.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
    ids=sorted({r['pair_id'] for r in rows});summary={}
    for name in ('all_visible_reference_points','exact_inside_local_window'):
        summary[name]=dict(cases=len(ids),pairs_with_any_correct_oracle_branch=sum(any(r['controls'][name]['pose_correct'] for r in rows if r['pair_id']==key) for key in ids))
    result=dict(scope=protocol['scope'],results=summary,branches=len(rows),results_sha256=sha256(root/'results.json.gz'),
        limitation='An exact-in-window control diagnoses search coverage; it is not deployable and not a formal upper bound on poses meeting a finite error threshold.')
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
