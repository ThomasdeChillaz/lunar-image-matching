"""Native-pixel SIFT/PnP baseline; no planar-affine prefilter on 3D matches."""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from dataset import sha256
from localization_diagnostics_v3 import map_points,point_metrics
from localization_robust_pose_v4 import CONFIGS,estimate
from lunar_geometry_v2 import correspondences,pose_error
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle


def points(a,b,clahe=False):
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('Equal grayscale images required')
    if min(a.std(),b.std())<1:return np.empty((0,2)),np.empty((0,2))
    if clahe:
        transform=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8));a,b=transform.apply(a),transform.apply(b)
    detector=cv2.SIFT_create(nfeatures=2000)
    ka,da=detector.detectAndCompute(a,None);kb,db=detector.detectAndCompute(b,None)
    if da is None or db is None or min(len(da),len(db))<2:return np.empty((0,2)),np.empty((0,2))
    raw=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
    good=sorted([pair[0] for pair in raw if len(pair)==2 and pair[0].distance<.75*pair[1].distance],key=lambda m:m.distance)
    # SIFT can assign several orientations to the same image location. Distinct
    # keypoint IDs must not multiply a single 2D/3D observation's pose support.
    unique=[];used_a=set();used_b=set()
    for m in good:
        pa,pb=tuple(ka[m.queryIdx].pt),tuple(kb[m.trainIdx].pt)
        if pa in used_a or pb in used_b:continue
        unique.append(m);used_a.add(pa);used_b.add(pb)
    good=unique
    return np.asarray([ka[m.queryIdx].pt for m in good]).reshape(-1,2),np.asarray([kb[m.trainIdx].pt for m in good]).reshape(-1,2)


def run(output):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);cv2.setNumThreads(1)
    manifest,tiles,data=load_bundle('reports/localization_training_data_v3.json')
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify')
    protocol=dict(scope='Development-only native512 SIFT/PnP comparison, not historical128 affine benchmark.',
        source_sha256=sha256(__file__),pose_source_sha256=sha256('localization_robust_pose_v4.py'),
        manifest_sha256=sha256('reports/localization_training_data_v3.json'),nfeatures=2000,ratio=.75,
        clahe_profiles=[False,True],pose_config=config,selection_bank=manifest['selection_bank'],
        deduplication='Unique native xy coordinates in both images; orientation IDs do not multiply pose support.',
        calibration_or_test_pixels_loaded=0,preprocessing='No depth mask or planar geometry prefilter; all finite grayscale pixels.')
    write_new(root/'protocol.json',protocol);rows=[]
    for c in [c for c in manifest['cases'] if c['split']=='model_selection']:
        t=tiles[c['tile_id']];z=data[c['tile_id']];a,b=t['cameras']
        for use_clahe in (False,True):
            tick=perf_counter();p,q=points(z['images'][2*c['light_a']],z['images'][2*c['light_b']+1],use_clahe)
            world,valid=map_points(p,z['depth_a'],a['intrinsics'],a['cam2world']);pose,info=estimate(world,q[valid],b['intrinsics'],config)
            elapsed=(perf_counter()-tick)*1000
            err=None if pose is None else pose_error(pose,np.asarray(b['cam2world']))
            _,target,_,visible=correspondences(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
            depth=z['depth_a'];distance=float(np.median(depth[np.isfinite(depth)&(depth>0)]))
            rows.append(dict(c,method='sift_clahe' if use_clahe else 'sift_native',**point_metrics(q,target,visible),
                pose_correct=bool(err and err['rotation_deg']<=2 and err['translation_m']<=.01*distance),
                strict_pose_correct=bool(err and err['rotation_deg']<=.5 and err['translation_m']<=.001*distance),
                pose=None if pose is None else pose.tolist(),pose_error=err,latency_ms=elapsed,reference_range_m=distance,
                reference_points_native=p.tolist(),query_points_native=q.tolist(),target_points_native=target.tolist(),visible=visible.tolist(),inlier_info=info))
    bank=set(manifest['selection_bank']);summary={}
    for method in ('sift_native','sift_clahe'):
        for stage in ('selection_bank','expanded_development'):
            part=[r for r in rows if r['method']==method and (stage=='expanded_development' or r['pair_id'] in bank)]
            summary[method+'/'+stage]=dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),strict=sum(r['strict_pose_correct'] for r in part),
                returned=sum(r['returned'] for r in part),median_total_ms=float(np.median([r['latency_ms'] for r in part])))
    winner=max(('sift_native','sift_clahe'),key=lambda k:(summary[k+'/selection_bank']['correct'],k=='sift_native'))
    with gzip.open(root/'results.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
    result=dict(scope=protocol['scope'],results=summary,selected_profile=winner,results_sha256=sha256(root/'results.json.gz'))
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
