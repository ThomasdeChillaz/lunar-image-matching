"""Pinned modern matcher vs compact control on original512px development images."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import cv2
import numpy as np

from dataset import sha256
from localization_diagnostics_v3 import point_metrics
from localization_robust_pose_v4 import estimate,CONFIGS
from lunar_geometry_v5 import map_points_z,correspondences_z,median_reference_range_z
from lunargeo_dataset import write_new


def propose_pose(p,q,reference_depth,reference_intrinsics,reference_pose,query_intrinsics):
    world,use=map_points_z(p,reference_depth,reference_intrinsics,reference_pose)
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify')
    return estimate(world,np.asarray(q).reshape(-1,2)[use],query_intrinsics,config)


def score(p,q,pose,z,cameras):
    a,b=cameras
    _,target,_,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
    distance=median_reference_range_z(z['depth_a'],a['intrinsics']);error=None;correct=strict=False
    if pose is not None:
        truth=np.array(b['cam2world']);angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
        translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]));error=dict(rotation_deg=angle,translation_m=translation)
        correct=angle<=2 and translation<=.01*distance;strict=angle<=.5 and translation<=.001*distance
    return dict(**point_metrics(q,target,visible),reference_points_native=np.asarray(p).reshape(-1,2).tolist(),query_points_native=np.asarray(q).reshape(-1,2).tolist(),
        target_points_native=target.tolist(),visible=visible.tolist(),estimated_cam2world=None if pose is None else pose.tolist(),
        pose_error=error,reference_range_m=distance,pose_correct=bool(correct),strict_pose_correct=bool(strict),acceptance_policy=None)


def controls(matcher):
    from synthetic import texture
    image=texture(20261101,512);truth=np.array([[1.,0,8.],[0,1.,-4.]])
    query=cv2.warpAffine(image,truth,(512,512));matcher.points(image,query);started=perf_counter()
    p,q=matcher.points(image,query);matcher.sync();elapsed=perf_counter()-started
    target=np.asarray(p).reshape(-1,2)+[8,-4];error=np.linalg.norm(np.asarray(q).reshape(-1,2)-target,axis=1)
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    pose,info=propose_pose(p,q,np.full(image.shape,10.),k,np.eye(4),k)
    pose_error=None if pose is None else float(np.linalg.norm(pose[:3,3]-np.array([-.2,.1,0])))
    rotation=None if pose is None else float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1,1))))
    blank=matcher.points(np.zeros_like(image),np.zeros_like(image));invalid=matcher.points(image,query,np.zeros_like(image,bool),np.zeros_like(query,bool))
    passed=len(error)>=6 and float(np.median(error))<=1 and pose_error is not None and pose_error<=.05 and rotation<=.5 and len(blank[0])==0 and len(invalid[0])==0
    return dict(passed=bool(passed),scope='Actual native512px known-transform and invalid-input software controls.',matches=len(error),
        median_native_error_px=None if not len(error) else float(np.median(error)),translation_error_m=pose_error,rotation_error_deg=rotation,
        warm_match_seconds=elapsed,blank_empty=len(blank[0])==0,invalid_mask_empty=len(invalid[0])==0,inliers=info['inliers'],
        reference_points_native=np.asarray(p).tolist(),query_points_native=np.asarray(q).tolist(),
        estimated_cam2world=None if pose is None else pose.tolist(),known_translation_native=[8.,-4.],image_seed=20261101)


def run(method,source,profile,output,max_seconds):
    import torch
    from compact_lunar import CompactMatcher
    from coarse_pipeline_v5 import infer
    from modern_baselines_v2 import ModernMatcher
    root=Path(output)
    if root.exists():raise FileExistsError('Keep every completed or partial baseline run')
    root.mkdir(parents=True);(root/'cases').mkdir();torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest_path=Path('reports/localization_training_data_v5.json');manifest=json.loads(manifest_path.read_text())
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'}
    cases=[c for c in manifest['cases'] if c['split']=='model_selection'];data={}
    for key,t in tiles.items():
        if sha256(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Native512px input changed')
        with np.load(t['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in ('images','depth_a','depth_b')}
    sources=['native_baseline_pose_v7.py','modern_baselines_v2.py','coarse_pipeline_v5.py','compact_lunar.py','contrastive_terrain.py',
        'localization_diagnostics_v3.py','lunar_geometry_v5.py','lunar_geometry_v2.py','localization_robust_pose_v4.py','localization_pose_refit_v3.py','rotation_bank_pose_v4.py']
    protocol=dict(scope='Original512px exposed development pose comparison; no final, calibrated acceptance or SOTA claim.',
        method=method,profile=profile,cpu_threads=2,case_ids=[c['pair_id'] for c in cases],
        manifest_sha256=sha256(manifest_path),source_sha256={p:sha256(p) for p in sources},
        input_sha256={t['native_cache']:t['native_cache_sha256'] for t in tiles.values()},
        pose_estimator='Same reference geometry and AP3P/RANSAC2000,3px,.99,LM,reclassification for both methods.',
        query_geometry='Query depth/true pose used only after inference for scoring.',max_comparison_seconds=max_seconds,
        started_utc=datetime.now(timezone.utc).isoformat(),public_execution_id=os.environ.get('GITHUB_RUN_ID'),source_commit=os.environ.get('GITHUB_SHA'),
        runtime=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,platform=platform.platform()))
    write_new(root/'protocol.json',protocol)
    matcher=ModernMatcher(method,source,profile,'cpu');write_new(root/'provenance.json',matcher.provenance)
    control=controls(matcher);write_new(root/'controls.json',control)
    if not control['passed']:raise ValueError('Actual native pretrained control failed; no lunar results inferred')
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval()
    from synthetic import texture
    warm=texture(733,512);k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    infer(coarse,coarse.model,warm,np.rot90(warm,1).copy(),np.full(warm.shape,10.),k,np.eye(4),k)
    rows=[];started=perf_counter()
    for i,c in enumerate(cases):
        if perf_counter()-started>max_seconds:
            write_new(root/'TIMEOUT.json',dict(completed_pairs=i,eligible_pairs=len(cases),comparison_seconds=perf_counter()-started));return
        z=data[c['tile_id']];a,b=tiles[c['tile_id']]['cameras'];ref,query=z['images'][2*c['light_a']],z['images'][2*c['light_b']+1]
        if ref.shape!=(512,512) or query.shape!=(512,512):raise ValueError('Native source content differs')
        for which in ([method,'compact_orientation'] if i%2==0 else ['compact_orientation',method]):
            tick=perf_counter();failure=None
            try:
                if which==method:
                    p,q=matcher.points(ref,query);pose,info=propose_pose(p,q,z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics']);matcher.sync()
                else:
                    selected,attempts=infer(coarse,coarse.model,ref,query,z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'])
                    branch=selected if selected is not None else attempts[0];p,q=branch['p'],branch['q'];pose=None if selected is None else branch['pose'];info=branch['info']
            except (RuntimeError,ValueError,cv2.error) as exc:
                p=q=np.empty((0,2));pose=None;info=dict(inliers=0,reason='inference_error');failure=dict(type=type(exc).__name__,message=str(exc))
            elapsed=(perf_counter()-tick)*1000
            # Query geometry first enters here, after prediction and timing.
            row=dict(c,method=which,elapsed_ms=elapsed,pose_info=info,inference_error=failure,**score(p,q,pose,z,tiles[c['tile_id']]['cameras']))
            rows.append(row);name=__import__('hashlib').sha256((c['pair_id']+which).encode()).hexdigest()[:20]
            write_new(root/'cases'/(name+'.json'),row)
        print(method,profile,'pair',i+1,'/',len(cases),'correct',[r['pose_correct'] for r in rows[-2:]],flush=True)
    results={m:dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),strict=sum(r['strict_pose_correct'] for r in part),
        estimated=sum(r['estimated_cam2world'] is not None for r in part),median_full_inference_ms=float(np.median([r['elapsed_ms'] for r in part])),
        inference_errors=sum(r['inference_error'] is not None for r in part)) for m in (method,'compact_orientation') if (part:=[r for r in rows if r['method']==m])}
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result=dict(scope=protocol['scope'],results=results,completed_comparison_seconds=perf_counter()-started,
        peak_joint_process_rss_bytes=int(rss if platform.system()=='Darwin' else rss*1024),
        memory_note='Peak includes both loaded models and software controls; not an isolated per-method memory measurement.',
        input_resolution=512,internal_pretrained_settings=matcher.provenance.get('settings'),calibration_or_test_pixels_loaded=0)
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['loma_r','romav2'],required=True);p.add_argument('--source',required=True)
    p.add_argument('--profile',choices=['common','author'],default='common');p.add_argument('--output',required=True);p.add_argument('--max-seconds',type=int,default=3600)
    a=p.parse_args();run(a.method,a.source,a.profile,a.output,a.max_seconds)
