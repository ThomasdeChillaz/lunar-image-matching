"""Operational fixed-range execution using unchanged v7 prediction/scoring."""
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
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import propose_pose,score,controls


def run(method,source,profile,output,max_seconds,case_start,case_count):
    import torch
    from compact_lunar import CompactMatcher
    from coarse_pipeline_v5 import infer
    from modern_baselines_v2 import ModernMatcher
    root=Path(output)
    if root.exists():raise FileExistsError('Keep every completed or partial baseline run')
    root.mkdir(parents=True);(root/'cases').mkdir();torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest_path=Path('reports/localization_training_data_v5.json');manifest=json.loads(manifest_path.read_text())
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'}
    all_cases=[c for c in manifest['cases'] if c['split']=='model_selection']
    if case_start<0 or case_count<1 or case_start+case_count>len(all_cases):raise ValueError('Invalid fixed case range')
    cases=all_cases[case_start:case_start+case_count];data={}
    for key,t in tiles.items():
        if sha256(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Native512px input changed')
        with np.load(t['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in ('images','depth_a','depth_b')}
    sources=['native_baseline_pose_v8.py','native_baseline_pose_v7.py','modern_baselines_v2.py','coarse_pipeline_v5.py','compact_lunar.py','contrastive_terrain.py',
        'localization_diagnostics_v3.py','lunar_geometry_v5.py','lunar_geometry_v2.py','localization_robust_pose_v4.py','localization_pose_refit_v3.py','rotation_bank_pose_v4.py']
    protocol=dict(scope='Original512px exposed development pose comparison; no final, calibrated acceptance or SOTA claim.',
        method=method,profile=profile,cpu_threads=2,case_ids=[c['pair_id'] for c in cases],
        case_start=case_start,case_count=case_count,all_case_ids=[c['pair_id'] for c in all_cases],
        execution_change='Fixed case-range execution only; v7 control, scoring, matcher, native coordinates, masks and final pose estimator are unchanged.',
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
    for local_i,c in enumerate(cases):
        i=case_start+local_i
        if perf_counter()-started>max_seconds:
            write_new(root/'TIMEOUT.json',dict(completed_pairs=local_i,eligible_pairs=len(cases),comparison_seconds=perf_counter()-started));return
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
            row=dict(c,global_case_index=i,method=which,elapsed_ms=elapsed,pose_info=info,inference_error=failure,**score(p,q,pose,z,tiles[c['tile_id']]['cameras']))
            rows.append(row);name=__import__('hashlib').sha256((c['pair_id']+which).encode()).hexdigest()[:20]
            write_new(root/'cases'/(name+'.json'),row)
        print(method,profile,'range pair',local_i+1,'/',len(cases),'correct',[r['pose_correct'] for r in rows[-2:]],flush=True)
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
    p.add_argument('--case-start',type=int,required=True);p.add_argument('--case-count',type=int,required=True)
    a=p.parse_args();run(a.method,a.source,a.profile,a.output,a.max_seconds,a.case_start,a.case_count)
