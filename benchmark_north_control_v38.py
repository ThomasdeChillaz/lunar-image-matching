"""Frozen compact/surface and SIFT-first controls on the new common cases."""
import argparse
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np

from north_training_bundle_v38 import load_bundle,sha


def inference_inputs(bundle,case):
    """Returns only images, reference depth/pose and camera intrinsics."""
    _,tiles,data=bundle;a=tiles[case['tile_id']];b=tiles[case.get('query_tile_id',case['tile_id'])]
    za=data[a['tile_id']];zb=data[b['tile_id']]
    return (za['images'][2*case['light_a']],zb['images'][2*case['light_b']+1],za['depth_a'],
            a['cameras'][0]['intrinsics'],a['cameras'][0]['cam2world'],b['cameras'][1]['intrinsics'])


def score_case(bundle,case,p,q,pose):
    from native_baseline_pose_v7 import score
    _,tiles,data=bundle;ta=tiles[case['tile_id']];tb=tiles[case.get('query_tile_id',case['tile_id'])]
    a=ta['cameras'][0];b=dict(tb['cameras'][1])
    if not case['positive']:
        # Convert the remote query camera to the reference component's ENU
        # frame only for scoring; inference never receives this transform.
        ra,rq=np.asarray(ta['enu_rows']),np.asarray(tb['enu_rows'])
        transform=np.eye(4);transform[:3,:3]=ra@rq.T
        transform[:3,3]=ra@(np.asarray(tb['body_origin'])-np.asarray(ta['body_origin']))
        b['cam2world']=(transform@np.asarray(b['cam2world'])).tolist()
    geometry=dict(depth_a=data[ta['tile_id']]['depth_a'],depth_b=data[tb['tile_id']]['depth_b'])
    return score(p,q,pose,geometry,[a,b])


def case_list(manifest,full):
    if full:return manifest['cases']+manifest['negative_cases']
    # Fixed metadata-only coverage of every selection component and all four
    # lighting patterns; no runtime or outcome based bank selection.
    bank=[]
    for index,component in enumerate(manifest['selection_components']):
        candidates=[c for c in manifest['cases']if c['component']==component]
        case=candidates[index%len(candidates)];bank.append(case)
        bank.append(next(c for c in manifest['negative_cases']if c['pair_id'].startswith(case['pair_id']+'/nonoverlap-')))
    return bank


def totals(rows):
    positive=[r for r in rows if r['positive']];negative=[r for r in rows if not r['positive']]
    return dict(positive_cases=len(positive),correct_raw_poses=sum(r['pose_correct']for r in positive),
        strict_raw_poses=sum(r['strict_pose_correct']for r in positive),
        wrong_raw_poses=sum(r['estimated_cam2world']is not None and not r['pose_correct']for r in positive),
        negative_cases=len(negative),raw_nonoverlap_proposals=sum(r['estimated_cam2world']is not None for r in negative),
        median_measured_inference_ms=None if not rows else float(np.median([r['elapsed_ms']for r in rows])),
        calibrated_acceptance_evaluated=False)


def run(manifest_path,output,commit,full,max_seconds):
    import torch
    from compact_lunar import CompactMatcher
    from evaluate_geometry_warp_v11 import control,SOURCES as SURFACE_SOURCES
    from geometry_warp_native_v11 import infer
    from native_sift_pose_v4 import points as sift_points
    from native_baseline_pose_v7 import propose_pose,controls
    from train_search_window_v18 import configure
    sources=list(dict.fromkeys(SURFACE_SOURCES+['native_sift_pose_v4.py','benchmark_north_control_v38.py',
        'north_training_bundle_v38.py','north_development_plan_v35.py']))
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact public source freeze required')
    for p in sources:
        if subprocess.check_output(['git','show',commit+':'+p])!=Path(p).read_bytes():raise ValueError('Baseline source differs from public freeze: '+p)
    configure();bundle=load_bundle(manifest_path,selection_only=True);m=bundle[0];cases=case_list(m,full)
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve every baseline attempt')
    root.mkdir(parents=True);(root/'cases').mkdir()
    protocol=dict(scope='Common new northern model-selection controls; raw proposals only, no final/calibrated/SOTA claim.',
        source_sha256={p:sha(p)for p in sources},manifest_sha256=sha(manifest_path),publication_commit=commit,
        full=full,case_ids=[c['pair_id']for c in cases],cpu_threads=2,max_calendar_seconds=max_seconds,
        primary='Correct raw poses/all eligible positive cases; includes failed poses.',
        inference_geometry='Reference axial surface/pose and query intrinsics only. Query depth/pose enters score_case after timing.',
        timing='Both methods loaded/warmed, alternate order by case index; fresh native feature cache per call. Hybrid executes SIFT first, then compact surface only if SIFT pose is null.',
        input_size=512,started_utc=datetime.now(timezone.utc).isoformat())
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    coarse=CompactMatcher(m['coarse_model']);coarse.model.eval().requires_grad_(False)
    def controlled_sift(a,b,ma=None,mb=None):
        p,q=sift_points(a,b)
        keep=np.ones(len(p),bool)
        for points,mask,image in ((p,ma,a),(q,mb,b)):
            if mask is None:continue
            if np.asarray(mask).shape!=image.shape:raise ValueError('Native SIFT control mask shape differs')
            xy=np.rint(points).astype(int);xy=np.clip(xy,0,np.array(image.shape[::-1])-1)
            keep&=np.asarray(mask,dtype=bool)[xy[:,1],xy[:,0]]
        return p[keep],q[keep]
    checks=dict(compact=control(coarse),sift=controls(SimpleNamespace(points=controlled_sift,sync=lambda:None)))
    (root/'controls.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(c['passed']for c in checks.values()):raise ValueError('Actual baseline native control failed')
    rows=[];tick=perf_counter();start=datetime.now(timezone.utc)
    for i,case in enumerate(cases):
        elapsed=(datetime.now(timezone.utc)-start).total_seconds()
        if elapsed<0 or elapsed>max_seconds or perf_counter()-tick>max_seconds:
            (root/'TIMEOUT.json').write_text(json.dumps(dict(completed_cases=i,eligible_cases=len(cases),calendar_seconds=elapsed),indent=2)+'\n')
            return
        inputs=inference_inputs(bundle,case)
        order=['frozen_surface','sift_first_surface']if i%2==0 else['sift_first_surface','frozen_surface']
        for method in order:
            before=perf_counter();pose=None;p=q=np.empty((0,2));info={};used='compact_surface';error=None
            try:
                if method=='sift_first_surface':
                    p,q=sift_points(inputs[0],inputs[1]);pose,info=propose_pose(p,q,inputs[2],inputs[3],inputs[4],inputs[5])
                    if pose is not None:used='sift'
                if pose is None:
                    selected,branches=infer(coarse,*inputs)
                    branch=selected if selected is not None else branches[0]
                    p,q=branch['p'],branch['q'];pose=None if selected is None else branch['pose'];info=branch['info']
            except (RuntimeError,ValueError,cv2.error)as exc:
                p=q=np.empty((0,2));pose=None;error=dict(type=type(exc).__name__,message=str(exc))
            elapsed_ms=1000*(perf_counter()-before)
            row=dict(case,method=method,used=used,elapsed_ms=elapsed_ms,pose_info=info,inference_error=error,
                     **score_case(bundle,case,p,q,pose))
            rows.append(row);name=__import__('hashlib').sha256((case['pair_id']+'/'+method).encode()).hexdigest()[:24]
            (root/'cases'/(name+'.json')).write_text(json.dumps(row,allow_nan=False)+'\n')
        print('NORTHERN CONTROL CASE',i+1,'/',len(cases),[r['pose_correct']for r in rows[-2:]],flush=True)
    with gzip.open(root/'results.json.gz','wt')as f:json.dump(rows,f,allow_nan=False)
    summary=dict(complete=True,full=full,eligible_cases_per_method=len(cases),
        results={method:totals([r for r in rows if r['method']==method])for method in ('frozen_surface','sift_first_surface')},
        results_sha256=sha(root/'results.json.gz'),protocol_sha256=sha(root/'protocol.json'),
        scope=protocol['scope'],calibration_or_test_pixels_loaded=0)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='reports/north_training_data_v38.json')
    p.add_argument('--output',default='runs/north-v38-baseline-cost-bank');p.add_argument('--publication-commit',required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--max-seconds',type=int,default=900)
    a=p.parse_args();run(a.manifest,a.output,a.publication_commit,a.full,a.max_seconds)
