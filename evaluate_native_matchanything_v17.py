"""Source-frozen native MatchAnything versus SIFT-first surface on common poses."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_geometry_warp_v11 import SOURCES as SURFACE_SOURCES
from geometry_warp_native_v11 import infer
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import controls,propose_pose,score
from native_matchanything_v17 import NativeMatchAnything
from native_sift_pose_v4 import points as sift_points

SOURCES=list(dict.fromkeys(SURFACE_SOURCES+['evaluate_native_matchanything_v17.py',
    'native_matchanything_v17.py','matchanything_matcher.py','matchanything_model_lock.json',
    'eloftr_matcher.py','loftr_matcher.py','native_sift_pose_v4.py','synthetic.py']))


def run(output,dependency_dir,commit,profile='native512',bank=False,max_seconds=7200):
    root=Path(output)
    for source in SOURCES:
        if subprocess.check_output(['git','show',commit+':'+source])!=Path(source).read_bytes():
            raise ValueError('Comparison source differs from published freeze: '+source)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    manifest_path=Path('reports/localization_training_data_v5.json');manifest=json.loads(manifest_path.read_text())
    if manifest['schema_version']!=5 or manifest['calibration_or_test_pixels_loaded'] or not manifest['train_selection_disjoint']:
        raise ValueError('Audited corrected development-only manifest required')
    if sha256(Path(manifest['coarse_model'])/'selected.pt')!=manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen compact checkpoint changed')
    cases=[c for c in manifest['cases'] if c['split']=='model_selection' and (not bank or c['pair_id'] in manifest['selection_bank'])]
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'};data={}
    for tile_id,tile in tiles.items():
        if sha256(tile['native_cache'])!=tile['native_cache_sha256']:raise ValueError('Native input changed')
        with np.load(tile['native_cache'],allow_pickle=False) as z:data[tile_id]={k:z[k].copy() for k in ('images','depth_a','depth_b')}
    if root.exists():
        protocol=json.loads((root/'protocol.json').read_text())
        if (commit,profile,bank,max_seconds)!=(protocol['publication_commit'],protocol['profile'],protocol['bank_only'],protocol['wall_cap_seconds']):
            raise ValueError('Resume must preserve the full comparison protocol')
        for source,digest in protocol['source_sha256'].items():
            if sha256(source)!=digest:raise ValueError('Comparison source changed')
    else:
        root.mkdir(parents=True);(root/'cases').mkdir();(root/'source').mkdir()
        protocol=dict(scope='Original512px exposed-development pose comparison; no fresh final, acceptance or SOTA claim.',
            publication_commit=commit,source_sha256={p:sha256(p) for p in SOURCES},profile=profile,bank_only=bank,
            case_ids=[c['pair_id'] for c in cases],manifest_sha256=sha256(manifest_path),
            input_sha256={t['native_cache']:t['native_cache_sha256'] for t in tiles.values()},
            cpu_threads=2,wall_cap_seconds=max_seconds,methods=['matchanything_native','sift_first_surface'],
            pose='Common reference axial geometry, AP3P/RANSAC2000,3nativepx,.99,LM/reclassification. Query truth is scoring only.',
            compact='SIFT-native ratio.75/unique xy; return its pose if available, otherwise frozen two-step reference-surface refinement. Never route using correctness.',
            content='Same original512px grayscale pixels/field of view. No planar geometry prefilter for either matcher.',
            timing='Both loaded/warmed, alternating order per pair, no cross-pair compact CNN cache. Other CPU jobs must be terminal for an isolated timing claim.',
            resumed_timing='Resume preserves accuracy rows, but any resumed or concurrently loaded run is not automatically an isolated latency study.',
            environment=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__),
            calibration_or_test_pixels_loaded=0,started_utc=datetime.now(timezone.utc).isoformat())
        write_new(root/'protocol.json',protocol)
        for source in SOURCES:shutil.copy2(source,root/'source'/source)
    if sha256(manifest_path)!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    if (root/'summary.json').exists():return json.loads((root/'summary.json').read_text())
    matcher=NativeMatchAnything(dependency_dir=dependency_dir,profile=profile,threads=2)
    provenance=dict(matcher.provenance,total_parameters=sum(p.numel() for p in matcher.model.parameters()))
    if (root/'provenance.json').exists():
        if json.loads((root/'provenance.json').read_text())!=provenance:raise ValueError('Loaded baseline changed')
    else:write_new(root/'provenance.json',provenance)
    if not (root/'controls.json').exists():
        controlled=controls(matcher);write_new(root/'controls.json',controlled)
    if not json.loads((root/'controls.json').read_text())['passed']:raise ValueError('Actual pretrained control failed')
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval()
    from evaluate_geometry_warp_v11 import control
    # This warms native encoding/pose estimation even if every actual SIFT case succeeds.
    if not control(coarse)['passed']:raise ValueError('Actual frozen compact control failed')
    from synthetic import texture
    warm=texture(733,512);sift_points(warm,np.roll(warm,8,axis=1))
    methods=protocol['methods'];started=perf_counter();rows=[]
    def timeout(*args):raise TimeoutError('Baseline wall cap reached; keep complete cases and resume unchanged')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(max_seconds)
    try:
        for i,case in enumerate(cases):
            tile=tiles[case['tile_id']];z=data[case['tile_id']];a,b=tile['cameras']
            reference,query=z['images'][2*case['light_a']],z['images'][2*case['light_b']+1]
            for method in (methods if i%2==0 else list(reversed(methods))):
                name=__import__('hashlib').sha256((case['pair_id']+method).encode()).hexdigest()[:20]
                path=root/'cases'/(name+'.json')
                if path.exists():
                    row=json.loads(path.read_text())
                    if row['pair_id']!=case['pair_id'] or row['method']!=method:raise ValueError('Saved case identity changed')
                    rows.append(row);continue
                tick=perf_counter();route=method;failure=None
                try:
                    if method=='matchanything_native':
                        p,q=matcher.points(reference,query)
                        pose,info=propose_pose(p,q,z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'])
                    else:
                        p,q=sift_points(reference,query)
                        pose,info=propose_pose(p,q,z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'])
                        if pose is None:
                            route='surface_fallback'
                            selected,branches=infer(coarse,reference,query,z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'])
                            branch=selected if selected is not None else branches[0]
                            p,q,info=branch['p'],branch['q'],branch['info'];pose=None if selected is None else branch['pose']
                except (RuntimeError,cv2.error) as exc:
                    p=q=np.empty((0,2));pose=None;info=dict(inliers=0,reason='inference_error');failure=dict(type=type(exc).__name__,message=str(exc))
                elapsed=perf_counter()-tick
                row=dict(case,method=method,route=route,elapsed_seconds=elapsed,inference_error=failure,pose_info=info,
                    **score(p,q,pose,z,tile['cameras']))
                write_new(path,row);rows.append(row)
            print('native comparison',profile,i+1,'/',len(cases),[(r['method'],r['pose_correct']) for r in rows[-2:]],flush=True)
        result={m:dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),strict=sum(r['strict_pose_correct'] for r in part),
            wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in part),
            median_full_inference_seconds=float(np.median([r['elapsed_seconds'] for r in part])),
            inference_errors=sum(r['inference_error'] is not None for r in part)) for m in methods if (part:=[r for r in rows if r['method']==m])}
        for source,digest in protocol['source_sha256'].items():
            if sha256(source)!=digest:raise ValueError('Pinned baseline source changed')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary=dict(scope=protocol['scope'],results=result,elapsed_this_invocation_seconds=perf_counter()-started,
            protocol_sha256=sha256(root/'protocol.json'),case_sha256={p.name:sha256(p) for p in sorted((root/'cases').glob('*.json'))},
            peak_joint_process_rss_bytes=int(rss if platform.system()=='Darwin' else rss*1024),
            memory_note='Joint peak of both models and controls, not separate model memory.',calibration_or_test_pixels_loaded=0)
        write_new(root/'summary.json',summary);print(json.dumps(summary,indent=2));return summary
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--dependency-dir');parser.add_argument('--source-commit',required=True)
    parser.add_argument('--profile',choices=['native512','training_size832'],default='native512')
    parser.add_argument('--bank',action='store_true');parser.add_argument('--max-seconds',type=int,default=7200);args=parser.parse_args()
    if not 600<=args.max_seconds<=72*3600:raise ValueError('Declared bounded comparison budget required')
    run(args.output,args.dependency_dir,args.source_commit,args.profile,args.bank,args.max_seconds)
