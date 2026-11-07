"""Gated coarse-proposal diagnosis before another learned architecture change.

Prepared only. Run after the full three-seed cohort and its independent audit,
the queued repository regression, and publication of this source. This uses
image-only 128px proposal inference, then exact BVH labels solely for scoring.
An explicitly separate oracle replaces reachable matches with true query points
and rejects the rest. Its pose results are counterfactual controls, never model
accuracy. No optimizer is run.
"""
import argparse
from datetime import datetime,timedelta,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def write_once(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Existing diagnostic evidence differs: '+str(path))
        return
    temporary=path.with_name(path.name+'.pending')
    if temporary.exists():
        temporary.rename(temporary.with_name(temporary.name+'.interrupted-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def rotate(points,k):
    points=np.asarray(points,float).reshape(-1,2).copy()
    for _ in range(k%4):points=np.column_stack([points[:,1],511-points[:,0]])
    return points


def source_indices(points):
    """Exact half-pixel proposal-grid lookup, not nearest-label interpolation."""
    p=np.asarray(points,float).reshape(-1,2);grid=(p-49.5)/16
    rounded=np.rint(grid)
    if (not np.isfinite(p).all()or not np.allclose(grid,rounded,rtol=0,atol=1e-6)
            or (rounded<0).any()or (rounded>25).any()):
        raise ValueError('Reference proposal is outside the exact fractional label grid')
    index=rounded[:,1].astype(int)*26+rounded[:,0].astype(int)
    if len(np.unique(index))!=len(index):raise ValueError('Duplicate source proposals')
    return index


def measure(p_native,q_rotated_native,target_native,visible,quarter_turn):
    p=np.asarray(p_native,float).reshape(-1,2);q=np.asarray(q_rotated_native,float).reshape(-1,2)
    target=np.asarray(target_native,float);visible=np.asarray(visible,bool)
    if p.shape!=q.shape or target.shape!=(676,2)or visible.shape!=(676,):
        raise ValueError('Complete exact labels and paired native proposals required')
    if not np.isfinite(q).all()or not np.isfinite(target[visible]).all():
        raise ValueError('Nonfinite visible labels or proposed query coordinates')
    if ((q<0)|(q>511)).any():raise ValueError('Native query proposal outside image')
    index=source_indices(p);good=visible[index]
    truth=rotate(target[index],quarter_turn)
    error=np.linalg.norm(q-truth,axis=1)
    center=np.rint(q)
    supported=((p>=12)&(p<[500,500])).all(1)
    supported&=((center-16>=12)&(center+16<[500,500])).all(1)
    reachable=good&supported&(np.abs(truth-center)<=16).all(1)
    return dict(source_indices=index.tolist(),visible_proposals=good.tolist(),
        reachable=reachable.tolist(),fine_support=supported.tolist(),returned=len(p),
        visible_grid_points=int(visible.sum()),nonvisible_returned=int((~good).sum()),
        correct_at_native_px={str(t):int((good&(error<=t)).sum())for t in (1,2,4,8,12,16)},
        reachable_proposals=int(reachable.sum()))


def union_reachability(branches):
    if len(branches)!=4 or {b['quarter_turn']for b in branches}!={0,1,2,3}:
        raise ValueError('All four fixed orientation hypotheses required')
    totals={b['measurement']['visible_grid_points']for b in branches}
    if len(totals)!=1:raise ValueError('Branch ground-truth denominator differs')
    reached=set();offered=set()
    for b in branches:
        m=b['measurement']
        if len(m['source_indices'])!=len(m['reachable']):raise ValueError('Invalid saved masks')
        reached.update(i for i,keep in zip(m['source_indices'],m['reachable'])if keep)
        offered.update(i for i,keep in zip(m['source_indices'],m['visible_proposals'])if keep)
    denominator=totals.pop()
    return dict(visible_grid_points=denominator,visible_source_points_offered=len(offered),
        oracle_source_points_reachable_in_any_branch=len(reached),
        oracle_reachable_fraction=None if not denominator else len(reached)/denominator,
        maximum_reachable_in_one_branch=max(b['measurement']['reachable_proposals']for b in branches),
        branches_with_at_least_six_reachable=sum(b['measurement']['reachable_proposals']>=6 for b in branches),
        actual_pose_success_claim=False,
        interpretation='Ground-truth-scored union across four proposal branches, not a usable branch selector or pose-success bound.')


def oracle_inputs(p_native,measurement,target_native):
    """Perfect local33px refinement/rejection control, not model inference."""
    p=np.asarray(p_native,float).reshape(-1,2)
    index=np.asarray(measurement['source_indices'],int)
    keep=np.asarray(measurement['reachable'],bool)
    if index.shape!=(len(p),)or keep.shape!=(len(p),):raise ValueError('Oracle mask shape differs')
    return p[keep],np.asarray(target_native,float)[index[keep]]


def verify_ended(state_path):
    state=read(state_path)
    if state['state']!='complete':raise ValueError('Required upstream controller is incomplete')
    for field in ('pid','child_pid'):
        if state.get(field):
            active=subprocess.run(['ps','-p',str(state[field]),'-o','command='],text=True,capture_output=True).stdout.strip()
            if active:raise ValueError('Upstream PID remains active or reused; inspect before inference')


def run(output,commit):
    invocation_started=datetime.now(timezone.utc);invocation_tick=perf_counter()
    # Imports are lazy so pure geometry/denominator tests do not load a network.
    import cv2
    import torch
    from compact_lunar import CompactMatcher
    from north_training_bundle_v38 import load_bundle
    from search_window_learning_v18 import SearchWindowMatcher
    from synthetic import texture
    from train_north_control_v39 import SOURCES as FIT_SOURCES,MANIFEST
    from train_search_window_v18 import configure,make_model
    from native_baseline_pose_v7 import propose_pose
    from lunar_geometry_v5 import median_reference_range_z
    sources=list(dict.fromkeys(FIT_SOURCES+['north_coarse_reachability_v43.py','synthetic.py']))
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full public source revision required')
    for name in sources:
        if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():
            raise ValueError('Diagnostic source differs from public freeze: '+name)
    verify_ended('runs/localization-v40-recovery-training-state.json')
    verify_ended('runs/north-v41-regression-state.json')
    qa=read('reports/north_software_checks_v41.json')
    audit=read('reports/north_expanded_control_audit_v39.json')
    cohort=Path('runs/north-v39-expanded-control')
    if (not qa['passed']or not audit['passed']or audit['recorded_updates']!=6000
            or audit['replayed_rows']!=1728
            or audit['summary_sha256']!=sha(cohort/'summary.json')):
        raise ValueError('Complete three-seed saved-result audit and new full regression required')
    configure();bundle=load_bundle(MANIFEST,selection_only=True);manifest,tiles,data=bundle
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
    seed_folder=cohort/'coarse_adaptation-20261101'
    for name,digest in read(seed_folder/'COMPLETE.json')['sha256'].items():
        if sha(seed_folder/name)!=digest:raise ValueError('Audited endpoint evidence changed')
    endpoint=seed_folder/'final.pt'
    model=make_model(coarse,'coarse_adaptation')
    model.load_state_dict(torch.load(endpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    modes={'frozen':coarse,'primary_endpoint':SearchWindowMatcher(coarse,model)}
    root=Path(output);static=dict(publication_commit=commit,source_sha256={p:sha(p)for p in sources},
        manifest_sha256=sha(MANIFEST),endpoint_checkpoint_sha256=sha(endpoint),
        upstream_audit_sha256=sha('reports/north_expanded_control_audit_v39.json'),
        regression_sha256=sha('reports/north_software_checks_v41.json'),
        case_ids=[c['pair_id']for c in manifest['cases']],modes=list(modes),quarter_turns=[0,1,2,3],
        max_calendar_seconds=1800,cpu_threads=2,
        environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,
            torch_threads=torch.get_num_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        scope='All96positive development pairs; image-only coarse proposals scored afterward with exact fractional BVH labels. Separate perfect-local-refinement/rejection oracle uses ground truth and the unchanged common pose estimator. No optimizer, learned fine refinement, reserve or acceptance inference.')
    if len(manifest['cases'])!=96:raise ValueError('Full positive denominator required')
    if root.exists():
        protocol=read(root/'protocol.json')
        if any(protocol[k]!=v for k,v in static.items()):raise ValueError('Different diagnostic on resume')
        if(root/'COMPLETE.json').exists():
            for name,digest in read(root/'COMPLETE.json')['sha256'].items():
                if sha(root/name)!=digest:raise ValueError('Completed diagnostic changed')
            return read(root/'summary.json')
    else:
        root.mkdir(parents=True);(root/'cases').mkdir();now=invocation_started
        protocol=dict(static,created_utc=now.isoformat(),absolute_deadline_utc=(now+timedelta(seconds=1800)).isoformat())
        write_once(root/'protocol.json',protocol)
    tick=invocation_tick
    def clock():
        now=datetime.now(timezone.utc)
        if (now<datetime.fromisoformat(protocol['created_utc'])or now>=datetime.fromisoformat(protocol['absolute_deadline_utc'])
                or perf_counter()-tick>1800):raise TimeoutError('Original fixed diagnostic allowance exhausted')
    controls={};rows=[]
    for mode,matcher in modes.items():
        clock();image=texture(733,128);query=cv2.warpAffine(image,np.array([[1.,0.,8.],[0.,1.,-4.]]),(128,128))
        p,q=matcher.points(image,query);error=np.linalg.norm(q-(p+[8.,-4.]),axis=1)
        blank=matcher.points(np.zeros_like(image),np.zeros_like(image))
        invalid=matcher.points(image,query,np.zeros_like(image,bool),np.zeros_like(image,bool))
        controls[mode]=dict(matches=len(p),median_error_low_px=None if not len(p)else float(np.median(error)),
            blank_empty=len(blank[0])==0,invalid_empty=len(invalid[0])==0,
            passed=bool(len(p)>=6 and np.median(error)<=1 and len(blank[0])==len(invalid[0])==0))
        # An endpoint failing controls is still a recorded negative finding.
        # The frozen software control is the admission gate.
        if mode=='frozen'and not controls[mode]['passed']:
            write_once(root/'failed_frozen_controls.json',controls)
            raise ValueError('Actual frozen proposal known-transform/invalid controls failed')
        for case in manifest['cases']:
            clock();z=data[case['tile_id']];branches=[]
            tile=tiles[case['tile_id']];ref,query_camera=tile['cameras']
            reference_range=median_reference_range_z(z['depth_a'],ref['intrinsics'])
            a=z['images'][2*case['light_a']];b=z['images'][2*case['light_b']+1]
            for k in range(4):
                key=hashlib.sha256((mode+'/'+case['pair_id']+'/'+str(k)).encode()).hexdigest()
                path=root/'cases'/(key+'.json')
                if path.exists():
                    row=read(path)
                    if (row['mode']!=mode or row['quarter_turn']!=k
                            or any(row[key]!=value for key,value in case.items())):
                        raise ValueError('Cached proposal identity differs')
                else:
                    low=[cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA)for im in(a,np.rot90(b,k).copy())]
                    # Images only. Geometry/labels are first accessed below, after points().
                    p,q=matcher.points(*low);pn=(np.asarray(p,float)+.5)*4-.5;qn=(np.asarray(q,float)+.5)*4-.5
                    row=dict(case,mode=mode,quarter_turn=k,reference_points_native=pn.tolist(),
                        query_points_rotated_native=qn.tolist())
                measured=measure(row['reference_points_native'],row['query_points_rotated_native'],
                    z['forward_target'],z['forward_visible'],k)
                if 'measurement'in row and row['measurement']!=measured:raise ValueError('Cached proposal metric differs')
                row['measurement']=measured
                # Exact query points and perfect outlier rejection define a
                # counterfactual control, not a learned matching result.
                op,oq=oracle_inputs(row['reference_points_native'],measured,z['forward_target'])
                pose,info=propose_pose(op,oq,z['depth_a'],ref['intrinsics'],ref['cam2world'],query_camera['intrinsics'])
                angle=translation=None;correct=False
                if pose is not None:
                    truth=np.asarray(query_camera['cam2world'])
                    angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
                    translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]))
                    correct=angle<=2 and translation<=.01*reference_range
                oracle=dict(scope='Ground-truth-assisted perfect33pxlocalrefinement plus perfectoutlierrejection; not matcher accuracy.',
                    supplied_matches=len(op),estimated_cam2world=None if pose is None else pose.tolist(),
                    pose_info=info,rotation_error_deg=angle,translation_error_m=translation,
                    correct=bool(correct),query_truth_used=True)
                if 'oracle_local_pose'in row and row['oracle_local_pose']!=oracle:
                    raise ValueError('Cached oracle pose control differs')
                row['oracle_local_pose']=oracle
                write_once(path,row)
                branches.append(row)
            rows.append(dict(case,mode=mode,**union_reachability(branches),
                oracle_any_branch_correct_pose=any(r['oracle_local_pose']['correct']for r in branches),
                oracle_branch_correct=[r['oracle_local_pose']['correct']for r in branches],
                oracle_query_truth_used=True))
            print('COARSE DIAGNOSTIC',mode,case['pair_id'],rows[-1]['oracle_source_points_reachable_in_any_branch'],flush=True)
    clock()
    write_once(root/'controls.json',controls)
    if(root/'per_case.json.gz').exists():
        with gzip.open(root/'per_case.json.gz','rt')as f:
            if json.load(f)!=rows:raise ValueError('Existing case summary differs')
    else:
        temporary=root/'per_case.json.gz.pending'
        if temporary.exists():temporary.rename(root/('per_case-interrupted-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.gz.pending'))
        with gzip.open(temporary,'wt')as f:json.dump(rows,f,allow_nan=False)
        temporary.replace(root/'per_case.json.gz')
    result=dict(complete=True,case_rows=len(rows),proposal_branch_rows=len(rows)*4,protocol_sha256=sha(root/'protocol.json'),
        results_sha256=sha(root/'per_case.json.gz'),controls=controls,scope=static['scope'],
        finer_stage_failure_attribution_established=False,calibration_or_test_pixels_loaded=0)
    write_once(root/'summary.json',result)
    sealed=[p for p in root.rglob('*')if p.is_file()and(p.name.endswith('.json')or p.name=='per_case.json.gz')]
    write_once(root/'COMPLETE.json',dict(sha256={str(p.relative_to(root)):sha(p)for p in sealed}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/north-v43-coarse-reachability')
    p.add_argument('--publication-commit',required=True);a=p.parse_args();run(a.output,a.publication_commit)
