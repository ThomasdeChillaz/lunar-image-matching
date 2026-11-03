"""Frozen coarse-proposal availability on already-qualified OLD camera pilots.

Truth is used AFTER image-only proposal generation, only for diagnostic scores.
The truth-assisted pose diagnostic is not an image-only matcher result or a
mathematical upper bound. No training/selection/reserve imagery is accessed.
"""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
import cv2
from compact_lunar import load_checkpoint
from solar_frontend_v63 import candidates,reference_anchors
from rotation_bank_pose_v4 import quarter_turn
from native_baseline_pose_v7 import propose_pose
from lunar_geometry_v5 import median_reference_range_z
from verify_north_camera_v59 import pose_errors
from solar_compute_budget_v67 import reserve

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
SOURCES=['diagnose_solar_candidates_v69.py','solar_frontend_v63.py','solar_assignment_v63.py','solar_native_v63.py','solar_response_v63.py','solar_physics_v63.py','solar_active_v63.py','solar_compute_budget_v67.py','compact_lunar.py','contrastive_terrain.py','rotation_bank_pose_v4.py','native_baseline_pose_v7.py','lunar_geometry_v5.py','localization_robust_pose_v4.py','verify_north_camera_v59.py']


def validate_cases(cases,tiles):
    expected={(f'np31-pilot-{i}/{family}-0',a,b)for i in range(4)for family in('training_motion','unseen_roll')for a,b in((0,0),(0,3))}
    actual=[(r['tile_id'],r['light_a'],r['light_b'])for r in cases]
    if len(cases)!=16 or set(actual)!=expected or len({r['case_id']for r in cases})!=16:raise ValueError('Exact four-old-terrain/two-family/two-light-pair diagnostic required')
    for tile_id,a,b in actual:
        t=tiles[tile_id]
        if t['split']!='pilot_camera_control'or t['pair_index']!=0 or t['family']not in('training_motion','unseen_roll'):raise ValueError('Excluded old pilot camera support required')


def support(query_candidates,rotations,target,visible):
    q=np.asarray(query_candidates,float);r=np.asarray(rotations);truth=np.asarray(target,float);visible=np.asarray(visible,bool)
    if q.ndim!=3 or q.shape[-1]!=2 or r.shape!=q.shape[:2]or truth.shape!=(len(q),2)or visible.shape!=(len(q),):raise ValueError('Native candidate/rotation and independent scoring-label schema required')
    gap=np.full(q.shape[:2],np.inf);inside=np.zeros(q.shape[:2],bool)
    for k in range(4):
        ids=np.where(r==k)
        centre=np.rint(quarter_turn(q[ids],k,512,False));t=quarter_turn(truth[ids[0]],k,512,False)
        complete=((centre>=28)&(centre<=483)).all(-1)
        delta=np.abs(t-centre);inside[ids]=complete&(delta<=16).all(-1)
        gap[ids]=np.where(complete,np.linalg.norm(np.maximum(delta-16,0),axis=-1),np.inf)
    return dict(top1_inside=visible&inside[:,0],top8_inside=visible&inside.any(-1),top8_distance_to_support=np.min(gap,axis=1),visible=visible)


def run(protocol_path,commit):
    torch.set_num_threads(2);cv2.setNumThreads(1);p=read(protocol_path);root=Path(p['output']);tick=time.monotonic();start=datetime.now(timezone.utc);deadline=start+timedelta(seconds=p['max_calendar_seconds'])
    if root.exists():raise FileExistsError('Preserve each diagnostic invocation')
    if p['max_calendar_seconds']!=300 or p['cpu_threads']!=2 or p['training_or_reserve_access']or len(p['cases'])!=16:raise ValueError('Fixed old-pilot300second diagnostic protocol required')
    if subprocess.check_output(['git','show',commit+':'+protocol_path])!=Path(protocol_path).read_bytes():raise ValueError('Diagnostic protocol not source-published')
    for n in SOURCES:
        if p['source_sha256'][n]!=sha(n)or subprocess.check_output(['git','show',commit+':'+n])!=Path(n).read_bytes():raise ValueError('Diagnostic source differs from freeze:'+n)
    for n,h in p['input_sha256'].items():
        if sha(n)!=h:raise ValueError('Pinned diagnostic input changed:'+n)
    audit=read(p['geometry_audit'])
    if not audit['passed']or audit['oracle_success']<.95:raise ValueError('Already-passed exact geometry control required')
    manifest=read(p['manifest']);tiles={t['tile_id']:t for t in manifest['tiles']}
    validate_cases(p['cases'],tiles)
    if manifest['fitting_allowed']or manifest['calibration_or_final_pixels_loaded']:raise ValueError('Strict old pilot-only bank required')
    reservation=reserve(p['compute_ledger'],p['compute_ledger_sha256'],p['reservation_journal'],sha(protocol_path)+'/candidate_diagnostic','baseline',300)
    model,metadata=load_checkpoint(p['model']);model.eval().requires_grad_(False);root.mkdir();cache={};rows=[]
    execution=dict(publication_commit=commit,protocol_sha256=sha(protocol_path),started_utc=start.isoformat(),absolute_deadline_utc=deadline.isoformat(),reservation=reservation,query_truth_is_scoring_only=True)
    (root/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    try:
        for case in p['cases']:
            if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=300:raise TimeoutError('Original candidate diagnostic allowance exhausted')
            tile=tiles[case['tile_id']]
            if tile['split']!='pilot_camera_control'or tile['family']not in('training_motion','unseen_roll')or tile['pair_index']!=0:raise ValueError('Fixed excluded old-pilot support required')
            if tile['tile_id']not in audit['qualified_pair_ids']:raise ValueError('Unqualified camera pair forbidden')
            if case['tile_id']not in cache:
                if sha(tile['native_cache'])!=tile['native_cache_sha256']:raise ValueError('Audited pilot cache changed')
                with np.load(tile['native_cache'],allow_pickle=False)as z:cache[case['tile_id']]={n:z[n].copy()for n in z.files}
            data=cache[case['tile_id']]
            # Primary inference accepts ONLY the two image arrays here.
            reference=data['images'][2*case['light_a']];query=data['images'][2*case['light_b']+1]
            before=time.monotonic();a,q,appearance,rotation=candidates(model,reference,query);elapsed=time.monotonic()-before
            # Read true correspondences/visibility/pose only AFTER inference.
            yy,xx=np.mgrid[:26,:26];full=np.column_stack([49.5+16*xx.ravel(),49.5+16*yy.ravel()]);indices=np.array([np.flatnonzero(np.all(full==point,axis=1))[0]for point in a])
            truth=data['forward_target'][indices];visible=data['forward_visible'][indices];availability=support(q,rotation,truth,visible)
            reference_camera,query_camera=tile['cameras'];use=availability['top8_inside']
            oracle,info=propose_pose(a[use],truth[use],data['depth_a'],reference_camera['intrinsics'],reference_camera['cam2world'],query_camera['intrinsics'])
            error=pose_errors(oracle,np.asarray(query_camera['cam2world']));distance=median_reference_range_z(data['depth_a'],reference_camera['intrinsics'])
            row=dict(case_id=case['case_id'],component=tile['component'],light_a=case['light_a'],light_b=case['light_b'],anchors=len(a),visible_anchors=int(visible.sum()),top1_exact_support=int(availability['top1_inside'].sum()),top8_exact_support=int(use.sum()),image_only_proposal_seconds=elapsed,truth_assisted_cam2world=None if oracle is None else oracle.tolist(),truth_assisted_pose_error=error,truth_assisted_pose_recovered=bool(error is not None and error['rotation_deg']<=2 and error['translation_m']<=.01*distance),reference_range_m=distance,oracle_inliers=info['inliers'])
            filename='case_'+str(len(rows))+'.npz';np.savez_compressed(root/filename,reference_xy=a,query_candidates=q,quarter_turns=rotation,appearance=appearance,scoring_query_xy=truth,scoring_visible=visible,top8_exact_support=use)
            row['saved_arrays']=filename;row['saved_arrays_sha256']=sha(root/filename);rows.append(row);print(json.dumps(row),flush=True)
        if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=300:raise TimeoutError('Diagnostic completed after original allowance')
        result=dict(complete=True,rows=rows,protocol_sha256=sha(protocol_path),source_publication_commit=commit,calendar_seconds=time.monotonic()-tick,matcher_lunar_cases=len(rows),optimizer_updates=0,calibration_or_final_access=False,scope='Four OLD excluded camera pilot terrains, sixteen frozen image-only coarse proposal cases. Ground truth used only after inference for support/pose ORACLE diagnostics. Neither a new learned matcher result nor a true performance upper bound; final research reserve remains untouched.')
        with(root/'summary.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        return result
    except BaseException as exc:
        with(root/'FAILED.json').open('x')as f:json.dump(dict(error_type=type(exc).__name__,error=str(exc),completed_cases=len(rows),calendar_seconds=time.monotonic()-tick),f,indent=2)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True);a=p.parse_args();run(a.protocol,a.publication_commit)
