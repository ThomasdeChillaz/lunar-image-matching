"""Independent saved-output replay for the coarse/local-oracle diagnostic.

No neural network is loaded. Grid lookup, rotations and window checks are
implemented separately; oracle poses replay the already fixed common estimator.
This does not prove a ground-truth-assisted oracle is an achievable matcher.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def independent_measure(p,q,target,visible,k):
    p=np.asarray(p,float).reshape(-1,2);q=np.asarray(q,float).reshape(-1,2)
    target=np.asarray(target,float);visible=np.asarray(visible,bool)
    if p.shape!=q.shape or target.shape!=(676,2)or visible.shape!=(676,):
        raise ValueError('Independent replay dimensions differ')
    if not np.isfinite(q).all()or not np.isfinite(target[visible]).all()or((q<0)|(q>511)).any():
        raise ValueError('Invalid replay query/visible target coordinates')
    grid={(49.5+16*x,49.5+16*y):26*y+x for y in range(26)for x in range(26)}
    try:index=np.asarray([grid[tuple(point)]for point in p],int)
    except KeyError:raise ValueError('Reference points do not belong to the exact canonical grid')from None
    if len(np.unique(index))!=len(index):raise ValueError('Duplicate replay reference points')
    matrices=[[[1,0,0],[0,1,0]],[[0,1,0],[-1,0,511]],
              [[-1,0,511],[0,-1,511]],[[0,-1,511],[1,0,0]]]
    if k not in (0,1,2,3):raise ValueError('Unknown rotation branch')
    truth=np.column_stack([target[index],np.ones(len(index))])@np.asarray(matrices[k]).T
    mask=visible[index];center=np.rint(q)
    support=(p[:,0]>=12)&(p[:,0]<500)&(p[:,1]>=12)&(p[:,1]<500)
    support&=(center[:,0]>=28)&(center[:,0]<484)&(center[:,1]>=28)&(center[:,1]<484)
    reach=mask&support&(truth[:,0]>=center[:,0]-16)&(truth[:,0]<=center[:,0]+16)
    reach&=(truth[:,1]>=center[:,1]-16)&(truth[:,1]<=center[:,1]+16)
    error=np.sqrt(np.square(q-truth).sum(1))
    return dict(source_indices=index.tolist(),visible_proposals=mask.tolist(),
        reachable=reach.tolist(),fine_support=support.tolist(),returned=len(p),
        visible_grid_points=int(visible.sum()),nonvisible_returned=int((~mask).sum()),
        correct_at_native_px={str(t):int(np.sum(mask&(error<=t)))for t in (1,2,4,8,12,16)},
        reachable_proposals=int(reach.sum()))


def check_measurement(row,target,visible):
    actual=independent_measure(row['reference_points_native'],row['query_points_rotated_native'],
        target,visible,row['quarter_turn'])
    if actual!=row['measurement']:raise ValueError('Saved coarse/window measurement differs')
    return actual


def audit(run,output):
    import cv2
    from native_baseline_pose_v7 import propose_pose
    from lunar_geometry_v5 import median_reference_range_z
    from north_training_bundle_v38 import load_bundle
    out=Path(output);root=Path(run)
    if out.exists():raise FileExistsError('Preserve each independent replay')
    cv2.setNumThreads(1)
    summary=read(root/'summary.json');protocol=read(root/'protocol.json')
    if not summary['complete']or summary['case_rows']!=192 or summary['proposal_branch_rows']!=768:
        raise ValueError('Complete two-model/all96case diagnostic required')
    if summary['controls']!=read(root/'controls.json')or not summary['controls']['frozen']['passed']:
        raise ValueError('Actual frozen diagnostic controls missing')
    for name,digest in read(root/'COMPLETE.json')['sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Sealed diagnostic changed: '+name)
    for name,digest in protocol['source_sha256'].items():
        if sha(name)!=digest or subprocess.check_output(['git','show',protocol['publication_commit']+':'+name])!=Path(name).read_bytes():
            raise ValueError('Executed/public diagnostic source changed: '+name)
    if summary['protocol_sha256']!=sha(root/'protocol.json')or summary['results_sha256']!=sha(root/'per_case.json.gz'):
        raise ValueError('Diagnostic summary seal differs')
    manifest='reports/north_training_data_v38.json'
    if protocol['manifest_sha256']!=sha(manifest):raise ValueError('Input manifest changed')
    endpoint=Path('runs/north-v39-expanded-control/coarse_adaptation-20261101/final.pt')
    if protocol['endpoint_checkpoint_sha256']!=sha(endpoint):raise ValueError('Executed endpoint changed')
    m,tiles,data=load_bundle(manifest,selection_only=True)
    if protocol['case_ids']!=[c['pair_id']for c in m['cases']]:raise ValueError('Frozen case list changed')
    with gzip.open(root/'per_case.json.gz','rt')as f:case_rows=json.load(f)
    index={(r['mode'],r['pair_id']):r for r in case_rows}
    if len(index)!=192 or len(case_rows)!=192:raise ValueError('Duplicate or missing case summary')
    if len(list((root/'cases').glob('*.json')))!=768:raise ValueError('Unexpected branch cache denominator')
    branch_count=oracle_correct=0;aggregate=[]
    for mode in ('frozen','primary_endpoint'):
        for case in m['cases']:
            z=data[case['tile_id']];tile=tiles[case['tile_id']];a,b=tile['cameras']
            reference_range=median_reference_range_z(z['depth_a'],a['intrinsics'])
            reached=set();offered=set();counts=[];pose_correct=[]
            for k in range(4):
                key=hashlib.sha256((mode+'/'+case['pair_id']+'/'+str(k)).encode()).hexdigest()
                row=read(root/'cases'/(key+'.json'))
                if row['mode']!=mode or row['quarter_turn']!=k or any(row[n]!=v for n,v in case.items()):
                    raise ValueError('Replay branch identity differs')
                measured=check_measurement(row,z['forward_target'],z['forward_visible'])
                ids=np.asarray(measured['source_indices'],int);keep=np.asarray(measured['reachable'],bool)
                p=np.asarray(row['reference_points_native'],float).reshape(-1,2)[keep]
                q=z['forward_target'][ids[keep]]
                pose,info=propose_pose(p,q,z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'])
                old=row['oracle_local_pose'];saved=old['estimated_cam2world']
                if (pose is None)!=(saved is None)or(pose is not None and not np.allclose(pose,saved,rtol=0,atol=1e-7)):
                    raise ValueError('Common-estimator oracle pose replay differs')
                angle=translation=None;correct=False
                if pose is not None:
                    truth=np.asarray(b['cam2world']);translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]))
                    angle=float(np.rad2deg(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
                    correct=bool(angle<=2 and translation<=reference_range*.01)
                if (old['supplied_matches']!=len(p)or old['pose_info']!=info or old['correct']!=correct
                        or not old['query_truth_used']):raise ValueError('Oracle metadata differs')
                for name,value in [('rotation_error_deg',angle),('translation_error_m',translation)]:
                    previous=old[name]
                    if (previous is None)!=(value is None)or(value is not None and not np.isclose(value,previous,rtol=0,atol=1e-8)):
                        raise ValueError('Oracle error replay differs')
                reached.update(i for i,v in zip(ids,keep)if v)
                offered.update(i for i,v in zip(ids,measured['visible_proposals'])if v)
                counts.append(measured['reachable_proposals']);pose_correct.append(correct);branch_count+=1
            actual=index[(mode,case['pair_id'])]
            if any(actual[n]!=v for n,v in case.items()):raise ValueError('Case-summary grouping differs')
            expected=dict(visible_grid_points=int(z['forward_visible'].sum()),visible_source_points_offered=len(offered),
                oracle_source_points_reachable_in_any_branch=len(reached),maximum_reachable_in_one_branch=max(counts),
                branches_with_at_least_six_reachable=sum(n>=6 for n in counts),
                oracle_reachable_fraction=None if not z['forward_visible'].any()else len(reached)/int(z['forward_visible'].sum()),
                actual_pose_success_claim=False,oracle_any_branch_correct_pose=any(pose_correct),
                oracle_branch_correct=pose_correct,oracle_query_truth_used=True)
            if any(actual[n]!=v for n,v in expected.items()):raise ValueError('Oracle/point case aggregate differs')
            aggregate.append(actual);oracle_correct+=any(pose_correct)
    result=dict(passed=True,independent_grid_rotation_window_replay=True,replayed_branch_rows=branch_count,
        replayed_case_rows=len(aggregate),replayed_oracle_pose_controls=branch_count,
        oracle_any_branch_correct_cases=oracle_correct,ground_truth_used_by_oracle=True,
        neural_forward_repeated=False,optimizer_run=False,calibration_or_test_pixels_loaded=0,
        source_sha256=sha(__file__),summary_sha256=sha(root/'summary.json'),protocol_sha256=sha(root/'protocol.json'),
        scope='Independent saved coarse-grid/window/aggregate replay plus common oracle-estimator replay. Oracle success is not model accuracy, a usable branch selector or a strict mathematical pose-success bound.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/north-v43-coarse-reachability')
    p.add_argument('--output',default='reports/north_coarse_reachability_audit_v43.json')
    a=p.parse_args();audit(a.run,a.output)
