"""Frozen all-case development ablations; query truth is passed to scoring only.

This runner produces RAW uncalibrated research measurements. No final reserve
or acceptance fitting is reachable. A separate published protocol admits cost
and case lists; partial runs/timeouts are preserved and explicitly incomplete.
"""
import hashlib,itertools,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
from solar_frontend_v63 import ReferenceGeometry
from localize_terrain import load_matcher,public_result,jsonable
from verify_north_camera_v59 import pose_errors
from lunar_geometry_v5 import median_reference_range_z
from solar_compute_budget_v67 import reserve

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def variants():
    return [dict(name=f'A{int(a)}B{int(b)}C{int(c)}',solar_mode='shared'if a else'off',response=b,active=c,pooled=False,budget=128,policy='active')for a,b,c in itertools.product((False,True),repeat=3)]


def score(result,case,geometry):
    pose=result.get('pose');positive=case['positive'];error=None
    if positive:error=pose_errors(pose,np.asarray(case['scoring_query_cam2world'],float))
    reference_range=median_reference_range_z(geometry.depth,geometry.intrinsics)
    correct=bool(positive and error is not None and error['rotation_deg']<=2 and error['translation_m']<=.01*reference_range)
    strict=bool(positive and error is not None and error['rotation_deg']<=.5 and error['translation_m']<=.001*reference_range)
    p=np.asarray(result.get('reference_xy',[])).reshape(-1,2);q=np.asarray(result.get('query_xy',[])).reshape(-1,2);precision={}
    if positive:
        with np.load(case['scoring_correspondences'],allow_pickle=False)as z:
            truth,visible,anchors=[z[k]for k in('query_xy','visible','reference_xy')]
        for threshold in(1,2,4,8,12):
            good=0;co_visible=0
            for a,b in zip(p,q):
                distances=np.linalg.norm(anchors-a,axis=1);index=int(distances.argmin())
                if distances[index]>.001:raise ValueError('Saved match has no independent native reference anchor')
                if visible[index]:co_visible+=1;good+=np.linalg.norm(b-truth[index])<=threshold
            precision[str(threshold)]=dict(correct=int(good),co_visible=co_visible,returned=len(p))
    return dict(correct=correct,strict=strict,pose_error=error,pose_returned=pose is not None,wrong_positive=bool(positive and pose is not None and not correct),raw_nonoverlap=bool(not positive and pose is not None),reference_range_m=reference_range,correspondence=precision)


def run(protocol_path,publication_commit):
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(1)
    protocol=json.loads(Path(protocol_path).read_text());tick=time.monotonic();root=Path(protocol['output'])
    if protocol['split']!='model_selection'or protocol['calibration_or_final_access']or not protocol['compute_admitted']:raise ValueError('Published, budget-admitted development-only protocol required')
    if protocol['max_calendar_seconds']<=0:raise ValueError('Original fixed wall allowance required')
    if subprocess.check_output(['git','show',publication_commit+':'+protocol_path])!=Path(protocol_path).read_bytes():raise ValueError('Protocol was not source-published')
    for n,h in protocol['source_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',publication_commit+':'+n])!=Path(n).read_bytes():raise ValueError('Executed source differs from public freeze:'+n)
    if root.exists():raise FileExistsError('Preserve prior experiments; explicit resume needs original fixed deadline')
    reservation=reserve(protocol['compute_ledger'],protocol['compute_ledger_sha256'],protocol['reservation_journal'],sha(protocol_path)+'/experiment',protocol['budget_category'],protocol['max_calendar_seconds'])
    root.mkdir(parents=True);rows=[]
    try:
        for case in protocol['cases']:
            if case['split']!='model_selection':raise ValueError('Selection-only cases required')
            for n,h in case['input_sha256'].items():
                if sha(n)!=h:raise ValueError('Frozen image/geometry/scoring input changed:'+n)
            images=[]
            for n in('reference','query'):
                with Image.open(case[n])as im:images.append(np.asarray(im.convert('L')).copy())
            geometry=ReferenceGeometry.load(case['reference_geometry']);k=np.asarray(case['query_intrinsics'],float)
            for profile in protocol['profiles']:
                if time.monotonic()-tick>=protocol['max_calendar_seconds']:raise TimeoutError('Original ablation execution allowance exhausted')
                config=profile['configuration'];matcher=load_matcher(profile['checkpoint'],config)
                known=None if profile['track']=='estimated_sun'else np.asarray(case['secondary_known_query_sun_world'],float)
                result=matcher.localize(*images,geometry,k,solar_mode=config['solar_mode'],response=config['response'],active=config['active'],budget=config['budget'],policy=config.get('policy','active'),known_sun=known)
                metrics=score(result,case,geometry)
                row=dict(case_id=case['case_id'],component=case['component'],query_component=case.get('query_component',case['component']),positive=case['positive'],profile=profile['name'],track=profile['track'],metrics=metrics,elapsed_seconds=result['elapsed_seconds'])
                path=root/(hashlib.sha256((case['case_id']+'/'+profile['name']+'/'+profile['track']).encode()).hexdigest()+'.json')
                with path.open('x')as f:json.dump(dict(row=row,result=public_result(result,True)),f,indent=2,allow_nan=False)
                rows.append(row);print(json.dumps(row),flush=True)
        summary=dict(complete=True,compute_reservation=reservation,rows=rows,protocol_sha256=sha(protocol_path),publication_commit=publication_commit,calendar_seconds=time.monotonic()-tick,scope='Raw development pose and correspondence ablations. No calibration, accepted poses, final performance or scientific superiority inferred automatically.')
        with(root/'summary.json').open('x')as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
        return summary
    except BaseException as exc:
        with(root/'FAILED.json').open('x')as f:json.dump(dict(error_type=type(exc).__name__,error=str(exc),completed_rows=len(rows),elapsed_seconds=time.monotonic()-tick),f,indent=2)
        raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True);a=p.parse_args();run(a.protocol,a.publication_commit)
