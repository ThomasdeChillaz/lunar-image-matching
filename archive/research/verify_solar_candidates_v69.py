"""Independent arithmetic replay of OLDpilot candidate-support/oracle metrics.

No matching model is imported or run. Native quarterturn mathematics is written
explicitly rather than calling the inference support helper.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def audit(run,protocol,output):
    root=Path(run);s=read(root/'summary.json');p=read(protocol);m=read(p['manifest']);tiles={t['tile_id']:t for t in m['tiles']};rows=[]
    expected={(f'np31-pilot-{i}/{family}-0',a,b)for i in range(4)for family in('training_motion','unseen_roll')for a,b in((0,0),(0,3))}
    actual=[(c['tile_id'],c['light_a'],c['light_b'])for c in p['cases']]
    if len(actual)!=16 or set(actual)!=expected or len({c['case_id']for c in p['cases']})!=16:raise ValueError('Independent exact16case/fourterrain diagnostic required')
    if not s['complete']or len(s['rows'])!=16 or s['protocol_sha256']!=sha(protocol):raise ValueError('Complete frozen16case diagnostic required')
    for record,case in zip(s['rows'],p['cases']):
        if record['case_id']!=case['case_id']:raise ValueError('Ordered complete case list changed')
        path=root/record['saved_arrays']
        if sha(path)!=record['saved_arrays_sha256']:raise ValueError('Saved model candidate array changed')
        with np.load(path,allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
        tile=tiles[case['tile_id']]
        if sha(tile['native_cache'])!=tile['native_cache_sha256']:raise ValueError('Independent geometric labels changed')
        yy,xx=np.mgrid[:26,:26];full=np.column_stack([49.5+16*xx.ravel(),49.5+16*yy.ravel()]);indices=np.array([np.flatnonzero(np.all(full==point,axis=1))[0]for point in a['reference_xy']])
        with np.load(tile['native_cache'],allow_pickle=False)as z:
            truth=z['forward_target'][indices];visible=z['forward_visible'][indices];depth=z['depth_a'].copy()
        if not np.array_equal(truth,a['scoring_query_xy'])or not np.array_equal(visible,a['scoring_visible']):raise ValueError('Scoring truth changed independently of saved model output')
        inside=np.zeros(a['quarter_turns'].shape,bool)
        for i in range(len(truth)):
            for c,r in enumerate(a['quarter_turns'][i]):
                def turn(v):
                    x,y=v
                    return np.array([(x,y),(y,511-x),(511-x,511-y),(511-y,x)][int(r)])
                centre=np.rint(turn(a['query_candidates'][i,c]));target=turn(truth[i]);inside[i,c]=bool(((centre>=28)&(centre<=483)).all()and(np.abs(target-centre)<=16).all())
        top1=visible&inside[:,0];top8=visible&inside.any(-1)
        if int(top1.sum())!=record['top1_exact_support']or int(top8.sum())!=record['top8_exact_support']or not np.array_equal(top8,a['top8_exact_support']):raise ValueError('Support counts differ from independent native replay')
        k=np.asarray(tile['cameras'][0]['intrinsics']);y,x=np.mgrid[:512,:512];rays=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])@np.linalg.inv(k).T;valid=np.isfinite(depth.ravel())&(depth.ravel()>0);distance=float(np.median(depth.ravel()[valid]*np.linalg.norm(rays[valid],axis=1)))
        pose=record['truth_assisted_cam2world'];error=None;correct=False
        if pose is not None:
            pose=np.asarray(pose);target=np.asarray(tile['cameras'][1]['cam2world']);angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@target[:3,:3])-1)/2,-1,1))));shift=float(np.linalg.norm(pose[:3,3]-target[:3,3]));error=dict(rotation_deg=angle,translation_m=shift);correct=angle<=2 and shift<=.01*distance
        if correct!=record['truth_assisted_pose_recovered']or(error is None)!=(record['truth_assisted_pose_error']is None):raise ValueError('Oracle outcome differs')
        if error is not None and not all(np.isclose(v,record['truth_assisted_pose_error'][k],rtol=0,atol=1e-9)for k,v in error.items()):raise ValueError('Oracle pose arithmetic differs')
        if not np.isclose(distance,record['reference_range_m'],rtol=0,atol=1e-9):raise ValueError('Native reference range differs')
        rows.append(dict(case_id=case['case_id'],visible=int(visible.sum()),top1_support=int(top1.sum()),top8_support=int(top8.sum()),truth_assisted_pose_recovered=bool(correct)))
    result=dict(passed=True,replayed_cases=len(rows),rows=rows,summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),scope='Independent saved-array/native-coordinate/pose arithmetic replay only. No new model inference or performance ceiling inferred.')
    with Path(output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(audit(a.run,a.protocol,a.output),indent=2))
