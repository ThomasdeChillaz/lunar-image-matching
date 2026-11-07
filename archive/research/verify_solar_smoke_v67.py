"""Independent replay of saved procedural correspondences and camera poses.

Does not import the matcher or trust its correctness flag. No real lunar
accuracy, model improvement, accepted-output risk or timing replay is inferred.
"""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root,output):
    root=Path(root);s=json.loads((root/'summary.json').read_text());rows=[]
    if len(s['rows'])!=8:raise ValueError('All eight software toggles required')
    for row in s['rows']:
        name=''.join(str(int(row[k]))for k in('solar','response','active'));p=root/(name+'.json');z=json.loads(p.read_text());raw=z['raw_research']
        pose=np.asarray(raw['pose'],float);a=np.asarray(raw['reference_xy']).reshape(-1,2);b=np.asarray(raw['query_xy']).reshape(-1,2)
        if pose.shape!=(4,4)or not np.isfinite(pose).all()or len(a)!=len(b)or not len(a):raise ValueError('Finite independently replayable pose/correspondences required')
        translation=float(np.linalg.norm(pose[:3,3]-[-2,1,0]));rotation=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1,1))))
        error=float(np.median(np.linalg.norm(b-a-[8,-4],axis=1)))
        if not np.isclose(translation,row['translation_error_m'],rtol=0,atol=1e-10)or not np.isclose(error,row['median_native_correspondence_error_px'],rtol=0,atol=1e-10):raise ValueError('Saved software metrics differ from independent native replay')
        passed=translation<1 and error<1
        if passed!=row['passed']or z['accepted']or z['pose']is not None or len(raw['measurements'])>32 or raw['proposal_samples_requested']>4096:raise ValueError('Software criterion, public null policy or budget differs')
        rows.append(dict(toggle=name,passed=passed,translation_error_m=translation,rotation_error_deg=rotation,median_native_correspondence_error_px=error,saved_result_sha256=sha(p)))
    if {r['toggle']for r in rows}!={''.join(map(str,x))for x in itertools.product((0,1),repeat=3)}:raise ValueError('Missing or repeated toggle')
    for p,h in s['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Procedural executed source changed:'+p)
    result=dict(passed=all(r['passed']for r in rows),rows=rows,source_sha256=sha(__file__),summary_sha256=sha(root/'summary.json'),scope='Independent arithmetic replay of eight PROCEDURAL image-to-pose controls. Rotation added as diagnostic; original<1m/<1px software gate retained. Does not replay runtimes or establish lunar accuracy/learning.')
    with Path(output).open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--output',required=True);a=p.parse_args();audit(a.run,a.output)
