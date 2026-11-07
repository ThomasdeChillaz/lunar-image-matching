"""Small generated wiring test; the image labels do not simulate sunlight."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from blind_matcher_v138 import run
from procedural_fixture_v138 import fixture


def control():
    cv2.setNumThreads(1)
    cat,q,depth,k,poses=fixture()
    raw=run(cat,q,depth,k,np.eye(4),k,reference_valid=np.ones_like(depth,bool),
        height_direction_reference=[0,0,-1],policies=('uniform','geometry','active'),budget=2,
        poses=poses,research_output=True)
    methods=raw['raw_research']['methods']; rows={}
    for name,m in methods.items():
        camera=None if m['pose'] is None else np.asarray(m['pose'])
        p,q=np.asarray(m['reference_xy']).reshape(-1,2),np.asarray(m['query_xy']).reshape(-1,2)
        truth=np.eye(4);truth[0,3]=20.
        errors=np.linalg.norm(q-(p-[8.,0.]),axis=1)
        passed=camera is not None and np.linalg.norm(camera[:3,3]-truth[:3,3])<2. and len(errors)>=6 and float(np.median(errors))<.2
        rows[name]=dict(passed=bool(passed),translation_m=None if camera is None else float(np.linalg.norm(camera[:3,3]-truth[:3,3])),
            correspondence_median_px=None if not len(errors) else float(np.median(errors)),returned_matches=len(errors))
    selected=raw['raw_research']['initialization']['photometric_hypotheses'][0]['light_id']
    return dict(schema='blind_catalogue_generated_v138',passed=selected==8 and all(v['passed'] for v in rows.values()),
        procedural_only=True,physical_sunlight_simulation=False,lunar_images_opened=False,
        calibrated_pose=raw['pose'],selected_catalogue_id=selected,rows=rows,result=raw,
        scoring_only_truth=dict(cam2world=truth.tolist(),horizontal_shift_pixels=8.),
        source_sha256={str(p.relative_to(Path.cwd())):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=control()
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('result','source_sha256')},indent=2))
    if not result['passed']: raise SystemExit(1)
