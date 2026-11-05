"""Independent saved procedural arithmetic; imports no matcher/pose estimator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def replay(path):
    source=json.loads(path.read_bytes())
    if (source['schema']!='blind_catalogue_generated_v138' or not source['procedural_only']
            or source['physical_sunlight_simulation'] or source['lunar_images_opened']
            or source['calibrated_pose'] is not None):
        raise ValueError('Only the generated, uncalibrated wiring control is eligible')
    for name,h in source['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h: raise ValueError('Changed procedural source')
    truth=np.asarray(source['scoring_only_truth']['cam2world'])
    shift=source['scoring_only_truth']['horizontal_shift_pixels']
    raw=source['result']['raw_research']; methods=raw['methods']
    if set(methods)!={'joint_no_extra','uniform','geometry','active'} or raw['query_truth_used']:
        raise ValueError('Incomplete methods or query-truth leakage')
    for name,m in methods.items():
        p,q=np.asarray(m['reference_xy']).reshape(-1,2),np.asarray(m['query_xy']).reshape(-1,2)
        error=np.linalg.norm(q-p+[shift,0.],axis=1)
        pose=None if m['pose'] is None else np.asarray(m['pose'])
        translation=None if pose is None else float(np.linalg.norm(pose[:3,3]-truth[:3,3]))
        expected=source['rows'][name]
        if translation!=expected['translation_m'] or len(error)!=expected['returned_matches'] or float(np.median(error))!=expected['correspondence_median_px']:
            raise ValueError('Saved procedural geometry metric differs')
        if translation is None or translation>=2. or len(error)<6 or np.median(error)>=.2:
            raise ValueError('Generated control failed, including null poses')
        if name!='joint_no_extra' and m['acquisition']['attempted_extra_patches']!=2:
            raise ValueError('Changed generated wiring budget')
    return dict(passed=True,method_case_cells=4,matching=False,pose_fitting=False,
                physical_sunlight_validation=False,lunar_accuracy_claim=False,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=replay(args.source)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
