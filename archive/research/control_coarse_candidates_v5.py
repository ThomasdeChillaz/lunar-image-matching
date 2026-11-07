"""Actual trained forward passes on analytic image/pose and blank controls."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from coarse_pipeline_v5 import infer
from compact_lunar import CompactMatcher
from dataset import sha256
from lunargeo_dataset import write_new
from synthetic import texture


def run(root,output):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads(Path('reports/localization_training_data_v5.json').read_text())
    torch.set_num_threads(2);cv2.setNumThreads(1)
    fine=CompactMatcher(manifest['coarse_model']).model.eval()
    image=texture(733,512);query=np.rot90(image,1).copy();depth=np.full(image.shape,10.)
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    expected=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    records=[]
    for entry in summary['results']:
        folder=root/f"{entry['variant']}-{entry['seed']}";matcher=CompactMatcher(manifest['coarse_model'])
        matcher.model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True))
        selected,_=infer(matcher,fine,image,query,depth,k,np.eye(4),k)
        rotation=None if selected is None else float(np.linalg.norm(selected['pose'][:3,:3]-expected))
        translation=None if selected is None else float(np.linalg.norm(selected['pose'][:3,3]))
        blank,candidates=infer(matcher,fine,np.zeros_like(image),np.zeros_like(query),depth,k,np.eye(4),k)
        blank_ok=blank is None and all(c['pose'] is None for c in candidates)
        pose_ok=selected is not None and rotation<.01 and translation<.05
        records.append(dict(variant=entry['variant'],seed=entry['seed'],selected_step=entry['selected_step'],
            checkpoint_sha256=sha256(folder/'selected.pt'),quarter_turn_pose_passed=bool(pose_ok),
            rotation_matrix_error=rotation,translation_error_m=translation,blank_no_pose=blank_ok))
    result=dict(passed=all(r['quarter_turn_pose_passed'] and r['blank_no_pose'] for r in records),
        scope='Analytic procedural software controls; not a fresh lunar-accuracy test or confidence policy.',
        source_sha256=sha256(__file__),runs=records)
    write_new(output,result);print(json.dumps(result,indent=2))
    if not result['passed']:raise ValueError('Candidate software control failed; result retained')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/localization-v5-coarse-development')
    p.add_argument('--output',default='reports/coarse_training_controls_v5.json');a=p.parse_args();run(a.run,a.output)
