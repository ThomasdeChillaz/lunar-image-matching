"""Actual selected native-refiner forward and rejection-path software controls."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from lunargeo_dataset import write_new
from native_refinement_v7 import NativeRefiner
from native_pipeline_v7 import infer
from synthetic import texture


def run(root,output):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());manifest=json.loads(Path('reports/localization_training_data_v5.json').read_text())
    torch.set_num_threads(2);cv2.setNumThreads(1);coarse=CompactMatcher(manifest['coarse_model'])
    image=texture(733,512);query=np.rot90(image,1).copy();depth=np.full(image.shape,10.)
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]]);expected=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    rows=[]
    for entry in summary['results']:
        folder=root/f"{entry['variant']}-{entry['seed']}";model=NativeRefiner(entry['variant'])
        model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True));model.eval()
        selected,_=infer(coarse,model,image,query,depth,k,np.eye(4),k)
        rotation=None if selected is None else float(np.linalg.norm(selected['pose'][:3,:3]-expected))
        translation=None if selected is None else float(np.linalg.norm(selected['pose'][:3,3]))
        blank,attempts=infer(coarse,model,np.zeros_like(image),np.zeros_like(query),depth,k,np.eye(4),k)
        passed=selected is not None and rotation<.01 and translation<.05 and blank is None and all(a['pose'] is None for a in attempts)
        rows.append(dict(variant=entry['variant'],seed=entry['seed'],selected_step=entry['selected_step'],passed=bool(passed),
            rotation_matrix_error=rotation,translation_error_m=translation,blank_no_pose=blank is None,
            checkpoint_sha256=sha256(folder/'selected.pt')))
    result=dict(passed=all(r['passed'] for r in rows),scope='Analytic actual-model software controls, not lunar generalization or a confidence guarantee.',
        runs=rows,source_sha256=sha256(__file__))
    write_new(output,result);print(json.dumps(result,indent=2))
    if not result['passed']:raise ValueError('Native model software control failed; result retained')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.run,a.output)
