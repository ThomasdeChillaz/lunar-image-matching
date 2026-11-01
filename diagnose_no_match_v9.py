"""Diagnose null-class granularity on fixed, exposed selection patches."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import sha256
from lunargeo_dataset import write_new
from native_refinement_v7 import NativeRefiner,joint_logits
from localization_refiner_v3 import decode_peak
from train_balanced_localization_v3 import load_bundle


def run(folder,output):
    folder=Path(folder);meta=json.loads((folder/'training.json').read_text());complete=json.loads((folder/'COMPLETE.json').read_text())
    if sha256(folder/'final.pt')!=complete['final_sha256']:raise ValueError('Completed final native checkpoint differs')
    torch.set_num_threads(2);cv2.setNumThreads(1);model=NativeRefiner('dustbin')
    model.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True));model.eval()
    manifest,tiles,data=load_bundle('reports/localization_training_data_v5.json');rng=np.random.default_rng(20261101);rows=[]
    for c in [c for c in manifest['cases'] if c['split']=='model_selection']:
        z=data[c['tile_id']];k=c['key'];p,q=z[f'p{k}'],z[f'q{k}'];target=z[f'target{k}']-np.rint(q)+16
        valid=z[f'visible{k}']&((target>=0)&(target<=32)).all(1)
        for positive in (True,False):
            eligible=np.flatnonzero(valid==positive)
            if not len(eligible):continue
            indices=rng.choice(eligible,min(4,len(eligible)),replace=False);rp=[];qp=[];xy=[]
            for i in indices:
                rc=np.rint(p[i]).astype(int);qc=np.rint(q[i]).astype(int)
                rp.append(cv2.getRectSubPix(z['images'][2*c['light_a']],(27,27),tuple(map(float,rc))).astype(np.float32)/255)
                qp.append(cv2.getRectSubPix(z['images'][2*c['light_b']+1],(57,57),tuple(map(float,qc))).astype(np.float32)/255);xy.append(p[i]-rc+13)
            with torch.inference_mode():
                local,no_match=model(torch.from_numpy(np.stack(rp))[:,None],torch.from_numpy(np.stack(qp))[:,None],torch.tensor(np.array(xy),dtype=torch.float32))
                probability=no_match.sigmoid().numpy();joint_reject=(joint_logits(local,no_match).argmax(1)==1089).numpy()
                binary_reject=(no_match>=0).numpy();peak=decode_peak(local).numpy()+16
            for j,i in enumerate(indices):
                rows.append(dict(pair_id=c['pair_id'],component=c['component'],proposal_index=int(i),window_has_visible_correspondence=positive,
                    no_match_probability=float(probability[j]),joint_argmax_reject=bool(joint_reject[j]),
                    aggregated_binary_reject=bool(binary_reject[j]),
                    decoded_error_px=None if not positive else float(np.linalg.norm(peak[j]-target[i]))))
    stats={}
    for positive in (True,False):
        part=[r for r in rows if r['window_has_visible_correspondence']==positive]
        stats['positive_window' if positive else 'negative_window']=dict(samples=len(part),
            median_no_match_probability=float(np.median([r['no_match_probability'] for r in part])),
            joint_argmax_rejections=sum(r['joint_argmax_reject'] for r in part),binary_rule_rejections=sum(r['aggregated_binary_reject'] for r in part))
    result=dict(scope='Fixed exposed-development diagnostic of a failed final checkpoint, not selected-model, pose, calibration or final-test accuracy.',
        final_checkpoint_sha256=complete['final_sha256'],source_sha256=sha256(__file__),model_seed=meta['seed'],training_step=meta['completed_steps'],
        hypothesis='The joint1090-class argmax compares one diffuse local bin with total no-match probability. The fixed binary rule instead compares any-match mass against no-match mass; no threshold search is performed.',
        results=stats,rows=rows,calibration_or_test_pixels_loaded=0)
    write_new(output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.model,a.output)
