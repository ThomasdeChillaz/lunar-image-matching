"""Actual selected v18 checkpoints: controls for all seeds and primary forward replay."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_geometry_warp_v11 import control
from lunargeo_dataset import write_new
from search_window_learning_v18 import SearchWindowMatcher
from train_balanced_localization_v3 import load_bundle
from train_search_window_v18 import configure,evaluate,make_model,read_rows


def run(cohort,output):
    cohort,root=Path(cohort),Path(output)
    if root.exists():raise FileExistsError('Preserve all selected-checkpoint control attempts')
    configure();protocol=json.loads((cohort/'protocol.json').read_text());summary=json.loads((cohort/'summary.json').read_text())
    if len(summary['results'])!=6 or summary['completed_optimizer_steps']!=12000:raise ValueError('Complete six-run cohort required')
    for name,digest in protocol['source_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Executed fitting source changed')
    audit_path=Path('reports/native_window_complete_audit_v18.json');audit=json.loads(audit_path.read_text())
    if not audit['passed'] or audit['partial'] or audit['recorded_updates']!=12000:raise ValueError('Complete saved-study audit required')
    root.mkdir(parents=True);bundle=load_bundle('reports/localization_training_data_v5.json')
    coarse=CompactMatcher(bundle[0]['coarse_model']);coarse.model.eval().requires_grad_(False);records=[]
    from synthetic import texture
    low=texture(994,128)
    for saved in summary['results']:
        folder=cohort/f'{saved["variant"]}-{saved["seed"]}';complete=json.loads((folder/'COMPLETE.json').read_text())
        meta=json.loads((folder/'training.json').read_text())
        if sha256(folder/'selected.pt')!=complete['sha256']['selected.pt']:raise ValueError('Selected checkpoint changed')
        model=make_model(coarse,saved['variant']);model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True));model.eval()
        matcher=SearchWindowMatcher(coarse,model);actual=control(matcher)
        p,q=matcher.points(low,low,np.zeros_like(low,bool),np.ones_like(low,bool))
        if len(p) or len(q):actual['passed']=False
        record=dict(variant=saved['variant'],seed=saved['seed'],selected_step=saved['selected_step'],
            checkpoint_sha256=sha256(folder/'selected.pt'),controls=actual,invalid_mask_empty=not(len(p) or len(q)))
        if saved['variant']==summary['winner'] and saved['seed']==20261101:
            evaluation,rows=evaluate(model,coarse,bundle,{},False)
            expected=read_rows(folder/f'selection_step_{meta["selected_step"]}.json.gz')
            if rows!=expected or evaluation!=saved['evaluation']:raise ValueError('Actual selected72-case primary forward differs')
            with gzip.open(root/'primary_forward.json.gz','wt') as stream:json.dump(rows,stream,allow_nan=False)
            record.update(actual_primary_forward_cases=72,actual_primary_forward_exact=True,evaluation=evaluation,
                forward_sha256=sha256(root/'primary_forward.json.gz'))
        records.append(record);print('selected window controls',saved['variant'],saved['seed'],actual['passed'],flush=True)
    result=dict(passed=all(r['controls']['passed'] and r['invalid_mask_empty'] for r in records),selected_models=6,records=records,
        source_sha256=sha256(__file__),cohort_summary_sha256=sha256(cohort/'summary.json'),cohort_audit_sha256=sha256(audit_path),
        scope='Actual selected-weight software controls and primary forward replay on exposed development data, not fresh geographic validation.')
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cohort',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.cohort,args.output)
