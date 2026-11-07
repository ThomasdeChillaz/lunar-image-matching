"""Actual selected weights: five controls plus complete primary forward replay."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import make_context
from dataset import sha256
from evaluate_geometry_warp_v11 import control
from global_context_v15 import ContextMatcher
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle
from train_global_context_v16 import evaluate


def run(cohort,output):
    cohort,root=Path(cohort),Path(output)
    if root.exists():raise FileExistsError('Preserve all selected-weight controls')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((cohort/'protocol.json').read_text());summary=json.loads((cohort/'summary.json').read_text())
    for source,digest in protocol['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Frozen cohort source changed')
    root.mkdir(parents=True);bundle=load_bundle('reports/localization_training_data_v5.json')
    coarse=CompactMatcher(bundle[0]['coarse_model']);coarse.model.eval();records=[]
    for saved in summary['results']:
        folder=cohort/f'{saved["variant"]}-{saved["seed"]}';meta=json.loads((folder/'training.json').read_text())
        proof=json.loads((folder/'COMPLETE.json').read_text())
        if sha256(folder/'selected.pt')!=proof['sha256']['selected.pt']:raise ValueError('Selected checkpoint changed')
        model=make_context(saved['variant']);model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True));model.eval()
        actual=control(ContextMatcher(coarse,model))
        if not actual['passed']:raise ValueError('Actual selected-weight control failed')
        record=dict(variant=saved['variant'],seed=saved['seed'],selected_step=saved['selected_step'],
            checkpoint_sha256=sha256(folder/'selected.pt'),controls=actual)
        if saved['variant']==summary['winner'] and saved['seed']==20261101:
            evaluation,rows=evaluate(model,coarse,bundle,{},False)
            with gzip.open(folder/f'selection_step_{meta["selected_step"]}.json.gz','rt') as stream:expected=json.load(stream)
            if rows!=expected or evaluation!=saved['evaluation']:raise ValueError('Actual72-case selected forward replay differs')
            with gzip.open(root/'primary_forward.json.gz','wt') as stream:json.dump(rows,stream,allow_nan=False)
            record.update(actual_primary_forward_cases=len(rows),actual_primary_forward_exact=True,
                evaluation=evaluation,forward_sha256=sha256(root/'primary_forward.json.gz'))
        records.append(record);print('selected controls',saved['variant'],saved['seed'],flush=True)
    result=dict(passed=True,selected_models=len(records),records=records,
        source_sha256=sha256(__file__),cohort_summary_sha256=sha256(cohort/'summary.json'),
        scope='Actual pretrained/fitted-checkpoint software control and selected primary native forward replay. Exposed development data, not fresh geographic validation.')
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cohort',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.cohort,args.output)
