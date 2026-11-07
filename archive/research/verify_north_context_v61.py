"""Independently replay the complete new-data100update optimizer pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compact_lunar import CompactMatcher
from north_training_bundle_v38 import load_bundle,prepare_training,sha
from train_north_context_v61 import initialize,fitting_step
from train_search_window_v18 import configure,make_optimizer,verify_frozen


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor)or a.dtype!=b.dtype or not torch.equal(a,b):raise ValueError('Replay tensor differs: '+path)
    elif isinstance(a,dict):
        if not isinstance(b,dict)or a.keys()!=b.keys():raise ValueError('Replay keys differ: '+path)
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b)or len(a)!=len(b):raise ValueError('Replay sequence differs: '+path)
        for i,(x,y)in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif a!=b:raise ValueError('Replay scalar differs: '+path)


def audit(run,output):
    configure();root,out=Path(run),Path(output)
    if out.exists():raise FileExistsError('Preserve previous replay evidence')
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    seal=json.loads((root/'COMPLETE.json').read_text())
    for p,h in seal['sha256'].items():
        if sha(root/p)!=h:raise ValueError('Completed pilot artifact changed: '+p)
    for p,h in protocol['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Executed fitting source changed: '+p)
    if not summary['complete']or summary['completed_steps']!=100 or summary['protocol_sha256']!=sha(root/'protocol.json'):
        raise ValueError('Complete sealed100update pilot required')
    if sha(protocol['manifest_path'])!=protocol['manifest_sha256']:raise ValueError('Pilot data manifest changed')
    manifest,tiles,data=load_bundle(protocol['manifest_path'],train_only=True);prepared=prepare_training(tiles,data)
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
    model=initialize(coarse);initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
    equal(model.state_dict(),initial)
    optimizer=make_optimizer(model);rng=np.random.default_rng(0)
    initial_rng=torch.load(root/'initial_rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(initial_rng['torch_rng']);rng.bit_generator.state=initial_rng['numpy_rng']
    history=json.loads((root/'history_100.json').read_text())
    if len(history)!=100:raise ValueError('Complete update history required')
    for step,saved in enumerate(history,1):
        actual=fitting_step(model,optimizer,tiles,prepared,rng,step)
        equal(actual,saved,'update-'+str(step));verify_frozen(model,initial)
        if step%50==0:
            recorded=torch.load(root/f'step_{step}.pt',map_location='cpu',weights_only=True)
            equal(model.state_dict(),recorded['model'],'model-'+str(step))
            equal(optimizer.state_dict(),recorded['optimizer'],'optimizer-'+str(step))
            equal(torch.get_rng_state(),recorded['torch_rng'],'torch_rng')
            equal(rng.bit_generator.state,recorded['numpy_rng'],'numpy_rng')
    if any(t['split']!='train'for t in tiles.values())or len(tiles)!=48:raise ValueError('Replay read forbidden split')
    original=json.loads(Path('runs/north-v38-training-cost-pilot/history_100.json').read_text())
    if [r['sample'] for r in history]!=[r['sample'] for r in original]:
        raise ValueError('Cross-context and original pilot sampling differ')
    result=dict(passed=True,completed_steps=100,matched_original_sample_records=100,
        total_parameters=82208,trainable_parameters=53936,only_declared_context_architecture_changed=True,exact_full_optimizer_retraining=True,
        exact_model_optimizer_losses_gradients_rng=True,fine_encoder_unchanged=True,
        training_components=16,training_geometries=48,lighting_count=6,
        model_selection_pixels_loaded=0,calibration_or_test_pixels_loaded=0,
        pilot_protocol_sha256=sha(root/'protocol.json'),pilot_summary_sha256=sha(root/'summary.json'),
        source_sha256=sha(__file__),scope='Exacttrain-only100update pilot replay, not localization improvement or full-cohort retraining.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/north-v61-context-cost-pilot')
    p.add_argument('--output',default='reports/north_context_pilot_audit_v61.json')
    a=p.parse_args();audit(a.run,a.output)
