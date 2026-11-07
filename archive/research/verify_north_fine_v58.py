"""Replay native fine controls; exact pilot optimizer replay, saved study metrics."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
from compact_lunar import CompactMatcher
from train_north_fine_v58 import (read,sha,MANIFEST,SEED,STEPS,INTERVAL,CAPS,
    configure,load_bundle,prepare,make_fine,fit_step,sample_batch,inputs_sha,
    bank_metrics,control,FineMatcher,counts,selection_key,read_rows,totals)
from verify_north_results_v39 import check_cases,replay_row,initial_equivalence


def identical(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(identical(v,b[k])for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b)and len(a)==len(b)and all(identical(x,y)for x,y in zip(a,b))
    return a==b


def audit(run,output):
    root=Path(run);out=Path(output)
    if out.exists():raise FileExistsError('Preserve previous replay')
    p=read(root/'protocol.json');s=read(root/'summary.json');mode=p['mode'];variant=p['variant']
    steps=100 if mode=='pilot'else STEPS
    if(not s['complete']or s['completed_steps']!=steps or s['model_promoted']
        or s['protocol_sha256']!=sha(root/'protocol.json')or s['calibration_or_test_pixels_loaded']!=0
        or not 0<=s['calendar_seconds']<=CAPS[mode]or p['seed']!=SEED
        or p['total_parameters']!=56544 or p['trainable_parameters']!=28272):
        raise ValueError('Complete bounded declared fine experiment required')
    sealed=read(root/'COMPLETE.json')['sha256']
    if set(sealed)!={str(x.relative_to(root))for x in root.rglob('*')if x.is_file()and x.name!='COMPLETE.json'}:
        raise ValueError('Sealed inventory differs')
    for n,h in sealed.items():
        if sha(root/n)!=h:raise ValueError('Sealed evidence changed:'+n)
    for n,h in p['source_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',p['publication_commit']+':'+n])!=Path(n).read_bytes():
            raise ValueError('Executed/public source differs:'+n)
    if sha(MANIFEST)!=p['manifest_sha256']:raise ValueError('Data manifest changed')
    configure();bundle=load_bundle(MANIFEST,train_only=True);tiles,data=bundle[1:];prepared=prepare(tiles,data)
    coarse=CompactMatcher(bundle[0]['coarse_model']);coarse.model.eval().requires_grad_(False)
    coarse_initial={k:v.clone()for k,v in coarse.model.state_dict().items()}
    torch.manual_seed(SEED);fine=make_fine(coarse)
    load=lambda name:torch.load(root/name,map_location='cpu',weights_only=True)
    initial=load('initial.pt');latest=load('latest.pt')
    if not identical(initial,fine.state_dict())or latest['step']!=steps or len(latest['history'])!=steps or latest['history']!=s['history']:
        raise ValueError('Initialization/optimizer boundary differs')
    if {int(v['step'])for v in latest['optimizer']['state'].values()}!={steps}:raise ValueError('Adam counts differ')
    groups=latest['optimizer']['param_groups']
    if len(groups)!=1 or groups[0]['lr']!=1e-4 or groups[0]['weight_decay']!=1e-4:raise ValueError('Optimizer settings differ')
    if bank_metrics(fine,tiles,data,prepared,variant)!=s['initial_bank']:raise ValueError('Initial training bank differs')
    rng=np.random.default_rng(SEED);optimizer=torch.optim.AdamW(fine.parameters(),lr=1e-4,weight_decay=1e-4)
    for step,saved in enumerate(s['history'],1):
        if mode=='pilot':
            actual=fit_step(fine,optimizer,tiles,data,prepared,rng,variant,step)
            if any(actual[k]!=saved[k]for k in ('step','loss','gradient_norm','samples','input_sha256')):
                raise ValueError('Exact optimizer replay differs at update'+str(step))
        else:
            arrays,meta=sample_batch(tiles,data,prepared,rng,variant)
            if saved['step']!=step or saved['samples']!=meta or saved['input_sha256']!=inputs_sha(arrays):
                raise ValueError('Training sample/input replay differs')
        if not np.isfinite([saved['loss'],saved['gradient_norm']]).all():raise ValueError('Nonfinite saved update')
    if rng.bit_generator.state!=latest['numpy_rng']:raise ValueError('Final data RNG differs')
    if mode=='pilot':
        if not identical(fine.state_dict(),latest['model'])or not identical(optimizer.state_dict(),latest['optimizer'])or not torch.equal(torch.get_rng_state(),latest['torch_rng']):
            raise ValueError('Exact final optimizer/model/RNG replay differs')
    fine.load_state_dict(latest['model'],strict=True)
    if bank_metrics(fine,tiles,data,prepared,variant)!=s['final_bank']:raise ValueError('Final training bank differs')
    if any(not torch.equal(v,coarse_initial[k])for k,v in coarse.model.state_dict().items()):raise ValueError('Coarse proposals changed')
    rows_replayed=0
    if mode=='pilot':
        observed=control(FineMatcher(coarse,fine))
        # Runtime fields are not reproducible; preserve the actual control verdict.
        if not observed['passed']or not s['post_control']['passed']or s['model_selection_pixels_loaded']!=0:
            raise ValueError('Actual post-training known-transform/blank controls failed')
    else:
        selection=load_bundle(MANIFEST,selection_only=True)
        if len(s['validation'])!=5:raise ValueError('All checkpoints required')
        for step,v in zip(range(0,STEPS+1,INTERVAL),s['validation']):
            path=root/f'selection_step_{step}.json.gz';rows=read_rows(path);check_cases(rows,selection[0]['cases'])
            if v['step']!=step or counts(rows)!=v['selection']or v['results_sha256']!=sha(path):raise ValueError('Selection aggregate differs')
            for row in rows:replay_row(selection,row)
            if step==0:initial_equivalence(rows,read_rows('runs/north-v39-full-baselines/results.json.gz'))
            rows_replayed+=len(rows)
        winner=max(s['validation'],key=lambda x:selection_key(x['selection'],x['step']))
        if winner['step']!=s['selected_step']or winner['selection']!=s['selection']or not identical(load('selected.pt'),load(f'checkpoint_step_{winner["step"]}.pt')):
            raise ValueError('Selected checkpoint differs')
        if not identical(load('final.pt'),latest['model']):raise ValueError('Final weights differ')
        neg=read_rows(root/'selected_negatives.json.gz');check_cases(neg,selection[0]['negative_cases'])
        for row in neg:replay_row(selection,row)
        if totals(neg)!=s['selected_negative_counts']:raise ValueError('Negative totals differ')
        rows_replayed+=len(neg)
        pilot=torch.load(Path(p['gate']['pilot_root'])/'latest.pt',map_location='cpu',weights_only=True)
        if s['history'][:100]!=pilot['history']or not identical(initial,torch.load(Path(p['gate']['pilot_root'])/'initial.pt',map_location='cpu',weights_only=True)):
            raise ValueError('Reused pilot prefix differs')
    result=dict(passed=True,mode=mode,variant=variant,replayed_updates=steps,replayed_rows=rows_replayed,
        exact_optimizer_retraining=mode=='pilot',exact_sampler_input_replay=True,coarse_weights_unchanged=True,
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),model_promoted=False,
        scope='Exact100update pilot optimizer replay or2000sample/saved576metric study replay. Training bank is not held-out pose accuracy. No calibrated/final/SOTA result.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.run,a.output)
