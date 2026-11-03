"""Source-published100/2000update response studies; no heldout illumination read.

The same trainer handles SH and capacity-matched pooled controls. Admission
requires a reconstructed cumulative compute ledger, hashed even-light bank and
published experiment protocol. No automatic checkpoint promotion/final test.
"""
import hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
from compact_lunar import load_checkpoint
from solar_response_v63 import LightingResponseModel
from solar_training_v63 import TrainingBank,fitting_step
from solar_compute_budget_v67 import reserve

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def initialize(protocol):
    torch.set_num_threads(2);torch.manual_seed(protocol['seed'])
    torch.use_deterministic_algorithms(True)
    encoder,metadata=load_checkpoint(protocol['initializer'])
    model=LightingResponseModel(encoder)
    optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    return model,optimizer,np.random.default_rng(protocol['seed'])


def snapshot(model,optimizer,rng,step):
    return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),step=step,format='solar-response-v63',training_complete=True)


def run(protocol_path,publication_commit):
    p=json.loads(Path(protocol_path).read_text());out=Path(p['output']);tick=time.monotonic();start=datetime.now(timezone.utc);deadline=start+timedelta(seconds=p['max_calendar_seconds'])
    if out.exists():raise FileExistsError('Preserve every prior training/checkpoint/failure')
    if p['updates']not in(100,2000)or p['seed']not in(20261101,20261102,20261103)or p['cpu_threads']!=2:raise ValueError('Fixed update, seed and CPU contract required')
    if subprocess.check_output(['git','show',publication_commit+':'+protocol_path])!=Path(protocol_path).read_bytes():raise ValueError('Training protocol not source-published')
    for n,h in p['source_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',publication_commit+':'+n])!=Path(n).read_bytes():raise ValueError('Training source differs from freeze:'+n)
    if sha(p['compute_ledger'])!=p['compute_ledger_sha256']:raise ValueError('Compute admission ledger changed')
    ledger=json.loads(Path(p['compute_ledger']).read_text())
    if not ledger['accounting_complete']or ledger['remaining_training_wall_seconds']<p['max_calendar_seconds']:raise ValueError('Cumulative48hour training budget not reconstructed or insufficient')
    if sha(p['bank'])!=p['bank_sha256']:raise ValueError('Training bank changed')
    if p['updates']==2000 and(not p['pilot_exact_replay_passed']or not p['measured_cost_fits']):raise ValueError('Actual100update reproducibility/cost gate required')
    reservation=reserve(p['compute_ledger'],p['compute_ledger_sha256'],p['reservation_journal'],sha(protocol_path)+'/train','training',p['max_calendar_seconds'])
    bank=TrainingBank(p['bank']);model,optimizer,rng=initialize(p);frozen={k:v.clone()for k,v in model.coarse.state_dict().items()}
    out.mkdir(parents=True);history=[]
    def save(n,z):
        with(out/n).open('x')as f:json.dump(z,f,indent=2,allow_nan=False);f.write('\n')
    def checkpoint(step):
        path=out/f'step_{step}.pt'
        if path.exists():raise FileExistsError(path)
        torch.save(snapshot(model,optimizer,rng,step),path)
    save('execution.json',dict(publication_commit=publication_commit,protocol_sha256=sha(protocol_path),started_utc=start.isoformat(),absolute_deadline_utc=deadline.isoformat(),total_parameters=model.total_parameters,compute_reservation=reservation));checkpoint(0)
    try:
        for step in range(1,p['updates']+1):
            if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Original response-study allowance exhausted')
            record=fitting_step(model,optimizer,bank,rng,p['pooled']);history.append(dict(step=step,**record))
            if any(not torch.equal(v,frozen[k])for k,v in model.coarse.state_dict().items()):raise ValueError('Frozen coarse proposal encoder changed')
            if step%100==0:print('RESPONSE',step,'loss',record['loss'],flush=True)
            if step in(100,500,1000,1500,2000):checkpoint(step);save(f'history_{step}.json',history)
        summary=dict(complete=True,updates=p['updates'],seed=p['seed'],pooled=p['pooled'],total_parameters=model.total_parameters,calendar_seconds=time.monotonic()-tick,all_history_sha256=sha(out/f'history_{p["updates"]}.json'),checkpoint_sha256={f'step_{s}.pt':sha(out/f'step_{s}.pt')for s in(0,100,500,1000,1500,2000)if(out/f'step_{s}.pt').exists()},heldout_image_reads=0,model_selection_performed=False,calibration_or_final_access=False,model_promoted=False)
        save('summary.json',summary);return summary
    except BaseException as exc:save('FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),updates=len(history),calendar_seconds=time.monotonic()-tick));raise


def replay_pilot(protocol_path,run_root):
    p=json.loads(Path(protocol_path).read_text());root=Path(run_root)
    if p['updates']!=100:raise ValueError('Exact100update pilot replay only')
    cap=p['replay_cap_seconds']
    if type(cap)is not int or not 0<cap<=900:raise ValueError('Fixed bounded replay allowance required')
    reserve(p['compute_ledger'],p['compute_ledger_sha256'],p['reservation_journal'],sha(protocol_path)+'/exact_replay','training',cap)
    tick=time.monotonic()
    bank=TrainingBank(p['bank']);model,optimizer,rng=initialize(p);history=[]
    initial=torch.load(root/'step_0.pt',map_location='cpu',weights_only=True)
    def identical(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys()and all(identical(a[k],b[k])for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b)and all(identical(x,y)for x,y in zip(a,b))
        return a==b
    if not identical(snapshot(model,optimizer,rng,0),initial):raise ValueError('Pilot initialization not exactly replayed')
    for step in range(1,101):
        if time.monotonic()-tick>=cap:raise TimeoutError('Original exact replay allowance exhausted')
        history.append(dict(step=step,**fitting_step(model,optimizer,bank,rng,p['pooled'])))
    final=torch.load(root/'step_100.pt',map_location='cpu',weights_only=True)
    if not identical(snapshot(model,optimizer,rng,100),final)or history!=json.loads((root/'history_100.json').read_text()):raise ValueError('Model, optimizer, random state, losses, gradients or sampler differ')
    return dict(passed=True,updates=100,exact_model_optimizer_rng_loss_gradient_sampler=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit');p.add_argument('--replay-pilot')
    a=p.parse_args();result=replay_pilot(a.protocol,a.replay_pilot)if a.replay_pilot else run(a.protocol,a.publication_commit);print(json.dumps(result,indent=2))
