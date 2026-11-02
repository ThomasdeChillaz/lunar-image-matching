"""Replay fixed training-fixture snapshots; never imply generalization."""
from pathlib import Path
import argparse, hashlib, json, subprocess
import numpy as np


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())


def independent_objective(scores, baseline, targets):
    y,x=np.mgrid[12:116:4,12:116:4]
    centers=np.rint((np.column_stack([x.ravel(),y.ravel()])+.5)*4-.5)
    def direction(s,b,t,v):
        good=np.asarray(v)[:,None] & (np.abs((np.asarray(t)[:,None]+.5)*4-.5-centers[None])<=16).all(-1)
        pos=good.any(1);loss=0.
        def lse(a):
            peak=a.max(1);return peak+np.log(np.exp(a-peak[:,None]).sum(1))
        if pos.any():
            a=s[pos]/.1;g=good[pos]
            loss+=float((lse(a)-lse(np.where(g,a,-np.inf))).mean())
            best=np.where(g,s[pos],-np.inf).max(1)
            loss+=.025*float(np.logaddexp(0,10*(.7-best)).mean())
        if (~pos).any():loss+=.025*float(np.logaddexp(0,10*(s[~pos].max(1)-.5)).mean())
        index=b.argmax(1);row=np.arange(676)
        keep=(b.argmax(0)[index]==row)&(b.max(1)>=.5)&good[row,index]
        if keep.any():
            wrong=np.where(good[keep],-np.inf,s[keep]).max(1)
            loss+=.05*float(np.logaddexp(0,10*(wrong-s[keep,index[keep]]+.1)).mean())
        return loss,dict(positive=int(pos.sum()),negative=int((~pos).sum()),anchors=int(keep.sum()))
    a,fa=direction(scores,baseline,targets['forward'],targets['forward_visible'])
    b,rb=direction(scores.T,baseline.T,targets['reverse'],targets['reverse_visible'])
    return .5*(a+b),dict(forward=fa,reverse=rb)


def audit(run,output):
    out=Path(output);assert not out.exists(), 'Preserve previous audit'
    root=Path(run);protocol=read(root/'protocol.json');summary=read(root/'summary.json')
    if not summary['complete'] or summary['logical_completed_updates']!=400 or summary['model_promoted']:
        raise ValueError('Complete diagnostic only, no promoted candidate')
    for n,h in read(root/'COMPLETE.json')['sha256'].items():
        if sha(root/n)!=h:raise ValueError('Sealed diagnostic changed:'+n)
    for n,h in protocol['source_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',protocol['publication_commit']+':'+n])!=Path(n).read_bytes():
            raise ValueError('Source differs from executed/public freeze')
    for n,h in protocol['upstream_check_sha256'].items():
        if sha(n)!=h:raise ValueError('Upstream evidence changed')
    import torch
    from compact_lunar import CompactMatcher
    from north_training_bundle_v38 import load_bundle
    from north_optimization_probe_v54 import fixed_specs,fixture_arrays,verify_fixture,score_metrics,decoded_indices
    from search_window_learning_v18 import SearchWindowMatcher
    from train_search_window_v18 import configure,make_model,verify_frozen
    configure();manifest,tiles,data=load_bundle('reports/north_training_data_v38.json',train_only=True)
    if fixed_specs(manifest)!=protocol['fixtures']:raise ValueError('Metadata fixture identities differ')
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
    snapshots=updates=0
    for spec,outcome in zip(protocol['fixtures'],summary['fixtures']):
        folder=root/spec['name'];a,b,target=fixture_arrays(data[spec['tile_id']],spec)
        verify_fixture(folder/'fixture.npz',a,b,target)
        images=torch.from_numpy(np.stack([a,b])[:,None]).float()/255
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
        latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if latest['step']!=100 or [r['step']for r in latest['history']]!=list(range(1,101)):
            raise ValueError('Missing logical update boundaries')
        if {int(s['step'])for s in latest['optimizer']['state'].values()}!={100}:
            raise ValueError('Incomplete optimizer counters')
        if outcome!=read(folder/'summary.json')or outcome['history']!=latest['history']:
            raise ValueError('History/summary differs')
        for row in latest['history']:
            if not np.isfinite([row['loss'],row['global_unclipped_gradient_norm'],*row['parameter_group_unclipped_gradient_norm'].values()]).all():
                raise ValueError('Nonfinite saved update')
            if row['parameter_group_unclipped_gradient_norm']['fine']!=0:raise ValueError('Frozen fine encoder received gradients')
        delta={prefix:float(sum(float((v-initial[n]).square().sum())for n,v in latest['model'].items()if n.startswith(prefix+'.'))**.5)for prefix in ('coarse','context','fine')}
        if delta!=outcome['parameter_group_weight_change_l2']or delta['fine']!=0:raise ValueError('Parameter changes differ')
        for step,saved in zip((0,25,50,75,100),outcome['evaluation']):
            cp=folder/f'checkpoint_{step}.pt';weights=torch.load(cp,map_location='cpu',weights_only=True)
            if saved!=read(folder/f'evaluation_{step}.json')or saved['checkpoint_sha256']!=sha(cp):raise ValueError('Snapshot differs')
            if step==100 and any(not torch.equal(v,latest['model'][k])for k,v in weights.items()):raise ValueError('Final snapshot differs')
            torch.manual_seed(20261101);model=make_model(coarse,'coarse_adaptation');model.load_state_dict(weights,strict=True);verify_frozen(model,initial);model.eval()
            with torch.inference_mode():
                scores,baseline=model(images);p,q=SearchWindowMatcher(coarse,model).points(a,b)
            ia,ib=decoded_indices(p,q)
            with np.load(folder/f'scores_{step}.npz',allow_pickle=False)as z:
                for name,value in [('scores',scores[0].numpy()),('baseline',baseline[0].numpy()),('reference_indices',ia),('query_indices',ib)]:
                    if not np.array_equal(z[name],value):raise ValueError('Actual forward/decoder replay differs:'+name)
            loss,labels=independent_objective(scores[0].numpy().astype(float),baseline[0].numpy().astype(float),target)
            if not np.isclose(loss,saved['objective'],rtol=0,atol=2e-6)or labels!=saved['labels']:raise ValueError('Independent objective/labels differ')
            if score_metrics(scores[0].numpy(),target,ia,ib)!=saved['metrics']:raise ValueError('Saved score metrics differ')
            snapshots+=1
        updates+=100
    if len(summary['fixtures'])!=4 or snapshots!=20 or updates!=400:raise ValueError('Diagnostic denominator differs')
    result=dict(passed=True,neural_snapshot_forwards_replayed=20,logical_recorded_updates=400,independent_numpy_objective_replay=True,full_optimizer_retraining=False,model_promoted=False,calibration_or_final_pixels_loaded=0,summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),scope='Fixed training-fixture numerical/forward replay only; no held-out pose, candidate, novelty or SOTA claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/north-v54-fixed-pair-optimization');p.add_argument('--output',default='reports/north_optimization_audit_v54.json');a=p.parse_args();audit(a.run,a.output)
