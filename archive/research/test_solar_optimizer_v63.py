"""Procedural optimizer and serialization replay, never a lunar training result."""
from copy import deepcopy
import numpy as np
import torch
from compact_lunar import CompactDescriptor
from solar_response_v63 import LightingResponseModel,counterfactual_loss


def execute():
    torch.manual_seed(20261101);model=LightingResponseModel(CompactDescriptor(False));optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=1e-4)
    p=torch.rand(4,1,33,33);condition=torch.zeros(4,9);condition[:,2]=1;condition[:,5]=1;condition[:,8]=1
    suns=torch.tensor([[[0.,0.,1.],[1.,0.,0.]]]).expand(4,-1,-1)
    available=torch.tensor([[True,False]]).expand(4,-1);teacher=torch.full((4,2,32),float('nan'));teacher[:,0]=torch.randn(4,32)
    frozen=deepcopy(model.coarse.state_dict());history=[]
    for _ in range(5):
        optimizer.zero_grad();loss,parts=counterfactual_loss(model,p,condition,suns,teacher,available,[[0,1],[2,3]])
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_([p for p in model.parameters()if p.requires_grad],5.);optimizer.step();history.append((float(loss.detach()),float(gradient)))
    assert all(torch.equal(v,frozen[k])for k,v in model.coarse.state_dict().items())
    return model.state_dict(),optimizer.state_dict(),history


def test_response_optimizer_replays_exactly_and_checkpoint_is_weights_only_loadable(tmp_path):
    a,oa,ha=execute();b,ob,hb=execute();assert ha==hb and all(torch.equal(v,b[k])for k,v in a.items())
    for k,v in oa['state'].items():
        for name,t in v.items():assert torch.equal(t,ob['state'][k][name])
    p=tmp_path/'checkpoint.pt';torch.save(dict(format='solar-response-v63',model=a,optimizer=oa,training_complete=True),p)
    saved=torch.load(p,weights_only=True,map_location='cpu');assert saved['format']=='solar-response-v63'
    assert all(torch.equal(v,saved['model'][k])for k,v in a.items())
