from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from compact_lunar import CompactDescriptor
from context_supervision_v15 import proposal_grid
from search_window_learning_v18 import SearchWindowModel,direction,objective,window_mask


def test_loss_accepts_alternative_coarse_cells_reachable_by_native_refinement():
    grid=torch.tensor([[12.,12.],[16.,12.],[20.,12.],[24.,12.]])
    target=torch.tensor([[16.,12.]])
    good=window_mask(target,torch.tensor([True]),grid)
    assert good.tolist()==[[True,True,False,False]]
    first=torch.tensor([[.9,0.,0.,0.]])
    second=torch.tensor([[0.,.9,0.,0.]])
    wrong=torch.tensor([[0.,0.,.9,0.]])
    a,_=direction(first,target,torch.tensor([True]),grid,anchor_weight=0)
    b,_=direction(second,target,torch.tensor([True]),grid,anchor_weight=0)
    c,_=direction(wrong,target,torch.tensor([True]),grid,anchor_weight=0)
    assert torch.isclose(a,b,atol=1e-6) and c>a+1


def test_occluded_and_out_of_window_rows_prefer_lower_scores():
    grid=torch.tensor([[12.,12.],[16.,12.]])
    target=torch.tensor([[14.,12.],[100.,100.]])
    visible=torch.tensor([False,True])
    low,labels=direction(torch.zeros(2,2),target,visible,grid,anchor_weight=0)
    high,_=direction(torch.full((2,2),.9),target,visible,grid,anchor_weight=0)
    assert labels==dict(positive=0,negative=2,anchors=0) and high>low


def test_anchor_protects_a_genuine_baseline_peak_from_a_new_distractor():
    grid=torch.tensor([[12.,12.],[16.,12.],[20.,12.],[24.,12.]])
    target=torch.tensor([[16.,12.]]);visible=torch.tensor([True])
    original=torch.tensor([[.9,.2,.1,.1]])
    learned=torch.tensor([[.0,.9,.7,.0]])
    no_anchor,_=direction(learned,target,visible,grid,anchor_weight=0)
    anchored,labels=direction(learned,target,visible,grid,baseline=original,anchor_weight=.5)
    assert labels['anchors']==1 and anchored>no_anchor+.3


@pytest.mark.parametrize('train_coarse',[False,True])
def test_initial_scores_are_exact_and_fine_encoder_stays_frozen_during_learning(train_coarse):
    torch.set_num_threads(1);torch.manual_seed(20261101)
    frozen=SimpleNamespace(model=CompactDescriptor(False).eval(),metadata=dict(variant='expanded'),device='cpu')
    model=SearchWindowModel(frozen,train_coarse=train_coarse)
    assert sum(p.numel() for p in model.parameters())==81696
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==25152+(28272 if train_coarse else 0)
    images=torch.rand(2,1,128,128);model.train();scores,baseline=model(images)
    assert torch.equal(scores,baseline)
    initial={k:v.clone() for k,v in model.fine.state_dict().items()}
    coarse_initial={k:v.clone() for k,v in model.coarse.state_dict().items()}
    targets=dict(forward=torch.from_numpy(proposal_grid()),reverse=torch.from_numpy(proposal_grid()),
        forward_visible=torch.ones(676,dtype=torch.bool),reverse_visible=torch.ones(676,dtype=torch.bool))
    loss,labels=objective(scores,targets,baseline);assert torch.isfinite(loss)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001)
    loss.backward();optimizer.step()
    assert all(torch.equal(v,initial[k]) for k,v in model.fine.state_dict().items())
    assert all(p.grad is None for p in model.fine.parameters())
    changed=any(not torch.equal(v,coarse_initial[k]) for k,v in model.coarse.state_dict().items())
    assert changed==train_coarse
    assert labels['forward']['positive']==676
