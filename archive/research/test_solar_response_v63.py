import math
import numpy as np
import pytest
import torch
from compact_lunar import CompactDescriptor
from solar_response_v63 import real_spherical_harmonics,LightingResponseModel,counterfactual_loss

torch.set_num_threads(2)


def test_real_harmonics_analytic_axes_and_addition_theorem():
    sun=torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.]])
    y=real_spherical_harmonics(sun)
    assert y.shape==(3,36)
    torch.testing.assert_close(y[:,0],torch.full((3,),1/math.sqrt(4*math.pi)))
    torch.testing.assert_close(y[:,:4],torch.tensor([[1/math.sqrt(4*math.pi),0,0,-math.sqrt(3/(4*math.pi))],[1/math.sqrt(4*math.pi),-math.sqrt(3/(4*math.pi)),0,0],[1/math.sqrt(4*math.pi),0,math.sqrt(3/(4*math.pi)),0]]),atol=1e-6,rtol=0)
    for degree in range(6):torch.testing.assert_close(y[:,degree*degree:(degree+1)**2].square().sum(-1),torch.full((3,),(2*degree+1)/(4*math.pi)),atol=2e-6,rtol=0)


def model():
    torch.manual_seed(20261101);return LightingResponseModel(CompactDescriptor(False))


def test_total_capacity_includes_frozen_backbone_and_all_heads():
    m=model();assert m.total_parameters==95808
    assert sum(p.numel()for p in m.parameters())<100000
    assert all(not p.requires_grad for p in m.coarse.parameters())
    assert all(p.requires_grad for p in m.fine.parameters())
    m.train();assert not m.coarse.training


def test_constant_initial_response_equals_fine_descriptor_and_pooled_control():
    m=model().eval();patch=torch.rand(3,1,33,33);condition=torch.rand(3,9);sun=torch.eye(3)
    with torch.inference_mode():
        d,_=m.fine(patch);a=m(patch,condition,sun);b=m(patch,condition,-sun);c=m(patch,condition,sun,True)
    torch.testing.assert_close(a,d[:,:,16,16],atol=1e-6,rtol=0)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=0);torch.testing.assert_close(a,c,atol=1e-6,rtol=0)


def test_response_can_encode_direction_while_pooling_removes_target_direction():
    m=model();coeff=torch.zeros(2,36,32);coeff[:,0,0]=2;coeff[:,3,1]=1
    suns=torch.tensor([[1.,0,0],[-1.,0,0]])
    a=m.decode(coeff,suns);b=m.decode(coeff,suns,True)
    assert a[0,1]<0<a[1,1];torch.testing.assert_close(b[0],b[1])


def training_fixture():
    m=model();p=torch.rand(8,1,33,33);c=torch.rand(8,9)
    sun=torch.tensor([[[1.,0,1],[0,1.,1]]]).expand(8,-1,-1).clone()
    available=torch.tensor([[True,False]]).expand(8,-1).clone()
    teacher=torch.full((8,2,32),float('nan'));teacher[:,0]=torch.rand(8,32)
    pairs=[[0,1],[2,3],[4,5],[6,7]]
    return m,p,c,sun,teacher,available,pairs


def test_training_gradient_is_finite_and_never_updates_original_coarse():
    m,p,c,s,t,a,pairs=training_fixture();before={k:v.clone()for k,v in m.coarse.state_dict().items()}
    opt=torch.optim.AdamW([v for v in m.parameters()if v.requires_grad],lr=1e-4)
    loss,report=counterfactual_loss(m,p,c,s,t,a,pairs);loss.backward()
    assert np.isfinite(list(report.values())).all()
    assert all(torch.isfinite(v.grad).all()for v in m.parameters()if v.grad is not None)
    opt.step();assert all(torch.equal(v,before[k])for k,v in m.coarse.state_dict().items())


def test_finite_heldout_teacher_target_is_rejected():
    m,p,c,s,t,a,pairs=training_fixture();t[:,1]=0
    with pytest.raises(ValueError,match='Heldout'):counterfactual_loss(m,p,c,s,t,a,pairs)


def test_no_common_training_light_or_different_canonical_frames_rejected():
    m,p,c,s,t,a,pairs=training_fixture();s[1,0]=torch.tensor([0.,1,0])
    with pytest.raises(ValueError,match='same target Sun frame'):counterfactual_loss(m,p,c,s,t,a,pairs)
