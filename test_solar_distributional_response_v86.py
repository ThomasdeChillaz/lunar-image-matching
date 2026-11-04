"""Draft pure-moment and Torch-autograd controls. NOT run during qualification.

Torch imports exist only inside explicitly named factory/autograd tests.
No dataset, checkpoint, ray caster or held-out lunar source is accessed.
"""
from copy import deepcopy
import hashlib
import inspect

import numpy as np
import pytest

from solar_distributional_response_v86 import (TrainingViewPrior,parameter_arithmetic,
    marginalize_numpy,gaussian_nll_numpy,gaussian_nll_torch,build_distributional_response)


def prior_fixture(constant=False):
    los=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]])
    up=np.tile([0.,0.,1.],(4,1))
    if constant:los[:]=[1.,0.,0.]
    return TrainingViewPrior.from_training_views(los,up,['trainA','trainA','trainB','trainB'],
        allowed_training_groups=['trainA','trainB'],source_sha256={'procedural_training_fixture':'0'*64}),los,up


def test_parameter_arithmetic_default_all_heads_fit_and_rank24_does_not():
    p=parameter_arithmetic()
    assert p['frozen_coarse']==28272 and p['reference_geometry_encoder']==28704
    assert p['observed_condition_film']==1264
    assert p['mean_patch_factors']+p['mean_angular']==20928
    assert p['noise_patch_factors']+p['noise_angular']==15948
    assert p['total_parameters']==95116 and p['trainable_parameters']==66844
    assert p['parameters_plus_fixed_buffers']==95221<100000
    with pytest.raises(ValueError,match='exceeds'):parameter_arithmetic(mean_rank=24)


def test_training_prior_moments_psd_unit_constraints_and_frozen_payload():
    prior,los,up=prior_fixture()
    basis=np.column_stack([np.ones(4),los,up])
    np.testing.assert_allclose(prior.mean,basis.mean(0),atol=1e-14,rtol=0)
    expected=(basis-basis.mean(0)).T@(basis-basis.mean(0))/4
    np.testing.assert_allclose(prior.covariance,expected,atol=1e-14,rtol=0)
    np.testing.assert_allclose(prior.factor@prior.factor.T,prior.covariance,atol=1e-14,rtol=0)
    assert np.linalg.eigvalsh(prior.covariance).min()>=-1e-14
    assert prior.mean[0]==1 and not prior.covariance[0].any()
    loaded=TrainingViewPrior.from_frozen_payload(prior.payload())
    np.testing.assert_allclose(loaded.covariance,prior.covariance,atol=1e-14,rtol=0)
    assert loaded.provenance['split']=='training'


def test_nontraining_sources_bad_directions_and_prior_tampering_rejected():
    prior,los,up=prior_fixture()
    with pytest.raises(ValueError,match='Training'):
        TrainingViewPrior.from_training_views(los,up,['trainA','trainA','final','trainB'],
            allowed_training_groups=['trainA','trainB'],source_sha256={'fixture':'0'*64})
    with pytest.raises(ValueError,match='unit'):
        TrainingViewPrior.from_training_views(los*2,up,['trainA']*4,
            allowed_training_groups=['trainA'],source_sha256={'fixture':'0'*64})
    p=prior.payload();p['covariance'][1][1]+=.1
    with pytest.raises(ValueError,match='bytes changed'):TrainingViewPrior.from_frozen_payload(p)
    p=prior.payload();p['provenance']['split']='calibration'
    with pytest.raises(ValueError,match='training-only'):TrainingViewPrior.from_frozen_payload(p)


@pytest.mark.parametrize('invalid',['negative','asymmetric','constant_variance','nonunit_moments'])
def test_invalid_covariance_is_not_repaired_into_a_prior(invalid):
    prior,_,_=prior_fixture();p=prior.payload();c=np.asarray(p['covariance'])
    if invalid=='negative':c[1,1]=-1.
    elif invalid=='asymmetric':c[1,2]=.1
    elif invalid=='constant_variance':c[0,0]=.1
    else:c[1:4,1:4]*=.5
    p['covariance']=c.tolist()
    p['moment_sha256']=hashlib.sha256(np.asarray(p['mean'],np.float64).tobytes()+c.astype(np.float64).tobytes()).hexdigest()
    with pytest.raises(ValueError):TrainingViewPrior.from_frozen_payload(p)


def test_marginal_moments_match_enumerated_training_views_and_full_diagonal():
    prior,los,up=prior_fixture();rng=np.random.default_rng(86)
    j=rng.normal(size=(3,32,7));d=np.full((3,32),.02);u=rng.normal(0,.1,(3,32,4))
    out=marginalize_numpy(j,d,u,prior)
    basis=np.column_stack([np.ones(4),los,up])
    conditional=np.einsum('bdv,nv->bnd',j,basis)
    mean=conditional.mean(1)
    centred=conditional-mean[:,None]
    expected=u@u.swapaxes(-1,-2)+np.eye(32)*d[:,None,:]+np.einsum('bnd,bne->bde',centred,centred)/4
    np.testing.assert_allclose(out['mean'],mean,atol=1e-12,rtol=0)
    np.testing.assert_allclose(out['covariance'],expected,atol=1e-12,rtol=0)
    np.testing.assert_allclose(out['diagonal'],np.diagonal(out['covariance'],axis1=-2,axis2=-1),atol=1e-12,rtol=0)
    assert (np.linalg.eigvalsh(out['covariance']).min(-1)>=.02-1e-12).all()
    cheap=marginalize_numpy(j,d,u,prior,diagonal_only=True)
    assert cheap['covariance']is None and 'approximation'in cheap['covariance_mode']
    np.testing.assert_array_equal(cheap['mean'],out['mean']);np.testing.assert_array_equal(cheap['diagonal'],out['diagonal'])


def test_cholesky_nll_matches_diagonal_closed_form_and_rejects_invalid_input():
    prior,_,_=prior_fixture(constant=True);rng=np.random.default_rng(2)
    j=rng.normal(size=(2,32,7));d=rng.uniform(.01,.2,(2,32));u=np.zeros((2,32,4))
    out=marginalize_numpy(j,d,u,prior);target=rng.normal(size=(2,32))
    a=gaussian_nll_numpy(target,out['mean'],covariance=out['covariance'])
    b=gaussian_nll_numpy(target,out['mean'],diagonal=out['diagonal'])
    expected=.5*np.sum((target-out['mean'])**2/d+np.log(2*np.pi*d),-1)
    np.testing.assert_allclose(a,expected,atol=1e-10,rtol=0);np.testing.assert_allclose(b,expected,atol=1e-10,rtol=0)
    bad=out['covariance'].copy();bad[0,0,0]=-1
    with pytest.raises(np.linalg.LinAlgError):gaussian_nll_numpy(target,out['mean'],covariance=bad)
    with pytest.raises(ValueError):gaussian_nll_numpy(target,out['mean'],diagonal=-d)
    with pytest.raises(ValueError):gaussian_nll_numpy(target,out['mean'],covariance=out['covariance'],diagonal=d)


def make_torch_model():
    import torch
    from compact_lunar import CompactDescriptor
    torch.manual_seed(86);torch.set_num_threads(2)
    prior,_,_=prior_fixture()
    return torch,build_distributional_response(CompactDescriptor(False).eval().requires_grad_(False),prior)


def test_torch_factory_actual_parameter_count_geometry_initialization_and_frozen_query():
    torch,model=make_torch_model()
    assert sum(p.numel()for p in model.parameters())==95116
    assert model.parameter_report['actual_state_elements']==95221
    assert sum(p.numel()for p in model.parameters()if p.requires_grad)==66844
    torch.testing.assert_close(model.reference_fine.layers[0].weight[:,:13],model.coarse.layers[0].weight,rtol=0,atol=0)
    assert not model.reference_fine.layers[0].weight[:,13:].count_nonzero()
    assert not any(p.requires_grad for p in model.coarse.parameters())
    model.train();assert not model.coarse.training and model.reference_fine.training


def test_torch_native_paths_require_selection_and_coarse_path_cannot_hide512_native_work():
    torch,model=make_torch_model()
    ref=torch.zeros((1,3,33,33));query=torch.zeros((1,1,33,33))
    with pytest.raises(TypeError):model.native_reference_features(ref)
    with pytest.raises(ValueError,match='selection'):model.native_reference_features(ref,native_requires_selected=False)
    with pytest.raises(ValueError,match='selection'):model.native_query_features(query,native_requires_selected=False)
    with pytest.raises(ValueError,match='128'):model.coarse_reference_features(torch.zeros(1,3,512,512),torch.zeros(1,1,2))
    with pytest.raises(ValueError,match='128'):model.coarse_query_features(torch.zeros(1,1,512,512),torch.zeros(1,1,2))
    assert model.native_reference_features(ref,native_requires_selected=True).shape==(1,32)
    assert model.native_query_features(query,native_requires_selected=True).shape==(1,32)
    points=torch.tensor([[[32.,32.],[64.,64.]]])
    assert model.coarse_reference_features(torch.zeros(1,3,128,128),points).shape==(1,2,32)
    assert model.coarse_query_features(torch.zeros(1,1,128,128),points).shape==(1,2,32)


def test_torch_marginal_mean_covariance_nll_and_autograd_are_not_posthoc_confidence():
    torch,model=make_torch_model()
    base=torch.randn(3,32,requires_grad=True)
    condition=torch.tensor([[1.,0.,0.,0.,0.,1.,0.,0.,-1.,0.]]).repeat(3,1)
    suns=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    out=model.marginal_distribution(base,condition,suns)
    j=out['view_jacobian']
    torch.testing.assert_close(out['mean'],j@model.view_mean)
    expected=torch.diag_embed(out['noise_diagonal'])+out['noise_low_rank']@out['noise_low_rank'].transpose(-1,-2)+j@model.view_covariance@j.transpose(-1,-2)
    torch.testing.assert_close(out['covariance'],expected,atol=1e-6,rtol=1e-5)
    assert (torch.linalg.eigvalsh(out['covariance']).min(-1).values>0).all()
    target=torch.nn.functional.normalize(torch.randn(3,32),dim=-1)
    loss=gaussian_nll_torch(target,out['mean'],covariance=out['covariance']).mean()
    assert torch.isfinite(loss);loss.backward()
    for value in (base,model.mean_factors.weight,model.mean_angular,model.noise_factors.weight,model.noise_angular,model.observed_film[-1].weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert all(p.grad is None for p in model.coarse.parameters())
    cheap=model.marginal_distribution(base.detach(),condition,suns,diagonal_only=True)
    torch.testing.assert_close(cheap['diagonal'],out['covariance'].diagonal(dim1=-2,dim2=-1))
    assert cheap['covariance']is None
    # The pre-pose marginal API does not accept a target/query view argument.
    assert 'target_view_basis'not in inspect.signature(model.marginal_distribution).parameters


def test_torch_explicit_training_view_has_no_extra_marginal_variance():
    torch,model=make_torch_model()
    base=torch.zeros(2,32);condition=torch.tensor([[1.,0.,0.,0.,0.,1.,0.,0.,-1.,0.]]).repeat(2,1)
    suns=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    view=torch.tensor([[1.,1.,0.,0.,0.,0.,1.]]).repeat(2,1)
    out=model.distribution(base,condition,suns,target_view_basis=view)
    torch.testing.assert_close(out['mean'],torch.einsum('bdv,bv->bd',out['view_jacobian'],view))
    assert not out['view_factor'].count_nonzero()
    assert 'explicit'in out['view_mode']


def test_torch_loss_reaches_reference_encoder_but_never_frozen_query_encoder():
    torch,model=make_torch_model()
    # Pure deterministic tensors, with no source image/checkpoint/data access.
    intensity=torch.linspace(0.,1.,128*128).reshape(1,1,128,128)
    field=torch.cat([intensity,torch.zeros(1,2,128,128)],1)
    points=torch.tensor([[[48.,48.],[72.,72.]]])
    base=model.coarse_reference_features(field,points)[0]
    target=model.coarse_query_features(1-intensity,points)[0]
    condition=torch.tensor([[1.,0.,0.,0.,0.,1.,0.,0.,-1.,0.]]).repeat(2,1)
    out=model.marginal_distribution(base,condition,torch.tensor([[1.,0.,0.],[0.,1.,0.]]))
    gaussian_nll_torch(target,out['mean'],covariance=out['covariance']).mean().backward()
    gradients=[p.grad for p in model.reference_fine.parameters()]
    assert all(g is not None and torch.isfinite(g).all()for g in gradients)
    assert sum(float(g.abs().sum())for g in gradients)>0
    assert all(p.grad is None for p in model.coarse.parameters())
