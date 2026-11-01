import inspect
import numpy as np
import pytest
torch=pytest.importorskip('torch')

from native_refinement_v7 import NativeRefiner,joint_logits,objective,refine


def test_no_match_is_a_competing_class_and_initial_policy_preserves_proposals():
    local=torch.zeros(3,33,33);local[0,16,16]=9;local[1,2,4]=3
    no_match=torch.full((3,),-7.5)
    assert (joint_logits(local,no_match).argmax(1)!=1089).all()
    no_match[2]=8
    assert joint_logits(local,no_match).argmax(1)[2]==1089
    labels=torch.tensor([1.,1.,0.]);targets=torch.tensor([[16.,16.],[4.,2.],[0.,0.]])
    good=objective(local,no_match,targets,labels,True)
    assert objective(local,-no_match,targets,labels,True)>good


def test_native_initial_encoder_matches_deterministic_support_and_peak():
    from localization_diagnostics_v3 import refine_native
    torch.set_num_threads(2);torch.manual_seed(5)
    model=NativeRefiner('encoder_3d');a=torch.nn.functional.normalize(torch.randn(1,32,80,80),dim=1);b=a.clone()
    p=np.array([[12.,20.],[32.5,36.5],[68.,30.],[40.,40.]],np.float32)
    q=np.array([[30.,30.],[32.,36.],[40.,40.],[41.,39.]],np.float32)
    with torch.inference_mode():
        pa,qa,idx=refine_native(p,q,a,b)
        pb,qb,local=refine(model,p,q,a,b)
    np.testing.assert_array_equal(pa,pb);np.testing.assert_array_equal(idx,local['indices'])
    np.testing.assert_allclose(qa,qb,atol=1e-4);assert local['no_match']==0


def test_joint_objective_has_finite_gradients_for_all_model_heads():
    torch.set_num_threads(2);torch.manual_seed(8);model=NativeRefiner('dustbin')
    local,no_match=model(torch.rand(4,1,27,27),torch.rand(4,1,57,57),torch.full((4,2),13.))
    loss=objective(local,no_match,torch.tensor([[16.25,16.5],[0.,0.],[24.,9.],[0.,0.]]),torch.tensor([1.,0.,1.,0.]),True)
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())+28272<100000


def test_native_inference_never_accepts_query_geometry():
    from native_pipeline_v7 import infer
    names=set(inspect.signature(infer).parameters)
    assert 'reference_depth' in names and 'reference_cam2world' in names
    assert not {'query_depth','query_pose','query_cam2world'}&names
