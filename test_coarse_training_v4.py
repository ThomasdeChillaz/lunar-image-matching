import numpy as np
import pytest
pytest.importorskip('torch')


def test_affine_supervision_preserves_native_coordinate_transform():
    from coarse_training_v4 import transform_points
    matrix=np.array([[0,-1,127],[1,0,0.]])
    p=np.array([[12.,20.],[60,80]])
    np.testing.assert_allclose(transform_points(p,matrix),[[107,12],[47,60]])


def test_true_3d_objective_has_finite_gradients_with_close_points():
    torch=pytest.importorskip('torch')
    from coarse_training_v4 import objective
    from compact_lunar import CompactDescriptor
    torch.set_num_threads(2);model=CompactDescriptor(False)
    image=torch.rand(2,1,128,128)
    p=torch.tensor([[20.,20.],[24.,20.],[60.,60.],[80.,80.]])
    q=torch.tensor([[22.,20.],[22.5,20.],[61.,60.],[80.,81.]])
    loss=objective(model,image,p,q);loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_uniform_3d_supervision_projects_a_known_plane_and_skips_selection_pixels():
    from coarse_training_v4 import build_supervision
    k=np.array([[618.,0,256],[0,618.,256],[0,0,1.]])
    a=np.eye(4);b=np.eye(4);b[0,3]=5
    y,x=np.mgrid[:512,:512];depth=100*np.sqrt(1+((x-256)/618)**2+((y-256)/618)**2)
    tiles={'train':dict(split='train',cameras=[dict(intrinsics=k,cam2world=a),dict(intrinsics=k,cam2world=b)]),
           'not_loaded':dict(split='model_selection')}
    data={'train':dict(depth_a=depth,depth_b=depth,images=np.zeros((6,512,512),np.uint8))}
    supervision,counts=build_supervision({},tiles,data)
    z=supervision['train'];valid=z['visible']
    assert counts['tiles']==1 and valid.sum()>400
    # The existing frozen geometry convention samples reference ray-depth at
    # the nearest pixel, even for fractional reference coordinates. Verify it
    # independently instead of pretending this is continuous-plane depth.
    native=(z['p'][valid]+.5)*4-.5;index=np.rint(native).astype(int)
    ray_norm=np.sqrt(1+((native[:,0]-256)/618)**2+((native[:,1]-256)/618)**2)
    sampled_z=depth[index[:,1],index[:,0]]/ray_norm
    expected=z['p'][valid].copy();expected[:,0]-=618*5/sampled_z/4
    np.testing.assert_allclose(z['q'][valid],expected,atol=1e-5)
    assert np.abs(z['q'][valid]-(z['p'][valid]-[618*5/100/4,0])).max()<.004
