import numpy as np
import torch
import pytest
from compact_lunar import CompactDescriptor
from solar_frontend_v63 import ReferenceGeometry,SolarMatcher,candidates,reference_anchors,response_evidence
from solar_response_v63 import LightingResponseModel
from solar_native_v63 import canonical_patch
from localize_terrain import public_result
from lunar_geometry_v5 import world_points_z


def geometry():
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    g=ReferenceGeometry(np.full((512,512),100.),k,np.eye(4),np.tile([0,0,-1.],(256,1)),np.array([[0.,0.,-1.]]),np.ones((256,1),np.uint8),np.ones((256,1,7),np.uint8),np.array([0.,0.,-1.]),np.repeat(world_points_z(reference_anchors(),np.full(256,100.),k,np.eye(4))[:,None],7,axis=1))
    return g,k


def test_reference_archive_forbids_query_truth(tmp_path):
    g,k=geometry();p=tmp_path/'reference.npz';np.savez(p,**{k:getattr(g,k)for k in g.__dataclass_fields__})
    loaded=ReferenceGeometry.load(p);np.testing.assert_array_equal(loaded.depth,g.depth)
    np.savez(p,**{k:getattr(g,k)for k in g.__dataclass_fields__},query_pose=np.eye(4))
    with pytest.raises(ValueError,match='query truth'):ReferenceGeometry.load(p)


def test_candidates_preserve_native_grid_and_query_rotation():
    torch.manual_seed(71);model=CompactDescriptor(False).eval();rng=np.random.default_rng(11)
    a=rng.integers(0,256,(512,512),dtype=np.uint8)
    p,q,s,r=candidates(model,a,a)
    assert p.shape==(256,2)and q.shape==(256,8,2)and s.shape==(256,8)
    np.testing.assert_array_equal(p,reference_anchors())
    assert ((r>=0)&(r<=3)).all();np.testing.assert_allclose(q[:,0],p,atol=1e-6)


def test_blank_never_acquires_patches_or_returns_public_pose():
    g,k=geometry();torch.manual_seed(73);model=CompactDescriptor(False)
    result=SolarMatcher(model,model).localize(np.zeros((512,512),np.uint8),np.zeros((512,512),np.uint8),g,k)
    assert result['pose']is None and result['reason']=='blank_or_uninformative_image'
    published=public_result(dict(pose=np.eye(4),reason=None,sun_enu=[1,0,0],solar_supported=False))
    assert published['pose']is None and not published['accepted']and 'raw_research'not in published
    assert published['estimated_sun_world']is None
    assert public_result(result,True)['raw_research']['pose']is None


def test_canonical_measurement_centres_on_candidate_without_query_truth():
    g,k=geometry();a=np.indices((512,512)).sum(0).astype(np.uint8)
    patch,meta=canonical_patch(a,a,[200.,200.],g.depth,k,g.cam2world,k,g.cam2world,query_center=[210,197])
    np.testing.assert_allclose(meta['query_xy'][16,16],[210,197]);assert patch is not None
    patch,meta=canonical_patch(a,a,[200.,200.],g.depth,k,g.cam2world,k,g.cam2world,query_center=[1,1])
    assert patch is None


def test_response_scoring_uses_canonical_images_and_trial_geometry_only():
    torch.manual_seed(74);model=LightingResponseModel(CompactDescriptor(False)).eval();g,k=geometry();g.query_intrinsics=k
    a=np.random.default_rng(1).integers(0,256,(512,512),dtype=np.uint8);p=reference_anchors()[100]
    score=response_evidence(model,a,a,p,p,g,np.eye(4),100)
    assert score.shape==(1,)and score[0]>.99
    matcher=SolarMatcher(model.coarse,model.fine,model);assert matcher.total_parameters==95808
