import numpy as np
import torch
from compact_lunar import CompactDescriptor
from solar_native_v63 import SparseNativeRefiner,canonical_patch
from localization_diagnostics_v3 import refine_native

torch.set_num_threads(2)


def test_sparse_integer_crop_has_bounded_full_map_agreement_and_cache_changes():
    torch.manual_seed(20261101);encoder=CompactDescriptor(False).eval()
    rng=np.random.default_rng(73);a=rng.integers(0,256,(96,96),dtype=np.uint8);b=np.roll(a,2,axis=1)
    p=np.array([[49.5,49.5]]);q=np.array([[50.5,49.5]])
    with torch.inference_mode():da,_=encoder(torch.from_numpy(a).float()[None,None]/255);db,_=encoder(torch.from_numpy(b.copy()).float()[None,None]/255)
    _,expected,_=refine_native(p,q,da,db)
    sparse=SparseNativeRefiner(encoder);actual,meta=sparse.refine(a,b,p[0],q[0]);np.testing.assert_allclose(actual,expected[0],atol=1e-3,rtol=0)
    # Different crop/global normalized-grid arithmetic perturbs a shallow
    # quadratic peak by0.0004px in this fixture. Not bit-exact legacy replay.
    local_ref=sparse.encode(a[36:63,36:63])
    torch.testing.assert_close(local_ref[:,:,12:15,12:15],da[:,:,48:51,48:51],atol=2e-6,rtol=0)
    assert sparse.forward_calls==2;sparse.refine(a,b,p[0],q[0]);assert sparse.forward_calls==2
    with torch.no_grad():next(encoder.parameters()).add_(.01)
    sparse.refine(a,b,p[0],q[0]);assert sparse.forward_calls==4
    bad,reason=sparse.refine(a,b,[1,1],[50,50]);assert bad is None and reason['reason']=='incomplete_native_patch_support'


def test_canonical_surface_projection_uses_only_reference_geometry_and_trial_pose():
    import cv2
    rng=np.random.default_rng(2026);a=rng.integers(0,256,(512,512),dtype=np.uint8)
    q=cv2.warpAffine(a,np.array([[1.,0,8.],[0,1.,-4.]]),(512,512))
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    ref=np.eye(4);trial=np.eye(4);trial[:3,3]=[-.2,.1,0]
    patches,meta=canonical_patch(a,q,[200,200],np.full(a.shape,10.),k,ref,k,trial)
    assert patches is not None and meta['reason']is None
    np.testing.assert_array_equal(*patches)
    np.testing.assert_allclose(meta['query_xy']-meta['reference_xy'],np.broadcast_to([8,-4],(33,33,2)),atol=1e-5)
