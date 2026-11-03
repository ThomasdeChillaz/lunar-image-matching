"""Procedural descriptor-coordinate invariance, not a learned lunar result."""
import math
import numpy as np
import torch
from compact_lunar import CompactDescriptor
from solar_response_v63 import LightingResponseModel
from solar_frontend_v63 import ReferenceGeometry,reference_anchors,response_evidence
from lunar_geometry_v5 import world_points_z


def test_identical_canonical_observations_match_after_decoder_feature_basis_changes():
    torch.manual_seed(20261101);model=LightingResponseModel(CompactDescriptor(False)).eval()
    q=torch.linalg.qr(torch.randn(32,32)).Q
    with torch.no_grad():model.coefficients.weight[:32].copy_(q*math.sqrt(4*math.pi))
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    g=ReferenceGeometry(np.full((512,512),100.),k,np.eye(4),np.tile([0.,0.,-1.],(256,1)),np.array([[0.,0.,-1.]]),np.ones((256,1),np.uint8),np.ones((256,1,7),np.uint8),np.array([0.,0.,-1.]),np.repeat(world_points_z(reference_anchors(),np.full(256,100.),k,np.eye(4))[:,None],7,axis=1));g.query_intrinsics=k
    a=np.random.default_rng(11).integers(0,256,(512,512),dtype=np.uint8);p=reference_anchors()[100]
    score=response_evidence(model,a,a,p,p,g,np.eye(4),100)
    assert score[0]>.999
