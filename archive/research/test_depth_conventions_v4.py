import pytest
pytest.importorskip('torch')
import cv2
import numpy as np
from audit_depth_conventions_v4 import closure


def test_depth_closure_distinguishes_ray_range_from_optical_z():
    k=np.array([[400.,0,128],[0,400.,128],[0,0,1.]])
    a=np.eye(4);b=np.eye(4);b[:3,:3]=cv2.Rodrigues(np.array([.07,-.03,.02]))[0];b[:3,3]=[5,1,.5]
    y,x=np.mgrid[:256,:256];rays=np.stack([(x-128)/400,(y-128)/400,np.ones(x.shape)],axis=-1)
    def depth(pose,ray):
        direction=rays@pose[:3,:3].T
        z=(100-pose[2,3])/direction[:,:,2]
        return z*np.linalg.norm(rays,axis=2) if ray else z
    for mode in ('ray','z'):
        da,db=[depth(p,mode=='ray') for p in (a,b)]
        correct=closure(da,db,k,a,k,b,mode)
        wrong=closure(da,db,k,a,k,b,'z' if mode=='ray' else 'ray')
        assert correct['projected_inside']>500
        assert correct['median_absolute_m']<.002
        assert wrong['median_absolute_m']>20*correct['median_absolute_m']
