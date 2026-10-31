import cv2
import numpy as np
import pytest

from localization_robust_pose_v4 import CONFIGS, estimate


@pytest.mark.parametrize('config',CONFIGS,ids=lambda c:c['name'])
def test_robust_pose_known_3d_geometry_and_outliers(config):
    rng=np.random.default_rng(52)
    world=rng.uniform([-4,-4,8],[4,4,20],(140,3))+[1200000,-900000,40000]
    center=np.array([1200000,-900000,40000.])
    k=np.array([[380,0,256],[0,380,256],[0,0,1.]])
    rv=np.array([.04,-.02,.01]);tv=np.array([.2,-.1,.5])
    q=cv2.projectPoints(world-center,rv,tv,k,None)[0].reshape(-1,2)
    q[:35]=rng.uniform(0,512,(35,2));q[35:]+=rng.normal(0,.05,(105,2))
    pose,info=estimate(world,q,k,config)
    assert pose is not None and info['inliers']>=95
    expected=np.eye(4);expected[:3,:3]=cv2.Rodrigues(rv)[0].T
    expected[:3,3]=center-expected[:3,:3]@tv
    assert np.linalg.norm(pose[:3,3]-expected[:3,3])<.02
    assert np.linalg.norm(pose[:3,:3]-expected[:3,:3])<.005


def test_robust_pose_invalid_and_degenerate_inputs():
    k=np.eye(3)
    assert estimate(np.empty((0,3)),np.empty((0,2)),k,CONFIGS[1])[0] is None
    assert estimate(np.zeros((10,3)),np.zeros((10,2)),k,CONFIGS[1])[1]['reason']=='degenerate_geometry'
    with pytest.raises(ValueError):estimate(np.zeros((8,3)),np.zeros((9,2)),k,CONFIGS[1])
