import cv2
import numpy as np

from pose_consensus import bilinear, camera_pose, measure, pose_parameters, refine
from smoke_v126 import generated


def test_coordinate_conversion_and_native_interpolation():
    pose=np.eye(4);pose[:3,3]=[10.,-20.,2000.]
    center=np.array([100.,200.,-30.])
    assert np.allclose(camera_pose(pose_parameters(pose,center),center),pose)
    image=np.arange(16,dtype=float).reshape(4,4)
    assert np.allclose(bilinear(image,np.array([[1.5,1.5]])),[7.5])


def test_shared_camera_optimization_recovers_generated_motion_without_query_truth():
    cv2.setNumThreads(1)
    args,_,truth=generated(False)
    reference,depth,k,old_pose,query,qk,_=args
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=2000.
    expected=pose@np.linalg.inv(old_pose)@truth
    hypotheses=refine(reference,query,depth,k,pose,qk,pose[None])
    assert hypotheses and np.linalg.norm(hypotheses[0]["pose"][:3,3]-expected[:3,3])<2.
    assert np.linalg.norm(hypotheses[0]["pose"][:3,:3]-expected[:3,:3])<.01
    measured=measure(reference,query,depth,k,pose,qk,hypotheses[0]["pose"][None],
                     reference_valid=np.ones(reference.shape,bool))
    for name,row in measured.items():
        assert row["pose"] is not None
        assert np.linalg.norm(np.asarray(row["pose"])[:3,3]-expected[:3,3])<2.


def test_blank_radiance_cannot_support_a_camera_hypothesis():
    args,_,_=generated(False)
    reference,depth,k,pose,query,qk,_=args
    assert refine(np.ones_like(reference),np.ones_like(query),depth,k,pose,qk,pose[None])==[]
