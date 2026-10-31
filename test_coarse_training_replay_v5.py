import numpy as np
import pytest
from verify_coarse_training_v5 import replay_row


def test_saved_pose_audit_detects_wrong_range_and_false_success_labels():
    k=np.array([[50.,0,15.5],[0,50.,15.5],[0,0,1.]])
    camera=dict(intrinsics=k,cam2world=np.eye(4));tile=dict(cameras=[camera,camera])
    depth=np.full((32,32),10.);data=dict(depth_a=depth,depth_b=depth)
    p=np.array([[8.,8.],[24.,8.],[8.,24.],[24.,24.]])
    y,x=np.mgrid[:32,:32];distance=float(np.median(10*np.sqrt(1+((x-15.5)/50)**2+((y-15.5)/50)**2)))
    row=dict(reference_points_native=p.tolist(),query_points_native=p.tolist(),target_points_native=p.tolist(),
        visible=[True]*4,correct_at_native_px={str(t):4 for t in (1,2,4,8,12)},returned=4,scorable=4,
        reference_range_m=distance,estimated_cam2world=np.eye(4).tolist(),pose_error=dict(rotation_deg=0.,translation_m=0.),
        pose_correct=True,strict_pose_correct=True,acceptance_policy=None)
    replay_row(row,tile,data)
    with pytest.raises(ValueError,match='Physical reference range'):replay_row(dict(row,reference_range_m=10.),tile,data)
    with pytest.raises(ValueError,match='Pose criterion'):replay_row(dict(row,estimated_cam2world=None),tile,data)
    with pytest.raises(ValueError,match='Correspondence precision'):replay_row(dict(row,query_points_native=(p+20).tolist()),tile,data)
