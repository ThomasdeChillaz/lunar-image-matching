import inspect
import numpy as np
import pytest
pytest.importorskip('torch')

from native_baseline_pose_v7 import propose_pose,score


def test_common_pose_uses_reference_geometry_and_native_coordinate_centers():
    names=set(inspect.signature(propose_pose).parameters)
    assert not {'query_depth','query_pose','query_cam2world'}&names
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    yy,xx=np.mgrid[48:464:32,48:464:32];p=np.column_stack([xx.ravel(),yy.ravel()]).astype(float);q=p+[8,-4]
    pose,_=propose_pose(p,q,np.full((512,512),10.),k,np.eye(4),k)
    assert pose is not None;np.testing.assert_allclose(pose[:3,3],[-.2,.1,0],atol=1e-5)
    a=dict(intrinsics=k,cam2world=np.eye(4));b=dict(intrinsics=k,cam2world=pose)
    row=score(p,q,pose,dict(depth_a=np.full((512,512),10.),depth_b=np.full((512,512),10.)),[a,b])
    assert row['pose_correct'] and row['strict_pose_correct'] and row['correct_at_native_px']['1']==len(p)
    failed=score(p,q,None,dict(depth_a=np.full((512,512),10.),depth_b=np.full((512,512),10.)),[a,b])
    assert not failed['pose_correct'] and failed['estimated_cam2world'] is None
