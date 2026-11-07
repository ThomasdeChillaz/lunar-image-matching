import pytest
pytest.importorskip('torch')
import numpy as np
from rotation_bank_pose_v4 import quarter_turn,choose_branch


def test_quarter_turn_native_pixel_centers_match_numpy_and_invert():
    image=np.arange(49).reshape(7,7);p=np.array([[0.,0],[2,5],[6,1]])
    for k in range(4):
        q=quarter_turn(p,k,7);rotated=np.rot90(image,k)
        assert np.array_equal(rotated[q[:,1].astype(int),q[:,0].astype(int)],image[p[:,1].astype(int),p[:,0].astype(int)])
        np.testing.assert_allclose(quarter_turn(q,k,7,True),p)
    fractional=np.array([[.5,1.5],[5.2,3.3]])
    np.testing.assert_allclose(quarter_turn(quarter_turn(fractional,1,7),1,7,True),fractional)


def test_rotation_selection_never_uses_correctness_label():
    items=[dict(pose=np.eye(4),quarter_turn=0,info=dict(inliers=30),median_inlier_residual=1.,pose_correct=True),
           dict(pose=np.eye(4),quarter_turn=1,info=dict(inliers=40),median_inlier_residual=2.,pose_correct=False)]
    assert choose_branch(items)['quarter_turn']==1
    assert choose_branch([]) is None
