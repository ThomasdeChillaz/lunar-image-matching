import pytest
pytest.importorskip('torch')
import cv2
import numpy as np
from native_sift_pose_v4 import points
from synthetic import texture


def test_native_sift_coordinates_preserve_known_shift_and_blank_failure():
    a=texture(122,256);matrix=np.array([[1.,0,8],[0,1,-4.]])
    b=cv2.warpAffine(a,matrix,(256,256))
    p,q=points(a,b)
    assert len(p)>=12 and len(np.unique(q,axis=0))==len(q)
    assert np.median(np.linalg.norm(q-p-[8,-4],axis=1))<.2
    blank=np.zeros((128,128),np.uint8)
    assert len(points(blank,blank)[0])==0
