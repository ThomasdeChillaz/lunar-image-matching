"""Off-axis controls prevent silent ray-range/camera-Z substitution."""
import numpy as np
import pytest

from renderer_depth_v34 import canonical_depth,control_convention


def test_measured_control_is_axial_and_stays_bound_to_original_audit():
    assert control_convention()=='axial'


def test_off_axis_ray_range_differs_from_camera_axis_depth():
    pose=np.eye(4);pose[:3,3]=[1.,2.,3.]
    position=np.array([[[1.,2.,13.],[7.,10.,13.]]])
    axial,ray,error=canonical_depth(position,pose,np.array([[10.,10.]]),'axial')
    np.testing.assert_array_equal(axial,[[10.,10.]])
    np.testing.assert_allclose(ray,[[10.,np.sqrt(200.)]])
    assert error==0
    with pytest.raises(ValueError):canonical_depth(position,pose,np.array([[10.,10.]]),'ray')


def test_camera_rotation_changes_axial_but_not_ray_distance():
    pose=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,10],[0,0,0,1]],float)
    xyz=np.array([[[3.,4.,0.]]]);a,r,_=canonical_depth(xyz,pose,np.array([[10.]]),'axial')
    assert a[0,0]==10 and r[0,0]==np.sqrt(125)


def test_invalid_or_unlabelled_depth_does_not_pass():
    xyz=np.array([[[0.,0.,10.]]]);pose=np.eye(4)
    for depth,convention in [(np.array([[np.nan]]),'axial'),(np.array([[0.]]),'axial'),(np.array([[10.]]),'unspecified')]:
        with pytest.raises(ValueError):canonical_depth(xyz,pose,depth,convention)
