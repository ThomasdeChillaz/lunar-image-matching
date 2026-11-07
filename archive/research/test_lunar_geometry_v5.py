import cv2
import numpy as np

from lunar_geometry_v5 import world_points_z,project_z,map_points_z,correspondences_z,median_reference_range_z


def test_axial_unprojection_and_ray_range_are_distinct():
    k=np.array([[100.,0,50],[0,100,50],[0,0,1.]])
    p=np.array([[50.,50.],[150,50]])
    world=world_points_z(p,np.array([10.,10.]),k,np.eye(4))
    np.testing.assert_allclose(world,[[0,0,10],[10,0,10]])
    q,z,front=project_z(world,k,np.eye(4))
    np.testing.assert_allclose(q,p);np.testing.assert_allclose(z,[10,10]);assert front.all()


def test_axial_plane_correspondence_with_known_camera_baseline_and_occlusion():
    k=np.array([[80.,0,32],[0,80,32],[0,0,1.]])
    a=np.eye(4);b=np.eye(4);b[0,3]=1
    depth=np.full((64,64),10.)
    p=np.array([[32.,32.],[40.5,40.5]])
    ref,q,world,visible=correspondences_z(depth,depth,k,a,k,b,points=p,relative_tolerance=0,absolute_tolerance=1e-6)
    np.testing.assert_allclose(q,p-[8,0]);assert visible.all()
    blocked=depth.copy();blocked[31:34,23:26]=5
    assert not correspondences_z(depth,blocked,k,a,k,b,points=p,relative_tolerance=0,absolute_tolerance=1e-6)[3][0]
    assert median_reference_range_z(depth,k)>10


def test_axial_mapping_rejects_invalid_depth_and_preserves_fractional_pixels():
    k=np.eye(3);depth=np.ones((4,4));depth[0,0]=np.inf
    world,valid=map_points_z([[0,0],[1.5,1.5],[8,8]],depth,k,np.eye(4))
    assert valid.tolist()==[False,True,False]
    np.testing.assert_allclose(world,[[1.5,1.5,1.]])
