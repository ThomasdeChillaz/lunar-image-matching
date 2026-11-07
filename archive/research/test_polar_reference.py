import numpy as np

from polar_reference import camera_points,score_points,visible


def test_external_camera_to_laser_transform_is_inverted_and_stereo_is_composed():
    cal=dict(CAM_LIDAR_TRANSLATION=np.array([1.,2,3]),CAM_LIDAR_ROTATION=np.eye(3),
        ROTATION_MATRIX=np.eye(3),TRANSLATION_VECTOR=np.array([-.3,0,0]))
    world=np.array([[2.,4,8]])
    assert np.allclose(camera_points(world,cal,'L'),[[1,2,5]])
    assert np.allclose(camera_points(world,cal,'R'),[[.7,2,5]])
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    cal['CAM_LIDAR_ROTATION']=rotation
    camera=np.array([[.5,1.,5.]])
    world=camera@rotation.T+cal['CAM_LIDAR_TRANSLATION']
    assert np.allclose(camera_points(world,cal,'L'),camera)


def test_external_reference_assignment_cannot_choose_using_query_prediction():
    a=np.array([[10.,10.],[10.2,10.]])
    b=np.array([[8.,10.],[120.,120.]])
    bins={(20,20):[0,1]}
    score=score_points(np.array([[10.,10.]]),np.array([[120.,120.]]),a,b,bins)
    assert score['laser_scorable_matches']==1
    assert score['nominal_within3px']==0
    assert score['median_nominal_error_px']>100


def test_external_visibility_excludes_occluded_and_out_of_view_points():
    points=np.array([[20.,20.],[20.,20.],[200.,20.],[20.,20.]])
    depth=np.array([1.,2.,1.,-1.])
    assert visible(points,depth).tolist()==[True,False,False,False]
