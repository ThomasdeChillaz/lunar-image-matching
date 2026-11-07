import numpy as np
import pytest
from diagnose_solar_candidates_v69 import support,validate_cases
from rotation_bank_pose_v4 import quarter_turn


def test_half_pixel_rotation_centres_and_native_window_support_are_not_interchanged():
    truth=np.array([[100.2,150.1],[32.,32.],[210.,210.]])
    q=np.repeat(np.array([[100.5,150.5],[20.,20.],[190.,210.]])[:,None],8,axis=1)
    r=np.tile(np.arange(8)%4,(3,1));visible=np.array([True,True,False]);out=support(q,r,truth,visible)
    assert out['top1_inside'].tolist()==[True,False,False]
    # Correctness requires usable57pxnativecontext, not merely nearby GT.
    assert not out['top8_inside'][1]
    for i in range(3):
        for k in range(4):
            native=quarter_turn(quarter_turn(q[i,:1],k,512),k,512,True)
            np.testing.assert_allclose(native,q[i,:1])


def test_support_never_calls_occluded_points_available_and_validates_schema():
    q=np.full((2,8,2),200.);r=np.zeros((2,8),int);truth=np.full((2,2),200.)
    out=support(q,r,truth,np.array([True,False]));assert out['top8_inside'].tolist()==[True,False]
    with pytest.raises(ValueError):support(q,r,np.zeros((3,2)),np.ones(2,bool))


def test_diagnostic_requires_every_fixed_old_terrain_family_and_light_pair():
    tiles={f'np31-pilot-{i}/{f}-0':dict(split='pilot_camera_control',pair_index=0,family=f)for i in range(4)for f in('training_motion','unseen_roll')}
    cases=[dict(case_id=t+str(b),tile_id=t,light_a=0,light_b=b)for t in tiles for b in(0,3)]
    validate_cases(cases,tiles)
    duplicate=cases.copy();duplicate[-1]=dict(cases[0],case_id='different-id')
    with pytest.raises(ValueError):validate_cases(duplicate,tiles)
    reserve=dict(tiles);reserve[cases[0]['tile_id']]=dict(reserve[cases[0]['tile_id']],split='final_test')
    with pytest.raises(ValueError):validate_cases(cases,reserve)
