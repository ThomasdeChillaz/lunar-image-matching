import numpy as np
import pytest
from solar_map_v63 import grid_envelope
from solar_physics_v63 import classify_solar_ray,SolarState
from north_terrain_pilot_v32 import grid_faces


def map_vertices(half):
    y,x=np.mgrid[-half:half+1:3j,-half:half+1:3j]
    return np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])


def test_crop_cannot_replace_original_scene_ceiling_with_its_lower_observed_maximum():
    vertices=map_vertices(10.);faces=grid_faces(3)
    no_bound=grid_envelope(vertices,faces)
    retained=grid_envelope(vertices,faces,scene_maximum_height=100.)
    assert not no_bound.ceiling_valid and retained.maximum_height==100.
    sun=np.array([1.,0,.1]);sun/=np.linalg.norm(sun)
    for envelope in(no_bound,retained):
        result=classify_solar_ray([0,0,0],[0,0,1],sun,envelope,lambda *args:None)
        assert result['state']==SolarState.UNKNOWN
    # A full finite scene explicitly bounded at0m can certify vertical clear.
    declared_flat=grid_envelope(vertices,faces,scene_maximum_height=0.)
    assert classify_solar_ray([0,0,0],[0,0,1],[0,0,1],declared_flat,lambda *args:None)['state']==SolarState.CLEAR
    raised=vertices.copy();raised[4,2]=20
    with pytest.raises(ValueError,match='does not bound'):grid_envelope(raised,faces,scene_maximum_height=10.)
