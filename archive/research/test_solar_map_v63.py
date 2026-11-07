import numpy as np
import pytest
from solar_map_v63 import grid_envelope,reference_solar_table
from north_terrain_pilot_v32 import grid_faces
from solar_physics_v63 import SolarState


def test_missing_triangle_never_gets_clear_coverage_certificate():
    y,x=np.mgrid[-100:101:20,-100:101:20];v=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    env=grid_envelope(v,grid_faces(11));assert env.maximum_height==0 and env.xmin<0<env.xmax
    with pytest.raises(ValueError):grid_envelope(v,grid_faces(11)[:-1])


def test_full_map_height_perturbation_reintersects_camera_rays_and_shadow_is_independent_of_intensity():
    y,x=np.mgrid[-100:101:20,-100:101:20];v=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)]);env=grid_envelope(v,grid_faces(11),scene_maximum_height=0.)
    def raycast(o,r,limit):
        if r[2]>=0:return None
        t=-o[2]/r[2]
        return (o+t*r,np.array([0.,0.,1.]))if 0<t<=limit else None
    pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[0,0,100]
    k=np.array([[100.,0,0],[0,100.,0],[0,0,1.]])
    table,reasons=reference_solar_table(np.array([[0.,0.],[2.,3.]]),k,pose,np.array([[0.,0.,1.],[1.,0.,0.]])/np.array([[1],[1]]),env,raycast)
    np.testing.assert_allclose(table['world_variants'][0,:,2],[0,-5,5,-10,10,-20,20])
    assert (table['solar_states'][:,0]==SolarState.CLEAR).all()
    assert (table['solar_states'][:,1]==SolarState.SHADOW).all()
    assert reasons['cleared_supported_terrain_ceiling']==14
