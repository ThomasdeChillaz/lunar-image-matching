import numpy as np
import pytest
from solar_physics_v63 import (SolarState,TerrainEnvelope,solar_direction,lighting_schedule,sun_candidates,classify_solar_ray,solar_log_evidence,soft_shadow_observation)
from north_solar_geometry_v37 import solar_configuration,HOUR_ANGLES_DEG


def test_existing_six_lunar_sun_vectors_are_preserved():
    site=dict(centre_latitude_deg=87.,centre_longitude_deg=30.)
    for i,h in enumerate(HOUR_ANGLES_DEG):np.testing.assert_allclose(solar_direction(87,h),solar_configuration(site,i)['solar_enu_direction'],atol=2e-16,rtol=0)


def test_24lights_even_training_and_daylight_candidate_grid():
    s=lighting_schedule(87);assert len(s)==24
    assert [x['index']for x in s if x['training_allowed']]==list(range(0,24,2))
    assert s[0]['hour_angle_deg']==-86.25 and s[-1]['hour_angle_deg']==86.25
    assert all(np.isclose(np.linalg.norm(x['sun_enu']),1)and x['sun_enu'][2]>0 for x in sun_candidates(87))


def obstacle(origin,ray,maximum):
    if ray[0]<=0:return None
    t=(5-origin[0])/ray[0]
    return t if 0<t<=maximum and origin[2]+t*ray[2]<=4 else None


def test_known_shadow_and_certified_clear_are_distinct_from_nohit_unknown():
    envelope=TerrainEnvelope(-10,10,-10,10,4.)
    shadow=classify_solar_ray([0,0,0],[0,0,1],[1,0,.2],envelope,obstacle)
    clear=classify_solar_ray([0,0,0],[0,0,1],[1,0,2],envelope,obstacle)
    unknown=classify_solar_ray([0,0,0],[0,0,1],[1,0,.2],envelope,lambda *a:None)
    assert shadow['state']==SolarState.SHADOW and clear['state']==SolarState.CLEAR and unknown['state']==SolarState.UNKNOWN


def test_cropped_missing_map_and_invalid_ceiling_never_prove_clear():
    full=TerrainEnvelope(-10,10,-10,10,4.)
    crop=TerrainEnvelope(-2,2,-2,2,4.)
    assert classify_solar_ray([0,0,0],[0,0,1],[1,0,.2],crop,obstacle)['state']==SolarState.UNKNOWN
    assert classify_solar_ray([0,0,0],[0,0,1],[1,0,2],full,lambda *a:None,coverage_limit=lambda *a:1.)['state']==SolarState.UNKNOWN
    assert classify_solar_ray([0,0,0],[0,0,1],[1,0,2],TerrainEnvelope(-10,10,-10,10,4.,False),lambda *a:None)['state']==SolarState.UNKNOWN


def test_ray_points_toward_sun_and_source_offset_avoids_self_intersection():
    seen=[]
    def trace(origin,direction,maximum):seen.append((origin,direction,maximum));return None
    r=classify_solar_ray([0,0,0],[0,0,1],[0,0,1],TerrainEnvelope(-10,10,-10,10,4),trace)
    assert r['state']==SolarState.CLEAR
    assert seen[0][0][2]>0 and seen[0][1][2]==1
    assert classify_solar_ray([0,0,0],[0,0,1],[0,0,-1],TerrainEnvelope(-10,10,-10,10,4),trace)['reason']=='surface_faces_away_from_sun'


def test_unknown_adds_no_solar_evidence_and_ambiguous_observation_is_neutral():
    assert solar_log_evidence([0,0],[.05,.95]).tolist()==[0.,0.]
    np.testing.assert_array_equal(solar_log_evidence([1,2],[.5,.5]),[0.,0.])
    assert solar_log_evidence([2],[.95])[0]>0 and solar_log_evidence([2],[.05])[0]<0
    p,meta=soft_shadow_observation(np.full((64,64),30,np.uint8));assert not meta['informative']and(p==.5).all()


@pytest.mark.parametrize('point,sun', [([0,0,5],[0,0,1]),([0,0,0],[0,0,0]),([np.nan,0,0],[0,0,1])])
def test_invalid_physical_inputs_fail_explicitly(point,sun):
    with pytest.raises(ValueError):classify_solar_ray(point,[0,0,1],sun,TerrainEnvelope(-10,10,-10,10,4),lambda *a:None)
