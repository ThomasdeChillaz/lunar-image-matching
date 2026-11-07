"""Independent body-to-ENU and solar-envelope checks at all prospective sites."""
import math
import numpy as np

from north_development_plan_v35 import development_plan
from north_render_plan_v31 import enu_frame
from north_solar_geometry_v37 import solar_configuration,HOUR_ANGLES_DEG


def test_body_vectors_reproduce_local_directions_and_lunar_solar_latitude():
    plan,_=development_plan()
    for site in plan['sites']:
        _,rows=enu_frame(site['x'],site['y'],0.)
        for light in range(6):
            s=solar_configuration(site,light);body=np.asarray(s['solar_body_direction'])
            np.testing.assert_allclose(np.asarray(rows)@body,s['solar_enu_direction'],rtol=0,atol=2e-14)
            assert abs(np.linalg.norm(body)-1)<1e-14
            assert abs(math.degrees(math.asin(body[2]))-1.5)<1e-12
            assert s['elevation_deg']>0


def test_hour_angles_are_fixed_before_images_or_model_results():
    plan,_=development_plan()
    for site in plan['sites']:
        for light,hour in enumerate(HOUR_ANGLES_DEG):
            s=solar_configuration(site,light);x,y,z=s['solar_body_direction']
            difference=(site['centre_longitude_deg']-math.degrees(math.atan2(y,x))+180)%360-180
            assert abs(difference-hour)<1e-12


def test_old_independent_angle_choices_can_violate_lunar_envelope():
    plan,_=development_plan();site=plan['sites'][0];phi=math.radians(site['centre_latitude_deg'])
    # A northward Sun5degrees above the horizontal implies a much larger
    # positive body latitude than the Moon's roughly1.5degree solar envelope.
    latitude=math.degrees(math.asin(math.sin(phi)*math.sin(math.radians(5))+math.cos(phi)*math.cos(math.radians(5))))
    assert latitude>5 and abs(solar_configuration(site,0)['subsolar_latitude_deg'])<1.6
