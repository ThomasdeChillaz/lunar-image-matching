import numpy as np
import pytest
from solar_data_v65 import schedule
from solar_physics_v63 import lighting_schedule,solar_direction
from north_camera_variation_v59 import rolled_pose


def site():return dict(component='software-fixture',split='training',centre_latitude_deg=80.)


def test_new_schedule_has_four_deterministic_audited_camera_pairs_and_no_reserve_access():
    a=schedule(site(),500.,2);assert a==schedule(site(),500.,2)
    assert [r['family']for r in a]==['training_motion']*2+['unseen_roll']*2
    for row in a:
        for role in('reference','query'):
            v=row[role];np.testing.assert_array_equal(v['cam2world'],rolled_pose(v['position'],v['target'],v['roll_degrees']))
    with pytest.raises(ValueError):schedule(dict(site(),split='final_test'),500.,2)
    assert all('20261104'in __import__('inspect').getsource(schedule)for _ in [0])


def test_physical24_light_schedule_even_train_and_fixed_selection_pairs():
    lights=lighting_schedule(80.);assert len(lights)==24
    assert [r['index']for r in lights if r['training_allowed']]==list(range(0,24,2))
    assert lights[0]['hour_angle_deg']==-86.25 and lights[-1]['hour_angle_deg']==86.25
    for row in lights:np.testing.assert_allclose(row['sun_enu'],solar_direction(80.,row['hour_angle_deg'],1.5),atol=1e-12)
