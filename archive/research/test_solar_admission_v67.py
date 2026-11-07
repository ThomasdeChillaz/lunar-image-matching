from copy import deepcopy
from datetime import datetime,timezone,timedelta
import pytest
from render_solar_development_v67 import admit_unit


def protocol():
    now=datetime(2026,11,3,12,tzinfo=timezone.utc)
    p=dict(phase='new_development_acquisition',compute_admitted=True,calibration_or_final_access=False,shared_cap_seconds=36000,absolute_deadline_utc=(now+timedelta(hours=10)).isoformat(),factor_two_projection_seconds=34000,units=[dict(site=dict(split='training'),calibration_or_final_access=False,max_calendar_seconds=900,cpu_threads=2)])
    return p,now


def test_admission_uses_shared_deadline_instead_of_resetting_on_later_unit():
    p,now=protocol();unit,deadline=admit_unit(p,0,now+timedelta(hours=9))
    assert deadline==now+timedelta(hours=10)
    with pytest.raises(TimeoutError):admit_unit(p,0,now+timedelta(hours=10))


def test_reserve_access_and_unaffordable_or_undeclared_units_are_rejected():
    p,now=protocol();bad=deepcopy(p);bad['units'][0]['site']['split']='final_test'
    with pytest.raises(ValueError):admit_unit(bad,0,now)
    bad=deepcopy(p);bad['factor_two_projection_seconds']=36001
    with pytest.raises(ValueError):admit_unit(bad,0,now)
    with pytest.raises(ValueError):admit_unit(p,1,now)
