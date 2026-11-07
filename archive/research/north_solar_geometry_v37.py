"""Lunar-envelope-consistent solar directions, not dated SPICE ephemerides.

Use a fixed subsolar latitude +1.5 degrees and six predetermined local hour
angles. Derive each site's ENU direction from one Moon-body solar vector;
elevation and azimuth cannot be chosen independently at polar latitudes.
"""
import hashlib
import json
import math
from datetime import datetime,timedelta
from pathlib import Path

from north_development_plan_v35 import development_plan

SUBSOLAR_LATITUDE_DEG=1.5
HOUR_ANGLES_DEG=(-80.,-50.,-20.,20.,50.,80.)


def solar_configuration(site,index):
    if index not in range(6):raise ValueError('Exactly the six predefined Sun indices required')
    latitude=math.radians(site['centre_latitude_deg']);longitude=math.radians(site['centre_longitude_deg'])
    declination=math.radians(SUBSOLAR_LATITUDE_DEG);hour=math.radians(HOUR_ANGLES_DEG[index])
    solar_longitude=longitude-hour
    body=[math.cos(declination)*math.cos(solar_longitude),
          math.cos(declination)*math.sin(solar_longitude),math.sin(declination)]
    local=[-math.cos(declination)*math.sin(hour),
           math.cos(latitude)*math.sin(declination)-math.sin(latitude)*math.cos(declination)*math.cos(hour),
           math.sin(latitude)*math.sin(declination)+math.cos(latitude)*math.cos(declination)*math.cos(hour)]
    elevation=math.degrees(math.asin(local[2]));azimuth=math.degrees(math.atan2(local[0],local[1]))%360
    if elevation<=0:raise ValueError('Declared daytime configuration below mathematical horizon')
    return dict(light=index,subsolar_latitude_deg=SUBSOLAR_LATITUDE_DEG,
        local_hour_angle_deg=HOUR_ANGLES_DEG[index],solar_body_direction=body,
        solar_enu_direction=local,elevation_deg=elevation,azimuth_deg=azimuth)


def write_contract(output):
    plan,sites=development_plan();out=Path(output)
    if out.exists():raise FileExistsError('Preserve earlier solar contracts')
    stopped=json.loads(Path('reports/north_v35_render_stop_v37.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    contract=dict(scope='Amended primary northern development illumination, selected by celestial geometry before matcher/fitting outcomes. Earlier v31/v35angles remain preserved stress settings.',
        source_plan_sha256=sha('reports/north_render_prospective_plan_v31.json'),
        solar_source_sha256=sha(__file__),diagnostic_sha256=sha('reports/north_solar_geometry_diagnostic_v37.json'),
        stop_record_sha256=sha('reports/north_v35_render_stop_v37.json'),
        acquired_data_summary_sha256=sha('data/north-render-v31/development-v35/summary.json'),
        subsolar_latitude_deg=SUBSOLAR_LATITUDE_DEG,hour_angles_deg=list(HOUR_ANGLES_DEG),
        development_solar_configurations={s['component']:[solar_configuration(s,i)for i in range(6)]for s in sites},
        prospective_all44_solar_configurations={s['component']:[solar_configuration(s,i)for i in range(6)]for s in plan['sites']},
        remaining_cumulative_render_allowance_seconds=stopped['remaining_cumulative_render_allowance_seconds'],
        original_total_render_allowance_seconds=21600,
        shared_absolute_render_deadline_utc=(datetime.fromisoformat(json.loads(Path('runs/north-v35-development/protocol.json').read_text())['started_utc'])+timedelta(seconds=21600)).isoformat(),
        unchanged_components=True,unchanged_camera_geometry=True,
        unchanged_primary_lighting_index_pairs=plan['primary_lighting_pairs'],model_outcomes_used=False,
        calibration_or_final_pixels_decoded=0,
        primary_source='https://ntrs.nasa.gov/api/citations/20230000854/downloads/Minow_SciTech_2023%20-%20presentation%20-%20final.pdf',
        limits=['This satisfies an idealized solar-centre latitude envelope, not a dated SPICE trajectory or complete ephemeris validation.',
                'Uniform diffuse material, constant weak ambient, point Sun and finite terrain halos remain simplified photometric assumptions.',
                'No source height, camera, component assignment, matching metric or case index was selected by model performance.'])
    out.write_text(json.dumps(contract,indent=2)+'\n');return contract


if __name__=='__main__':write_contract('reports/north_solar_amendment_v37.json')
