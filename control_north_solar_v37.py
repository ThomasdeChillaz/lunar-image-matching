"""Check actual Blender Sun orientation against the frozen lunar-body vectors."""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0,str(Path.cwd()))
from north_development_plan_v35 import development_plan
from north_render_plan_v31 import enu_frame
from north_solar_geometry_v37 import solar_configuration
from north_terrain_pilot_v32 import sha


def control(output):
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve earlier actual Sun controls')
    if bpy.app.version[:3]!=(4,5,9)or bpy.app.build_hash.decode()!='8bf95cbd38d1':raise ValueError('Pinned renderer required')
    _,sites=development_plan();bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.light_add(type='SUN');sun=bpy.context.object;records=[]
    for site in sites:
        _,rows=enu_frame(site['x'],site['y'],0.);rows=np.asarray(rows)
        for index in range(6):
            solar=solar_configuration(site,index);toward=Vector(solar['solar_enu_direction'])
            sun.rotation_euler=(-toward).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()
            emitted=np.asarray(sun.matrix_world.to_3x3()@Vector((0.,0.,-1.)))
            maximum=float(np.linalg.norm(emitted+np.asarray(toward)))
            toward_body=rows.T@(-emitted);toward_body/=np.linalg.norm(toward_body)
            actual_latitude=math.degrees(math.asin(toward_body[2]))
            if maximum>2e-6 or abs(actual_latitude-1.5)>.0002:
                raise ValueError('Actual lamp direction disagrees with body-coordinate solar contract')
            records.append(dict(component=site['component'],light=index,
                emission_vector_error=maximum,actual_implied_solar_body_latitude_deg=actual_latitude))
    result=dict(passed=True,actual_configurations=len(records),records=records,
        source_sha256={p:sha(p)for p in ['control_north_solar_v37.py','north_solar_geometry_v37.py','north_development_plan_v35.py']},
        source_height_pixels_decoded=0,lunar_images_rendered=0,matcher_inferences=0,
        scope='Actual Blender lamp orientation and ideal solar-body latitude only, not dated ephemerides, radiometric calibration or model performance.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,actual_configurations=len(records),maximum_emission_error=max(r['emission_vector_error']for r in records))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/north_solar_actual_control_v37.json')
    control(p.parse_args(sys.argv[sys.argv.index('--')+1:]).output)
