"""One frozen unit of the new24-light development acquisition, never a restart.

Geometry uses independent integer-centre BVH rays, preserving raw beauty renders
and avoiding interpolated position-pass pixels at depth discontinuities. Exact
3D labels and the common pose gate must be audited before any matcher or fitting.
"""
import argparse,hashlib,json,sys,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from solar_data_v65 import schedule
from solar_physics_v63 import lighting_schedule
from north_camera_correspondences_v59 import mesh_correspondences
from north_development_plan_v35 import proposal_grid_native,world_rays
from solar_physics_v63 import solar_direction

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES=['render_solar_development_v67.py','render_solar_data_v65.py','solar_data_v65.py','solar_physics_v63.py','north_camera_scene_v59.py','north_camera_correspondences_v59.py','north_development_plan_v35.py','north_camera_variation_v59.py','north_render_plan_v31.py','blender_camera_control_v31.py']


from render_solar_data_v65 import ray_geometry


def admit_unit(master,unit_index,now):
    if master['phase']!='new_development_acquisition' or not master['compute_admitted']or master['calibration_or_final_access']:raise ValueError('New development-only acquisition admission required')
    if master['shared_cap_seconds']!=36000:raise ValueError('Original shared ten-hour data allowance required')
    deadline=datetime.fromisoformat(master['absolute_deadline_utc'])
    if now>=deadline:raise TimeoutError('Original shared acquisition deadline expired; never reset')
    if type(unit_index)is not int or not 0<=unit_index<len(master['units']):raise ValueError('Frozen unit index outside schedule')
    unit=master['units'][unit_index]
    if unit['site']['split']not in('training','model_selection')or unit['calibration_or_final_access']:raise ValueError('Reserve terrain cannot enter development')
    if unit['max_calendar_seconds']!=900 or unit['cpu_threads']!=2:raise ValueError('Fixed per-unit data budget required')
    if master['factor_two_projection_seconds']>master['shared_cap_seconds']:raise ValueError('Measured cost does not fit original data allowance')
    return unit,deadline


def run(protocol_path,commit,unit_index):
    import bpy
    from mathutils import Vector,Matrix
    from north_camera_scene_v59 import setup_scene
    master=json.loads(Path(protocol_path).read_text());start=datetime.now(timezone.utc);p,global_deadline=admit_unit(master,unit_index,start);out=Path(p['output']);tick=time.monotonic();deadline=start+timedelta(seconds=p['max_calendar_seconds'])
    if out.exists():raise FileExistsError('Preserve all prior attempts; no automatic restart')
    if p['phase']!='development_acquisition_unit'or p['max_calendar_seconds']!=900 or p['cpu_threads']!=2 or p['calibration_or_final_access']:raise ValueError('Only separately frozen900second development unit is admitted')
    for n in SOURCES:
        if p['source_sha256'][n]!=sha(n)or subprocess.check_output(['git','show',commit+':'+n])!=Path(n).read_bytes():raise ValueError('Published rendering source differs:'+n)
    if subprocess.check_output(['git','show',commit+':'+protocol_path])!=Path(protocol_path).read_bytes():raise ValueError('Cost protocol not source-published')
    if bpy.app.version[:3]!=(4,5,9)or bpy.app.build_hash.decode()!='8bf95cbd38d1':raise ValueError('Pinned official renderer required')
    if sha(p['mesh'])!=p['mesh_sha256']:raise ValueError('Acquired pilot terrain changed')
    if len(p['camera_pairs'])!=1 or p['camera_pairs'][0]not in schedule(p['site'],p['core_max_height'],2):raise ValueError('Fixed first fresh-seed motion pair required')
    if p['site']['split']not in('training','model_selection')or master['calibration_or_final_access']or not master['compute_admitted']:raise ValueError('Only admitted original development terrain allowed')
    global_deadline=datetime.fromisoformat(master['absolute_deadline_utc'])
    if start>=global_deadline:raise TimeoutError('Original shared acquisition deadline expired; never reset')
    out.mkdir(parents=True);images=[];views={}

    def clock():
        if datetime.now(timezone.utc)>=min(deadline,global_deadline)or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Original solar cost-pilot deadline exhausted')
    def save(path,z):
        with Path(path).open('x')as f:json.dump(z,f,indent=2,allow_nan=False);f.write('\n')
    save(out/'execution.json',dict(protocol_sha256=sha(protocol_path),publication_commit=commit,started_utc=start.isoformat(),absolute_deadline_utc=deadline.isoformat(),cpu_threads=2))
    try:
        with np.load(p['mesh'],allow_pickle=False)as z:vertices,faces=z['vertices'].copy(),z['faces'].copy();origin=z['body_origin'].copy();enu=z['enu_rows'].copy()
        pair=p['camera_pairs'][0];scene,camera,sun,nodes,bvh=setup_scene(out,vertices,faces,np.asarray(pair['reference']['intrinsics']))
        # Geometry is now explicit centre rays. EXR channels are unnecessary for
        # training; removing file-output nodes avoids writing them at24lights.
        for node in nodes:scene.node_tree.nodes.remove(node)
        lights=lighting_schedule(p['site']['centre_latitude_deg'])
        for role in('reference','query'):
            clock();folder=out/role;folder.mkdir();pose=np.asarray(pair[role]['cam2world']);k=np.asarray(pair[role]['intrinsics'])
            before=time.monotonic();geometry=ray_geometry(bvh,pose,k,clock);geometry.update(body_origin=origin,enu_rows=enu)
            np.savez_compressed(folder/'geometry.npz',**geometry)
            views[role]=dict(path=str((folder/'geometry.npz').relative_to(out)),sha256=sha(folder/'geometry.npz'),complete_surface=bool(geometry['valid'].all()),centre_ray_seconds=time.monotonic()-before)
            camera.data.lens=36*k[0,0]/512;m=pose.copy();m[:3,:3]=pose[:3,:3]@np.diag([1.,-1.,-1.]);camera.matrix_world=Matrix(m.tolist())
            for light in lights:
                clock();path=folder/('light-'+str(light['index'])+'.png');sun.rotation_euler=(-Vector(light['sun_enu'])).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(path);bpy.context.view_layer.update();before=time.monotonic();bpy.ops.render.render(write_still=True)
                images.append(dict(role=role,light=light['index'],path=str(path.relative_to(out)),sha256=sha(path),render_seconds=time.monotonic()-before))
        labels=[]
        for role,other in(('reference','query'),('query','reference')):
            clock();z=mesh_correspondences(bvh,np.asarray(pair[role]['cam2world']),np.asarray(pair[other]['cam2world']),np.asarray(pair[role]['intrinsics']),np.asarray(pair[other]['intrinsics']),proposal_grid_native())
            path=out/(role+'-to-'+other+'.npz');np.savez_compressed(path,**z);labels.append(dict(source=role,path=path.name,sha256=sha(path),visible=int(z['visible'].sum()),points=len(z['visible'])))
        clock();result=dict(complete=True,phase=p['phase'],images=images,views=views,labels=labels,rendered_images=len(images),calendar_seconds=time.monotonic()-tick,protocol_sha256=sha(protocol_path),source_sha256=p['source_sha256'],geometry_audit_pending=True,training_allowed=False,matcher_forwards=0,optimizer_updates=0,calibration_or_final_access=False)
        save(out/'summary.json',result);print(json.dumps(result,indent=2));return result
    except BaseException as exc:save(out/'FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-tick,rendered_images=len(images)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--unit-index',required=True,type=int);args=sys.argv[sys.argv.index('--')+1:]if '--'in sys.argv else sys.argv[1:];a=p.parse_args(args);run(a.protocol,a.publication_commit,a.unit_index)
