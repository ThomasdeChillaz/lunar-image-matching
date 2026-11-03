"""Render only missing units under a separately declared14hour completion cap.

Originalv67outputs/failure/expiry remain unchanged. The physical rendering body
and integer-centre BVH export are reused; only strict cross-runtime replay and
new output/provenance/deadline admission differ. No trained matcher is executed.
"""
import argparse,hashlib,json,sys,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from solar_physics_v63 import lighting_schedule
from north_camera_correspondences_v59 import mesh_correspondences
from north_development_plan_v35 import proposal_grid_native
from render_solar_data_v65 import ray_geometry
from solar_completion_v68 import verify_publication,admit,sha


def run(protocol_path,commit,unit_index):
    import bpy
    from mathutils import Vector,Matrix
    from north_camera_scene_v59 import setup_scene
    contract=verify_publication(protocol_path,commit);start=datetime.now(timezone.utc)
    p,out,global_deadline=admit(contract,unit_index,start);tick=time.monotonic();deadline=min(start+timedelta(seconds=900),global_deadline)
    if out.exists():raise FileExistsError('Preserve every completed/partial completion unit; no automatic restart')
    if bpy.app.version[:3]!=(4,5,9)or bpy.app.build_hash.decode()!='8bf95cbd38d1':raise ValueError('Pinned official renderer required')
    if sha(p['mesh'])!=p['mesh_sha256']:raise ValueError('Original acquired terrain changed')
    out.mkdir(parents=True);images=[];views={}

    def clock():
        if datetime.now(timezone.utc)>=min(deadline,global_deadline)or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Original solar cost-pilot deadline exhausted')
    def save(path,z):
        with Path(path).open('x')as f:json.dump(z,f,indent=2,allow_nan=False);f.write('\n')
    save(out/'execution.json',dict(protocol_sha256=sha(contract['original_master']),completion_contract_sha256=sha(protocol_path),publication_commit=commit,original_unit_index=unit_index,started_utc=start.isoformat(),absolute_deadline_utc=deadline.isoformat(),cpu_threads=2))
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
        clock();result=dict(complete=True,phase='missing_only_completion_unit',images=images,views=views,labels=labels,rendered_images=len(images),calendar_seconds=time.monotonic()-tick,protocol_sha256=sha(contract['original_master']),completion_contract_sha256=sha(protocol_path),original_unit_index=unit_index,source_sha256=contract['render_source_sha256'],geometry_audit_pending=True,training_allowed=False,matcher_forwards=0,optimizer_updates=0,calibration_or_final_access=False)
        save(out/'summary.json',result);print(json.dumps(result,indent=2));return result
    except BaseException as exc:save(out/'FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-tick,rendered_images=len(images)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--unit-index',required=True,type=int)
    args=sys.argv[sys.argv.index('--')+1:]if '--'in sys.argv else sys.argv[1:];a=p.parse_args(args);run(a.protocol,a.publication_commit,a.unit_index)
