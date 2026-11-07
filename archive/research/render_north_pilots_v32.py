"""Render the four declared pilots, with explicit geometry-only visibility.

Run with the pinned Blender; 1,800 calendar/monotonic seconds total. The four
pilots measure feasibility, and are never used as fresh validation terrain.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from blender_camera_control_v31 import output_node, read_exr
from north_render_plan_v31 import camera_to_world
from north_terrain_pilot_v32 import pilot_plan, sha


def run(data, output):
    plan, sites = pilot_plan(); source = Path(data); root = Path(output).resolve()
    acquired = json.loads((source/'summary.json').read_text())
    if not acquired['complete'] or len(acquired['records']) != 4:
        raise ValueError('Complete four-pilot acquisition required')
    if root.exists():
        raise FileExistsError('Preserve every previous run or failure')
    if bpy.app.version[:3] != (4, 5, 9) or bpy.app.build_hash.decode() != '8bf95cbd38d1':
        raise ValueError('Pinned official renderer required')
    root.mkdir(parents=True); tick = time.monotonic(); start = datetime.now(timezone.utc)
    def deadline():
        wall = (datetime.now(timezone.utc)-start).total_seconds()
        if wall < 0 or wall > 1800 or time.monotonic()-tick > 1800:
            raise TimeoutError('Original pilot calendar/monotonic allowance exceeded')
    protocol = dict(source_sha256={p: sha(p) for p in ['render_north_pilots_v32.py',
        'north_terrain_pilot_v32.py', 'north_render_plan_v31.py', 'blender_camera_control_v31.py']},
        data_summary_sha256=sha(source/'summary.json'), cpu_threads=2,
        started_utc=start.isoformat(), max_calendar_seconds=1800,
        components=[s['component'] for s in sites],
        visibility='Geometric mesh BVH first hit within0.05m; never derived from brightness.',
        scope='Four excluded geometry/cost pilots only. No matcher, training, calibration or test.')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n'); records=[]
    try:
        for site, record in zip(sites, acquired['records']):
            deadline(); folder=root/site['component']; folder.mkdir()
            path=source/site['component']/'mesh.npz'
            if record['component'] != site['component'] or sha(path) != record['mesh_sha256']:
                raise ValueError('Acquired mesh changed or reordered')
            with np.load(path, allow_pickle=False) as f:
                vertices=f['vertices'].copy(); faces=f['faces'].copy(); clearance=float(f['core_max_height'])
                body_origin=f['body_origin'].copy(); enu_rows=f['enu_rows'].copy()
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene=bpy.context.scene; scene.render.engine='CYCLES'; scene.cycles.device='CPU'
            scene.render.threads_mode='FIXED'; scene.render.threads=2; scene.cycles.samples=16
            scene.cycles.use_denoising=False; scene.cycles.seed=20261101; scene.cycles.max_bounces=2
            scene.cycles.pixel_filter_type='BOX'; scene.cycles.filter_width=.01; scene.render.filter_size=.01
            size=plan['image_size']; k=np.asarray(plan['intrinsics'])
            scene.render.resolution_x=size; scene.render.resolution_y=size; scene.render.resolution_percentage=100
            scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1.
            scene.view_settings.view_transform='Standard'; scene.view_settings.exposure=0.; scene.view_settings.gamma=1.
            world=bpy.data.worlds.new('Uniform weak world'); world.use_nodes=True
            world.node_tree.nodes['Background'].inputs['Color'].default_value=(1.,1.,1.,1.)
            world.node_tree.nodes['Background'].inputs['Strength'].default_value=.001; scene.world=world
            mesh=bpy.data.meshes.new('Acquired sphere-corrected terrain')
            mesh.vertices.add(len(vertices)); mesh.vertices.foreach_set('co',vertices.ravel())
            mesh.loops.add(faces.size); mesh.loops.foreach_set('vertex_index', faces.ravel())
            mesh.polygons.add(len(faces)); mesh.polygons.foreach_set('loop_start', np.arange(len(faces), dtype=np.int32)*3)
            mesh.polygons.foreach_set('loop_total',np.full(len(faces),3,dtype=np.int32)); mesh.update()
            obj=bpy.data.objects.new('Finite terrain support',mesh); scene.collection.objects.link(obj)
            mat=bpy.data.materials.new('Constant diffuse'); mat.use_nodes=True; nodes=mat.node_tree.nodes; nodes.clear()
            diffuse=nodes.new('ShaderNodeBsdfDiffuse'); diffuse.inputs['Color'].default_value=(.5,.5,.5,1.)
            diffuse.inputs['Roughness'].default_value=0.; out=nodes.new('ShaderNodeOutputMaterial')
            mat.node_tree.links.new(diffuse.outputs[0],out.inputs['Surface']); mesh.materials.append(mat)
            bpy.ops.object.camera_add(); camera=bpy.context.object; scene.camera=camera
            camera.data.type='PERSP'; camera.data.sensor_fit='HORIZONTAL'; camera.data.sensor_width=36.
            camera.data.lens=36*k[0,0]/size; camera.data.clip_start=.1; camera.data.clip_end=50000.
            camera.data.dof.use_dof=False
            bpy.ops.object.light_add(type='SUN'); sun=bpy.context.object; sun.data.energy=3.; sun.data.angle=0.
            view=scene.view_layers[0]; view.use_pass_z=True; view.use_pass_position=True
            scene.use_nodes=True; tree=scene.node_tree; tree.nodes.clear(); layer=tree.nodes.new('CompositorNodeRLayers')
            comp=tree.nodes.new('CompositorNodeComposite'); tree.links.new(layer.outputs['Image'],comp.inputs['Image'])
            output_node(tree,folder,'depth',layer.outputs['Depth'])
            output_node(tree,folder,'position',layer.outputs['Position'])
            output_nodes=[n for n in tree.nodes if n.bl_idname=='CompositorNodeOutputFile']
            scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'
            bpy.context.view_layer.update(); depsgraph=bpy.context.evaluated_depsgraph_get()
            bvh=BVHTree.FromObject(obj,depsgraph)
            views=[]; images=[]
            for spec in plan['camera_specs']:
                position=np.asarray(spec['position_offset'],float); position[2]+=clearance
                pose=np.asarray(camera_to_world(position.tolist(),spec['target']))
                matrix=pose.copy(); matrix[:3,:3]=pose[:3,:3]@np.diag([1.,-1.,-1.]); camera.matrix_world=Matrix(matrix.tolist())
                view_folder=folder/f'view-{spec["view"]}'; view_folder.mkdir()
                for light,(elevation,azimuth) in enumerate(plan['rendered_sun_elevation_azimuth_degrees']):
                    deadline(); light_folder=view_folder/f'light-{light}'; light_folder.mkdir()
                    e,a=math.radians(elevation),math.radians(azimuth)
                    toward_sun=Vector((math.cos(e)*math.sin(a),math.cos(e)*math.cos(a),math.sin(e)))
                    sun.rotation_euler=(-toward_sun).to_track_quat('-Z','Y').to_euler()
                    for node in output_nodes:node.base_path=str(light_folder)
                    scene.render.filepath=str(light_folder/'image.png'); before=time.monotonic()
                    bpy.context.view_layer.update(); bpy.ops.render.render(write_still=True)
                    images.append(dict(view=spec['view'],light=light,seconds=time.monotonic()-before,
                                       image_sha256=sha(light_folder/'image.png')))
                    if light==0:
                        world_points=read_exr(next(light_folder.glob('position_*.exr')),size)
                        depth=read_exr(next(light_folder.glob('depth_*.exr')),size)[...,0]
                        camera_points=(world_points-pose[:3,3])@pose[:3,:3]
                        valid=np.isfinite(depth)&(depth>0)&(depth<50000)&np.isfinite(world_points).all(2)
                        if not valid.all():raise ValueError('Pilot image leaves finite terrain; preserve geometry failure')
                        np.savez_compressed(view_folder/'geometry.npz', position=world_points,
                            axial_depth=camera_points[...,2],ray_depth=depth,cam2world=pose,intrinsics=k,
                            valid=valid,body_origin=body_origin,enu_rows=enu_rows)
                        views.append(dict(pose=pose,world=world_points,folder=view_folder))
            # Exact native pixel reference samples; no darkness-based visibility.
            yy,xx=np.mgrid[8:size-8:8,8:size-8:8]; p=np.stack([xx.ravel(),yy.ravel()],1).astype(float)
            xyz=views[0]['world'][yy.ravel(),xx.ravel()].astype(float); labels=[]
            for view_index in (1,2,3):
                deadline(); pose=views[view_index]['pose']; cam=(xyz-pose[:3,3])@pose[:3,:3]
                q=cam[:,:2]/cam[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
                in_frame=(cam[:,2]>0)&((q>=0)&(q<=size-1)).all(1); visible=np.zeros(len(p),bool)
                for i in np.flatnonzero(in_frame):
                    direction=Vector((xyz[i]-pose[:3,3]).tolist()); distance=direction.length; direction.normalize()
                    hit,normal,index,travel=bvh.ray_cast(Vector(pose[:3,3].tolist()),direction,50000.)
                    visible[i]=hit is not None and abs(travel-distance)<=.05
                dest=folder/f'correspondences-0-{view_index}.npz'
                np.savez_compressed(dest,reference=p,query=q,world=xyz,in_frame=in_frame,visible=visible)
                labels.append(dict(query_view=view_index,points=len(p),in_frame=int(in_frame.sum()),
                                   visible=int(visible.sum()),sha256=sha(dest)))
            item=dict(component=site['component'],images=images,labels=labels,
                      geometry_sha256={str(v['folder'].relative_to(folder)/'geometry.npz'):sha(v['folder']/'geometry.npz') for v in views})
            records.append(item); (folder/'summary.json').write_text(json.dumps(item,indent=2)+'\n')
            print('PILOT COMPONENT COMPLETE',site['component'],flush=True)
        deadline(); result=dict(complete=True,records=records,protocol_sha256=sha(root/'protocol.json'),
            calendar_seconds=(datetime.now(timezone.utc)-start).total_seconds(),cpu_threads=2,
            rendered_images=sum(len(r['images']) for r in records),matcher_inferences=0,
            calibration_or_test_images_rendered=0,fresh_final_gate_passed=False)
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result
    except Exception as exc:
        (root/'FAILED.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),
            completed_components=len(records),elapsed_seconds=time.monotonic()-tick),indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/north-render-v31/pilots-v32')
    p.add_argument('--output',default='runs/north-v32-terrain-pilots')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.data,a.output)
