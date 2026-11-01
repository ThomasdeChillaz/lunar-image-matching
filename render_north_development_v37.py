"""Render only16training/eightselection supports with exact3D supervision.

Same cameras/photometric settings, with solar directions corrected before model use. Correct rawZ,
axial depth and Euclidean range are explicit. No calibration/final imagery.
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
from north_terrain_pilot_v32 import sha
from north_development_plan_v35 import (development_plan,prerequisite_gate,proposal_grid_native,
                                        quarter_turn_points,world_rays)
from renderer_depth_v34 import canonical_depth,control_convention
from north_bvh_supervision_v35 import mesh_correspondences
from north_solar_geometry_v37 import solar_configuration


def run(data, output):
    gates=prerequisite_gate();plan, sites = development_plan(); source = Path(data); root = Path(output).resolve()
    convention=control_convention()
    solar_path=Path('reports/north_solar_amendment_v37.json');solar_contract=json.loads(solar_path.read_text())
    if solar_contract['solar_source_sha256']!=sha('north_solar_geometry_v37.py'):
        raise ValueError('Frozen solar derivation changed')
    if solar_contract['acquired_data_summary_sha256']!=sha(source/'summary.json'):
        raise ValueError('Original24acquired terrain meshes changed')
    budget=solar_contract['remaining_cumulative_render_allowance_seconds']
    absolute_deadline=datetime.fromisoformat(solar_contract['shared_absolute_render_deadline_utc'])
    acquired = json.loads((source/'summary.json').read_text())
    if not acquired['complete'] or len(acquired['records']) != 24:
        raise ValueError('Complete original24component development acquisition required')
    if root.exists():
        raise FileExistsError('Preserve every previous run or failure')
    if bpy.app.version[:3] != (4, 5, 9) or bpy.app.build_hash.decode() != '8bf95cbd38d1':
        raise ValueError('Pinned official renderer required')
    root.mkdir(parents=True); tick = time.monotonic(); start = datetime.now(timezone.utc)
    def deadline():
        wall = (datetime.now(timezone.utc)-start).total_seconds()
        if wall < 0 or wall > budget or time.monotonic()-tick > budget or datetime.now(timezone.utc)>absolute_deadline:
            raise TimeoutError('Original six-hour development render allowance exceeded')
    protocol = dict(source_sha256={p: sha(p) for p in ['render_north_development_v37.py','north_development_plan_v35.py',
        'north_terrain_pilot_v32.py', 'north_render_plan_v31.py', 'blender_camera_control_v31.py', 'renderer_depth_v34.py','north_bvh_supervision_v35.py','north_solar_geometry_v37.py']},
        data_summary_sha256=sha(source/'summary.json'), cpu_threads=2,
        started_utc=start.isoformat(), max_calendar_seconds=budget,shared_absolute_deadline=absolute_deadline.isoformat(),
        solar_contract_sha256=sha(solar_path),prerequisite_sha256=gates,raw_depth_convention=convention,
        components=[s['component'] for s in sites],
        visibility='Geometric mesh BVH first hit within0.05m; never derived from brightness.',
        scope='Original16training/eightselection components; data generation only, no matcher/fitting/calibration/final imagery.')
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
                for light in range(6):
                    deadline(); light_folder=view_folder/f'light-{light}'; light_folder.mkdir()
                    solar=solar_configuration(site,light)
                    expected_solar=solar_contract['development_solar_configurations'][site['component']][light]
                    if solar!=expected_solar:raise ValueError('Solar configuration differs from pre-inference contract')
                    direction=np.asarray(enu_rows)@np.asarray(solar['solar_body_direction'])
                    np.testing.assert_allclose(direction,solar['solar_enu_direction'],rtol=0,atol=1e-10)
                    toward_sun=Vector(direction.tolist())
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
                        axial,ray,raw_error=canonical_depth(world_points,pose,depth,convention)
                        np.savez_compressed(view_folder/'geometry.npz', position=world_points,
                            axial_depth=axial,ray_depth=ray,renderer_depth=depth,
                            renderer_depth_convention=convention,cam2world=pose,intrinsics=k,
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
            # Exact fractional-grid source intersections for training. Every
            # target uses the actual mesh/visibility, independent of darkness.
            exact_labels=[]
            def exact_pair(source_view,target_view,source_points,rotation,name):
                deadline();source_pose=views[source_view]['pose'];target_pose=views[target_view]['pose']
                labels=mesh_correspondences(bvh,source_pose,target_pose,k,source_points,size)
                dest=folder/(name+'.npz')
                np.savez_compressed(dest,**labels,
                    source_view=source_view,target_view=target_view,source_grid_rotation=rotation)
                exact_labels.append(dict(file=dest.name,sha256=sha(dest),source_view=source_view,target_view=target_view,
                    source_grid_rotation=rotation,points=len(source_points),source_valid=int(labels['source_valid'].sum()),
                    in_frame=int(labels['in_frame'].sum()),visible=int(labels['visible'].sum())))
            grid=proposal_grid_native()
            for query_view in (1,2,3):
                exact_pair(0,query_view,grid,0,f'supervision-forward-0-{query_view}')
                for rotation in range(4):
                    original_query_grid=quarter_turn_points(grid,-rotation)
                    exact_pair(query_view,0,original_query_grid,rotation,
                               f'supervision-reverse-{query_view}-0-rotation-{rotation}')
            item=dict(component=site['component'],split=site['split'],images=images,labels=labels,exact_supervision=exact_labels,
                      solar_configurations=[solar_configuration(site,i)for i in range(6)],
                      geometry_sha256={str(v['folder'].relative_to(folder)/'geometry.npz'):sha(v['folder']/'geometry.npz') for v in views})
            records.append(item); (folder/'summary.json').write_text(json.dumps(item,indent=2)+'\n')
            print('DEVELOPMENT COMPONENT COMPLETE',site['component'],flush=True)
        deadline(); result=dict(complete=True,records=records,protocol_sha256=sha(root/'protocol.json'),
            calendar_seconds=(datetime.now(timezone.utc)-start).total_seconds(),cpu_threads=2,
            rendered_images=sum(len(r['images']) for r in records),matcher_inferences=0,
            calibration_or_test_images_rendered=0,fresh_final_gate_passed=False)
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result
    except Exception as exc:
        (root/'FAILED.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),
            completed_components=len(records),elapsed_seconds=time.monotonic()-tick),indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/north-render-v31/development-v35')
    p.add_argument('--output',default='runs/north-v37-development')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.data,a.output)
