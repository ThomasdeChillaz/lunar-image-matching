"""Actual Blender camera/depth/occluder controls, invoked with official Blender.

No lunar terrain, matcher or reserve images are loaded. Freeze this script and
its pure geometry dependency publicly before running; preserve failed outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix,Vector
import numpy as np

sys.path.insert(0,str(Path.cwd()))
from north_render_plan_v31 import camera_intrinsics,camera_to_world

VIEWS=((128,[0.,0.,10.],[0.,0.,0.]),(512,[2.,-1.,11.],[.5,0.,0.]))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytic_hits(position,rotation,k,size):
    yy,xx=np.mgrid[:size,:size]
    camera=np.stack([(xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones_like(xx)],axis=-1).astype(float)
    camera/=np.linalg.norm(camera,axis=-1,keepdims=True)
    rays=camera@rotation.T
    floor=-position[2]/rays[...,2]
    # Independent axis-aligned solid-box intersections, including first hits.
    low=np.array([.5,-.6,0.]);high=np.array([2.5,1.4,2.]);near=[];far=[]
    for axis in range(3):
        d=rays[...,axis];nonzero=np.abs(d)>1e-14
        a=np.divide(low[axis]-position[axis],d,out=np.zeros_like(d),where=nonzero)
        b=np.divide(high[axis]-position[axis],d,out=np.zeros_like(d),where=nonzero)
        inside=low[axis]<=position[axis]<=high[axis]
        near.append(np.where(nonzero,np.minimum(a,b),-np.inf if inside else np.inf))
        far.append(np.where(nonzero,np.maximum(a,b),np.inf if inside else -np.inf))
    tn=np.max(near,axis=0);tf=np.min(far,axis=0)
    box=(tf>=np.maximum(tn,0))&(tn>0)&(tn<floor)
    distance=np.where(box,tn,floor)
    if not(np.isfinite(distance).all()and(distance>0).all()):raise ValueError('Analytic fixture has unsupported background rays')
    points=position+rays*distance[...,None]
    # A fixed two-pixel guard around exact geometric silhouettes, never darkness.
    boundary=np.zeros(box.shape,bool)
    for dy in range(-2,3):
        for dx in range(-2,3):
            shifted=np.roll(np.roll(box,dy,0),dx,1)
            boundary|=shifted!=box
    boundary[:2]=True;boundary[-2:]=True;boundary[:,:2]=True;boundary[:,-2:]=True
    return points,distance,box,~boundary


def read_exr(path,size):
    im=bpy.data.images.load(str(path),check_existing=False)
    im.colorspace_settings.name='Non-Color'
    array=np.asarray(im.pixels[:],dtype=np.float32).reshape(size,size,4)[::-1,:,:3].copy()
    bpy.data.images.remove(im)
    return array


def output_node(tree,root,name,source):
    node=tree.nodes.new('CompositorNodeOutputFile');node.base_path=str(root)
    node.format.file_format='OPEN_EXR';node.format.color_depth='32';node.format.color_mode='RGB'
    node.file_slots[0].path=name+'_';tree.links.new(source,node.inputs[0])


def run(output):
    root=Path(output).resolve()
    if root.exists():raise FileExistsError('Preserve every renderer control or failure')
    if bpy.app.version[:3]!=(4,5,9) or bpy.app.build_hash.decode()!='8bf95cbd38d1':
        raise ValueError('Expected the official pinned Blender4.5.9build')
    root.mkdir(parents=True);started=perf_counter()
    protocol=dict(scope='Actual analytic plane+solid box rendering only. No lunar/source terrain or matcher inference.',
        source_sha256={p:sha(p)for p in ('blender_camera_control_v31.py','north_render_plan_v31.py')},
        blender=bpy.app.version_string,build_hash=bpy.app.build_hash.decode(),numpy=np.__version__,
        cpu_threads=2,views=[dict(size=s,position=p,target=t)for s,p,t in VIEWS],
        coordinate_rule='Native integer pixel centres, Blender edge coordinates minus0.5; image rows top-down; CVcamera axes right/down/forward.',
        tolerances=dict(interior_world_position_max_m=.002,interior_native_projection_max_px=.02,
            interior_depth_relative_error=.0005,minimum_interior_pixels_fraction=.8),
        silhouette_guard='Fixed two-pixel analytic geometric-silhouette and image-edge guard. All-pixel errors are also retained.',
        depth_convention='Determine only by analytic camera/scene geometry; never matcher outcomes.',
        max_control_seconds=600,source_height_pixels_decoded=0,lunar_images_rendered=0,matcher_inferences=0)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]
    try:
        for index,(size,position,target)in enumerate(VIEWS):
            if perf_counter()-started>600:raise TimeoutError('Renderer control cap exceeded')
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU'
            scene.render.threads_mode='FIXED';scene.render.threads=2;scene.cycles.samples=8
            scene.cycles.use_denoising=False;scene.cycles.seed=20261101;scene.cycles.max_bounces=2
            scene.cycles.pixel_filter_type='BOX';scene.cycles.filter_width=.01;scene.render.filter_size=.01
            scene.render.resolution_x=size;scene.render.resolution_y=size;scene.render.resolution_percentage=100
            scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1.
            scene.view_settings.view_transform='Standard';scene.view_settings.exposure=0.;scene.view_settings.gamma=1.
            world=bpy.data.worlds.new('Fixed dark background');world.use_nodes=True
            world.node_tree.nodes['Background'].inputs['Color'].default_value=(.001,.001,.001,1.)
            scene.world=world
            material=bpy.data.materials.new('Uniform diffuse fixture');material.use_nodes=True
            nodes=material.node_tree.nodes;nodes.clear();diffuse=nodes.new('ShaderNodeBsdfDiffuse')
            diffuse.inputs['Color'].default_value=(.5,.5,.5,1.);diffuse.inputs['Roughness'].default_value=0.
            out=nodes.new('ShaderNodeOutputMaterial');material.node_tree.links.new(diffuse.outputs[0],out.inputs['Surface'])
            bpy.ops.mesh.primitive_plane_add(size=100.,location=(0,0,0));bpy.context.object.data.materials.append(material)
            bpy.ops.mesh.primitive_cube_add(size=1.,location=(1.5,.4,1.));box=bpy.context.object
            box.scale=(2.,2.,2.);box.data.materials.append(material)
            bpy.ops.object.light_add(type='SUN',location=(0,0,10));sun=bpy.context.object
            sun.data.energy=3.;sun.data.angle=0.;sun.rotation_euler=Vector((-1.,-.5,-1.)).to_track_quat('-Z','Y').to_euler()
            pose=np.asarray(camera_to_world(position,target),float);k=np.asarray(camera_intrinsics(size),float)
            bpy.ops.object.camera_add();camera=bpy.context.object
            matrix=pose.copy();matrix[:3,:3]=pose[:3,:3]@np.diag([1.,-1.,-1.]);camera.matrix_world=Matrix(matrix.tolist())
            camera.data.type='PERSP';camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36.
            camera.data.lens=36*k[0,0]/size;camera.data.clip_start=.01;camera.data.clip_end=1000.
            camera.data.dof.use_dof=False;scene.camera=camera
            view=scene.view_layers[0];view.use_pass_z=True;view.use_pass_position=True
            scene.use_nodes=True;tree=scene.node_tree;tree.nodes.clear();layer=tree.nodes.new('CompositorNodeRLayers')
            composite=tree.nodes.new('CompositorNodeComposite');tree.links.new(layer.outputs['Image'],composite.inputs['Image'])
            folder=root/f'view-{index}';folder.mkdir()
            output_node(tree,folder,'depth',layer.outputs['Depth']);output_node(tree,folder,'position',layer.outputs['Position'])
            scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.filepath=str(folder/'image.png')
            bpy.context.view_layer.update();bpy.ops.render.render(write_still=True)
            files={name:next(folder.glob(name+'_*.exr'))for name in ('depth','position')}
            actual=read_exr(files['position'],size);depth=read_exr(files['depth'],size)[...,0]
            truth,ray,box_mask,interior=analytic_hits(pose[:3,3],pose[:3,:3],k,size)
            if interior.mean()<.8:raise ValueError('Too little eligible analytic control support')
            error=np.linalg.norm(actual-truth,axis=-1);cam=(actual-pose[:3,3])@pose[:3,:3]
            predicted=cam[...,:2]/cam[...,2,None]*np.array([k[0,0],k[1,1]])+np.array([k[0,2],k[1,2]])
            yy,xx=np.mgrid[:size,:size];pixel_error=np.linalg.norm(predicted-np.stack([xx,yy],-1),axis=-1)
            axial=((truth-pose[:3,3])@pose[:3,:3])[...,2]
            candidates={name:float(np.max(np.abs(depth[interior]-expected[interior])/expected[interior]))for name,expected in [('ray',ray),('axial',axial)]}
            convention=min(candidates,key=candidates.get)
            sdk_errors=[]
            for v,u in [(size//4,size//4),(size//2,size//2),(3*size//4,3*size//4)]:
                projected=world_to_camera_view(scene,camera,Vector(truth[v,u].tolist()))
                sdk_errors.append(float(np.hypot(projected.x*size-.5-u,(1-projected.y)*size-.5-v)))
            passed=bool(error[interior].max()<=.002 and pixel_error[interior].max()<=.02 and candidates[convention]<=.0005 and max(sdk_errors)<.002)
            np.savez_compressed(folder/'geometry.npz',position=actual,renderer_depth=depth,
                expected_position=truth,analytic_box_mask=box_mask,interior=interior,intrinsics=k,cam2world=pose)
            record=dict(view=index,size=size,passed=passed,depth_convention=convention,
                depth_relative_error_candidates=candidates,interior_pixels=int(interior.sum()),all_pixels=size*size,
                all_pixel_max_world_error_m=float(error.max()),interior_max_world_error_m=float(error[interior].max()),
                interior_max_native_projection_error_px=float(pixel_error[interior].max()),
                independent_blender_projection_max_native_px=max(sdk_errors),
                geometry_sha256=sha(folder/'geometry.npz'),image_sha256=sha(folder/'image.png'))
            records.append(record);(folder/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
            if not passed:raise ValueError('Actual renderer geometry/pixel/depth control failed; preserve all outputs')
        if len({r['depth_convention']for r in records})!=1:raise ValueError('Depth convention changes across declared camera settings')
        result=dict(passed=True,records=records,protocol_sha256=sha(root/'protocol.json'),
            elapsed_seconds=perf_counter()-started,source_height_pixels_decoded=0,lunar_images_rendered=0,matcher_inferences=0,
            scope='Actual renderer analytic geometry controls only; no terrain freshness, learned result, optical lunar validation or pose-oracle gate yet.')
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result
    except Exception as exc:
        (root/'FAILED.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),completed_views=len(records),elapsed_seconds=perf_counter()-started),indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.output)
