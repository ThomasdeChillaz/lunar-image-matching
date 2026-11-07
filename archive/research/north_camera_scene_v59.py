"""Draft varied-camera scene using the unchanged v40 photometric settings."""
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from blender_camera_control_v31 import output_node


def setup_scene(folder,vertices,faces,k,size=512):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene; scene.render.engine='CYCLES'; scene.cycles.device='CPU'
    scene.render.threads_mode='FIXED'; scene.render.threads=2; scene.cycles.samples=16
    scene.cycles.use_denoising=False; scene.cycles.seed=20261101; scene.cycles.max_bounces=2
    scene.cycles.pixel_filter_type='BOX'; scene.cycles.filter_width=.01; scene.render.filter_size=.01
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
    return scene,camera,sun,output_nodes,bvh
