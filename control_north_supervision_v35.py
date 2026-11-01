"""Actual Blender BVH fractional-grid controls; no new terrain or rendering."""
import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0,str(Path.cwd()))
from north_bvh_supervision_v35 import mesh_correspondences
from north_development_plan_v35 import proposal_grid_native,quarter_turn_points,world_rays
from north_render_plan_v31 import camera_intrinsics,camera_to_world
from north_terrain_pilot_v32 import sha
from verify_blender_camera_control_v31 import triangles,trace


def control(output):
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve prior BVH controls')
    if bpy.app.version[:3]!=(4,5,9)or bpy.app.build_hash.decode()!='8bf95cbd38d1':
        raise ValueError('Pinned official renderer required')
    faces=triangles();vertices=faces.reshape(-1,3);indices=np.arange(len(vertices)).reshape(-1,3)
    bvh=BVHTree.FromPolygons(vertices.tolist(),indices.tolist(),all_triangles=True)
    a=np.asarray(camera_to_world([0.,0.,10.],[0.,0.,0.]));b=np.asarray(camera_to_world([2.,-1.,11.],[.5,0.,0.]))
    k=np.asarray(camera_intrinsics());records=[]
    for source,target in ((a,b),(b,a)):
        for rotation in range(4):
            p=quarter_turn_points(proposal_grid_native(),-rotation)
            labels=mesh_correspondences(bvh,source,target,k,p)
            rays=world_rays(p,k,source);ranges=trace(source[:3,3],rays)
            expected=source[:3,3]+rays*ranges[:,None]
            maximum=float(np.linalg.norm(expected-labels['world'],axis=1).max())
            if not labels['source_valid'].all()or maximum>.002:raise ValueError('Actual fractional BVH first hits differ from independent triangles')
            cam=(expected-target[:3,3])@target[:3,:3];h=cam@k.T;q=h[:,:2]/h[:,2,None]
            direction=expected-target[:3,3];distance=np.linalg.norm(direction,axis=1);direction/=distance[:,None]
            first=trace(target[:3,3],direction)
            visible=(cam[:,2]>0)&((q>=0)&(q<=511)).all(1)&(np.abs(first-distance)<=.05)
            if not np.array_equal(visible,labels['visible']):raise ValueError('Actual BVH visibility differs from independent analytic fixture')
            if np.linalg.norm(q-labels['target_points'],axis=1).max()>.02:raise ValueError('Fractional native projection differs')
            records.append(dict(direction=len(records)//4,rotation=rotation,points=len(p),visible=int(visible.sum()),maximum_world_error_m=maximum))
    result=dict(passed=True,records=records,source_sha256={p:sha(p)for p in
        ['control_north_supervision_v35.py','north_bvh_supervision_v35.py','north_development_plan_v35.py','verify_blender_camera_control_v31.py']},
        source_height_pixels_decoded=0,lunar_images_rendered=0,matcher_inferences=0,
        scope='Actual BVH fractional-grid/quarter-turn visibility versus independent analytic triangles only; no learned or terrain-generalization result.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/north_fractional_bvh_control_v35.json')
    control(p.parse_args(sys.argv[sys.argv.index('--')+1:]).output)
