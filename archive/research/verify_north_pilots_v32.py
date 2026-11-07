"""Replay pilot labels, local mesh intersections and common-estimator oracle.

These are excluded data/geometry pilots, not model-selection or test results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from north_render_plan_v31 import RADIUS
from north_terrain_pilot_v32 import pilot_plan, sha


def triangle_hits(origin, ray, triangles):
    a,b,c=triangles.transpose(1,0,2);e1=b-a;e2=c-a
    cross=np.cross(np.broadcast_to(ray,e2.shape),e2);det=np.sum(e1*cross,1)
    inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-12)
    tvec=origin-a;u=np.sum(tvec*cross,1)*inv;q=np.cross(tvec,e1)
    v=q@ray*inv;t=np.sum(e2*q,1)*inv
    valid=(np.abs(det)>1e-12)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>0)
    return float(np.min(np.where(valid,t,np.inf)))


def local_surface_check(points, pose, mesh, site, offsets):
    """Find neighbouring source-grid faces independently of Blender's BVH."""
    vertices=mesh['vertices'].reshape(len(offsets),len(offsets),3).astype(float)
    body=points@mesh['enu_rows']+mesh['body_origin']
    radii=np.linalg.norm(body,axis=1);longitude=np.arctan2(body[:,1],body[:,0])
    latitude=np.arcsin(body[:,2]/radii);rho=2*RADIUS*np.tan((np.pi/2-latitude)/2)
    x=rho*np.sin(longitude);y=-rho*np.cos(longitude)
    columns=np.searchsorted(offsets,(x-site['x'])/10.,side='right')-1
    rows=np.searchsorted(offsets,(site['y']-y)/10.,side='right')-1
    errors=[]
    for point,row,col in zip(points,rows,columns):
        triangles=[]
        for r in range(max(0,row-1),min(len(offsets)-1,row+2)):
            for c in range(max(0,col-1),min(len(offsets)-1,col+2)):
                a,b,d,e=vertices[r,c],vertices[r,c+1],vertices[r+1,c],vertices[r+1,c+1]
                triangles.extend([[a,b,d],[b,e,d]])
        direction=point-pose[:3,3];distance=np.linalg.norm(direction);direction/=distance
        hit=triangle_hits(pose[:3,3],direction,np.asarray(triangles))
        errors.append(abs(hit-distance))
    return np.asarray(errors)


def verify(data,run,output):
    from native_baseline_pose_v7 import propose_pose
    import cv2
    cv2.setNumThreads(1)
    plan,sites=pilot_plan();data,root,out=Path(data),Path(run),Path(output)
    if out.exists():raise FileExistsError('Preserve previous audit or failure')
    summary=json.loads((root/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if not summary['complete'] or summary['rendered_images']!=96 or summary['protocol_sha256']!=sha(root/'protocol.json'):
        raise ValueError('All96 pilot images and complete sealed summary required')
    for p,h in protocol['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Executed pilot renderer changed')
    if protocol['data_summary_sha256']!=sha(data/'summary.json'):raise ValueError('Source samples changed')
    records=[];poses=[]
    for site,record in zip(sites,summary['records']):
        folder=root/site['component'];views=[]
        with np.load(data/site['component']/'mesh.npz',allow_pickle=False)as f:mesh={k:f[k]for k in f.files}
        for item in record['images']:
            if sha(folder/f'view-{item["view"]}'/f'light-{item["light"]}'/'image.png')!=item['image_sha256']:
                raise ValueError('Rendered pilot image changed')
        for view in range(4):
            path=folder/f'view-{view}'/'geometry.npz'
            if sha(path)!=record['geometry_sha256'][f'view-{view}/geometry.npz']:raise ValueError('Geometry changed')
            with np.load(path,allow_pickle=False)as f:v={k:f[k]for k in f.files}
            pose,k=v['cam2world'],v['intrinsics'];points=v['position'][32:-32:64,32:-32:64].reshape(-1,3)
            errors=local_surface_check(points.astype(float),pose,mesh,site,np.asarray(plan['mesh_native_offset_indices']))
            if not np.isfinite(errors).all() or errors.max()>.05:raise ValueError('Independent terrain triangle/ray check failed')
            yy,xx=np.mgrid[:512,:512];cam=(v['position']-pose[:3,3])@pose[:3,:3]
            projected=cam[...,:2]/cam[...,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
            pixel_error=np.linalg.norm(projected-np.stack([xx,yy],-1),axis=-1)
            if not np.isfinite(pixel_error).all() or pixel_error.max()>.02:raise ValueError('Native pixel-center convention failed')
            depth_error=np.abs(np.linalg.norm(v['position']-pose[:3,3],axis=-1)-v['ray_depth'])/v['ray_depth']
            if depth_error.max()>.0005:raise ValueError('Renderer ray-depth convention changed')
            records.append(dict(component=site['component'],view=view,independent_mesh_rays=len(points),
                maximum_mesh_error_m=float(errors.max()),maximum_native_projection_error_px=float(pixel_error.max()),
                maximum_ray_depth_relative_error=float(depth_error.max())))
            views.append(v)
        for qview in (1,2,3):
            path=folder/f'correspondences-0-{qview}.npz'
            if sha(path)!=record['labels'][qview-1]['sha256']:raise ValueError('Saved labels changed')
            with np.load(path,allow_pickle=False)as f:labels={k:f[k]for k in f.files}
            visible=labels['visible'];p=labels['reference'][visible];q=labels['query'][visible]
            a,b=views[0],views[qview]
            pose,info=propose_pose(p,q,a['axial_depth'],a['intrinsics'],a['cam2world'],b['intrinsics'])
            rotation=translation=None;correct=False
            if pose is not None:
                truth=b['cam2world'];translation=float(np.linalg.norm(pose[:3,3]-truth[:3,3]))
                rotation=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1))))
                reference_range=float(np.median(a['ray_depth']))
                correct=rotation<=2 and translation<=.01*reference_range
            poses.append(dict(component=site['component'],query_view=qview,visible_points=len(p),
                correct=bool(correct),translation_error_m=translation,rotation_error_deg=rotation,pose_info=info))
    count=sum(p['correct']for p in poses);rate=count/12
    result=dict(passed=rate>=.95,eligible_geometry_pairs=12,correct_oracle_poses=count,oracle_pose_rate=rate,
        geometry_controls=records,oracle_poses=poses,source_sha256=sha(__file__),renderer_summary_sha256=sha(root/'summary.json'),
        pilot_render_calendar_seconds=summary['calendar_seconds'],rendered_images=96,
        full44component_render_linear_projection_seconds=summary['calendar_seconds']*11,
        scope='Excluded pilots and common-estimator software oracle only. No model result or original fresh-LunarGeo gate.',
        fresh_test_gate_passed=False,matcher_inferences=0,calibration_or_test_pixels_decoded=0)
    out.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError('Actual terrain oracle fails95% gate; preserve report')
    print(json.dumps(dict(passed=True,correct_oracle_poses=count,eligible=12,render_seconds=summary['calendar_seconds'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/north-render-v31/pilots-v32')
    p.add_argument('--run',default='runs/north-v32-terrain-pilots');p.add_argument('--output',default='reports/north_terrain_pilot_audit_v32.json')
    a=p.parse_args();verify(a.data,a.run,a.output)
