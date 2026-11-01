"""Independent triangle-ray fixture and common AP3P replay of actual renders."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np



def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def triangles():
    # Independent explicit triangles rather than the renderer's plane/slab code.
    floor=np.array([[-50,-50,0],[50,-50,0],[50,50,0],[-50,50,0]],float)
    cube=np.array([[x,y,z]for z in (0.,2.)for y in (-.6,1.4)for x in (.5,2.5)])
    faces=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    out=[floor[[0,1,2]],floor[[0,2,3]]]
    for a,b,c,d in faces:out.extend([cube[[a,b,c]],cube[[a,c,d]]])
    return np.array(out)


def trace(origins,directions):
    """Independent double-sided Moller-Trumbore first intersections."""
    origins=np.broadcast_to(origins,directions.shape);best=np.full(len(directions),np.inf)
    for a,b,c in triangles():
        e1,e2=b-a,c-a;p=np.cross(directions,e2);det=p@e1
        valid=np.abs(det)>1e-12;inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
        tvec=origins-a;u=np.sum(tvec*p,axis=1)*inv;q=np.cross(tvec,e1)
        v=np.sum(directions*q,axis=1)*inv;t=(q@e2)*inv
        use=valid&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(t>0)
        best=np.minimum(best,np.where(use,t,np.inf))
    return best


def project(world,pose,k):
    camera=(world-pose[:3,3])@pose[:3,:3];h=camera@k.T
    return h[:,:2]/h[:,2,None],camera[:,2]


def audit(run,output):
    from native_baseline_pose_v7 import propose_pose
    import cv2
    cv2.setNumThreads(1)
    root,out=Path(run),Path(output);summary=json.loads((root/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if out.exists():raise FileExistsError('Preserve prior actual renderer audits')
    if not summary['passed'] or summary['protocol_sha256']!=sha(root/'protocol.json') or len(summary['records'])!=2:
        raise ValueError('All actual renderer controls must pass before replay')
    for source,digest in protocol['source_sha256'].items():
        if sha(source)!=digest:raise ValueError('Executed renderer source changed')
    datasets=[];checks=[]
    for record in summary['records']:
        folder=root/f'view-{record["view"]}';path=folder/'geometry.npz'
        if sha(path)!=record['geometry_sha256'] or sha(folder/'image.png')!=record['image_sha256']:
            raise ValueError('Saved renderer bytes changed')
        with np.load(path,allow_pickle=False)as f:data={k:f[k].copy()for k in f.files}
        h,w=data['interior'].shape;k,pose=data['intrinsics'],data['cam2world']
        yy,xx=np.mgrid[4:h-4:4,4:w-4:4];points=np.stack([xx.ravel(),yy.ravel()],1).astype(float)
        points=points[data['interior'][points[:,1].astype(int),points[:,0].astype(int)]]
        directions=np.c_[(points[:,0]-k[0,2])/k[0,0],(points[:,1]-k[1,2])/k[1,1],np.ones(len(points))]@pose[:3,:3].T
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        distance=trace(pose[:3,3],directions);expected=pose[:3,3]+directions*distance[:,None]
        actual=data['position'][points[:,1].astype(int),points[:,0].astype(int)]
        error=np.linalg.norm(actual-expected,axis=1)
        if not len(error) or not np.isfinite(error).all() or error.max()>.002:
            raise ValueError('Independent triangle intersections differ from actual render positions')
        query,z=project(actual,pose,k)
        if np.linalg.norm(query-points,axis=1).max()>.02:raise ValueError('Native coordinate/vertical convention differs')
        checks.append(dict(view=record['view'],analytic_triangle_rays=len(points),maximum_world_error_m=float(error.max())))
        data.update(points=points,independent_world=expected);datasets.append(data)
    poses=[]
    for index,(a,b)in enumerate((datasets,datasets[::-1])):
        points,world=a['points'],a['independent_world'];k,pose=b['intrinsics'],b['cam2world'];q,front=project(world,pose,k)
        rays=world-pose[:3,3];ranges=np.linalg.norm(rays,axis=1);rays/=ranges[:,None]
        visible=(front>0)&(np.linalg.norm((pose[:3,3]+rays*trace(pose[:3,3],rays)[:,None])-world,axis=1)<.001)
        height,width=b['interior'].shape;visible&=((q>=2)&(q<np.array([width-2,height-2]))).all(1)
        ref_pose=a['cam2world'];axial=((a['position']-ref_pose[:3,3])@ref_pose[:3,:3])[...,2]
        estimated,info=propose_pose(points[visible],q[visible],axial,a['intrinsics'],ref_pose,k)
        if estimated is None:raise ValueError('Exact visible correspondences did not recover control pose')
        translation=float(np.linalg.norm(estimated[:3,3]-pose[:3,3]))
        rotation=float(np.degrees(np.arccos(np.clip((np.trace(estimated[:3,:3].T@pose[:3,:3])-1)/2,-1,1))))
        if translation>.002 or rotation>.02:raise ValueError('Oracle control pose outside declared stringent software tolerance')
        poses.append(dict(pair=index,visible_points=int(visible.sum()),translation_error_m=translation,rotation_error_deg=rotation,pose_info=info))
    result=dict(passed=True,actual_render_views_replayed=len(checks),independent_triangle_controls=checks,
        common_AP3P_oracle_control_poses=poses,renderer_summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),
        source_height_pixels_decoded=0,lunar_images_rendered=0,matcher_inferences=0,
        scope='Analytic plane+box software camera/depth/visibility and common AP3P oracle controls only. Two constructed poses are not fresh lunar terrain or the required95%development oracle gate.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.run,a.output)
