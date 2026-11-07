"""Persistent, source-pinned Blender BVH backend for reference-only Sun rays.

The owner publishes source/protocol and admits measured work before launching.
No image, query camera, query depth, or true query lighting file is loaded.
"""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from solar_lookup_v70 import validate_protocol
from solar_map_v63 import grid_envelope
from solar_physics_v63 import TerrainEnvelope,classify_solar_ray
from solar_active_v63 import HEIGHT_VARIANTS_M


def reference_anchor_intersections(intrinsics,pose,raycast):
    axis=49.5+16*np.rint(np.linspace(0,25,16));yy,xx=np.meshgrid(axis,axis,indexing='ij')
    points=np.column_stack([xx.ravel(),yy.ravel()]);rays=np.column_stack([points,np.ones(256)])@np.linalg.inv(intrinsics).T@pose[:3,:3].T
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    world=np.empty((256,7,3));normals=np.empty_like(world)
    for v,dz in enumerate(HEIGHT_VARIANTS_M):
        offset=np.array([0,0,dz])
        for i,ray in enumerate(rays):
            hit=raycast(pose[:3,3]-offset,ray,1e8)
            if hit is None:raise ValueError('Reference map lacks eligible anchor/height intersection')
            point,normal=map(lambda x:np.asarray(x,float),hit)
            if normal[2]<0:normal=-normal
            world[i,v]=point+offset;normals[i,v]=normal/np.linalg.norm(normal)
    return world,normals


def lookup_states(world,normals,suns,envelope,raycast,check=None):
    states=np.empty((len(world),len(suns),7),np.uint8);rays=0
    for v,dz in enumerate(HEIGHT_VARIANTS_M):
        offset=np.array([0,0,dz]);bound=TerrainEnvelope(envelope.xmin,envelope.xmax,envelope.ymin,envelope.ymax,envelope.maximum_height+dz,envelope.ceiling_valid)
        for i in range(len(world)):
            if check is not None:check()
            def first(origin,direction,limit):
                nonlocal rays
                rays+=1;hit=raycast(origin-offset,direction,limit)
                return None if hit is None else float(np.linalg.norm(np.asarray(hit[0])-(origin-offset)))
            for j,sun in enumerate(suns):states[i,j,v]=classify_solar_ray(world[i,v],normals[i,v],sun,bound,first)['state']
    return dict(solar_states=states[:,:,0].tolist(),solar_variants=states.tolist(),solar_rays=rays,computed_columns=len(suns))


def main(protocol_path,publication_commit,owner_pid):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    p,key=validate_protocol(protocol_path,publication_commit);started=time.monotonic();request_deadline=started+p['max_request_seconds']
    with np.load(p['reference_geometry'],allow_pickle=False)as z:k,pose=[z[name].copy()for name in('intrinsics','cam2world')]
    with np.load(p['mesh'],allow_pickle=False)as z:vertices,faces=z['vertices'].copy(),z['faces'].copy()
    with np.load(p['scene_mesh'],allow_pickle=False)as z:ceiling=float(z['vertices'][:,2].max())
    envelope=grid_envelope(vertices,faces,scene_maximum_height=ceiling)
    bvh=BVHTree.FromPolygons(vertices.tolist(),faces.tolist(),all_triangles=True,epsilon=0.)
    def check():
        if time.monotonic()>min(started+p['max_lookup_calendar_seconds'],request_deadline):raise TimeoutError('Original bounded solar-worker deadline exhausted')
        try:os.kill(owner_pid,0)
        except ProcessLookupError:raise RuntimeError('Solar lookup owner exited')
    def raycast(origin,direction,limit):
        check()
        point,normal,face,distance=bvh.ray_cast(Vector(origin),Vector(direction),float(limit))
        return None if point is None else(np.asarray(point,float),np.asarray(normal,float))
    world,normals=reference_anchor_intersections(k,pose,raycast)
    def emit(value):print('SOLAR_REPLY '+json.dumps(value,separators=(',',':'),allow_nan=False),flush=True)
    emit(dict(kind='ready',physical_key=key,reference_rays=256*7,bpy_version=bpy.app.version_string))
    for line in sys.stdin:
        request=json.loads(line)
        if set(request)!={'sun_vectors'}:raise ValueError('Sun vectors are the only lazy lookup input')
        suns=np.asarray(request['sun_vectors'],float)
        if suns.ndim!=2 or suns.shape[1]!=3 or not len(suns)or not np.isfinite(suns).all()or not np.allclose(np.linalg.norm(suns,axis=1),1,atol=1e-6):raise ValueError('Exact unit Sun vectors required')
        tick=time.monotonic();request_deadline=tick+p['max_request_seconds']
        result=lookup_states(world,normals,suns,envelope,raycast,check);result['worker_seconds']=time.monotonic()-tick
        emit(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--publication-commit',required=True);parser.add_argument('--owner-pid',required=True,type=int)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]if '--'in sys.argv else sys.argv[1:])
    try:main(args.protocol,args.publication_commit,args.owner_pid)
    except BaseException as error:
        print('SOLAR_REPLY '+json.dumps(dict(error=type(error).__name__+': '+str(error))),flush=True);raise
