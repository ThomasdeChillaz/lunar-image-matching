"""Reference-map solar lookup construction; no images, query cameras or truth.

The map is a complete structured triangulated height surface. A conservative
interior square certifies ray coverage; no-hit outside it stays unknown. The
ceiling is a declared finite-scene bound, not knowledge of unmapped lunar terrain.
Height perturbations rigidly shift the entire reference surface by +/-5/10/20m;
we re-intersect reference rays, rather than just shifting correspondence points.
"""
import math
import numpy as np
from solar_physics_v63 import TerrainEnvelope,classify_solar_ray
from solar_active_v63 import HEIGHT_VARIANTS_M


def grid_envelope(vertices,faces,scene_maximum_height=None):
    v=np.asarray(vertices,float);f=np.asarray(faces)
    n=math.isqrt(len(v))
    if n*n!=len(v)or v.shape!=(n*n,3)or n<3 or f.shape!=(2*(n-1)**2,3)or not np.isfinite(v).all():raise ValueError('Complete finite square structured terrain grid required')
    from north_terrain_pilot_v32 import grid_faces
    expected=np.sort(grid_faces(n),axis=1)
    if not np.array_equal(np.sort(f,axis=1),expected):raise ValueError('Missing, extra or reordered terrain triangles forbidden')
    tri=v[f];area=np.cross(tri[:,1,:2]-tri[:,0,:2],tri[:,2,:2]-tri[:,0,:2])
    if not((area>1e-8).all()or(area< -1e-8).all()):raise ValueError('Folded/degenerate projected terrain has no certified coverage')
    g=v.reshape(n,n,3)
    boundary=np.concatenate([g[0,:-1],g[:-1,-1],g[-1,:0:-1],g[:0:-1,0]])[:,:2]
    centre=g[n//2,n//2,:2];edges=np.roll(boundary,-1,axis=0)-boundary
    inward=np.column_stack([-edges[:,1],edges[:,0]])
    orientation=np.sign(np.sum(boundary[:,0]*np.roll(boundary[:,1],-1)-boundary[:,1]*np.roll(boundary[:,0],-1)))
    inward*=orientation;distance=np.sum((centre-boundary)*inward,axis=1)
    if (distance<=0).any():raise ValueError('No certified star-shaped central map support')
    half=float(np.min(distance/(np.abs(inward).sum(1))))*.999
    observed_maximum=float(v[:,2].max())
    if scene_maximum_height is None:
        # A cropped map maximum is not a bound on the original renderer scene.
        # It can establish shadows from observed hits, never a clear ray.
        ceiling=observed_maximum;certified=False
    else:
        ceiling=float(scene_maximum_height)
        if not np.isfinite(ceiling)or ceiling<observed_maximum-1e-6:raise ValueError('Declared original-scene ceiling does not bound map heights')
        certified=True
    return TerrainEnvelope(centre[0]-half,centre[0]+half,centre[1]-half,centre[1]+half,ceiling,certified)


def reference_solar_table(points,intrinsics,cam2world,sun_vectors,envelope,raycast):
    """raycast(origin,direction,maxdist)->(worldpoint,normal) or None.

    One immutable base-surface accelerator suffices: shifting ray origins by
    -dz is exactly equivalent to globally perturbing all terrain heights by dz.
    No camera visibility labels, brightness or query metadata are consumed.
    """
    points=np.asarray(points,float);k=np.asarray(intrinsics,float);pose=np.asarray(cam2world,float);suns=np.asarray(sun_vectors,float)
    if points.ndim!=2 or points.shape[1]!=2 or k.shape!=(3,3)or pose.shape!=(4,4)or suns.ndim!=2 or suns.shape[1]!=3:raise ValueError('Reference points/camera and Sun catalogue required')
    rays=np.column_stack([points,np.ones(len(points))])@np.linalg.inv(k).T@pose[:3,:3].T;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    world=np.full((len(points),7,3),np.nan);normals=np.full((len(points),3),np.nan);states=np.zeros((len(points),len(suns),7),np.uint8);reasons={}
    for v,dz in enumerate(HEIGHT_VARIANTS_M):
        offset=np.array([0,0,dz]);bound=TerrainEnvelope(envelope.xmin,envelope.xmax,envelope.ymin,envelope.ymax,envelope.maximum_height+dz,envelope.ceiling_valid)
        for i,ray in enumerate(rays):
            hit=raycast(pose[:3,3]-offset,ray,1e8)
            if hit is None:raise ValueError('Reference or perturbed map misses an anchor ray; qualify pair before inference')
            point,normal=map(lambda a:np.asarray(a,float),hit);point=point+offset
            if normal[2]<0:normal=-normal
            world[i,v]=point
            if v==0:normals[i]=normal/np.linalg.norm(normal)
            def first(origin,direction,limit):
                candidate=raycast(origin-offset,direction,limit)
                if candidate is None:return None
                return float(np.linalg.norm(np.asarray(candidate[0])-(origin-offset)))
            for j,sun in enumerate(suns):
                r=classify_solar_ray(point,normal,sun,bound,first);states[i,j,v]=r['state'];reasons[r['reason']]=reasons.get(r['reason'],0)+1
    return dict(world_variants=world,normals=normals,solar_states=states[:,:,0],solar_variants=states),reasons
