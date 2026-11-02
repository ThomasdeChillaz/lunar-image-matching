"""Sampled independent full-mesh first-hit checks, with bounded chunk memory."""
import numpy as np
from verify_north_development_v35 import triangle_hits


def first_mesh_hit(origin,ray,vertices,faces,chunk_size=16384):
 origin=np.asarray(origin,float);ray=np.asarray(ray,float);vertices=np.asarray(vertices);faces=np.asarray(faces)
 if origin.shape!=(3,)or ray.shape!=(3,)or not np.isfinite(origin).all()or not np.isfinite(ray).all()or abs(np.linalg.norm(ray)-1)>1e-10:raise ValueError('Finite origin and unit world ray required')
 if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer):raise ValueError('Triangular mesh vertices/indices required')
 if type(chunk_size)is not int or chunk_size<1:raise ValueError('Positive bounded triangle chunk required')
 best=float('inf')
 for start in range(0,len(faces),chunk_size):
  triangles=vertices[faces[start:start+chunk_size]].astype(float)
  best=min(best,triangle_hits(origin,ray,triangles))
 return best


def visibility_check(origin,point,vertices,faces):
 origin,point=np.asarray(origin,float),np.asarray(point,float);direction=point-origin;distance=float(np.linalg.norm(direction))
 if not np.isfinite(distance)or distance<=0:raise ValueError('Positive finite target distance required')
 hit=first_mesh_hit(origin,direction/distance,vertices,faces)
 visible=bool(np.isfinite(hit)and abs(hit-distance)<=.05);occluded=bool(np.isfinite(hit)and hit<distance-.05)
 return dict(visible=visible,geometrically_occluded=occluded,visibility_uncertain=not visible and not occluded,
  first_hit_m=None if not np.isfinite(hit)else hit,target_distance_m=distance)
