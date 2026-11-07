"""Exact BVH ray labels with separate source/target camera calibration."""
import numpy as np
from north_development_plan_v35 import world_rays


def mesh_correspondences(bvh,source_pose,target_pose,source_intrinsics,target_intrinsics,source_points,size=512):
 p=np.asarray(source_points,float);a=np.asarray(source_pose,float);b=np.asarray(target_pose,float);ka=np.asarray(source_intrinsics,float);kb=np.asarray(target_intrinsics,float)
 if p.ndim!=2 or p.shape[1]!=2 or ka.shape!=(3,3)or kb.shape!=(3,3):raise ValueError('Native Nx2 rays and both3x3 camera calibrations required')
 rays=world_rays(p,ka,a);world=np.zeros((len(p),3));source_valid=np.zeros(len(p),bool)
 for i,direction in enumerate(rays):
  hit,normal,index,travel=bvh.ray_cast(a[:3,3].tolist(),direction.tolist(),50000.)
  if hit is not None:world[i]=hit[:];source_valid[i]=True
 cam=(world-b[:3,3])@b[:3,:3];homogeneous=cam@kb.T;q=np.zeros((len(p),2));front=source_valid&(cam[:,2]>0)
 q[front]=homogeneous[front,:2]/homogeneous[front,2,None]
 in_frame=front&((q>=0)&(q<=size-1)).all(1);visible=np.zeros(len(p),bool)
 occluded=np.zeros(len(p),bool);hit_distance=np.full(len(p),np.nan)
 for i in np.flatnonzero(in_frame):
  direction=world[i]-b[:3,3];distance=np.linalg.norm(direction);direction/=distance
  hit,normal,index,travel=bvh.ray_cast(b[:3,3].tolist(),direction.tolist(),50000.)
  visible[i]=hit is not None and abs(travel-distance)<=.05
  if hit is not None:
   hit_distance[i]=travel;occluded[i]=travel<distance-.05
 return dict(source_points=p,target_points=q,world=world,source_valid=source_valid,in_frame=in_frame,visible=visible,
  geometrically_occluded=occluded,target_hit_distance_m=hit_distance,visibility_uncertain=in_frame&~visible&~occluded)
