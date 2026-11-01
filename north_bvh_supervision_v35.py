"""Geometric fractional-pixel supervision, independent of rendered brightness."""
import numpy as np

from north_development_plan_v35 import world_rays


def mesh_correspondences(bvh,source_pose,target_pose,intrinsics,source_points,size=512):
    p=np.asarray(source_points,float);a=np.asarray(source_pose,float);b=np.asarray(target_pose,float)
    k=np.asarray(intrinsics,float);rays=world_rays(p,k,a)
    world=np.zeros((len(p),3));source_valid=np.zeros(len(p),bool)
    for i,direction in enumerate(rays):
        hit,normal,index,travel=bvh.ray_cast(a[:3,3].tolist(),direction.tolist(),50000.)
        if hit is not None:world[i]=hit[:];source_valid[i]=True
    cam=(world-b[:3,3])@b[:3,:3];q=np.zeros((len(p),2))
    front=source_valid&(cam[:,2]>0)
    q[front]=cam[front,:2]/cam[front,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    in_frame=front&((q>=0)&(q<=size-1)).all(1);visible=np.zeros(len(p),bool)
    for i in np.flatnonzero(in_frame):
        direction=world[i]-b[:3,3];distance=np.linalg.norm(direction);direction/=distance
        hit,normal,index,travel=bvh.ray_cast(b[:3,3].tolist(),direction.tolist(),50000.)
        visible[i]=hit is not None and abs(travel-distance)<=.05
    return dict(source_points=p,target_points=q,world=world,source_valid=source_valid,
                in_frame=in_frame,visible=visible)
