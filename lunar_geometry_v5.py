"""Explicit axial-depth geometry for the audited released LunarGeo EXRs.

Legacy ray-range functions remain unchanged for historical replay. Source
intrinsics are retained exactly; no performance-fitted pixel-center shift.
"""
import cv2
import numpy as np

from lunar_geometry_v2 import camera_check


def world_points_z(points,depths,k,pose):
    k,pose=camera_check(k,pose);points=np.asarray(points,float).reshape(-1,2);depths=np.asarray(depths,float)
    if depths.shape!=(len(points),):raise ValueError('One optical-axis z depth per point required')
    rays=np.column_stack([points,np.ones(len(points))])@np.linalg.inv(k).T
    return (rays*depths[:,None])@pose[:3,:3].T+pose[:3,3]


def project_z(world,k,pose):
    k,pose=camera_check(k,pose);camera=(np.asarray(world,float)-pose[:3,3])@pose[:3,:3]
    homogeneous=camera@k.T
    with np.errstate(divide='ignore',invalid='ignore'):q=homogeneous[:,:2]/homogeneous[:,2,None]
    return q,camera[:,2],camera[:,2]>0


def map_points_z(points,depth,k,pose):
    p=np.asarray(points,float).reshape(-1,2);indices=np.rint(p).astype(int)
    inside=((indices>=0)&(indices<np.array(depth.shape[::-1]))).all(1)
    indices=indices.clip(0,np.array(depth.shape[::-1])-1);z=depth[indices[:,1],indices[:,0]]
    inside&=np.isfinite(z)&(z>0)
    return world_points_z(p[inside],z[inside],k,pose),inside


def correspondences_z(da,db,ka,ca,kb,cb,points=None,stride=4,relative_tolerance=.01,absolute_tolerance=.05):
    """Same historical sampling/support/tolerance rules, corrected depth meaning.

    Visibility-tolerance sensitivity is a separate experiment. Brightness is
    never a visibility label. No pose/matcher is used to build correspondence GT.
    """
    da,db=np.asarray(da,float),np.asarray(db,float)
    if points is None:
        y,x=np.mgrid[0:da.shape[0]:stride,0:da.shape[1]:stride];points=np.column_stack([x.ravel(),y.ravel()])
    p=np.asarray(points,float).reshape(-1,2)
    if not len(p):return p.copy(),p.copy(),np.empty((0,3)),np.zeros(0,bool)
    world,valid=map_points_z(p,da,ka,ca)
    all_world=np.zeros((len(p),3));all_world[valid]=world
    q,predicted,front=project_z(all_world,kb,cb)
    valid&=front&np.isfinite(q).all(1)&((q>=0)&(q<np.array(db.shape[::-1])-1)).all(1)
    safe=np.where(np.isfinite(q),q,-1).astype(np.float32)
    observed=cv2.remap(np.where(np.isfinite(db),db,0).astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR).ravel()
    support=cv2.remap((np.isfinite(db)&(db>0)).astype(np.float32),safe[:,0,None],safe[:,1,None],cv2.INTER_LINEAR).ravel()
    valid&=(support>=.9999)&(np.abs(observed-predicted)<=absolute_tolerance+relative_tolerance*predicted)
    return p[valid],q[valid],all_world[valid],valid


def median_reference_range_z(depth,k):
    """Physical Euclidean camera-to-surface range, not median axial depth."""
    y,x=np.mgrid[:depth.shape[0],:depth.shape[1]];p=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])
    rays=p@np.linalg.inv(np.asarray(k,float)).T;z=depth.ravel();valid=np.isfinite(z)&(z>0)
    if not valid.any():raise ValueError('No finite reference surface')
    return float(np.median(z[valid]*np.linalg.norm(rays[valid],axis=1)))
