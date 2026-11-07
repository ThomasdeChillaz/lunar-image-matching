"""Shared-camera multiscale photometric refinement, followed by measured native matches."""

from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solar-local-v127"))
import pipeline
from lunar_geometry_v5 import map_points_z, project_z

SIGMAS = (16., 8., 4., 2.)
MAX_STEPS = 35
MAX_SEEDS = 8
STRIDE = 8


def bilinear(image, xy):
    safe = np.clip(xy, [0., 0.], [image.shape[1]-1.000001, image.shape[0]-1.000001])
    x, y = np.floor(safe).astype(int).T
    dx, dy = (safe-np.floor(safe)).T
    return ((1-dx)*(1-dy)*image[y,x]+dx*(1-dy)*image[y,x+1]
            +(1-dx)*dy*image[y+1,x]+dx*dy*image[y+1,x+1])


def unit(values):
    centered = values-values.mean()
    return centered/max(float(np.sqrt(np.mean(centered*centered))), 1e-6)


def pose_parameters(pose, center):
    rotation = pose[:3,:3].T
    return np.r_[cv2.Rodrigues(rotation)[0].ravel(), rotation@(center-pose[:3,3])]


def camera_pose(parameters, center):
    rotation = cv2.Rodrigues(parameters[:3])[0]
    pose = np.eye(4)
    pose[:3,:3] = rotation.T
    pose[:3,3] = center-rotation.T@parameters[3:]
    return pose


def optimize(residual, initial):
    """Small six-variable robust Levenberg–Marquardt solver using NumPy only."""
    scale=np.r_[np.full(3,.05),np.full(3,50.)]
    parameters=initial.copy()
    damping=.01
    value=residual(parameters); evaluations=1
    def cost(r):
        return float(np.mean(2*.5**2*(np.sqrt(1+(r/.5)**2)-1)))
    current=cost(value)
    for _ in range(MAX_STEPS):
        columns=[]
        for axis in range(6):
            step=np.zeros(6);step[axis]=scale[axis]*1e-3
            columns.append((residual(parameters+step)-value)/1e-3)
        evaluations+=6
        jacobian=np.column_stack(columns)
        weights=1/np.sqrt(1+(value/.5)**2)
        normal=jacobian.T@(weights[:,None]*jacobian)
        rhs=jacobian.T@(weights*value)
        try:
            delta=np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal),1e-9)),-rhs)
        except np.linalg.LinAlgError:
            break
        delta*=min(1.,2./max(float(np.abs(delta).max()),1e-9))
        trial=parameters+scale*delta
        observed=residual(trial);evaluations+=1
        candidate=cost(observed)
        if candidate<current:
            improvement=current-candidate
            parameters,value,current=trial,observed,candidate
            damping=max(1e-6,damping*.4)
            if np.abs(delta).max()<1e-3 or improvement<1e-5*max(current,1e-6):
                break
        else:
            damping=min(1e8,damping*4.)
    return parameters,evaluations


def refine(reference, query, depth, k, reference_pose, query_k, seeds, *, reference_valid=None):
    """Optimize image evidence, never query depth/pose/Sun; no accepted pose is returned."""
    support=np.isfinite(depth)&(depth>0)
    if reference_valid is not None:
        support &= reference_valid
    a, b, _ = pipeline.common_images(reference, query, support)
    a, b = a.astype(float)/255., b.astype(float)/255.
    if min(float(a.std()),float(b.std()))<1e-6:
        return []
    y,x = np.mgrid[100:416:STRIDE,100:416:STRIDE]
    points = np.c_[x.ravel(),y.ravel()].astype(float)
    world, valid = map_points_z(points,depth,k,reference_pose)
    available=support[points[:,1].astype(int),points[:,0].astype(int)]
    world=world[available[valid]]
    points = points[valid & available]
    center = world.mean(axis=0) if len(world) else np.zeros(3)
    levels = [(cv2.GaussianBlur(a,(0,0),s),cv2.GaussianBlur(b,(0,0),s)) for s in SIGMAS]
    hypotheses=[]
    for index, seed in enumerate(seeds[:MAX_SEEDS]):
        projected,_,front=project_z(world,query_k,seed)
        keep=front & np.isfinite(projected).all(axis=1)&(projected>32.).all(axis=1)&(projected<[480.,480.]).all(axis=1)
        if keep.sum()<256:
            continue
        # Every hypothesis optimizes and is ranked on the same reference points.
        # Initial image overlap only decides whether optimization is feasible.
        chosen=world
        camera_points=points
        x0=pose_parameters(seed,center)
        calls=[]
        for ref,obs in levels:
            expected=unit(bilinear(ref,camera_points))
            def residual(parameters):
                pose=camera_pose(parameters,center)
                xy,z,front=project_z(chosen,query_k,pose)
                # Always retain the same reference samples during a level.
                # Moving points outside the image cannot discard a residual.
                samples=bilinear(obs,np.nan_to_num(xy,nan=0.,posinf=511.,neginf=0.))
                photometry=np.clip(unit(samples)-expected,-4.,4.)
                border=np.maximum(-xy,0.)+np.maximum(xy-[511.,511.],0.)
                penalty=np.minimum(np.linalg.norm(np.nan_to_num(border,nan=512.),axis=1)/16.,32.)
                penalty=np.where(front,penalty,32.)
                return np.r_[photometry,penalty]
            x0,evaluations=optimize(residual,x0)
            calls.append(evaluations)
        pose=camera_pose(x0,center)
        error=residual(x0)
        score=float(np.mean(2*.5**2*(np.sqrt(1+(error/.5)**2)-1)))
        hypotheses.append(dict(seed_index=index,pose=pose,score=score,points=len(chosen),evaluations=calls))
    hypotheses.sort(key=lambda row:(row["score"],row["seed_index"]))
    return hypotheses


ORIGINAL_RUN = pipeline.run


def run(reference,query,depth,k,reference_pose,query_k,seeds,library,**kwargs):
    result=ORIGINAL_RUN(reference,query,depth,k,reference_pose,query_k,seeds,library,**kwargs)
    tick=perf_counter()
    hypotheses=refine(reference,query,depth,k,reference_pose,query_k,seeds,
                      reference_valid=kwargs.get("reference_valid"))
    seconds=perf_counter()-tick
    chosen=np.asarray([h["pose"] for h in hypotheses[:1]]).reshape(-1,4,4)
    # The refined camera only initializes image measurement. Final poses still
    # come from the same AP3P estimator on actual reciprocal patch observations.
    measured=measure(reference,query,depth,k,reference_pose,query_k,chosen,
                     reference_valid=kwargs["reference_valid"])
    for method,row in measured.items():
        row["inference_seconds"]+=seconds+kwargs.get("initialization_seconds",0.)
        result["methods"]["shared_camera_"+method]=row
    result["photometric_initialization"]=[dict(h,pose=h["pose"].tolist()) for h in hypotheses]
    result["shared_camera_seconds"]=seconds
    return result


def measure(reference,query,depth,k,reference_pose,query_k,seeds,*,reference_valid):
    tick=perf_counter()
    a,b,_=pipeline.common_images(reference,query,reference_valid)
    prepared=None
    if len(seeds):
        warper=pipeline.previous.ReferencePatchWarper(a.astype(float),depth,k,reference_pose,
            pipeline.previous.ANCHORS,reference_valid=reference_valid)
        plane=warper.warp(query_k,seeds[0],reference.shape)["planar"]
        eligible=np.array([v and pipeline.previous.template_available(plane.detector_valid,c)
                           for v,c in zip(plane.anchor_visible,plane.predicted_xy,strict=True)])
        selection=pipeline.select(plane.reference_world,pipeline.previous.ray_plane_points(warper),
                                  pipeline.previous.ANCHORS,query_k,seeds,eligible,budget=12)
        prepared=plane,selection
    setup=perf_counter()-tick
    outputs={}
    for name in ("pooled_uniform","pooled_parallax"):
        tick=perf_counter()
        if prepared is None:
            row=dict(**pipeline.solve([],[],depth,k,reference_pose,query_k),failures=[],searches=[],
                     selected_anchors=[],attempted_patches=0)
        else:
            plane,selection=prepared
            indices=selection["active"] if name.endswith("parallax") else selection["uniform"]
            row=pipeline.patch_measure(plane,b.astype(float),indices,pipeline.search_patch,
                                       depth,k,reference_pose,query_k)
        outputs[name]=dict(row,inference_seconds=perf_counter()-tick+setup)
    return outputs
