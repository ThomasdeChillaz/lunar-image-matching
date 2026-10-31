"""Image/reference-map pose proposals with audited axial-depth geometry.

No query depth or true query pose is accepted by this inference interface.
Returned poses are uncalibrated research proposals, not accepted navigation.
"""
import cv2
import numpy as np
import torch

from localization_diagnostics_v3 import native_points,refine_native
from localization_robust_pose_v4 import CONFIGS,estimate,residuals
from lunar_geometry_v5 import map_points_z
from rotation_bank_pose_v4 import quarter_turn,choose_branch


def infer(matcher,fine,reference,query,reference_depth,reference_intrinsics,reference_cam2world,
          query_intrinsics,features=None,feature_ids=('reference','query'),rotations=(0,1,2,3)):
    if reference.shape!=query.shape or reference.ndim!=2 or reference.shape[0]!=reference.shape[1]:
        raise ValueError('Equal square grayscale input required')
    if features is None:features={}
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify');candidates=[]
    def encoded(image,identity,k):
        key=(identity,k)
        if key not in features:
            with torch.inference_mode():features[key]=fine(torch.from_numpy(np.rot90(image,k).copy()).float()[None,None]/255)[0]
        return features[key]
    da=encoded(reference,feature_ids[0],0)
    for k in rotations:
        rotated=np.rot90(query,k).copy()
        low=[cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA) for im in (reference,rotated)]
        p,q=matcher.points(*low);p,q=[native_points(v,reference.shape) for v in (p,q)]
        with torch.inference_mode():p,q,_=refine_native(p,q,da,encoded(query,feature_ids[1],k))
        q=quarter_turn(q,k,query.shape[0],True)
        world,use=map_points_z(p,reference_depth,reference_intrinsics,reference_cam2world)
        pose,info=estimate(world,q[use],query_intrinsics,config);median=None
        if pose is not None:
            r=pose[:3,:3].T;t=-r@pose[:3,3]
            error=residuals(world,q[use],np.asarray(query_intrinsics),cv2.Rodrigues(r)[0],t)
            median=float(np.median(error[error<=3])) if (error<=3).any() else float('inf')
        candidates.append(dict(p=p,q=q,pose=pose,info=info,quarter_turn=k,median_inlier_residual=median))
    return choose_branch(candidates),candidates
