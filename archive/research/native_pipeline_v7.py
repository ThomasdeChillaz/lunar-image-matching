"""Frozen coarse proposals, learned native refinement and reference-map pose."""
import cv2
import numpy as np
import torch

from localization_diagnostics_v3 import native_points
from localization_robust_pose_v4 import CONFIGS,estimate,residuals
from lunar_geometry_v5 import map_points_z
from native_refinement_v7 import refine
from rotation_bank_pose_v4 import quarter_turn,choose_branch


def infer(coarse,fine,reference,query,reference_depth,reference_intrinsics,reference_cam2world,
          query_intrinsics,features=None,feature_ids=('reference','query')):
    if reference.shape!=query.shape or reference.ndim!=2 or reference.shape[0]!=reference.shape[1]:
        raise ValueError('Equal square grayscale images required')
    if features is None:features={}
    config=next(c for c in CONFIGS if c['name']=='ap3p_reclassify');attempts=[]
    def encoded(image,identity,k):
        key=(identity,k)
        if key not in features:
            with torch.inference_mode():features[key]=fine.encoder(torch.from_numpy(np.rot90(image,k).copy()).float()[None,None]/255)
        return features[key]
    da=encoded(reference,feature_ids[0],0)
    for k in range(4):
        p,q=coarse.points(*[cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA) for im in (reference,np.rot90(query,k).copy())])
        p,q=[native_points(v,reference.shape) for v in (p,q)]
        with torch.inference_mode():p,q,local=refine(fine,p,q,da,encoded(query,feature_ids[1],k))
        q=quarter_turn(q,k,query.shape[0],True)
        world,use=map_points_z(p,reference_depth,reference_intrinsics,reference_cam2world)
        pose,info=estimate(world,q[use],query_intrinsics,config);median=None
        if pose is not None:
            r=pose[:3,:3].T;t=-r@pose[:3,3]
            error=residuals(world,q[use],np.asarray(query_intrinsics),cv2.Rodrigues(r)[0],t)
            median=float(np.median(error[error<=3])) if (error<=3).any() else float('inf')
        attempts.append(dict(p=p,q=q,pose=pose,info=info,quarter_turn=k,median_inlier_residual=median,
            supported_matches=local['supported'],no_match_count=local['no_match'],point_reliability=local['point_reliability']))
    return choose_branch(attempts),attempts
