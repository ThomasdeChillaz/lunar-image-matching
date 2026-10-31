"""Versioned development pose hypotheses; inference never receives query truth."""
import cv2
import numpy as np

from localization_pose_refit_v3 import estimate_refined_pose


CONFIGS = [
    dict(name='frozen_epnp',solver='legacy',iterations=2000,reclassify=False),
    dict(name='ap3p',solver='ap3p',iterations=2000,reclassify=False),
    dict(name='ap3p_reclassify',solver='ap3p',iterations=2000,reclassify=True),
    dict(name='ap3p_10k',solver='ap3p',iterations=10000,reclassify=True),
    dict(name='usac_msac',solver='msac',iterations=2000,reclassify=True),
    dict(name='usac_magsac_10k',solver='magsac',iterations=10000,reclassify=True),
]
SEED=20261101


def residuals(world, pixels, k, rvec, tvec):
    rotation=cv2.Rodrigues(rvec)[0]
    camera=world@rotation.T+np.asarray(tvec).reshape(1,3)
    projected=camera@k.T
    with np.errstate(divide='ignore',invalid='ignore'):
        error=np.linalg.norm(projected[:,:2]/projected[:,2,None]-pixels,axis=1)
    return np.where((camera[:,2]>0)&np.isfinite(error),error,np.inf)


def estimate(world, pixels, intrinsics, config):
    world=np.ascontiguousarray(world,dtype=np.float64).reshape(-1,3)
    pixels=np.ascontiguousarray(pixels,dtype=np.float64).reshape(-1,2)
    k=np.asarray(intrinsics,dtype=np.float64)
    if len(world)!=len(pixels) or k.shape!=(3,3):raise ValueError('Paired 3D/2D points and 3x3 intrinsics required')
    if len(world)<6 or not np.isfinite(world).all() or not np.isfinite(pixels).all():return None,dict(inliers=0,reason='insufficient_or_invalid_matches')
    if config['solver']=='legacy':
        pose,n=estimate_refined_pose(world,pixels,k)
        return pose,dict(inliers=int(n),reason=None if pose is not None else 'ransac_failed')
    center=world.mean(0); object_points=world-center
    if np.linalg.matrix_rank(object_points,tol=1e-7)<2:return None,dict(inliers=0,reason='degenerate_geometry')
    cv2.setRNGSeed(SEED)
    try:
        if config['solver']=='ap3p':
            ok,rv,tv,indices=cv2.solvePnPRansac(object_points,pixels,k,None,
                iterationsCount=config['iterations'],reprojectionError=3.,confidence=.99,flags=cv2.SOLVEPNP_AP3P)
        else:
            params=cv2.UsacParams();params.randomGeneratorState=SEED;params.isParallel=False
            params.maxIterations=config['iterations'];params.threshold=3.;params.confidence=.99
            params.score=cv2.SCORE_METHOD_MAGSAC if config['solver']=='magsac' else cv2.SCORE_METHOD_MSAC
            ok,_,rv,tv,indices=cv2.solvePnPRansac(object_points,pixels,k,None,params=params)
        if not ok or indices is None or len(indices)<6:return None,dict(inliers=0,reason='ransac_failed')
        use=indices.ravel()
        rv,tv=cv2.solvePnPRefineLM(object_points[use],pixels[use],k,None,rv,tv)
        err=residuals(object_points,pixels,k,rv,tv)
        best=(float(np.minimum(err**2,9).sum()),rv.copy(),tv.copy())
        if config['reclassify']:
            for _ in range(3):
                use=np.where(err<=3.)[0]
                if len(use)<6:break
                rv,tv=cv2.solvePnPRefineLM(object_points[use],pixels[use],k,None,rv,tv)
                err=residuals(object_points,pixels,k,rv,tv)
                cost=float(np.minimum(err**2,9).sum())
                if cost<best[0]:best=(cost,rv.copy(),tv.copy())
        _,rv,tv=best;err=residuals(object_points,pixels,k,rv,tv);n=int((err<=3).sum())
        if n<6:return None,dict(inliers=n,reason='insufficient_front_facing_inliers')
        rotation=cv2.Rodrigues(rv)[0];pose=np.eye(4)
        pose[:3,:3]=rotation.T;pose[:3,3]=center-rotation.T@tv.ravel()
        return pose,dict(inliers=n,reason=None,clipped_squared_reprojection_cost=best[0])
    except cv2.error as exc:
        return None,dict(inliers=0,reason='opencv_failure',detail=str(exc).splitlines()[-1])
