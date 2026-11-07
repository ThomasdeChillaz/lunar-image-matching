"""Training-only orientation canonicalization, leaving v5 inference unchanged."""
import cv2
import numpy as np
import torch

from coarse_training_v5 import sample_training as wide_sample,transform_points
from contrastive_terrain import valid_centers
from rotation_bank_pose_v4 import quarter_turn
from train_balanced_localization_v3 import appearance


def canonical_quarter_turn(p,q,visible,size=128):
    """Choose the discrete query orientation nearest reference using train GT.

    No matcher prediction, pose success, brightness or evaluation data is used.
    Centering removes displacement; positive-scale least squares ranks turns.
    This helper belongs only to data augmentation, never inference.
    """
    p,q=np.asarray(p,float),np.asarray(q,float);visible=np.array(visible,bool,copy=True)
    if p.shape!=q.shape or p.ndim!=2 or p.shape[1]!=2 or visible.shape!=(len(p),):
        raise ValueError('Paired points and visibility mask required')
    visible&=np.isfinite(p).all(1)&np.isfinite(q).all(1)
    if visible.sum()<3:raise ValueError('At least three finite training correspondences required')
    a=p[visible]-p[visible].mean(0);den=float((a*a).sum())
    if den<=1e-8:raise ValueError('Degenerate training geometry')
    scores=[]
    for k in range(4):
        b=quarter_turn(q[visible],k,size);b-=b.mean(0)
        scale=max(0,float((a*b).sum())/max(float((b*b).sum()),1e-8))
        scores.append(float(((a-scale*b)**2).sum())/den)
    return min(range(4),key=lambda k:(scores[k],k))


def sample_training(variant,manifest,tiles,supervision,rng):
    if variant=='true_3d':return wide_sample('true_3d',manifest,tiles,supervision,rng)
    if variant!='canonical_3d':raise ValueError('Unknown orientation ablation')
    groups={}
    for t in tiles.values():
        if t['split']=='train':groups.setdefault(t['component'],[]).append(t['tile_id'])
    names=sorted(groups)
    for _ in range(200):
        group=names[int(rng.integers(len(names)))];ids=groups[group];key=ids[int(rng.integers(len(ids)))];z=supervision[key]
        la,lb=rng.integers(0,3,2);reference=z['images'][2*la];query=z['images'][2*lb+1]
        p=z['p'];use=z['visible'].copy()
        k=canonical_quarter_turn(p,z['q'],use)
        q=quarter_turn(z['q'],k,128);query=np.rot90(query,k).copy();source_mask=np.rot90(z['masks'][1],k).copy()
        angle=float(rng.uniform(-15,15));scale=float(np.exp(rng.uniform(np.log(.7),np.log(1.4))))
        matrix=cv2.getRotationMatrix2D((63.5,63.5),angle,scale);matrix[:,2]+=rng.uniform(-8,8,2)
        q=transform_points(q,matrix)
        warped=cv2.warpAffine(source_mask.astype(np.float32),matrix,(128,128),flags=cv2.INTER_LINEAR)>=.9999
        support=valid_centers(warped,12);inside=((q>=12)&(q<116)).all(1);safe=np.rint(q).astype(int).clip(0,127)
        use&=inside&support[safe[:,1],safe[:,0]];eligible=np.where(use)[0]
        if len(eligible)<8:continue
        indices=rng.choice(eligible,min(96,len(eligible)),replace=False)
        query=cv2.warpAffine(query,matrix,(128,128),flags=cv2.INTER_LINEAR)
        images=np.stack([appearance(im,rng) for im in (reference,query)])[:,None]
        return torch.from_numpy(images),torch.from_numpy(p[indices].astype(np.float32)),torch.from_numpy(q[indices].astype(np.float32)),dict(
            tile_id=key,component=group,light_a=int(la),light_b=int(lb),points=len(indices),training_quarter_turn=k)
    raise ValueError('Could not sample enough canonicalized 3D training correspondences')
