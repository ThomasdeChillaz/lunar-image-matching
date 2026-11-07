"""Corrected axial-depth coarse proposal learning; legacy v4 retained unchanged."""
from collections import Counter
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from contrastive_terrain import sample_descriptors,valid_centers
from lunar_geometry_v5 import correspondences_z as correspondences
from train_balanced_localization_v3 import appearance


def build_supervision(manifest,tiles,data):
    """Use uniform reference locations, not matcher-selected training positives."""
    yy,xx=np.mgrid[12:116:4,12:116:4]
    p=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
    native=(p+.5)*4-.5;result={};counts=Counter()
    for key,t in tiles.items():
        if t['split']!='train':continue
        z=data[key];a,b=t['cameras']
        _,target,_,visible=correspondences(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=native)
        q=np.zeros_like(p);q[visible]=(target+.5)/4-.5
        masks=[]
        for depth in (z['depth_a'],z['depth_b']):
            mask=cv2.resize((np.isfinite(depth)&(depth>0)).astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)>=.9999
            masks.append(mask)
        ma,mb=[valid_centers(m,12) for m in masks]
        use=visible&ma[p[:,1].astype(int),p[:,0].astype(int)]
        safe=np.rint(q).astype(int).clip(0,127);use&=mb[safe[:,1],safe[:,0]]
        result[key]=dict(p=p.copy(),q=q,visible=use,masks=masks,
                         images=np.stack([cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA) for im in z['images']]))
        counts.update(dict(tiles=1,uniform_points=len(p),visible_supported_points=int(use.sum()),fewer_than_eight=int(use.sum()<8)))
    return result,dict(counts)


def transform_points(points,matrix):
    return np.asarray(points)@matrix[:,:2].T+matrix[:,2]


def sample_training(variant,manifest,tiles,supervision,rng):
    groups={}
    for t in tiles.values():
        if t['split']=='train':groups.setdefault(t['component'],[]).append(t['tile_id'])
    names=sorted(groups)
    for _ in range(200):
        group=names[int(rng.integers(len(names)))];ids=groups[group];key=ids[int(rng.integers(len(ids)))];z=supervision[key]
        la,lb=rng.integers(0,3,2);reference=z['images'][2*la]
        query=z['images'][2*lb+(variant=='true_3d')]
        p=z['p'];q=z['q'] if variant=='true_3d' else p
        use=z['visible'].copy() if variant=='true_3d' else valid_centers(z['masks'][0],12)[p[:,1].astype(int),p[:,0].astype(int)]
        # Identical augmentation distribution for both supervision ablations.
        angle=float(rng.uniform(-90,90));scale=float(np.exp(rng.uniform(np.log(.7),np.log(1.4))))
        matrix=cv2.getRotationMatrix2D((63.5,63.5),angle,scale);matrix[:,2]+=rng.uniform(-8,8,2)
        q=transform_points(q,matrix)
        source_mask=z['masks'][1 if variant=='true_3d' else 0]
        warped_mask=cv2.warpAffine(source_mask.astype(np.float32),matrix,(128,128),flags=cv2.INTER_LINEAR)>=.9999
        query_valid=valid_centers(warped_mask,12)
        inside=((q>=12)&(q<116)).all(1);safe=np.rint(q).astype(int).clip(0,127)
        use&=inside&query_valid[safe[:,1],safe[:,0]]
        eligible=np.where(use)[0]
        if len(eligible)<8:continue
        indices=rng.choice(eligible,min(96,len(eligible)),replace=False)
        query=cv2.warpAffine(query,matrix,(128,128),flags=cv2.INTER_LINEAR)
        images=np.stack([appearance(im,rng) for im in (reference,query)])[:,None]
        return torch.from_numpy(images),torch.from_numpy(p[indices].astype(np.float32)),torch.from_numpy(q[indices].astype(np.float32)),dict(tile_id=key,component=group,light_a=int(la),light_b=int(lb),points=len(indices))
    raise ValueError('Could not sample enough independently projected training correspondences')


def objective(model,images,p,q):
    dense,_=model(images)
    a=sample_descriptors(dense[:1],p[None])[0]
    b=sample_descriptors(dense[1:],q[None])[0]
    scores=a@b.T/.1;labels=torch.arange(len(p))
    # Nearby projected points are ambiguous negatives under foreshortening.
    close=(torch.cdist(p,p)<2)|(torch.cdist(q,q)<2);close.fill_diagonal_(False)
    scores=scores.masked_fill(close,-1e4)
    return .5*(F.cross_entropy(scores,labels)+F.cross_entropy(scores.T,labels))
