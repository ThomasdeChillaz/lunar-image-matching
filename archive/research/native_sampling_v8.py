"""Uniform projected positives; preserve the v7 negative sampler and model."""
from collections import Counter
import cv2
import numpy as np
import torch

from lunar_geometry_v5 import correspondences_z
from train_balanced_localization_v3 import appearance,training_batch as original_batch


def prepare_uniform(tiles,data):
    yy,xx=np.meshgrid(np.arange(49.5,450,16),np.arange(49.5,450,16),indexing='ij')
    p=np.column_stack([xx.ravel(),yy.ravel()]);counts={}
    for key,t in tiles.items():
        if t['split']!='train':continue
        z=data[key];a,b=t['cameras']
        _,gt,_,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
        target=np.zeros_like(p);target[visible]=gt
        use=visible&((target>=44)&(target<468)).all(1)
        if use.sum()<8:raise ValueError('Insufficient uniform exact positives in a training geometry')
        z['uniform_reference_native']=p[use].copy();z['uniform_target_native']=target[use].copy()
        counts[key]=dict(component=t['component'],visible_supported_points=int(use.sum()))
    return dict(tiles=len(counts),points=sum(c['visible_supported_points'] for c in counts.values()),groups=len({c['component'] for c in counts.values()}),per_tile=counts)


def uniform_example(tiles,data,rng):
    groups={}
    for t in tiles.values():
        if t['split']=='train':groups.setdefault(t['component'],[]).append(t['tile_id'])
    names=sorted(groups);group=names[int(rng.integers(len(names)))];ids=groups[group];key=ids[int(rng.integers(len(ids)))];z=data[key]
    index=int(rng.integers(len(z['uniform_reference_native'])));p=z['uniform_reference_native'][index];target=z['uniform_target_native'][index]
    la,lb=rng.integers(0,3,2);ref_center=np.rint(p).astype(int);center=np.rint(target).astype(int)+rng.integers(-12,13,2)
    if (center<28).any() or (center>=484).any():raise ValueError('Uniform patch exceeds image bounds')
    ref=cv2.getRectSubPix(z['images'][2*la],(27,27),tuple(map(float,ref_center)))
    query=cv2.getRectSubPix(z['images'][2*lb+1],(57,57),tuple(map(float,center)))
    local=target-center+16
    if ((local<0)|(local>32)).any():raise ValueError('Exact uniform positive outside search window')
    arrays=[appearance(ref,rng)[None],appearance(query,rng)[None],(p-ref_center+13).astype(np.float32),local.astype(np.float32)]
    return arrays,dict(tile_id=key,component=group,reference_native=p.tolist(),target_native=target.tolist(),query_center_native=center.tolist(),light_a=int(la),light_b=int(lb))


def training_batch(manifest,tiles,data,rng,size):
    batch,kinds,positive=original_batch(manifest,tiles,data,rng,size)
    kinds=Counter({k:v for k,v in kinds.items() if k not in ('balanced_positive','balanced_positive_jitter')})
    for slot in range(0,size,2):
        arrays,meta=uniform_example(tiles,data,rng)
        for destination,value in zip(batch[:4],arrays):destination[slot]=torch.from_numpy(value)
        batch[4][slot]=1.;kinds['uniform_3d_positive/'+meta['component']]+=1
    return batch,kinds,positive
