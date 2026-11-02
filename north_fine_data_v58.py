"""Matched native/fractional-query sampling for exact3D fine supervision.

This is a controlled sampling ablation, not a reproduction of the oldv13
trainer. Reference features always sample the original half-pixel ray grid.
"""
import cv2
import numpy as np
import torch
from torch import nn

from native_crop_alignment_v57 import native_example
from north_training_bundle_v38 import proposal_grid_native
from train_balanced_localization_v3 import appearance

VARIANTS=('aligned','fractional')


def eligible_points(z):
    targets=np.asarray(z['forward_target']);visible=np.asarray(z['forward_visible'],bool)
    if targets.shape!=(676,2)or visible.shape!=(676,):raise ValueError('Exact native-grid BVH labels required')
    # Margin includes +/-12integer jitter, +/-0.5fractional phase and57px crop.
    return np.flatnonzero(visible & np.isfinite(targets).all(1)
        & (targets>=42).all(1)&(targets<=470).all(1))


def prepare(tiles,data):
    groups={};eligible={}
    for key,t in tiles.items():
        if t['split']!='train':raise ValueError('Only training geometry may enter fine fitting')
        indices=eligible_points(data[key])
        if len(indices)<8:raise ValueError('Every training geometry must retain eight supported exact points')
        eligible[key]=indices;groups.setdefault(t['component'],[]).append(key)
    groups={k:sorted(v)for k,v in sorted(groups.items())}
    return groups,eligible


def sample_batch(tiles,data,prepared,rng,variant,size=8):
    if variant not in VARIANTS or size<1:raise ValueError('Declared sampling variant and positive batch size required')
    groups,eligible=prepared;names=sorted(groups);grid=proposal_grid_native()
    refs=[];queries=[];reference_xy=[];targets=[];metadata=[]
    for _ in range(size):
        group=names[int(rng.integers(len(names)))];ids=groups[group]
        key=ids[int(rng.integers(len(ids)))];z=data[key]
        la,lb=map(int,rng.integers(0,6,2));indices=eligible[key]
        index=int(indices[int(rng.integers(len(indices)))]);p=grid[index];truth=z['forward_target'][index]
        jitter=rng.integers(-12,13,2);phase=rng.uniform(-.5,.5,2)
        # Both variants consume exactly the same draws, even when phase is unused.
        center=np.rint(truth).astype(int)+jitter
        raw_a,raw_b=z['images'][2*la],z['images'][2*lb+1]
        e=native_example(raw_a,raw_b,p,center,truth,True)
        a,b=appearance(raw_a,rng),appearance(raw_b,rng)
        x,y=e['reference_origin'];ref=a[y:y+27,x:x+27].copy()
        actual_center=center.astype(float)+(phase if variant=='fractional'else 0)
        if variant=='aligned':
            x,y=e['query_origin'];query=b[y:y+57,x:x+57].copy()
        else:
            query=cv2.getRectSubPix(b,(57,57),tuple(map(float,actual_center)))
        target=(truth-actual_center+16).astype(np.float32)
        if ref.shape!=(27,27)or query.shape!=(57,57)or not np.isfinite(target).all()or((target<0)|(target>32)).any():
            raise ValueError('Fine input support or exact local label invalid')
        if ref.dtype!=np.float32 or query.dtype!=np.float32:raise ValueError('Full-image sensor augmentation must return float32[0,1]')
        refs.append(ref[None]);queries.append(query[None]);reference_xy.append(e['reference_xy']);targets.append(target)
        metadata.append(dict(tile_id=key,component=group,point_index=index,light_a=la,light_b=lb,
            reference_native=p.tolist(),target_native=truth.tolist(),integer_query_center=center.tolist(),
            drawn_fractional_phase=phase.tolist(),effective_query_center=actual_center.tolist(),
            variant=variant,geometrically_visible=True))
    return tuple(np.stack(v)for v in (refs,queries,reference_xy,targets)),metadata


def physical_samples(metadata):
    """Compare treatment-independent draws, not labels shifted by the treatment."""
    return [{k:v for k,v in row.items()if k not in ('variant','effective_query_center')}for row in metadata]


def make_fine(coarse):
    from native_refinement_v7 import NativeRefiner
    fine=NativeRefiner('encoder_3d')
    fine.initialize_encoder(coarse)
    fine.train().requires_grad_(True)
    if coarse.metadata['variant']!='expanded' or coarse.model.reliability:
        raise ValueError('Declared original encoder-only starting model required')
    total=sum(p.numel()for p in fine.parameters())+sum(p.numel()for p in coarse.model.parameters())
    if total!=56544 or total>=100000:raise ValueError('Combined frozen-coarse/fine parameter budget differs')
    return fine


class FineEncoderAdapter(nn.Module):
    def __init__(self,fine):
        super().__init__();self.encoder=fine.encoder
    def forward(self,image):
        dense=self.encoder(image)
        return dense,torch.zeros_like(dense[:,:1])


class FineMatcher:
    """Same original coarse proposals; only native fine descriptors may change."""
    def __init__(self,coarse,fine):
        self.coarse=coarse;self.model=FineEncoderAdapter(fine).eval();self.metadata=coarse.metadata;self.device=coarse.device
    def points(self,*args,**kwargs):return self.coarse.points(*args,**kwargs)


class CheckpointFeatureCache:
    """Never reuse full-image fine maps across different learned weights."""
    def __init__(self,capacity=64):self.capacity=capacity;self.invalidate()
    def invalidate(self):self.token=None;self.cache=None
    def for_weights(self,token):
        if not token:raise ValueError('Explicit checkpoint identity required')
        if token!=self.token:
            from train_north_control_v39 import FineFeatureCache
            self.cache=FineFeatureCache(self.capacity);self.token=token
        return self.cache
