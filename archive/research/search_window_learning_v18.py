"""Search-window coarse supervision and a separately frozen native fine encoder.

Prepared hypothesis, not a trained result. Keep the ongoing v16 source frozen.
The new objective treats every coarse candidate whose native33px search can
reach the projected point as positive, rather than demanding bilinear mass
at one grid cell. Optional anchors preserve genuine frozen mutual proposals.
"""
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from context_supervision_v15 import make_context,proposal_grid
from contrastive_terrain import sample_descriptors
from global_context_v15 import ContextMatcher


def window_mask(target128,visible,candidate_grid128):
    target_native=(target128+.5)*4-.5
    centers=((candidate_grid128+.5)*4-.5).round()
    return visible[:,None] & ((target_native[:,None]-centers[None]).abs()<=16).all(-1)


def direction(scores,target,visible,query_grid,baseline=None,anchor_weight=.5):
    good=window_mask(target,visible,query_grid);positive=good.any(1);negative=~positive
    loss=scores.sum()*0;anchors=0
    if positive.any():
        logits=scores[positive]/.1;allowed=good[positive]
        loss=loss+(logits.logsumexp(-1)-logits.masked_fill(~allowed,-torch.inf).logsumexp(-1)).mean()
        best=scores[positive].masked_fill(~allowed,-torch.inf).max(-1).values
        loss=loss+.25*F.softplus(10*(.7-best)).mean()/10
    if negative.any():
        loss=loss+.25*F.softplus(10*(scores[negative].max(-1).values-.5)).mean()/10
    if anchor_weight:
        if baseline is None or baseline.shape!=scores.shape:raise ValueError('Frozen baseline scores required for anchors')
        with torch.no_grad():
            top,index=baseline.max(-1);reverse=baseline.argmax(0)
            row=torch.arange(len(scores),device=scores.device)
            keep=(reverse[index]==row)&(top>=.5)&good[row,index]
        anchors=int(keep.sum())
        if anchors:
            # Protect only independently geometry-labelled genuine baseline
            # proposals. This is training supervision, never evaluation truth.
            candidate=scores[keep,index[keep]]
            wrong=scores[keep].masked_fill(good[keep],-torch.inf).max(-1).values
            loss=loss+anchor_weight*F.softplus(10*(wrong-candidate+.1)).mean()/10
    return loss,dict(positive=int(positive.sum()),negative=int(negative.sum()),anchors=anchors)


def objective(scores,targets,baseline_scores=None,anchor_weight=.5):
    grid=torch.from_numpy(proposal_grid()).to(scores)
    if scores.shape!=(1,676,676):raise ValueError('Complete paired proposal grids required')
    if baseline_scores is not None:baseline_scores=baseline_scores.detach()
    forward,fs=direction(scores[0],targets['forward'],targets['forward_visible'],grid,
        None if baseline_scores is None else baseline_scores[0],anchor_weight)
    reverse,rs=direction(scores[0].T,targets['reverse'],targets['reverse_visible'],grid,
        None if baseline_scores is None else baseline_scores[0].T,anchor_weight)
    return .5*(forward+reverse),dict(forward=fs,reverse=rs)


class SearchWindowModel(nn.Module):
    def __init__(self,frozen_coarse,variant='pointwise',train_coarse=False):
        super().__init__()
        if frozen_coarse.metadata['variant']!='expanded' or frozen_coarse.model.reliability:
            raise ValueError('Declared expanded frozen backbone required')
        self.train_coarse=train_coarse;self.variant=variant
        self.fine=deepcopy(frozen_coarse.model).eval().requires_grad_(False)
        self.coarse=deepcopy(frozen_coarse.model).eval().requires_grad_(train_coarse)
        self.context=make_context(variant)
        if sum(p.numel() for p in self.parameters())>=100000:raise ValueError('Combined model exceeds100k')

    def train(self,mode=True):
        super().train(mode);self.fine.eval();self.coarse.train(mode and self.train_coarse)
        return self

    def forward(self,images):
        if images.shape!=(2,1,128,128):raise ValueError('Paired128px model inputs required')
        grid=torch.from_numpy(proposal_grid()).to(images)[None]
        with torch.set_grad_enabled(self.training and self.train_coarse):
            dense,_=self.coarse(images)
            a,b=sample_descriptors(dense[:1],grid),sample_descriptors(dense[1:],grid)
        scores=self.context(a,b,grid,grid)
        with torch.no_grad():
            frozen,_=self.fine(images)
            fa,fb=sample_descriptors(frozen[:1],grid),sample_descriptors(frozen[1:],grid)
            baseline=torch.stack([aa@bb.T for aa,bb in zip(fa,fb)])
        return scores,baseline


class SearchWindowMatcher:
    """Geometry-warp inference keeps the original native fine CNN unchanged."""
    def __init__(self,frozen_coarse,model):
        model.eval();self.model=model.fine;self.metadata=frozen_coarse.metadata;self.device=frozen_coarse.device
        proposal=SimpleNamespace(model=model.coarse,metadata=self.metadata,device=self.device)
        self.proposal_matcher=ContextMatcher(proposal,model.context)

    def points(self,*args,**kwargs):
        return self.proposal_matcher.points(*args,**kwargs)
