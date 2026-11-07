"""Corrected-depth local refinement with an explicit categorical no-match bin."""
import numpy as np
import torch
from torch.nn import functional as F

from contrastive_terrain import sample_descriptors
from localization_refiner_v3 import LocalRefiner,decode_peak

VARIANTS=('encoder_3d','pair_refinement','dustbin')


class NativeRefiner(LocalRefiner):
    def __init__(self,variant):
        if variant not in VARIANTS:raise ValueError('Unknown native refinement experiment')
        super().__init__(dict(encoder_3d='descriptor_3d',pair_refinement='refinement',dustbin='reliability')[variant])
        self.experiment_variant=variant
        if variant=='dustbin':
            # log(1089)<7.5, so the initial no-match bin is strictly below
            # every local maximum, regardless of the image similarity values.
            # The learned head predicts no-match log odds, not matchability.
            with torch.no_grad():self.matchability[-1].bias.fill_(-7.5)


def joint_logits(local,no_match):
    flat=local.flatten(1)
    return torch.cat([flat,(flat.logsumexp(1)+no_match)[:,None]],dim=1)


def objective(local,no_match,target,visible,dustbin):
    positive=visible.bool();flat=local.flatten(1)
    if dustbin:flat=joint_logits(local,no_match)
    logp=flat.log_softmax(1);loss=local.sum()*0
    if positive.any():
        xy=target[positive]
        if not torch.isfinite(xy).all() or ((xy<0)|(xy>32)).any():raise ValueError('Positive exact target outside33px window')
        low=xy.floor().long();high=(low+1).clamp(max=32);fraction=xy-low
        values=local.new_zeros(len(xy))
        for x,y,w in [(low[:,0],low[:,1],(1-fraction[:,0])*(1-fraction[:,1])),
                      (high[:,0],low[:,1],fraction[:,0]*(1-fraction[:,1])),
                      (low[:,0],high[:,1],(1-fraction[:,0])*fraction[:,1]),
                      (high[:,0],high[:,1],fraction[:,0]*fraction[:,1])]:
            values-=w*logp[positive].gather(1,(y*33+x)[:,None])[:,0]
        loss=values.sum()
    if dustbin:
        loss=loss-logp[~positive,-1].sum()
        return loss/len(local)
    return loss/positive.sum().clamp_min(1)


def refine(model,reference_points,query_points,reference_dense,query_dense,batch_size=32):
    """Image features only; use exactly the deterministic baseline's support."""
    p,q=[np.asarray(v,np.float32).reshape(-1,2) for v in (reference_points,query_points)]
    if p.shape!=q.shape:raise ValueError('Paired native Nx2 coordinates required')
    h,w=query_dense.shape[-2:];hr,wr=reference_dense.shape[-2:];center=np.rint(q).astype(int)
    valid=((p>=12)&(p<[wr-12,hr-12])).all(1)
    valid&=((center-16>=12)&(center+16<[w-12,h-12])).all(1)
    indices=np.flatnonzero(valid);offset=torch.arange(-16,17,device=query_dense.device)
    predicted=[];reliability=[];accepted=[]
    for start in range(0,len(indices),batch_size):
        selected=indices[start:start+batch_size];centers=torch.from_numpy(center[selected]).to(query_dense.device)
        ys=centers[:,1,None,None]+offset[None,:,None];xs=centers[:,0,None,None]+offset[None,None,:]
        windows=query_dense[0][:,ys,xs].permute(1,0,2,3).contiguous()
        vectors=sample_descriptors(reference_dense,torch.from_numpy(p[selected]).to(query_dense.device)[None])[0]
        logits,no_match=model.score(vectors,windows)
        prediction=centers+decode_peak(logits)
        if model.experiment_variant=='dustbin':
            all_logits=joint_logits(logits,no_match);keep=all_logits.argmax(1)!=1089
            score=1-no_match.sigmoid()
        else:keep=torch.ones(len(selected),dtype=torch.bool,device=query_dense.device);score=torch.ones(len(selected),device=query_dense.device)
        predicted.append(prediction.cpu().numpy());reliability.append(score.cpu().numpy());accepted.append(keep.cpu().numpy())
    pred=np.concatenate(predicted) if predicted else np.empty((0,2),np.float32)
    score=np.concatenate(reliability) if reliability else np.empty(0,np.float32)
    keep=np.concatenate(accepted) if accepted else np.empty(0,bool)
    return p[indices][keep],pred[keep],dict(supported=len(indices),no_match=int((~keep).sum()),
        indices=indices[keep],point_reliability=score[keep],all_point_reliability=score)
