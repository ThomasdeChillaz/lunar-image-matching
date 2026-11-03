"""Robust disagreement-directed patch acquisition; no query truth or labels.

Predictions have Hhypotheses x Nactions x Vheightvariants x Dchannels. The
first2channels are projected native positions divided by16; remaining channels
may contain predicted shadows/appearance. Invalid physics is unknown, not0.
"""
import numpy as np

HEIGHT_VARIANTS_M=(0.,-5.,5.,-10.,10.,-20.,20.)
POLICIES=('active','uniform','appearance','geometry')


def disagreement_scores(predictions,weights,valid=None,geometry_only=False):
    x=np.asarray(predictions,float);w=np.asarray(weights,float)
    if x.ndim!=4 or x.shape[2]!=7 or x.shape[-1]<2 or w.shape!=(x.shape[0],)or not np.isfinite(w).all()or(w<0).any()or w.sum()<=0:raise ValueError('HxNx7xDpredictions and nonnegative hypothesis weights required')
    mask=np.isfinite(x)if valid is None else np.asarray(valid,bool)&np.isfinite(x)
    if mask.shape!=x.shape:raise ValueError('Prediction validity shape differs')
    if geometry_only:x=x[...,:2];mask=mask[...,:2]
    complete=mask.all(2)
    low=np.where(mask,x,np.inf).min(2);high=np.where(mask,x,-np.inf).max(2)
    low=np.where(complete,low,0.);high=np.where(complete,high,0.)
    w=w/w.sum();scores=np.zeros(x.shape[1])
    for a in range(len(w)):
        for b in range(a+1,len(w)):
            known=complete[a]&complete[b]
            gap=np.maximum(0,np.maximum(low[a]-high[b],low[b]-high[a]))
            scores+=w[a]*w[b]*np.where(known,gap*gap,0.).mean(-1)
    return scores


def choose_action(predictions,weights,points,used,policy='active',appearance=None,valid=None):
    if policy not in POLICIES:raise ValueError('Unknown patch acquisition policy')
    p=np.asarray(points,float);used=np.asarray(used,bool)
    if p.ndim!=2 or p.shape[1]!=2 or used.shape!=(len(p),)or not np.isfinite(p).all():raise ValueError('Finite native action points and used mask required')
    eligible=np.flatnonzero(~used)
    if not len(eligible):return None
    if policy in('active','geometry'):
        scores=disagreement_scores(predictions,weights,valid,policy=='geometry')
        if scores.shape!=used.shape:raise ValueError('Prediction/action counts differ')
        return int(min(eligible,key=lambda i:(-scores[i],i)))
    if policy=='appearance':
        a=np.asarray(appearance,float)
        if a.shape!=used.shape or not np.isfinite(a).all():raise ValueError('Finite coarse appearance evidence required')
        return int(min(eligible,key=lambda i:(-a[i],i)))
    # Deterministic4x4coverage, avoiding a brightness/confidence preference.
    cell=np.clip(np.floor(p/128).astype(int),0,3);codes=cell[:,1]*4+cell[:,0]
    occupied=np.bincount(codes[used],minlength=16)
    return int(min(eligible,key=lambda i:(occupied[codes[i]],codes[i],i)))


def acquire_patch_actions(predict,measure,update,points,budget=128,policy='active',appearance=None):
    """Callbacks keep high-resolution extraction strictly inside measure()."""
    if budget not in(32,64,128):raise ValueError('Declared32/64/128query budget required')
    p=np.asarray(points,float);used=np.zeros(len(p),bool);records=[]
    for _ in range(min(budget,len(p))):
        predictions,weights,valid=predict()
        index=choose_action(predictions,weights,p,used,policy,appearance,valid)
        if index is None:break
        used[index]=True;measurement=measure(index);update(index,measurement)
        records.append(dict(action=index,measurement=measurement))
    return records
