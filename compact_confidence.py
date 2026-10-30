"""Regularized image-evidence confidence, separate fitting and calibration splits."""
import numpy as np

FEATURE_NAMES=('estimated','support_log','support_fraction','spatial_fraction',
               'median_residual','descriptor_margin','matchability','reverse_cycle')


def evidence_vector(match, shape, extra=None, reverse=None):
    extra={} if extra is None else extra
    if match.matrix is None:
        return np.array([0,0,0,0,10,0,0,10.],float)
    a,b=np.asarray(match.reference_points),np.asarray(match.query_points)
    if len(a):
        residual=np.median(np.linalg.norm(a@match.matrix[:,:2].T+match.matrix[:,2]-b,axis=1))
        spatial=np.prod(np.ptp(a,axis=0))/np.prod(shape)
    else:
        residual,spatial=10.,0.
    cycle=10.
    if reverse is not None and reverse.matrix is not None:
        sample=np.array([[shape[1]*x,shape[0]*y] for x in (.25,.5,.75) for y in (.25,.5,.75)])
        moved=sample@match.matrix[:,:2].T+match.matrix[:,2]
        restored=moved@reverse.matrix[:,:2].T+reverse.matrix[:,2]
        cycle=float(np.percentile(np.linalg.norm(restored-sample,axis=1),95))
    return np.array([1,np.log1p(match.support),match.support/max(extra.get('raw_matches',match.support),1),
        spatial,min(residual,10),extra.get('descriptor_margin',0),extra.get('matchability',0),min(cycle,10)],float)


def fit_confidence(rows):
    if not rows or any(r['split']!='model_selection' for r in rows):
        raise ValueError('Confidence fitting requires model-selection groups only')
    x=np.array([r['features'] for r in rows],float)
    y=np.array([r['correct'] and not r['negative'] for r in rows],float)
    if x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid image-evidence features')
    mean=x.mean(0);scale=np.maximum(x.std(0),.01)
    xx=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    w=np.zeros(xx.shape[1])
    # Fixed budget shared by all methods. No search over regularization strengths.
    for _ in range(1000):
        prediction=1/(1+np.exp(-np.clip(xx@w,-30,30)))
        gradient=xx.T@(prediction-y)/len(y)+.01*np.r_[0,w[1:]]
        w-=.05*gradient
    return dict(feature_names=list(FEATURE_NAMES),mean=mean.tolist(),scale=scale.tolist(),weights=w.tolist(),
                l2=.01,iterations=1000,fitted_split='model_selection')


def scores(policy,rows):
    if policy['feature_names']!=list(FEATURE_NAMES):
        raise ValueError('Confidence feature order changed')
    x=np.array([r['features'] for r in rows],float)
    z=(x-np.array(policy['mean']))/np.array(policy['scale'])
    result=1/(1+np.exp(-np.clip(np.column_stack([np.ones(len(z)),z])@policy['weights'],-30,30)))
    result[x[:,0]==0]=-1.
    return result


def counts(rows,accepted):
    positive=np.array([not r['negative'] for r in rows])
    correct=np.array([r['correct'] and not r['negative'] for r in rows])
    accepted=np.asarray(accepted,bool)
    wrong=int((accepted&~correct).sum())
    good=int((accepted&correct).sum())
    return dict(positives=int(positive.sum()),negatives=int((~positive).sum()),
        correct_accepted=good,wrong_accepted=wrong,negative_accepted=int((accepted&~positive).sum()),
        correct_acceptance=good/max(int(positive.sum()),1),accepted_error=wrong/max(good+wrong,1),
        negative_acceptance=int((accepted&~positive).sum())/max(int((~positive).sum()),1))


def calibrate(policy,rows):
    if not rows or any(r['split']!='calibration' for r in rows):
        raise ValueError('Threshold selection requires calibration groups only')
    prediction=scores(policy,rows)
    best=dict(threshold=1.000001,counts=counts(rows,np.zeros(len(rows),bool)))
    curve=[]
    for threshold in np.r_[1.000001,np.unique(prediction[prediction>=0])[::-1]]:
        stat=counts(rows,prediction>=threshold)
        curve.append(dict(threshold=float(threshold),**stat))
        if stat['accepted_error']<=.01 and stat['negative_acceptance']<=.01 and stat['correct_accepted']>best['counts']['correct_accepted']:
            best=dict(threshold=float(threshold),counts=stat)
    return dict(policy,**best,calibrated_split='calibration',curve=curve,
                warning='Empirical operating point; no deployment risk guarantee')


def accepts(policy,row):
    return bool(scores(policy,[row])[0]>=policy['threshold'])
