"""Globally shared Sun during correspondence assignment and pose refinement.

Image frontends supply finite candidate arrays and a LAZY patch-measurement
callback. No query depth/pose/Sun truth enters this engine. It is a raw research
solver; calibration/accepted public outputs are a separate frozen interface.
"""
from dataclasses import dataclass
import numpy as np
from solar_physics_v63 import SolarState,solar_log_evidence
from solar_active_v63 import choose_action,HEIGHT_VARIANTS_M
from localization_robust_pose_v4 import estimate


@dataclass
class CandidateGraph:
    reference_xy:np.ndarray
    world:np.ndarray
    query_xy:np.ndarray
    appearance:np.ndarray
    observed_shadow:np.ndarray
    solar_states:np.ndarray
    sun_vectors:np.ndarray
    world_variants:np.ndarray
    solar_variants:np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value=np.asarray(getattr(self,name))
            if name not in('solar_states','solar_variants'):value=value.astype(float)
            setattr(self,name,value)
        n,k=self.appearance.shape;s=len(self.sun_vectors)
        expected=dict(reference_xy=(n,2),world=(n,3),query_xy=(n,k,2),observed_shadow=(n,k),solar_states=(n,s),sun_vectors=(s,3),world_variants=(n,7,3),solar_variants=(n,s,7))
        if n<1 or k<1 or s<1:raise ValueError('Nonempty candidate graph required')
        for name,shape in expected.items():
            if getattr(self,name).shape!=shape:raise ValueError('Candidate graph shape differs:'+name)
        if any(not np.isfinite(getattr(self,name)).all()for name in ('reference_xy','world','query_xy','appearance','observed_shadow','sun_vectors','world_variants')):raise ValueError('Finite image/map/prediction candidates required')
        if ((self.observed_shadow<0)|(self.observed_shadow>1)).any():raise ValueError('Observed shadow probabilities must lie in[0,1]')
        if not np.isin(self.solar_states,[0,1,2]).all()or not np.isin(self.solar_variants,[0,1,2]).all():raise ValueError('Solar states must distinguish unknown/clear/shadow')
        if any(((getattr(self,name)<0)|(getattr(self,name)>511)).any()for name in ('reference_xy','query_xy')):raise ValueError('Candidate coordinates must use native512pixel centres')
        if not np.allclose(np.linalg.norm(self.sun_vectors,axis=-1),1.,atol=1e-6,rtol=0):raise ValueError('Unit candidate Sun vectors required')


def project(world,pose,k):
    camera=(np.asarray(world)-pose[:3,3])@pose[:3,:3];h=camera@np.asarray(k).T
    with np.errstate(divide='ignore',invalid='ignore'):pixels=h[...,:2]/h[...,2,None]
    return pixels,(camera[...,2]>0)&np.isfinite(pixels).all(-1)


def height_variants(world,reference_pose):
    """Change reference heights along calibrated rays, preserving ref pixels."""
    world=np.asarray(world,float);origin=np.asarray(reference_pose)[:3,3];ray=world-origin
    if world.ndim!=2 or world.shape[1]!=3 or not np.isfinite(world).all()or(np.abs(ray[:,2])<1e-8).any():raise ValueError('Reference rays must admit finite height perturbations')
    return world[:,None]+np.asarray(HEIGHT_VARIANTS_M)[None,:,None]*ray[:,None]/ray[:,None,2:3]


def evidence_tensor(graph):
    return solar_log_evidence(graph.solar_states[:,:,None],graph.observed_shadow[:,None,:])


def assignments(graph,sun_index,solar_mode='shared',pose=None,intrinsics=None,response=None,precomputed_evidence=None,candidate_rank=0):
    if solar_mode not in('off','shared','independent'):raise ValueError('Unknown solar consistency control')
    score=graph.appearance.copy()if response is None else np.asarray(response,float).copy()
    if score.shape!=graph.appearance.shape or not np.isfinite(score).all():raise ValueError('Finite candidate appearance/response score required')
    evidence=evidence_tensor(graph)if precomputed_evidence is None else precomputed_evidence
    if solar_mode=='shared':score+=evidence[:,sun_index]
    elif solar_mode=='independent':score+=evidence.max(1)
    if pose is not None:
        projected,front=project(graph.world,pose,intrinsics)
        error=np.linalg.norm(graph.query_xy-projected[:,None],axis=-1)
        score-=np.where(front[:,None],np.minimum((error/3.)**2,9.),9.)
    if not 0<=candidate_rank<score.shape[1]:raise ValueError('Initial candidate rank outside candidate bank')
    index=np.argsort(-score,axis=1,kind='stable')[:,candidate_rank];values=score[np.arange(len(score)),index]
    # One query grid point cannot represent two distinct reference anchors.
    keep=[];seen=set()
    for i in sorted(range(len(index)),key=lambda i:(-values[i],i)):
        key=tuple(graph.query_xy[i,index[i]])
        if key not in seen:keep.append(i);seen.add(key)
    keep=np.asarray(sorted(keep),int)
    known=(graph.solar_states[:,sun_index]!=SolarState.UNKNOWN)&(np.abs(graph.observed_shadow[np.arange(len(index)),index]-.5)>=.1)
    cells=np.floor(graph.reference_xy[known]/128).astype(int)
    support=bool(solar_mode=='shared'and known.sum()>=32 and len(set(map(tuple,cells)))>=4)
    return dict(candidate=index,keep=keep,total=float(values[keep].sum()/len(graph.world)),solar_supported=support,informative_anchors=int(known.sum()))


def initial_beams(graph,solar_mode='shared',maximum=8):
    candidates=[];evidence=evidence_tensor(graph)
    suns=range(len(graph.sun_vectors))
    for rank in range(min(8,graph.appearance.shape[1])):
        for si in suns:
            a=assignments(graph,si,solar_mode,precomputed_evidence=evidence,candidate_rank=rank)
            candidates.append(dict(sun_index=si,assignment=a,pose=None,score=a['total'],initial_candidate_rank=rank))
    candidates.sort(key=lambda b:(-b['score'],b['initial_candidate_rank'],b['sun_index']))
    chosen=[];seen=set()
    for b in candidates:
        key=tuple(b['assignment']['candidate'])
        # Distinct assignment OR separated lighting constitutes diversity.
        if any(key==tuple(x['assignment']['candidate'])and np.dot(graph.sun_vectors[b['sun_index']],graph.sun_vectors[x['sun_index']])>np.cos(np.radians(10.))for x in chosen):continue
        chosen.append(b)
        if len(chosen)==maximum:break
    return chosen


def _matched(graph,beam,measurements):
    p=[];q=[];w=[];ids=[]
    for i in beam['assignment']['keep']:
        key=(int(i),int(beam['assignment']['candidate'][i]));m=measurements.get(key)
        if m is not None and m.get('query_xy')is not None:
            p.append(graph.reference_xy[i]);q.append(m['query_xy']);w.append(graph.world[i]);ids.append(i)
    return np.asarray(p).reshape(-1,2),np.asarray(q).reshape(-1,2),np.asarray(w).reshape(-1,3),ids


def _lm_update(pose,world,pixels,k):
    import cv2
    if pose is None or len(world)<6:return pose
    predicted,front=project(world,pose,k);use=front&(np.linalg.norm(predicted-pixels,axis=-1)<=3.)
    if use.sum()<6:return pose
    rv=cv2.Rodrigues(pose[:3,:3].T)[0];tv=-pose[:3,:3].T@pose[:3,3]
    try:rv,tv=cv2.solvePnPRefineLM(np.ascontiguousarray(world[use]),np.ascontiguousarray(pixels[use]),np.asarray(k),None,rv,tv)
    except cv2.error:return pose
    r=cv2.Rodrigues(rv)[0];out=np.eye(4);out[:3,:3]=r.T;out[:3,3]=-r.T@tv.ravel();return out


def acquisition_predictions(graph,beams,actions,intrinsics):
    predictions=np.full((len(beams),len(actions),7,4),np.nan)
    for h,b in enumerate(beams):
        if b['pose']is None:continue
        for j,(i,c)in enumerate(actions):
            xy,front=project(graph.world_variants[i],b['pose'],intrinsics)
            delta=(xy-graph.query_xy[i,c])/16.
            predictions[h,j,:,:2]=np.where(front[:,None],delta,np.nan)
            state=graph.solar_variants[i,b['sun_index']]
            predictions[h,j,:,2]=np.where(state==0,np.nan,(state==SolarState.SHADOW).astype(float))
            predictions[h,j,:,3]=np.where(front,np.exp(-.5*np.sum(delta*delta,-1)),np.nan)
    scores=np.asarray([b['score']for b in beams]);weights=np.exp(scores-scores.max())
    return predictions,weights,np.isfinite(predictions)


def solve(graph,intrinsics,measure,solar_mode='shared',active=False,policy='active',budget=128,response_enabled=False):
    """measure(i,c,trial_pose,sun_index)->{query_xy, response_scores(optional)}.

    Each measurement consumes budget even if invalid. Response scores must be
    physically conditioned by the image frontend; unavailable scores do not
    silently become a trained-model result. Seed queries use the frozen model.
    """
    if budget not in(32,64,128):raise ValueError('Declared32/64/128patch budget required')
    graph=CandidateGraph(**{name:getattr(graph,name).copy()for name in graph.__dataclass_fields__})
    beams=initial_beams(graph,solar_mode,8);measurements={};history=[]
    if not beams:return dict(pose=None,reason='no_candidate_assignments',measurements=[],hypotheses=[])
    n,k=graph.appearance.shape;actions=[(i,c)for i in range(n)for c in range(k)]
    points=np.repeat(graph.reference_xy,k,axis=0);used=np.zeros(len(actions),bool)
    appearances=graph.appearance.reshape(-1)
    def take(action,beam):
        i,c=actions[action];m=measure(i,c,beam['pose'],beam['sun_index']);used[action]=True
        if not isinstance(m,dict):raise ValueError('Patch measurement must be an explicit record')
        q=m.get('query_xy')
        if q is not None and(np.asarray(q).shape!=(2,)or not np.isfinite(q).all()):raise ValueError('Finite native query correspondence required')
        if q is not None:
            if ((np.asarray(q)<0)|(np.asarray(q)>511)).any():raise ValueError('Measured correspondence outside native image')
            graph.query_xy[i,c]=q
            if 'observed_shadow'in m:graph.observed_shadow[i,c]=float(m['observed_shadow'])
        measurements[(i,c)]=m;history.append(dict(anchor=i,candidate=c,trial_sun_index=beam['sun_index'],response_used=bool(m.get('response_scores')is not None)))
    # Eight spatially distributed seeds per initial solar/assignment beam.
    # Fixed proposal allocation BEFORE native acquisition. Coarse poses are
    # tentative inference hypotheses, never query ground truth. This permits
    # eight beams even for the32patch control without spending64seed patches.
    proposal_requested=0
    for b in beams:
        a=b['assignment'];ids=a['keep'];q=graph.query_xy[ids,a['candidate'][ids]];allocation=1024//len(beams);proposal_requested+=allocation
        b['pose'],b['pose_info']=estimate(graph.world[ids],q,intrinsics,dict(solver='ap3p',iterations=allocation,reclassify=True))
        b['native_pose']=False
    seed_count=min(budget//2,8*len(beams));orders=[]
    for b in beams:
        a=b['assignment'];order=sorted(a['keep'],key=lambda i:(int(graph.reference_xy[i,1]//128)*4+int(graph.reference_xy[i,0]//128),i))
        positions=np.rint(np.linspace(0,max(0,len(order)-1),8)).astype(int)
        orders.append([int(order[j])*k+int(a['candidate'][order[j]])for j in positions]if order else[])
    # First cover eight spatial anchors under the best current assignment.
    # Then diversify beam/candidate measurements, rather than spending eight
    # queries on one anchor before obtaining a camera constraint elsewhere.
    best_seed=max(range(len(beams)),key=lambda j:beams[j]['score'])
    for position in range(8):
        order=orders[best_seed]
        if len(history)>=seed_count:break
        if order and not used[order[position]]:take(order[position],beams[best_seed])
    for turn in range(8*len(beams)):
        if len(history)>=seed_count:break
        j=turn%len(beams);position=(turn//len(beams)+j)%8;order=orders[j]
        if order and not used[order[position]]:take(order[position],beams[j])
    rounds=[]
    for round_index in range(3):
        target=min(budget,len(history)+int(np.ceil((budget-len(history))/(3-round_index))))
        while len(history)<target and not used.all():
            predictions,weights,valid=acquisition_predictions(graph,beams,actions,intrinsics)
            chosen_policy=policy if active and any(b['pose']is not None for b in beams)else'uniform'
            # A search at the same reference location supplies no new spatial
            # camera constraint. Apply this common acquisition guard to every
            # policy; revisit candidates only after all anchors were visited.
            visited={i for i,c in measurements};unavailable=used.copy()
            if len(visited)<n:
                unavailable|=np.array([i in visited for i,c in actions])
            action=choose_action(predictions,weights,points,unavailable,chosen_policy,appearances,valid)
            best=max(beams,key=lambda b:b['score']);take(action,best)
        evidence=evidence_tensor(graph)
        for b in beams:
            # The Sun is updated jointly with pose and correspondence choices.
            suns=range(len(graph.sun_vectors))if solar_mode=='shared'or response_enabled else(0,)
            choices=[]
            for si in suns:
                response=graph.appearance.copy()
                if response_enabled:
                    for (i,c),m in measurements.items():
                        if m.get('response_scores')is not None:
                            scores=np.asarray(m['response_scores'],float)
                            if scores.shape!=(len(graph.sun_vectors),):raise ValueError('One learned response score per candidate Sun required')
                            response[i,c]=scores[si]
                a=assignments(graph,si,solar_mode,b['pose']if b['native_pose']else None,intrinsics,response,evidence)
                choices.append((a['total'],-si,si,a))
            _,_,b['sun_index'],b['assignment']=max(choices,key=lambda x:x[:2])
            _,q,w,_=_matched(graph,b,measurements)
            allocation=1024//len(beams);proposal_requested+=allocation
            native_pose,info=estimate(w,q,intrinsics,dict(solver='ap3p',iterations=allocation,reclassify=True))
            if native_pose is not None:
                b['pose']=_lm_update(native_pose,w,q,intrinsics);b['native_pose']=True;b['pose_info']=info
            b['score']=b['assignment']['total']
        rounds.append(dict(round=round_index+1,queries=len(history),live_poses=sum(b['pose']is not None for b in beams)))
    # Common final estimator remains the unchanged3px AP3P/RANSAC2000control.
    candidates=[b for b in beams if b['pose']is not None]
    if not candidates:return dict(pose=None,reason='no_pose_hypothesis',measurements=history,hypotheses=beams,proposal_samples_requested=proposal_requested,rounds=rounds)
    best=max(candidates,key=lambda b:b['score']);p,q,w,_=_matched(graph,best,measurements)
    pose,info=estimate(w,q,intrinsics,dict(solver='ap3p',iterations=2000,reclassify=True))
    return dict(pose=pose,reason=info['reason'],reference_xy=p,query_xy=q,sun_enu=None if solar_mode!='shared'else graph.sun_vectors[best['sun_index']],solar_supported=best['assignment']['solar_supported'],measurements=history,hypotheses=beams,proposal_samples_requested=proposal_requested,final_estimator_iterations=2000,rounds=rounds,response_measurements=sum(h['response_used']for h in history))
