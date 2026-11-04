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
from solar_sun_search_v68 import bounded_sun_indices


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
    sun_parameters:np.ndarray=None
    coarse_sun_mask:np.ndarray=None
    solar_materialized_mask:np.ndarray=None

    def __post_init__(self):
        for name in ('reference_xy','world','query_xy','appearance','observed_shadow','solar_states','sun_vectors','world_variants','solar_variants'):
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
        self.sun_parameters=np.empty((0,2))if self.sun_parameters is None else np.asarray(self.sun_parameters,float)
        self.coarse_sun_mask=np.ones(s,bool)if self.coarse_sun_mask is None else np.asarray(self.coarse_sun_mask)
        if self.sun_parameters.shape not in((0,2),(s,2))or not np.isfinite(self.sun_parameters).all():raise ValueError('Empty legacy or Sx2Sun search coordinates required')
        if self.coarse_sun_mask.shape!=(s,)or self.coarse_sun_mask.dtype!=bool or not self.coarse_sun_mask.any():raise ValueError('At least one declared coarse Sun required')
        self.solar_materialized_mask=np.ones(s,bool)if self.solar_materialized_mask is None else np.asarray(self.solar_materialized_mask)
        if self.solar_materialized_mask.shape!=(s,)or self.solar_materialized_mask.dtype!=bool:raise ValueError('Explicit computed-column mask required; uncomputed differs from coverage-unknown')

    @property
    def has_fine_sun(self):
        return bool(len(self.sun_parameters)and not self.coarse_sun_mask.all())

    def search_indices(self,centre,round_index):
        if not self.has_fine_sun:return np.arange(len(self.sun_vectors))
        return bounded_sun_indices(self.sun_parameters,self.sun_parameters[centre],round_index)



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
    evidence=solar_log_evidence(graph.solar_states[:,:,None],graph.observed_shadow[:,None,:])
    evidence[:,~graph.solar_materialized_mask]=np.nan
    return evidence


def assignments(graph,sun_index,solar_mode='shared',pose=None,intrinsics=None,response=None,precomputed_evidence=None,candidate_rank=0,response_by_sun=None,independent_sun_indices=None,measurement_state=None,native_reprojection_gate=None):
    if solar_mode not in('off','shared','independent'):raise ValueError('Unknown solar consistency control')
    score=graph.appearance.copy()if response is None else np.asarray(response,float).copy()
    if score.shape!=graph.appearance.shape or not np.isfinite(score).all():raise ValueError('Finite candidate appearance/response score required')
    if measurement_state is not None:
        measurement_state=np.asarray(measurement_state)
        if measurement_state.shape!=score.shape or not np.isin(measurement_state,[0,1,2]).all():raise ValueError('Explicit unobserved/native/failed local measurement states required')
    if native_reprojection_gate is not None and(pose is None or native_reprojection_gate!=3.):raise ValueError('Verified native support must retain the fixed3pxgeometry gate')
    evidence=evidence_tensor(graph)if precomputed_evidence is None else precomputed_evidence
    if solar_mode=='shared':
        if not graph.solar_materialized_mask[sun_index]:raise ValueError('Requested solar column is uncomputed, not coverage-unknown')
        score+=evidence[:,sun_index]
    independent_choices=None
    if solar_mode=='independent':
        suns=np.arange(len(graph.sun_vectors))if independent_sun_indices is None else np.asarray(independent_sun_indices,int)
        if suns.ndim!=1 or not len(suns)or(suns<0).any()or(suns>=len(graph.sun_vectors)).any():raise ValueError('Valid independent Sun search columns required')
        if not graph.solar_materialized_mask[suns].all():raise ValueError('Requested independent solar columns are uncomputed')
        if response_by_sun is None:combined=score[:,None,:]+evidence[:,suns]
        else:
            response_by_sun=np.asarray(response_by_sun,float)
            if response_by_sun.shape!=(len(score),len(suns),score.shape[1])or not np.isfinite(response_by_sun).all():raise ValueError('Finite independent NxSxKresponse scores required')
            combined=response_by_sun+evidence[:,suns]
        independent_choices=suns[combined.argmax(1)];score=combined.max(1)
    if pose is not None:
        projected,front=project(graph.world,pose,intrinsics)
        error=np.linalg.norm(graph.query_xy-projected[:,None],axis=-1)
        if measurement_state is not None:
            # An unrefined candidate denotes a33x33search, not a precise
            # native correspondence. This rotation-independent outer bound
            # includes16pxhalfwidth plus <=.5px centre rounding per axis.
            radius=np.sqrt(2.)*16.5
            error=np.where(measurement_state==0,np.maximum(error-radius,0.),error)
        score-=np.where(front[:,None],np.minimum((error/3.)**2,9.),9.)
    if measurement_state is not None:score=np.where(measurement_state==2,-np.inf,score)
    if not 0<=candidate_rank<score.shape[1]:raise ValueError('Initial candidate rank outside candidate bank')
    index=np.argsort(-score,axis=1,kind='stable')[:,candidate_rank];values=score[np.arange(len(score)),index]
    # One query grid point cannot represent two distinct reference anchors.
    keep=[];seen=set()
    for i in sorted(range(len(index)),key=lambda i:(-values[i],i)):
        key=tuple(graph.query_xy[i,index[i]])
        if np.isfinite(values[i])and key not in seen:keep.append(i);seen.add(key)
    keep=np.asarray(sorted(keep),int)
    known=(graph.solar_states[:,sun_index]!=SolarState.UNKNOWN)&(np.abs(graph.observed_shadow[np.arange(len(index)),index]-.5)>=.1)
    if measurement_state is not None:known&=np.isin(np.arange(len(index)),keep)
    cells=np.floor(graph.reference_xy[known]/256).astype(int)
    support=bool(solar_mode=='shared'and known.sum()>=32 and len(set(map(tuple,cells)))>=4)
    result=dict(candidate=index,keep=keep,total=float(values[keep].sum()/len(graph.world)),solar_supported=support,informative_anchors=int(known.sum()),independent_sun_indices=None if independent_choices is None else independent_choices[np.arange(len(index)),index])
    if measurement_state is not None:
        # Pose fitting requires actual native measurements. An unmeasured
        # alternative may still lead the latent coarse assignment, but cannot
        # erase the native evidence already acquired at the same anchor.
        native_scores=np.where(measurement_state==1,score,-np.inf)
        native_index=np.argmax(native_scores,axis=1)
        native_values=native_scores[np.arange(len(index)),native_index]
        native_keep=[];seen=set()
        for i in sorted(range(len(index)),key=lambda i:(-native_values[i],i)):
            key=tuple(graph.query_xy[i,native_index[i]])
            if np.isfinite(native_values[i])and key not in seen:native_keep.append(i);seen.add(key)
        result.update(native_candidate=native_index,native_keep=np.asarray(sorted(native_keep),int))
        if native_reprojection_gate is not None:
            # Current inliers verify a hypothesis, but all paid measurements
            # stay available to the next RANSAC update so a wrong early pose
            # cannot permanently censor corrective observations.
            native_keep=np.asarray(sorted(native_keep),int)
            residual=np.linalg.norm(graph.query_xy[np.arange(len(index)),native_index]-projected,axis=-1)
            result['native_inlier_keep']=native_keep[front[native_keep]&(residual[native_keep]<=3.)]
    return result


def initial_beams(graph,solar_mode='shared',maximum=8,response_enabled=False):
    candidates=[];evidence=evidence_tensor(graph)
    coarse=np.flatnonzero(graph.coarse_sun_mask)
    suns=coarse if solar_mode=='shared'or(response_enabled and solar_mode!='independent')else coarse[:1]
    for rank in range(min(8,graph.appearance.shape[1])):
        for si in suns:
            a=assignments(graph,si,solar_mode,precomputed_evidence=evidence,candidate_rank=rank,independent_sun_indices=coarse)
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
    a=beam['assignment'];keep=a.get('native_keep',a['keep']);candidate=a.get('native_candidate',a['candidate'])
    for i in keep:
        key=(int(i),int(candidate[i]));m=measurements.get(key)
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


def acquisition_predictions(graph,beams,actions,intrinsics,solar_mode='shared'):
    """Vectorized Hxactionsx7 terrain predictions, with no query truth."""
    actions=np.asarray(actions,int);i,c=actions.T;h=len(beams)
    poses=np.stack([np.eye(4)if b['pose']is None else b['pose']for b in beams]);live=np.array([b['pose']is not None for b in beams])
    world=graph.world_variants[i]
    camera=np.einsum('havc,hcd->havd',world[None]-poses[:,None,None,:3,3],poses[:,:3,:3],optimize=True)
    homogeneous=camera@np.asarray(intrinsics).T
    with np.errstate(divide='ignore',invalid='ignore'):xy=homogeneous[...,:2]/homogeneous[...,2,None]
    front=live[:,None,None]&(camera[...,2]>0)&np.isfinite(xy).all(-1)
    delta=(xy-graph.query_xy[i,c][None,:,None,:])/16.
    predictions=np.full((h,len(actions),7,4),np.nan)
    predictions[...,:2]=np.where(front[...,None],delta,np.nan)
    for j,b in enumerate(beams):
        independent=b['assignment'].get('independent_sun_indices')
        sun=np.full(len(i),b['sun_index'],int)if independent is None else independent[i]
        states=graph.solar_variants[i,sun]
        if solar_mode!='off'and not graph.solar_materialized_mask[sun].all():raise ValueError('Active solar predictions require computed rays')
        observable=front[j]&((xy[j]>=0)&(xy[j]<=511)).all(-1)&(states!=0)&(solar_mode!='off')
        predictions[j,:,:,2]=np.where(observable,(states==SolarState.SHADOW).astype(float),np.nan)
    predictions[:,:,:,3]=np.where(front,np.exp(-.5*np.sum(delta*delta,-1)),np.nan)
    scores=np.asarray([b['score']for b in beams]);weights=np.exp(scores-scores.max())
    return predictions,weights,np.isfinite(predictions)


def solve(graph,intrinsics,measure,solar_mode='shared',active=False,policy='active',budget=128,
          response_enabled=False,response_refresh=None,final_estimator=None,solar_lookup=None,
          pose_initialization='strict3'):
    """Joint research inference without query depth, true pose or Sun labels.

    Each local search counts once. Responses are refreshed on the selected
    patches under every CURRENT trial pose, never carried over from an old
    canonicalization. After every measurement, correspondence assignment and
    hypothesis weights update before the next active decision. Actual pose
    RANSAC/LM updates occur at the three common round boundaries, identically
    across ablations. The proposal allocation stays <=4096 plus one common
    final2000iteration estimator. final_estimator(p,q) can supply the exact same
    reference-depth geometry/estimator used by all competing matchers.
    """
    if budget not in(32,64,128):raise ValueError('Declared32/64/128patch budget required')
    if pose_initialization not in('strict3','quantization_aware'):raise ValueError('Explicit supported initial-pose policy required')
    if response_enabled and not callable(response_refresh):raise ValueError('Pair-specific current-pose response refresh callback required')
    graph=CandidateGraph(**{name:getattr(graph,name).copy()for name in graph.__dataclass_fields__})
    def ensure_suns(suns):
        missing=np.unique(np.asarray(suns,int));missing=missing[~graph.solar_materialized_mask[missing]]
        if not len(missing):return
        if not callable(solar_lookup):raise ValueError('Uncomputed solar columns require a source-pinned physical lookup')
        table=solar_lookup(graph.sun_vectors[missing])
        states=np.asarray(table['solar_states']);variants=np.asarray(table['solar_variants'])
        if states.shape!=(len(graph.world),len(missing))or variants.shape!=(len(graph.world),len(missing),7)or not np.isin(states,[0,1,2]).all()or not np.isin(variants,[0,1,2]).all():raise ValueError('Physical lookup returned invalid explicit solar states')
        graph.solar_states[:,missing]=states;graph.solar_variants[:,missing]=variants;graph.solar_materialized_mask[missing]=True
    if solar_mode!='off':ensure_suns(np.flatnonzero(graph.coarse_sun_mask))
    beams=initial_beams(graph,solar_mode,8,response_enabled);measurements={};history=[];incremental=[]
    if not beams:return dict(pose=None,reason='no_candidate_assignments',measurements=[],hypotheses=[])
    n,k=graph.appearance.shape;actions=np.array([(i,c)for i in range(n)for c in range(k)],int)
    points=np.repeat(graph.reference_xy,k,axis=0);used=np.zeros(len(actions),bool);appearances=graph.appearance.reshape(-1)
    evidence=evidence_tensor(graph);response_calls=0;response_edges=set()
    measurement_state=np.zeros((n,k),np.uint8)if pose_initialization=='quantization_aware'else None

    def response_cube(beam,suns):
        nonlocal response_calls
        if not response_enabled or beam['pose']is None:
            return np.repeat(graph.appearance[:,None,:],len(suns),axis=1)
        # Poses stay fixed within an acquisition round. Cache the accumulated
        # response matrix and evaluate only newly selected measurements; a new
        # pose/Sun neighborhood creates a different entry and refreshes it.
        token=getattr(response_refresh,'cache_token',None)
        token=token()if callable(token)else None
        key=(beam['pose'].tobytes(),tuple(suns),token)
        cache=beam.setdefault('_response_matrices',{})
        if key not in cache:
            if len(cache)>=4:cache.clear()
            cache[key]=(np.repeat(graph.appearance[:,None,:],len(suns),axis=1),set())
        result,done=cache[key]
        for(i,c),measurement in measurements.items():
            edge=(i,c)
            if edge in done:continue
            done.add(edge)
            if measurement.get('query_xy')is None:continue
            response_calls+=1
            values=response_refresh(i,c,measurement['query_xy'],beam['pose'],suns)
            if values is None:continue
            values=np.asarray(values,float)
            if values.shape!=(len(suns),)or not np.isfinite(values).all():raise ValueError('Finite one-response-score-per-requested-Sun required')
            result[i,:,c]=values;response_edges.add(edge)
        return result

    def update_beam(beam,suns):
        suns=np.asarray(suns,int)
        if solar_mode!='off':
            ensure_suns(suns)
            evidence[:,suns]=solar_log_evidence(graph.solar_states[:,suns,None],graph.observed_shadow[:,None,:])
        responses=None if solar_mode=='independent'and not response_enabled else response_cube(beam,suns)
        pose=beam['pose']if beam['native_pose']or pose_initialization=='quantization_aware'else None
        if solar_mode=='independent':
            a=assignments(graph,beam['sun_index'],solar_mode,pose,intrinsics,None,evidence,
                response_by_sun=responses,independent_sun_indices=suns,measurement_state=measurement_state,
                native_reprojection_gate=3. if measurement_state is not None and beam['native_pose']else None)
            beam['assignment']=a;beam['score']=a['total'];return
        choices=[]
        for j,si in enumerate(suns):
            a=assignments(graph,int(si),solar_mode,pose,intrinsics,responses[:,j,:],evidence,measurement_state=measurement_state,
                native_reprojection_gate=3. if measurement_state is not None and beam['native_pose']else None)
            choices.append((a['total'],-int(si),int(si),a))
        _,_,beam['sun_index'],beam['assignment']=max(choices,key=lambda x:x[:2]);beam['score']=beam['assignment']['total']

    def take(action,beam):
        i,c=map(int,actions[action]);measurement=measure(i,c,beam['pose'],beam['sun_index']);used[action]=True
        if not isinstance(measurement,dict):raise ValueError('Patch measurement must be an explicit record')
        q=measurement.get('query_xy')
        if q is not None and(np.asarray(q).shape!=(2,)or not np.isfinite(q).all()):raise ValueError('Finite native query correspondence required')
        if q is not None:
            if ((np.asarray(q)<0)|(np.asarray(q)>511)).any():raise ValueError('Measured correspondence outside native image')
            graph.query_xy[i,c]=q
            if 'observed_shadow'in measurement:graph.observed_shadow[i,c]=float(measurement['observed_shadow'])
            computed=graph.solar_materialized_mask
            evidence[i,computed,c]=solar_log_evidence(graph.solar_states[i,computed],graph.observed_shadow[i,c])
        measurements[(i,c)]=measurement;history.append(dict(anchor=i,candidate=c,trial_sun_index=beam['sun_index']))
        if measurement_state is not None:measurement_state[i,c]=1 if q is not None else 2
        # Assimilate the new measurement before the next acquisition. Poses
        # remain fixed within a round, allowing exact reuse of responses.
        for current in beams:
            if solar_mode=='independent':suns=current.get('_independent_search',np.flatnonzero(graph.coarse_sun_mask))
            else:suns=np.array([current['sun_index']])
            update_beam(current,suns)
        scores=np.array([b['score']for b in beams]);weights=np.exp(scores-scores.max());weights/=weights.sum()
        incremental.append(dict(after_search=len(history),weights=weights.tolist(),candidate_assignments=[b['assignment']['candidate'].copy()for b in beams]))

    proposal_requested=0
    for b in beams:
        a=b['assignment'];ids=a['keep'];q=graph.query_xy[ids,a['candidate'][ids]];allocation=1024//len(beams);proposal_requested+=allocation
        if pose_initialization=='quantization_aware':
            from solar_pose_proposals_v73 import propose
            b['pose'],b['pose_info']=propose(graph.world[ids],q,intrinsics,allocation)
        else:b['pose'],b['pose_info']=estimate(graph.world[ids],q,intrinsics,dict(solver='ap3p',iterations=allocation,reclassify=True))
        b['native_pose']=False
    initialization=dict(policy=pose_initialization,initial_hypothesis_count=len(beams),
        initial_live_poses=sum(b['pose']is not None for b in beams),
        proposal_info=[dict(b['pose_info'])for b in beams],final_reprojection_threshold_native_px=3.)
    seed_count=min(budget//2,8*len(beams));orders=[]
    for b in beams:
        a=b['assignment'];order=sorted(a['keep'],key=lambda i:(int(graph.reference_xy[i,1]//128)*4+int(graph.reference_xy[i,0]//128),i))
        positions=np.rint(np.linspace(0,max(0,len(order)-1),8)).astype(int)
        orders.append([int(order[j])*k+int(a['candidate'][order[j]])for j in positions]if order else[])
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
            predictions,weights,valid=acquisition_predictions(graph,beams,actions,intrinsics,solar_mode)
            chosen_policy=policy if active and any(b['pose']is not None for b in beams)else'uniform'
            visited={i for i,c in measurements};unavailable=used.copy()
            if len(visited)<n:unavailable|=np.array([int(i)in visited for i,c in actions])
            action=choose_action(predictions,weights,points,unavailable,chosen_policy,appearances,valid)
            if action is None:break
            take(action,max(beams,key=lambda b:b['score']))
        for b in beams:
            if solar_mode=='independent':
                centres=np.unique(b['assignment']['independent_sun_indices'])
                suns=np.unique(np.concatenate([graph.search_indices(si,round_index)for si in centres]));b['_independent_search']=suns
            elif solar_mode=='shared'or response_enabled:suns=graph.search_indices(b['sun_index'],round_index)
            else:suns=np.array([b['sun_index']])
            update_beam(b,suns)
            _,q,w,_=_matched(graph,b,measurements);allocation=1024//len(beams);proposal_requested+=allocation
            native_pose,info=estimate(w,q,intrinsics,dict(solver='ap3p',iterations=allocation,reclassify=True))
            if native_pose is not None:
                b['pose']=_lm_update(native_pose,w,q,intrinsics);b['native_pose']=True;b['pose_info']=info
            # No stale pre-pose score is used by the next acquisition.
            update_beam(b,np.array([b['sun_index']])if solar_mode!='independent'else b['_independent_search'])
        rounds.append(dict(round=round_index+1,queries=len(history),live_poses=sum(b['pose']is not None for b in beams),sun_indices=[b['sun_index']for b in beams]))
    candidates=[b for b in beams if b['pose']is not None]
    if not candidates:
        p,q,_,_=_matched(graph,max(beams,key=lambda b:b['score']),measurements)
        for b in beams:b.pop('_response_matrices',None);b.pop('_independent_search',None)
        return dict(pose=None,reason='no_pose_hypothesis',reference_xy=p,query_xy=q,measurements=history,hypotheses=beams,proposal_samples_requested=proposal_requested,rounds=rounds,incremental_updates=incremental,response_refresh_calls=response_calls,initialization=initialization)
    # Re-score all surviving beams under their LAST updated poses and bounded
    # Sun support before ranking; no extra RANSAC budget is introduced.
    for b in candidates:
        if solar_mode=='independent':suns=b.get('_independent_search',np.flatnonzero(graph.coarse_sun_mask))
        elif solar_mode=='shared'or response_enabled:suns=graph.search_indices(b['sun_index'],2)
        else:suns=np.array([b['sun_index']])
        update_beam(b,suns)
    if pose_initialization=='quantization_aware':
        # A free coarse proposal must not outrank a native-verified hypothesis
        # just because it escaped the geometric likelihood or lacks6matches.
        supported=[b for b in candidates if len(_matched(graph,b,measurements)[0])>=6]
        verified=[b for b in supported if b['native_pose']and len(b['assignment'].get('native_inlier_keep',[]))>=6]
        if verified:candidates=verified
        elif supported:candidates=supported
        else:
            p,q,_,_=_matched(graph,max(candidates,key=lambda b:b['score']),measurements)
            for b in beams:b.pop('_response_matrices',None);b.pop('_independent_search',None)
            return dict(pose=None,reason='insufficient_supported_native_matches',reference_xy=p,query_xy=q,
                hypotheses=beams,measurements=history,proposal_samples_requested=proposal_requested,
                rounds=rounds,initialization=initialization,response_refresh_calls=response_calls)
    best=max(candidates,key=lambda b:b['score']);p,q,w,_=_matched(graph,best,measurements)
    if final_estimator is None:
        pose,info=estimate(w,q,intrinsics,dict(solver='ap3p',iterations=2000,reclassify=True));final_geometry='exact_reference_map_anchors';final_iterations=2000
    else:
        pose,info=final_estimator(p,q);final_geometry='caller_supplied_common_reference_geometry';final_iterations=info.get('iterations_requested')
    for b in beams:b.pop('_response_matrices',None);b.pop('_independent_search',None)
    for record in history:record['response_used']=(record['anchor'],record['candidate'])in response_edges
    return dict(pose=pose,reason=info['reason'],reference_xy=p,query_xy=q,initialization=initialization,
        sun_enu=None if solar_mode!='shared'else graph.sun_vectors[best['sun_index']],solar_supported=best['assignment']['solar_supported'],
        measurements=history,hypotheses=beams,proposal_samples_requested=proposal_requested,final_estimator_iterations=final_iterations,
        final_geometry=final_geometry,rounds=rounds,incremental_updates=incremental,response_measurements=len(response_edges),
        response_refresh_calls=response_calls,fine_sun_search_available=graph.has_fine_sun,solar_columns_materialized=int(graph.solar_materialized_mask.sum()),
        estimated_sun_parameters=None if not len(graph.sun_parameters)or solar_mode!='shared'else graph.sun_parameters[best['sun_index']])
