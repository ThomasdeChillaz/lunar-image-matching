"""Native image frontend for the experimental shared-Sun/response/active solver.

Only reference geometry enters inference. Precomputed ray tables are physical
reference-map lookups, never query depth, query pose or true query Sun labels.
"""
from dataclasses import dataclass,replace
from collections import OrderedDict
from time import perf_counter
import hashlib
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from contrastive_terrain import sample_descriptors
from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import map_points_z
from rotation_bank_pose_v4 import quarter_turn
from solar_assignment_v63 import CandidateGraph,height_variants,solve
from solar_native_v63 import SparseNativeRefiner,canonical_patch,state_digest
from solar_physics_v63 import soft_shadow_observation
from solar_response_v63 import FusedLightingDecoder
from solar_rotation_v71 import RotationBank,proposal_angles as validate_proposal_angles


def reference_anchors():
    axis=49.5+16*np.rint(np.linspace(0,25,16)).astype(int)
    y,x=np.meshgrid(axis,axis,indexing='ij')
    return np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)


@dataclass
class ReferenceGeometry:
    depth:np.ndarray
    intrinsics:np.ndarray
    cam2world:np.ndarray
    normals:np.ndarray
    sun_vectors:np.ndarray
    solar_states:np.ndarray
    solar_variants:np.ndarray
    reference_sun:np.ndarray
    world_variants:np.ndarray
    sun_parameters:np.ndarray=None
    coarse_sun_mask:np.ndarray=None
    solar_materialized_mask:np.ndarray=None

    def __post_init__(self):
        self.intrinsics,self.cam2world=camera_check(self.intrinsics,self.cam2world)
        self.depth=np.asarray(self.depth,float);self.normals=np.asarray(self.normals,float)
        self.world_variants=np.asarray(self.world_variants,float)
        if self.world_variants.shape!=(256,7,3)or not np.isfinite(self.world_variants).all():raise ValueError('Reference-ray intersections for all seven terrain-height variants required')
        self.sun_vectors=np.asarray(self.sun_vectors,float);self.reference_sun=np.asarray(self.reference_sun,float)
        self.solar_states=np.asarray(self.solar_states);self.solar_variants=np.asarray(self.solar_variants)
        if self.depth.shape!=(512,512)or self.normals.shape!=(256,3):raise ValueError('Native reference axial depth and256anchor normals required')
        if not np.isfinite(self.normals).all()or not np.allclose(np.linalg.norm(self.normals,axis=1),1,atol=1e-6):raise ValueError('Unit reference-world surface normals required')
        s=len(self.sun_vectors)
        if self.sun_vectors.shape!=(s,3)or s<1 or self.reference_sun.shape!=(3,)or not np.isfinite(self.reference_sun).all()or not np.isfinite(self.sun_vectors).all():raise ValueError('Finite reference and trial Sun vectors required')
        if not np.allclose(np.linalg.norm(self.sun_vectors,axis=-1),1,atol=1e-6)or not np.isclose(np.linalg.norm(self.reference_sun),1):raise ValueError('Unit world-frame solar vectors required')
        if self.solar_states.shape!=(256,s)or self.solar_variants.shape!=(256,s,7)or not np.isin(self.solar_states,[0,1,2]).all()or not np.isin(self.solar_variants,[0,1,2]).all():raise ValueError('Explicit reference-map solar states and seven height variants required')
        self.sun_parameters=np.empty((0,2))if self.sun_parameters is None else np.asarray(self.sun_parameters,float)
        self.coarse_sun_mask=np.ones(s,bool)if self.coarse_sun_mask is None else np.asarray(self.coarse_sun_mask)
        if self.coarse_sun_mask.shape!=(s,)or self.coarse_sun_mask.dtype!=bool or not self.coarse_sun_mask.any():raise ValueError('Explicit nonempty coarse Sun search mask required')
        if self.sun_parameters.shape not in((0,2),(s,2))or not np.isfinite(self.sun_parameters).all():raise ValueError('Empty legacy or Sx2hour/decl metadata required')
        self.solar_materialized_mask=np.ones(s,bool)if self.solar_materialized_mask is None else np.asarray(self.solar_materialized_mask)
        if self.solar_materialized_mask.shape!=(s,)or self.solar_materialized_mask.dtype!=bool:raise ValueError('Explicit solar computed-column mask required')
        if len(self.sun_parameters):
            if (np.abs(self.sun_parameters[:,0])>90+1e-9).any()or(np.abs(self.sun_parameters[:,1])>1.5+1e-9).any():raise ValueError('Sun catalogue outside declared lunar search range')
            expected=np.isclose(self.sun_parameters[:,0]/5,np.rint(self.sun_parameters[:,0]/5),atol=1e-9)&np.isclose(self.sun_parameters[:,1]/.5,np.rint(self.sun_parameters[:,1]/.5),atol=1e-9)
            if not np.array_equal(expected,self.coarse_sun_mask):raise ValueError('Primary initialization mask must use only5degree/.5degree Sun states')

    def with_reference_sun(self,sun):
        """Actual REFERENCE image metadata; never supplies a query Sun label."""
        value=np.asarray(sun,float)
        if value.shape!=(3,)or not np.isfinite(value).all()or not np.isclose(np.linalg.norm(value),1,atol=1e-6):raise ValueError('Unit observed reference-image Sun metadata required')
        return replace(self,reference_sun=value.copy())


    @classmethod
    def load(cls,path):
        allowed=set(cls.__dataclass_fields__)
        with np.load(path,allow_pickle=False)as z:
            legacy=allowed-{'sun_parameters','coarse_sun_mask','solar_materialized_mask'}
            schemas=(allowed,allowed-{'solar_materialized_mask'},legacy,legacy|{'solar_materialized_mask'})
            if set(z.files)not in schemas:raise ValueError('Reference-only archive schema required; query truth and extra keys forbidden')
            return cls(**{k:z[k].copy()for k in z.files})


def candidates(encoder,reference,query,proposal_angles=None):
    """Frozen128pxproposal maps; native high-resolution features stay lazy."""
    p=reference_anchors();yy,xx=np.mgrid[12:116:4,12:116:4];low=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
    native=(low+.5)*4-.5
    bank=RotationBank(query,proposal_angles);valid=bank.coarse_mask(low).reshape(-1)
    images=[reference]+bank.images
    batch=np.stack([cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA)for im in images])
    with torch.inference_mode():
        dense,_=encoder(torch.from_numpy(batch).float()[:,None]/255)
        pa=torch.from_numpy(((p+.5)/4-.5).astype(np.float32))[None]
        a=sample_descriptors(dense[:1],pa)[0]
        qpoints=torch.from_numpy(low)[None].expand(len(bank.angles),-1,-1)
        b=sample_descriptors(dense[1:],qpoints).reshape(-1,32)
        score=a@b.T
        score[:,~torch.as_tensor(valid)]=-torch.inf
        # Stable sorting fixes ties independently of topk implementation.
        rank=torch.argsort(score,dim=1,descending=True,stable=True)[:,:8]
        values=score.gather(1,rank).cpu().numpy();ids=rank.cpu().numpy()
    rotations=ids//len(low);indices=ids%len(low);q=np.empty((len(p),8,2))
    if not np.isfinite(values).all():raise ValueError('Insufficient fully supported rotation proposals')
    for r in range(len(bank.angles)):
        use=rotations==r;q[use]=bank.transform(native[indices[use]],r,True)
    return p,q,values,rotations


def response_evidence(model,reference,query,point,query_center,geometry,trial_pose,anchor,pooled=False,
                      sun_indices=None,query_intrinsics=None,counters=None,feature_cache=None,
                      decoder=None,reference_decoder=None):
    if counters is not None:counters['canonicalizations']=counters.get('canonicalizations',0)+1
    k_query=getattr(geometry,'query_intrinsics',None)if query_intrinsics is None else query_intrinsics
    if k_query is None:raise ValueError('Explicit query intrinsics required for pair-specific canonicalization')
    patches,meta=canonical_patch(reference,query,point,geometry.depth,geometry.intrinsics,geometry.cam2world,
        k_query,trial_pose,query_center=query_center)
    if patches is None:return None
    r=geometry.cam2world[:3,:3];normal=geometry.normals[anchor]@r
    world=geometry.world_variants[anchor,0]
    reference_view=geometry.cam2world[:3,3]-world;view=trial_pose[:3,3]-world
    if min(np.linalg.norm(reference_view),np.linalg.norm(view))<1e-10:raise ValueError('Exact reference anchor must be separated from camera centres')
    reference_view=reference_view/np.linalg.norm(reference_view)@r;view=view/np.linalg.norm(view)@r
    selected=np.arange(len(geometry.sun_vectors))if sun_indices is None else np.asarray(sun_indices,int)
    if selected.ndim!=1 or not len(selected)or(selected<0).any()or(selected>=len(geometry.sun_vectors)).any():raise ValueError('Explicit valid target Sun columns required')
    reference_sun=geometry.reference_sun@r;suns=geometry.sun_vectors[selected]@r;n=len(suns)
    dtype=model.coefficients.weight.dtype;device=model.coefficients.weight.device
    key=(state_digest(model.fine),hashlib.sha256(patches[0].tobytes()+patches[1].tobytes()).hexdigest())
    with torch.inference_mode():
        if feature_cache is not None and key in feature_cache:
            features=feature_cache[key]
            if counters is not None:counters['canonical_feature_cache_hits']=counters.get('canonical_feature_cache_hits',0)+1
        else:
            if counters is not None:
                counters['patch_encoder_calls']=counters.get('patch_encoder_calls',0)+1;counters['patches_encoded']=counters.get('patches_encoded',0)+2
            features=model.patch_features(torch.as_tensor(np.stack(patches).copy(),dtype=dtype,device=device)[:,None]/255)
            if feature_cache is not None:
                feature_cache[key]=features
                while len(feature_cache)>512:feature_cache.popitem(last=False)
        ref_condition=torch.tensor(np.concatenate([reference_sun,normal,reference_view]),dtype=dtype,device=device)[None]
        query_condition=torch.tensor(np.column_stack([suns,np.repeat(normal[None],n,0),np.repeat(view[None],n,0)]),dtype=dtype,device=device)
        if decoder is None:
            ref_coeff=model.condition_features(features[:1],ref_condition).expand(n,-1,-1)
            query_coeff=model.condition_features(features[1:].expand(n,-1),query_condition)
            target=torch.tensor(suns,dtype=dtype,device=device);ref_target=torch.tensor(np.repeat(reference_sun[None],n,0),dtype=dtype,device=device)
            reference_at_query=model.decode(ref_coeff,target,pooled);query_at_query=model.decode(query_coeff,target,pooled)
            query_at_reference=model.decode(query_coeff,ref_target,pooled);reference_at_reference=model.decode(ref_coeff,ref_target,pooled)
        else:
            if reference_decoder is None:raise ValueError('Reference-Sun fused decoder required')
            reference_at_query=decoder(features[:1].expand(n,-1),ref_condition.expand(n,-1),selected)
            query_at_query=decoder(features[1:].expand(n,-1),query_condition,selected)
            query_at_reference=reference_decoder(features[1:].expand(n,-1),query_condition,np.zeros(n,int))
            reference_at_reference=reference_decoder(features[:1],ref_condition,np.zeros(1,int)).expand(n,-1)
        # BOTH operands are predictions in the frozen-teacher coordinate system.
        # No raw trainable-encoder basis is compared with a decoded prediction.
        scores=.5*((reference_at_query*query_at_query).sum(-1)+(query_at_reference*reference_at_reference).sum(-1))
    return scores.cpu().numpy()


class PairResponseCache:
    """Selected measurements only; exact pose and checkpoint-sensitive cache.

    A repeated query of the same selected patch under a changed trial pose is
    re-canonicalized and re-encoded. Search budgets count local searches; all
    refresh work is separately counted and included in the complete timer.
    """
    def __init__(self,model,reference,query,geometry,k_query,anchors,graph_sun_ids,pooled=False,capacity=512):
        self.model=model;self.reference=reference;self.query=query;self.geometry=geometry;self.k_query=k_query
        self.anchors=np.asarray(anchors,int);self.graph_sun_ids=np.asarray(graph_sun_ids,int);self.pooled=pooled
        r=geometry.cam2world[:3,:3]
        self.decoder=FusedLightingDecoder(model,geometry.sun_vectors@r,pooled);self.reference_decoder=FusedLightingDecoder(model,(geometry.reference_sun@r)[None],pooled)
        self.features=OrderedDict()
        self.capacity=capacity;self.cache=OrderedDict();self.cost=dict(refresh_requests=0,cache_hits=0,canonicalizations=0,patch_encoder_calls=0,patches_encoded=0)

    def cache_token(self):
        return state_digest(self.model)

    def __call__(self,i,c,query_xy,trial_pose,sun_indices):
        self.cost['refresh_requests']+=1
        if trial_pose is None:return None
        sun_indices=np.asarray(sun_indices,int)
        key=(state_digest(self.model),int(i),int(c),np.asarray(query_xy,float).tobytes(),np.asarray(trial_pose,float).tobytes(),tuple(sun_indices))
        if key not in self.cache:
            anchor=int(self.anchors[i]);point=reference_anchors()[anchor]
            self.cache[key]=response_evidence(self.model,self.reference,self.query,point,query_xy,self.geometry,trial_pose,anchor,self.pooled,self.graph_sun_ids[sun_indices],self.k_query,self.cost,self.features,self.decoder,self.reference_decoder)
        else:self.cost['cache_hits']+=1
        self.cache.move_to_end(key)
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        result=self.cache[key]
        return None if result is None else result.copy()


class SolarMatcher:
    inference_revision='solar-v68'
    def __init__(self,coarse,fine,response=None,pooled=False,proposal_angles=None,pose_initialization='strict3'):
        if pose_initialization not in('strict3','quantization_aware'):raise ValueError('Unsupported initial-pose policy')
        self.pose_initialization=pose_initialization
        self.inference_revision='solar-v68'if pose_initialization=='strict3'else'solar-v73-quantized-init'
        self.proposal_angles=validate_proposal_angles(proposal_angles)
        self.pooled=bool(pooled);self.response=response.eval()if response is not None else None
        if response is not None:
            if state_digest(coarse)!=state_digest(response.coarse):raise ValueError('Response model must own the exact frozen proposal encoder')
            if fine is not response.fine and state_digest(fine)!=state_digest(response.coarse):raise ValueError('No hidden third native encoder is permitted')
            # Keep A's original proposal/refinement foundation frozen. B's
            # learned encoder was trained on canonical patches and must not
            # silently replace ordinary uncanonicalized native refinement.
            self.coarse=response.coarse.eval().requires_grad_(False);self.fine=self.coarse
            self.native_refinement_encoder='frozen_original_shared_descriptor'
            count=sum(p.numel()for p in response.parameters())
        else:
            self.coarse=coarse.eval().requires_grad_(False);self.fine=fine.eval().requires_grad_(False)
            self.native_refinement_encoder='caller_fine_encoder'
            count=sum(p.numel()for p in coarse.parameters())+sum(p.numel()for p in fine.parameters())
        if count>=100000:raise ValueError('Complete experimental matcher exceeds100kparameters')
        self.total_parameters=count

    def localize(self,reference,query,geometry,query_intrinsics,solar_mode='shared',response=False,active=False,budget=128,policy='active',known_sun=None,reference_sun=None,final_estimator=None,solar_lookup=None):
        tick=perf_counter()
        for image in(reference,query):
            if np.asarray(image).shape!=(512,512)or np.asarray(image).dtype!=np.uint8:raise ValueError('Native512uint8grayscale images required')
        query_intrinsics,_=camera_check(query_intrinsics,np.eye(4))
        if reference_sun is not None:geometry=geometry.with_reference_sun(reference_sun)
        if response and self.response is None:raise ValueError('Learned response requested without a frozen response checkpoint')
        if known_sun is not None:
            known_sun=np.asarray(known_sun,float)
            if known_sun.shape!=(3,)or not np.isfinite(known_sun).all()or not np.isclose(np.linalg.norm(known_sun),1,atol=1e-6):raise ValueError('Finite known-Sun metadata required')
            matches=np.flatnonzero(np.linalg.norm(geometry.sun_vectors-known_sun,axis=1)<1e-6)
            if len(matches)>1:raise ValueError('Ambiguous exact known-Sun catalogue direction')
            if len(matches)==0:
                if not callable(solar_lookup):raise ValueError('Exact known Sun outside catalogue requires a source-pinned physical lookup; snapping forbidden')
                table=solar_lookup(known_sun[None])
                geometry=replace(geometry,sun_vectors=known_sun[None].copy(),solar_states=np.asarray(table['solar_states']),solar_variants=np.asarray(table['solar_variants']),sun_parameters=np.empty((0,2)),coarse_sun_mask=np.ones(1,bool),solar_materialized_mask=np.ones(1,bool))
                ids=np.array([0])
            else:ids=matches
        else:ids=np.arange(len(geometry.sun_vectors))
        if min(reference.std(),query.std())<1:return dict(pose=None,reason='blank_or_uninformative_image',reference_xy=np.empty((0,2)),query_xy=np.empty((0,2)),elapsed_seconds=perf_counter()-tick,inference_revision=self.inference_revision)
        p,q,appearance,rotation=candidates(self.coarse,reference,query,self.proposal_angles)
        world,use=map_points_z(p,geometry.depth,geometry.intrinsics,geometry.cam2world)
        anchors=np.flatnonzero(use)
        if len(anchors)<6:return dict(pose=None,reason='insufficient_reference_surface',reference_xy=np.empty((0,2)),query_xy=np.empty((0,2)),elapsed_seconds=perf_counter()-tick,inference_revision=self.inference_revision)
        p,q,appearance,rotation=p[use],q[use],appearance[use],rotation[use]
        # Exact reference-ray intersections are supplied by the physical map
        # producer; nearest depth pixels only establish available image support.
        world=geometry.world_variants[use,0]
        from solar_assignment_v63 import project
        reprojection,front=project(world,geometry.cam2world,geometry.intrinsics)
        if not front.all()or np.max(np.linalg.norm(reprojection-p,axis=1))>.02:raise ValueError('Reference-map anchor geometry violates native pixel centres')
        shadow,observation=soft_shadow_observation(query)
        observed=cv2.remap(shadow.astype(np.float32),q[:,:,0].astype(np.float32),q[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        graph=CandidateGraph(p,world,q,appearance,observed,geometry.solar_states[use][:,ids],geometry.sun_vectors[ids],
            geometry.world_variants[use],geometry.solar_variants[use][:,ids],
            geometry.sun_parameters[ids]if len(geometry.sun_parameters)else None,
            np.ones(len(ids),bool)if known_sun is not None else geometry.coarse_sun_mask[ids],geometry.solar_materialized_mask[ids])
        refiner=SparseNativeRefiner(self.fine);query_bank=RotationBank(query,self.proposal_angles)
        responder=PairResponseCache(self.response,reference,query,geometry,query_intrinsics,anchors,ids,self.pooled)if response else None
        def measure(i,c,trial_pose,sun):
            r=int(rotation[i,c]);proposal=query_bank.transform(q[i,c][None],r,False)[0]
            if not query_bank.native_support(proposal,r):return dict(query_xy=None,reason='incomplete_rotated_native_patch_support')
            measured,meta=refiner.refine(reference,query_bank.images[r],p[i],proposal)
            if measured is None:return dict(query_xy=None,reason=meta['reason'])
            measured=query_bank.transform(measured[None],r,True)[0]
            if not((measured>=0)&(measured<=511)).all():return dict(query_xy=None,reason='refinement_outside_image')
            record=dict(query_xy=measured,observed_shadow=float(cv2.remap(shadow.astype(np.float32),np.float32([[measured[0]]]),np.float32([[measured[1]]]),cv2.INTER_LINEAR)[0,0]))
            return record
        if final_estimator is None:
            from native_baseline_pose_v7 import propose_pose
            def final_estimator(native_p,native_q):
                pose,info=propose_pose(native_p,native_q,geometry.depth,geometry.intrinsics,geometry.cam2world,query_intrinsics)
                return pose,dict(info,iterations_requested=2000,reprojection_threshold_native_px=3.,solver='ap3p_reclassify')
            estimator_label='common_reference_axial_depth_AP3P_3px'
        else:estimator_label='caller_supplied_common_final_estimator'
        def graph_lookup(suns):
            table=solar_lookup(suns)
            return dict(solar_states=np.asarray(table['solar_states'])[use],solar_variants=np.asarray(table['solar_variants'])[use])
        result=solve(graph,query_intrinsics,measure,solar_mode,active,policy,budget,response,response_refresh=responder,final_estimator=final_estimator,solar_lookup=graph_lookup if callable(solar_lookup)else None,pose_initialization=self.pose_initialization)
        result['final_geometry']=estimator_label
        result['proposal_bank']=dict(angles_degrees=list(self.proposal_angles),anchors=256,candidates_per_anchor=8,native_image_size=512,engineering_control=True,validity='Entire coarse25px low-resolution receptive field and native57px refinement crop must have valid rotated source support; inverse affine returns native pixel centres.')
        result.update(elapsed_seconds=perf_counter()-tick,inference_revision=self.inference_revision,total_parameters=self.total_parameters,native_encoder_forwards=refiner.forward_calls,
            observation=observation,input_digest=hashlib.sha256(reference.tobytes()+query.tobytes()).hexdigest(),
            inference_inputs='reference image/depth/pose/normals/solar ray table; query image/intrinsics; optional known-Sun metadata; no query truth',known_sun_supplied=known_sun is not None,
            response_cost=None if responder is None else responder.cost,
            solar_lookup_cost=None if solar_lookup is None else dict(getattr(solar_lookup,'stats',{})),
            sun_search_limit='Coarse5degree/.5degree initialization, then bounded1.25degree/.25degree catalogue refinement.'if graph.has_fine_sun else'Legacy discrete catalogue; no fine columns available.',
            reference_sun_world=geometry.reference_sun.copy(),native_refinement_encoder=self.native_refinement_encoder)
        return result
