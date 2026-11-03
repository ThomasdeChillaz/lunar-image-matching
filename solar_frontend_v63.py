"""Native image frontend for the experimental shared-Sun/response/active solver.

Only reference geometry enters inference. Precomputed ray tables are physical
reference-map lookups, never query depth, query pose or true query Sun labels.
"""
from dataclasses import dataclass
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

    @classmethod
    def load(cls,path):
        allowed=set(cls.__dataclass_fields__)
        with np.load(path,allow_pickle=False)as z:
            if set(z.files)!=allowed:raise ValueError('Reference-only archive schema required; query truth and extra keys forbidden')
            return cls(**{k:z[k].copy()for k in z.files})


def candidates(encoder,reference,query):
    """Frozen128pxproposal maps; native high-resolution features stay lazy."""
    p=reference_anchors();yy,xx=np.mgrid[12:116:4,12:116:4];low=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
    native=(low+.5)*4-.5
    images=[reference]+[np.rot90(query,r).copy()for r in range(4)]
    batch=np.stack([cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA)for im in images])
    with torch.inference_mode():
        dense,_=encoder(torch.from_numpy(batch).float()[:,None]/255)
        pa=torch.from_numpy(((p+.5)/4-.5).astype(np.float32))[None]
        a=sample_descriptors(dense[:1],pa)[0]
        qpoints=torch.from_numpy(low)[None].expand(4,-1,-1)
        b=sample_descriptors(dense[1:],qpoints).reshape(-1,32)
        score=a@b.T
        # Stable sorting fixes ties independently of topk implementation.
        rank=torch.argsort(score,dim=1,descending=True,stable=True)[:,:8]
        values=score.gather(1,rank).cpu().numpy();ids=rank.cpu().numpy()
    rotations=ids//len(low);indices=ids%len(low);q=np.empty((len(p),8,2))
    for r in range(4):
        use=rotations==r;q[use]=quarter_turn(native[indices[use]],r,512,True)
    return p,q,values,rotations


def response_evidence(model,reference,query,point,query_center,geometry,trial_pose,anchor,pooled=False):
    patches,meta=canonical_patch(reference,query,point,geometry.depth,geometry.intrinsics,geometry.cam2world,
        geometry.query_intrinsics,trial_pose,query_center=query_center)
    if patches is None:return None
    r=geometry.cam2world[:3,:3];normal=geometry.normals[anchor]@r
    world,use=map_points_z(np.asarray(point)[None],geometry.depth,geometry.intrinsics,geometry.cam2world)
    if not use.all():return None
    view=trial_pose[:3,3]-world[0];view/=max(np.linalg.norm(view),1e-12);view=view@r
    reference_sun=geometry.reference_sun@r;suns=geometry.sun_vectors@r;s=len(suns)
    with torch.inference_mode():
        features=model.patch_features(torch.from_numpy(np.stack(patches).copy()).float()[:,None]/255)
        ref_condition=torch.tensor(np.concatenate([reference_sun,normal,(geometry.cam2world[:3,3]-world[0])/np.linalg.norm(geometry.cam2world[:3,3]-world[0])@r]),dtype=torch.float32)[None]
        ref_coeff=model.condition_features(features[:1],ref_condition).expand(s,-1,-1)
        query_condition=torch.tensor(np.column_stack([suns,np.repeat(normal[None],s,0),np.repeat(view[None],s,0)]),dtype=torch.float32)
        query_coeff=model.condition_features(features[1:].expand(s,-1),query_condition)
        forward=model.decode(ref_coeff,torch.tensor(suns,dtype=torch.float32),pooled)
        reverse=model.decode(query_coeff,torch.tensor(np.repeat(reference_sun[None],s,0),dtype=torch.float32),pooled)
        # Both directions compare at the same physical target illumination.
        observed=F.normalize(features,dim=-1)
        scores=.5*((forward*observed[1]).sum(-1)+(reverse*observed[0]).sum(-1))
    return scores.cpu().numpy()


class SolarMatcher:
    def __init__(self,coarse,fine,response=None,pooled=False):
        self.coarse=coarse.eval().requires_grad_(False);self.fine=fine.eval().requires_grad_(False)
        self.response=response.eval()if response is not None else None;self.pooled=bool(pooled)
        # The response network owns both encoders; no hidden third encoder.
        if response is not None:
            if state_digest(self.coarse)!=state_digest(response.coarse)or state_digest(self.fine)!=state_digest(response.fine):raise ValueError('Response model must own the exact coarse and fine encoders')
            count=sum(p.numel()for p in response.parameters())
        else:count=sum(p.numel()for p in coarse.parameters())+sum(p.numel()for p in fine.parameters())
        if count>=100000:raise ValueError('Complete experimental matcher exceeds100kparameters')
        self.total_parameters=count

    def localize(self,reference,query,geometry,query_intrinsics,solar_mode='shared',response=False,active=False,budget=128,policy='active',known_sun=None):
        tick=perf_counter()
        for image in(reference,query):
            if np.asarray(image).shape!=(512,512)or np.asarray(image).dtype!=np.uint8:raise ValueError('Native512uint8grayscale images required')
        query_intrinsics,_=camera_check(query_intrinsics,np.eye(4))
        if response and self.response is None:raise ValueError('Learned response requested without a frozen response checkpoint')
        if known_sun is not None:
            known_sun=np.asarray(known_sun,float)
            if known_sun.shape!=(3,)or not np.isfinite(known_sun).all():raise ValueError('Finite known-Sun metadata required')
            matches=np.flatnonzero(np.linalg.norm(geometry.sun_vectors-known_sun,axis=1)<1e-6)
            if len(matches)!=1:raise ValueError('Known Sun must have an exact independently prepared reference-map ray column')
            ids=matches
        else:ids=np.arange(len(geometry.sun_vectors))
        if min(reference.std(),query.std())<1:return dict(pose=None,reason='blank_or_uninformative_image',reference_xy=np.empty((0,2)),query_xy=np.empty((0,2)),elapsed_seconds=perf_counter()-tick)
        p,q,appearance,rotation=candidates(self.coarse,reference,query)
        world,use=map_points_z(p,geometry.depth,geometry.intrinsics,geometry.cam2world)
        anchors=np.flatnonzero(use)
        if len(anchors)<6:return dict(pose=None,reason='insufficient_reference_surface',reference_xy=np.empty((0,2)),query_xy=np.empty((0,2)),elapsed_seconds=perf_counter()-tick)
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
            geometry.world_variants[use],geometry.solar_variants[use][:,ids])
        refiner=SparseNativeRefiner(self.fine);query_rotations=[np.rot90(query,r).copy()for r in range(4)]
        # Explicit trial camera calibration is inference metadata, not a truth pose.
        geometry.query_intrinsics=query_intrinsics
        def measure(i,c,trial_pose,sun):
            r=int(rotation[i,c]);proposal=quarter_turn(q[i,c][None],r,512,False)[0]
            measured,meta=refiner.refine(reference,query_rotations[r],p[i],proposal)
            if measured is None:return dict(query_xy=None,reason=meta['reason'])
            measured=quarter_turn(measured[None],r,512,True)[0]
            if not((measured>=0)&(measured<=511)).all():return dict(query_xy=None,reason='refinement_outside_image')
            record=dict(query_xy=measured,observed_shadow=float(cv2.remap(shadow.astype(np.float32),np.float32([[measured[0]]]),np.float32([[measured[1]]]),cv2.INTER_LINEAR)[0,0]))
            if response and trial_pose is not None:
                scores=response_evidence(self.response,reference,query,p[i],measured,geometry,trial_pose,int(anchors[i]),self.pooled)
                if scores is not None:record['response_scores']=scores[ids]
            return record
        result=solve(graph,query_intrinsics,measure,solar_mode,active,policy,budget,response)
        result.update(elapsed_seconds=perf_counter()-tick,total_parameters=self.total_parameters,native_encoder_forwards=refiner.forward_calls,
            observation=observation,input_digest=hashlib.sha256(reference.tobytes()+query.tobytes()).hexdigest(),
            inference_inputs='reference image/depth/pose/normals/solar ray table; query image/intrinsics; optional known-Sun metadata; no query truth',known_sun_supplied=known_sun is not None,
            sun_search_limit='Supplied discrete reference-map catalogue; bounded fine Sun refinement not yet implemented.')
        return result
