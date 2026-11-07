"""Untrained geometry/Sun/view feature-distribution transport prototype.

Importing this module does not import Torch. The default architecture has an
ARITHMETIC estimate of95116 parameters; the lazy factory asserts its actual
count when separately qualified. No checkpoint, data loader, training job,
scientific admission, solver integration or novelty claim is provided.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

import numpy as np


FEATURE_DIM = 32
VIEW_DIM = 7
SH_DIM = 36
NOISE_OUTPUTS = 32+32*4
PRIOR_FORMAT = 'training-target-view-prior-v86'
_PRIOR_TOKEN = object()


def parameter_arithmetic(mean_rank=16, noise_rank=3):
    if type(mean_rank) is not int or type(noise_rank) is not int or min(mean_rank,noise_rank) < 1:
        raise ValueError('Positive integer factor ranks required')
    parts = dict(frozen_coarse=28272,reference_geometry_encoder=28272+2*24*3*3,
                 observed_condition_film=(10+1)*16+(16+1)*64,
                 mean_patch_factors=(32+1)*mean_rank*32,
                 mean_angular=36*7*mean_rank,
                 noise_patch_factors=(32+1)*noise_rank*NOISE_OUTPUTS,
                 noise_angular=36*noise_rank)
    parts['total_parameters'] = sum(parts.values())
    parts['trainable_parameters'] = parts['total_parameters']-parts['frozen_coarse']
    # Mean7 + covariance7x7 + PSD factor7x7, not learned, but counted separately.
    parts['fixed_prior_buffer_elements'] = 7+49+49
    parts['parameters_plus_fixed_buffers'] = parts['total_parameters']+105
    if parts['parameters_plus_fixed_buffers'] >= 100000:
        raise ValueError('Declared encoder/head/prior state exceeds100000 elements')
    return parts


def _provenance(provenance):
    if not isinstance(provenance,dict) or provenance.get('split') != 'training':
        raise ValueError('Explicit training-only view-prior provenance required')
    if type(provenance.get('sample_count')) is not int or provenance['sample_count'] < 2:
        raise ValueError('At least two training view samples required')
    allowed = provenance.get('allowed_training_groups')
    fitted = provenance.get('fitted_groups')
    hashes = provenance.get('source_sha256')
    if (not isinstance(allowed,list) or not allowed or any(type(g)is not str or not g for g in allowed)
            or len(set(allowed)) != len(allowed) or not isinstance(fitted,list) or not fitted
            or not set(fitted).issubset(allowed) or not isinstance(hashes,dict) or not hashes
            or any(type(k)is not str or not k or type(v)is not str
                   or re.fullmatch('[0-9a-f]{64}',v)is None for k,v in hashes.items())):
        raise ValueError('Training group inventory and pinned source hashes required')


def _validated_moments(mean,covariance):
    mean = np.asarray(mean,np.float64)
    covariance = np.asarray(covariance,np.float64)
    if mean.shape != (7,) or covariance.shape != (7,7) or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('Finite mean7 and covariance7x7 required')
    if not np.isclose(mean[0],1.,atol=1e-12,rtol=0) or not np.allclose(covariance[0],0.,atol=1e-12,rtol=0) or not np.allclose(covariance[:,0],0.,atol=1e-12,rtol=0):
        raise ValueError('View-basis constant must have mean1 and zero variance')
    if not np.allclose(covariance,covariance.T,atol=1e-12,rtol=0):
        raise ValueError('Prior covariance must be symmetric')
    covariance = .5*(covariance+covariance.T)
    eigenvalues,vectors = np.linalg.eigh(covariance)
    tolerance = 64*np.finfo(np.float64).eps*max(1.,float(np.linalg.norm(covariance,2)))
    if eigenvalues.min() < -tolerance:
        raise ValueError('Prior covariance is not positive semidefinite')
    # Only floating-point negative roundoff is clipped, with its magnitude
    # recorded. The model evaluates JLL^T J^T using this explicit PSD factor.
    factor = vectors*np.sqrt(np.maximum(eigenvalues,0.))[None,:]
    factor[0] = 0.
    psd_covariance = factor@factor.T
    for sl in (slice(1,4),slice(4,7)):
        if not np.isclose(np.trace(psd_covariance[sl,sl])+mean[sl]@mean[sl],1.,atol=5e-6,rtol=0):
            raise ValueError('Prior moments must derive from unit LOS and camera-up directions')
    return mean.copy(),psd_covariance,factor,float(np.linalg.norm(psd_covariance-covariance))


class TrainingViewPrior:
    """Frozen first/second moments of TRAINING target-view basis[1,LOS3,up3].

    Group/hash metadata is mandatory. Loading frozen metadata does not itself
    establish actual split membership; the future dataset/protocol must verify
    the pinned training sources. Query labels never enter marginal inference.
    """
    def __init__(self,mean,covariance,provenance,*,_token=None):
        if _token is not _PRIOR_TOKEN:
            raise TypeError('Construct from training samples or a frozen training payload')
        _provenance(provenance)
        self.mean,self.covariance,self.factor,self.roundoff_adjustment = _validated_moments(mean,covariance)
        self.provenance = deepcopy(provenance)
        for a in (self.mean,self.covariance,self.factor): a.setflags(write=False)
        self._seal = self._digest()
    def _digest(self):
        provenance=json.dumps(self.provenance,sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(self.mean.tobytes()+self.covariance.tobytes()+self.factor.tobytes()+provenance).hexdigest()
    def validate(self):
        _provenance(self.provenance)
        if self._digest() != self._seal:
            raise ValueError('Frozen training-view moments/provenance changed')
        return self
    @classmethod
    def from_training_views(cls,los,camera_up,group_ids,*,allowed_training_groups,source_sha256):
        los = np.asarray(los,np.float64); up = np.asarray(camera_up,np.float64)
        groups = np.asarray(group_ids)
        if (los.ndim != 2 or los.shape[1] != 3 or up.shape != los.shape or len(los)<2
                or groups.shape != (len(los),) or not np.isfinite(los).all() or not np.isfinite(up).all()
                or not np.allclose(np.linalg.norm(los,axis=1),1.,atol=1e-6,rtol=0)
                or not np.allclose(np.linalg.norm(up,axis=1),1.,atol=1e-6,rtol=0)):
            raise ValueError('Paired finite unit training LOS/up vectors and group IDs required')
        allowed = list(allowed_training_groups)
        fitted = sorted(set(groups.tolist()))
        provenance = dict(split='training',sample_count=len(los),allowed_training_groups=allowed,
                          fitted_groups=fitted,source_sha256=dict(source_sha256),weighting='uniform_per_supplied_sample')
        _provenance(provenance)
        basis = np.column_stack([np.ones(len(los)),los,up])
        mean = basis.mean(0); centred = basis-mean
        covariance = centred.T@centred/len(basis)
        provenance['training_view_array_sha256'] = hashlib.sha256(basis.tobytes()).hexdigest()
        return cls(mean,covariance,provenance,_token=_PRIOR_TOKEN)
    def payload(self):
        self.validate()
        return dict(format=PRIOR_FORMAT,mean=self.mean.tolist(),covariance=self.covariance.tolist(),
                    provenance=deepcopy(self.provenance),
                    moment_sha256=hashlib.sha256(self.mean.tobytes()+self.covariance.tobytes()).hexdigest(),
                    numerical_psd_adjustment=self.roundoff_adjustment)
    @classmethod
    def from_frozen_payload(cls,payload):
        if not isinstance(payload,dict) or payload.get('format') != PRIOR_FORMAT:
            raise ValueError('Versioned frozen training-view prior required')
        raw_mean=np.asarray(payload['mean'],np.float64)
        raw_covariance=np.asarray(payload['covariance'],np.float64)
        digest=hashlib.sha256(raw_mean.tobytes()+raw_covariance.tobytes()).hexdigest()
        if payload.get('moment_sha256') != digest:
            raise ValueError('Frozen training-view moment bytes changed')
        prior = cls(payload['mean'],payload['covariance'],payload['provenance'],_token=_PRIOR_TOKEN)
        # Eigenvectors need not be byte-stable on another LAPACK/host. The
        # payload's pinned external hash is authoritative; validate moments.
        return prior


def marginalize_numpy(jacobian,noise_diagonal,noise_low_rank,prior,*,diagonal_only=False):
    """Exact affine view moments; a Gaussian likelihood remains an approximation."""
    prior.validate()
    j = np.asarray(jacobian,np.float64); d = np.asarray(noise_diagonal,np.float64)
    u = np.asarray(noise_low_rank,np.float64)
    if (j.shape[-2:] != (32,7) or d.shape != j.shape[:-1] or u.shape != j.shape[:-1]+(4,)
            or not np.isfinite(j).all() or not np.isfinite(d).all() or not np.isfinite(u).all()
            or (d<=0).any()):
        raise ValueError('Finite J[...,32,7], positive diagonal[...,32], U[...,32,4] required')
    mean = j@prior.mean
    view_factor = j@prior.factor
    diagonal = d+np.sum(u*u,-1)+np.sum(view_factor*view_factor,-1)
    covariance = None
    if not diagonal_only:
        covariance = u@np.swapaxes(u,-1,-2)+view_factor@np.swapaxes(view_factor,-1,-2)
        covariance = covariance+np.eye(32)*d[...,None,:]
    if not np.isfinite(mean).all() or not np.isfinite(diagonal).all():
        raise ValueError('Nonfinite transported moments')
    return dict(mean=mean,diagonal=diagonal,covariance=covariance,view_factor=view_factor,
                covariance_mode='diagonal_marginal_approximation' if diagonal_only else 'full_marginal')


def gaussian_nll_numpy(observed,mean,*,covariance=None,diagonal=None):
    observed = np.asarray(observed,np.float64); mean = np.asarray(mean,np.float64)
    if observed.shape != mean.shape or observed.shape[-1] != 32 or not np.isfinite(observed).all() or not np.isfinite(mean).all():
        raise ValueError('Finite paired32D observed/mean features required')
    if (covariance is None) == (diagonal is None):
        raise ValueError('Choose full covariance or diagonal approximation explicitly')
    residual = observed-mean
    if diagonal is not None:
        diagonal = np.asarray(diagonal,np.float64)
        if diagonal.shape != mean.shape or not np.isfinite(diagonal).all() or (diagonal<=0).any():
            raise ValueError('Finite positive diagonal required')
        return .5*np.sum(residual*residual/diagonal+np.log(diagonal)+math.log(2*math.pi),-1)
    covariance = np.asarray(covariance,np.float64)
    if covariance.shape != mean.shape+(32,) or not np.isfinite(covariance).all() or not np.allclose(covariance,np.swapaxes(covariance,-1,-2),atol=1e-10,rtol=0):
        raise ValueError('Finite symmetric32x32 covariance required')
    # Model construction supplies a strictly positive diagonal floor; invalid
    # external covariances are rejected rather than repaired into confidence.
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol,residual[...,None])[...,0]
    return .5*(np.sum(whitened*whitened,-1)+2*np.log(np.diagonal(chol,axis1=-2,axis2=-1)).sum(-1)+32*math.log(2*math.pi))


def gaussian_nll_torch(observed,mean,*,covariance=None,diagonal=None):
    """Differentiable Cholesky NLL; fails on invalid matrices, no inverse/logdet."""
    import torch
    if observed.shape != mean.shape or mean.shape[-1] != 32 or not torch.isfinite(observed).all() or not torch.isfinite(mean).all():
        raise ValueError('Finite paired32D observed/mean tensors required')
    if observed.dtype!=mean.dtype or observed.device!=mean.device:
        raise ValueError('Observed/mean feature dtype and device must agree')
    if (covariance is None) == (diagonal is None):
        raise ValueError('Choose one covariance representation')
    residual = observed-mean
    if diagonal is not None:
        if diagonal.shape != mean.shape or not torch.isfinite(diagonal).all() or (diagonal<=0).any():
            raise ValueError('Finite positive diagonal required')
        if diagonal.dtype!=mean.dtype or diagonal.device!=mean.device:
            raise ValueError('Diagonal dtype/device must match the features')
        return .5*(residual.square()/diagonal+diagonal.log()+math.log(2*math.pi)).sum(-1)
    if covariance.shape != mean.shape+(32,) or not torch.isfinite(covariance).all():
        raise ValueError('Finite full covariance required')
    if covariance.dtype!=mean.dtype or covariance.device!=mean.device:
        raise ValueError('Covariance dtype/device must match the features')
    scale = max(1.,float(covariance.detach().abs().amax().cpu()))
    if not torch.allclose(covariance,covariance.transpose(-1,-2),atol=1e-6*scale,rtol=0):
        raise ValueError('Symmetric covariance required')
    chol,info = torch.linalg.cholesky_ex(covariance)
    if (info!=0).any():
        raise ValueError('Covariance failed Cholesky; model floor/conditioning must be characterized')
    whitened = torch.linalg.solve_triangular(chol,residual.unsqueeze(-1),upper=False).squeeze(-1)
    return .5*(whitened.square().sum(-1)+2*chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)+32*math.log(2*math.pi))


def build_distributional_response(frozen_descriptor,prior,*,mean_rank=16,noise_rank=3,
                                  variance_floor=1e-4):
    """Lazy actual NN factory. No actual model is built merely by importing.

    Primary coarse path is128 only. Every33px reference/query path requires
    native_requires_selected=True, a call-site paid-C assertion (not an
    independent proof of selection). Future integration must enforce the event.
    """
    expected = parameter_arithmetic(mean_rank,noise_rank)
    if not isinstance(prior,TrainingViewPrior):
        raise ValueError('Explicit training-derived target-view prior required')
    prior.validate()
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError('Finite fixed positive covariance floor required')
    import torch
    from torch import nn
    from torch.nn import functional as F
    from contrastive_terrain import structural_channels,sample_descriptors
    from solar_response_v63 import real_spherical_harmonics
    if (getattr(frozen_descriptor,'reliability',True) or sum(p.numel()for p in frozen_descriptor.parameters())!=28272
            or frozen_descriptor.layers[0].in_channels!=13):
        raise ValueError('Original expanded28272-parameter CompactDescriptor required')
    if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in frozen_descriptor.parameters()):
        raise ValueError('Existing free-CPU float32 descriptor required')

    class GeometryReferenceEncoder(nn.Module):
        def __init__(self,source):
            super().__init__();self.layers=deepcopy(source.layers)
            old=self.layers[0]
            new=nn.Conv2d(15,old.out_channels,old.kernel_size,stride=old.stride,padding=old.padding,
                          dilation=old.dilation,bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
            with torch.no_grad():
                new.weight.zero_();new.weight[:,:13].copy_(old.weight)
                if old.bias is not None:new.bias.copy_(old.bias)
            self.layers[0]=new
        def forward(self,field,*,native_requires_selected=False):
            if field.ndim!=4 or field.shape[1]!=3 or field.shape[-2:]not in((128,128),(33,33)):
                raise ValueError('Reference field is intensity/nx/ny at128 or selected33px')
            if field.shape[-2:]==(33,33)and native_requires_selected is not True:
                raise ValueError('Native reference patches require a paid prior C selection')
            if field.dtype!=self.layers[0].weight.dtype or field.device!=self.layers[0].weight.device:
                raise ValueError('Reference field must match the encoder dtype/device')
            if (not torch.isfinite(field).all() or(field[:,:1]<0).any()or(field[:,:1]>1).any()
                    or(field[:,1:].abs()>1+1e-6).any()or(field[:,1:].square().sum(1)>1+1e-5).any()):
                raise ValueError('Finite normalized intensity and reference normal components required')
            structural=structural_channels(field[:,:1])
            return F.normalize(self.layers(torch.cat([structural,field[:,1:]],1)),dim=1,eps=1e-8)

    class DistributionalResponse(nn.Module):
        def __init__(self):
            super().__init__()
            self.coarse=deepcopy(frozen_descriptor).eval().requires_grad_(False)
            self.reference_fine=GeometryReferenceEncoder(frozen_descriptor).requires_grad_(True)
            self.observed_film=nn.Sequential(nn.Linear(10,16,dtype=torch.float32),nn.SiLU(),nn.Linear(16,64,dtype=torch.float32))
            self.mean_factors=nn.Linear(32,mean_rank*32,dtype=torch.float32)
            self.mean_angular=nn.Parameter(torch.empty(36,7,mean_rank,dtype=torch.float32))
            self.noise_factors=nn.Linear(32,noise_rank*NOISE_OUTPUTS,dtype=torch.float32)
            self.noise_angular=nn.Parameter(torch.empty(36,noise_rank,dtype=torch.float32))
            self.register_buffer('view_mean',torch.tensor(prior.mean.copy(),dtype=torch.float32))
            factor=torch.tensor(prior.factor.copy(),dtype=torch.float32)
            self.register_buffer('view_covariance',factor@factor.T)
            self.register_buffer('view_factor',factor)
            self.prior_provenance=deepcopy(prior.provenance);self.variance_floor=float(variance_floor)
            self.mean_rank=mean_rank;self.noise_rank=noise_rank
            nn.init.zeros_(self.observed_film[-1].weight);nn.init.zeros_(self.observed_film[-1].bias)
            nn.init.normal_(self.mean_factors.weight,std=.01);nn.init.zeros_(self.mean_factors.bias)
            with torch.no_grad():self.mean_factors.weight[:32].copy_(torch.eye(32))
            # Small nonzero view terms avoid a permanently dead marginalized
            # view variance when the training prior has zero directional mean.
            nn.init.normal_(self.mean_angular,std=.01)
            with torch.no_grad():self.mean_angular[0,0,0].add_(math.sqrt(4*math.pi))
            nn.init.normal_(self.noise_factors.weight,std=.01);nn.init.zeros_(self.noise_factors.bias)
            nn.init.normal_(self.noise_angular,std=.01)
            with torch.no_grad():
                self.noise_angular[0,0].add_(math.sqrt(4*math.pi))
                self.noise_factors.bias[:32].fill_(math.log(math.expm1(.05)))
            actual=sum(p.numel()for p in self.parameters())
            trainable=sum(p.numel()for p in self.parameters()if p.requires_grad)
            state=actual+sum(b.numel()for b in self.buffers())
            if (actual!=expected['total_parameters']or trainable!=expected['trainable_parameters']
                    or state>=100000):
                raise ValueError('Actual encoder/head/state count violates declared architecture/cap')
            self.parameter_report=dict(expected,actual_parameters=actual,actual_trainable_parameters=trainable,actual_state_elements=state)
        def train(self,mode=True):
            super().train(mode);self.coarse.eval();return self
        def coarse_reference_features(self,field,points):
            if field.shape[-2:]!=(128,128):raise ValueError('Pre-C reference encoding is128px only')
            dense=self.reference_fine(field)
            return sample_descriptors(dense,points)
        def native_reference_features(self,patches,*,native_requires_selected):
            if patches.shape[-2:]!=(33,33):raise ValueError('Selected native patch size is33')
            return self.reference_fine(patches,native_requires_selected=native_requires_selected)[:,:,16,16]
        def coarse_query_features(self,image,points):
            if image.ndim!=4 or image.shape[1:]!=(1,128,128)or not torch.isfinite(image).all()or(image<0).any()or(image>1).any():
                raise ValueError('Image-only normalized128px query field required')
            if image.dtype!=self.coarse.layers[0].weight.dtype or image.device!=self.coarse.layers[0].weight.device:raise ValueError('Query image dtype/device mismatch')
            with torch.no_grad():dense,_=self.coarse(image)
            return sample_descriptors(dense,points)
        def native_query_features(self,patches,*,native_requires_selected):
            if native_requires_selected is not True or patches.ndim!=4 or patches.shape[1:]!=(1,33,33):
                raise ValueError('Native query33px encoder requires paid prior C selection')
            if not torch.isfinite(patches).all()or(patches<0).any()or(patches>1).any():raise ValueError('Normalized selected query patches required')
            if patches.dtype!=self.coarse.layers[0].weight.dtype or patches.device!=self.coarse.layers[0].weight.device:raise ValueError('Query patch dtype/device mismatch')
            with torch.no_grad():dense,_=self.coarse(patches)
            return dense[:,:,16,16]
        def distribution(self,base,observed_condition,trial_sun,*,target_view_basis=None,diagonal_only=False):
            if (base.ndim!=2 or base.shape[1]!=32 or observed_condition.shape!=(len(base),10)
                    or trial_sun.shape!=(len(base),3)or not torch.isfinite(base).all()
                    or not torch.isfinite(observed_condition).all()or not torch.isfinite(trial_sun).all()):
                raise ValueError('Finite base32, KNOWN reference condition10 and trial-Sun3 tensors required')
            if any(x.dtype!=self.mean_factors.weight.dtype or x.device!=self.mean_factors.weight.device
                   for x in(base,observed_condition,trial_sun)):
                raise ValueError('Feature/condition/Sun tensors must match the model dtype/device')
            for sl in(slice(0,3),slice(3,6),slice(6,9)):
                if not torch.allclose(observed_condition[:,sl].norm(dim=-1),torch.ones(len(base),device=base.device,dtype=base.dtype),atol=1e-5,rtol=0):
                    raise ValueError('Observed Sun/normal/reference-view conditioning must be unit')
            if not torch.allclose(trial_sun.norm(dim=-1),torch.ones(len(base),device=base.device,dtype=base.dtype),atol=1e-5,rtol=0):raise ValueError('Trial Sun directions must be unit')
            gamma,beta=self.observed_film(observed_condition).chunk(2,-1)
            z=base*(1+.1*gamma.tanh())+.1*beta
            sh=real_spherical_harmonics(trial_sun)
            patch=self.mean_factors(z).reshape(-1,mean_rank,32)
            angular=torch.einsum('bk,kvr->bvr',sh,self.mean_angular)
            j=torch.einsum('bvr,brd->bdv',angular,patch)
            noise=self.noise_factors(z).reshape(-1,noise_rank,NOISE_OUTPUTS)
            weights=sh@self.noise_angular
            noise=torch.einsum('br,bro->bo',weights,noise)
            d=F.softplus(noise[:,:32])+self.variance_floor
            u=noise[:,32:].reshape(-1,32,4)
            if target_view_basis is None:
                mean=j@self.view_mean
                vf=j@self.view_factor
                view_mode='training_prior_marginal'
            else:
                if target_view_basis.shape!=(len(base),7)or not torch.isfinite(target_view_basis).all()or not torch.allclose(target_view_basis[:,0],torch.ones(len(base),device=base.device,dtype=base.dtype),atol=1e-6,rtol=0):raise ValueError('Conditioned view basis must be[1,unitLOS,unitUp]')
                for sl in(slice(1,4),slice(4,7)):
                    if not torch.allclose(target_view_basis[:,sl].norm(dim=-1),torch.ones(len(base),device=base.device,dtype=base.dtype),atol=1e-5,rtol=0):raise ValueError('Conditioned LOS/up must be unit')
                mean=torch.einsum('bdv,bv->bd',j,target_view_basis)
                vf=j.new_zeros((len(base),32,7));view_mode='explicit_training_or_estimated_view'
            diagonal=d+u.square().sum(-1)+vf.square().sum(-1)
            covariance=None if diagonal_only else torch.diag_embed(d)+u@u.transpose(-1,-2)+vf@vf.transpose(-1,-2)
            if not torch.isfinite(mean).all()or not torch.isfinite(diagonal).all():raise ValueError('Nonfinite feature distribution')
            return dict(mean=mean,diagonal=diagonal,covariance=covariance,noise_diagonal=d,
                        noise_low_rank=u,view_jacobian=j,view_factor=vf,view_mode=view_mode,
                        covariance_mode='diagonal_marginal_approximation'if diagonal_only else'full')
        def marginal_distribution(self,base,observed_condition,trial_sun,*,diagonal_only=False):
            return self.distribution(base,observed_condition,trial_sun,diagonal_only=diagonal_only)

    return DistributionalResponse()
