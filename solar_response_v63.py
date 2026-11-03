"""Compact counterfactual feature response; SH approximation, not exact equivariance.

All backbones/heads count toward the100k limit. New actual lunar fitting is
forbidden until this code, inputs and experiment protocol are source-published.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F


def real_spherical_harmonics(directions,degree=5):
    if not isinstance(directions,torch.Tensor)or directions.shape[-1]!=3 or degree not in range(6):raise ValueError('Tensor of Sun three-vectors and degree0..5 required')
    if not directions.is_floating_point()or not torch.isfinite(directions).all():raise ValueError('Finite floating Sun directions required')
    length=directions.norm(dim=-1,keepdim=True)
    if (length<1e-10).any():raise ValueError('Nonzero Sun direction required')
    u=directions/length;z=u[...,2].clamp(-1,1);phi=torch.atan2(u[...,1],u[...,0]);p={}
    p[0,0]=torch.ones_like(z)
    for m in range(1,degree+1):p[m,m]=-(2*m-1)*torch.sqrt((1-z*z).clamp(min=0))*p[m-1,m-1]
    for m in range(degree+1):
        if m<degree:p[m+1,m]=(2*m+1)*z*p[m,m]
        for l in range(m+2,degree+1):p[l,m]=((2*l-1)*z*p[l-1,m]-(l+m-1)*p[l-2,m])/(l-m)
    result=[]
    for l in range(degree+1):
        for m in range(-l,l+1):
            a=abs(m);n=math.sqrt((2*l+1)/(4*math.pi)*math.factorial(l-a)/math.factorial(l+a))
            value=n*p[l,a]
            if m<0:value=value*math.sqrt(2)*torch.sin(a*phi)
            elif m>0:value=value*math.sqrt(2)*torch.cos(a*phi)
            result.append(value)
    return torch.stack(result,-1)


class LightingResponseModel(nn.Module):
    """Native33px canonical patches; condition=[observedSun,normal,view] (3each)."""
    def __init__(self,frozen_descriptor):
        super().__init__()
        if frozen_descriptor.reliability:raise ValueError('Original expanded no-reliability descriptor required')
        self.coarse=deepcopy(frozen_descriptor).eval().requires_grad_(False)
        self.fine=deepcopy(frozen_descriptor).requires_grad_(True)
        self.condition=nn.Sequential(nn.Linear(9,16),nn.SiLU(),nn.Linear(16,64))
        self.coefficients=nn.Linear(32,36*32)
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        nn.init.zeros_(self.coefficients.weight);nn.init.zeros_(self.coefficients.bias)
        with torch.no_grad():self.coefficients.weight[:32].copy_(torch.eye(32)*math.sqrt(4*math.pi))
        self.total_parameters=sum(p.numel()for p in self.parameters())
        if self.total_parameters>=100000:raise ValueError('Combined encoders and response heads exceed100k')

    def train(self,mode=True):
        super().train(mode);self.coarse.eval();return self

    def encode(self,patches,condition):
        if patches.ndim!=4 or patches.shape[1:]!=(1,33,33)or condition.shape!=(len(patches),9):raise ValueError('Bx1x33x33canonical patches and Bx9conditioning required')
        if not torch.isfinite(patches).all()or not torch.isfinite(condition).all():raise ValueError('Finite patches/physical conditioning required')
        if (patches<0).any()or(patches>1).any():raise ValueError('Canonical patch intensities must lie in[0,1]')
        return self.condition_features(self.patch_features(patches),condition)

    def patch_features(self,patches):
        if patches.ndim!=4 or patches.shape[1:]!=(1,33,33) or not torch.isfinite(patches).all() or (patches<0).any() or (patches>1).any():raise ValueError('Finite normalized33px canonical patches required')
        features,_=self.fine(patches)
        return features[:,:,16,16]

    def condition_features(self,base,condition):
        if base.shape!=(len(condition),32)or condition.shape!=(len(base),9)or not torch.isfinite(condition).all():raise ValueError('Paired32Dfeatures and9Dphysical conditioning required')
        gamma,beta=self.condition(condition).chunk(2,-1)
        adjusted=base*(1+.1*gamma.tanh())+.1*beta
        return self.coefficients(adjusted).reshape(-1,36,32)

    def decode(self,coefficients,target_sun,pooled=False):
        if coefficients.ndim!=3 or coefficients.shape[1:]!=(36,32):raise ValueError('Bx36x32response coefficients required')
        if target_sun.shape!=(len(coefficients),3):raise ValueError('One candidate Sun direction per patch required')
        if not torch.isfinite(coefficients).all():raise ValueError('Finite response coefficients required')
        basis=real_spherical_harmonics(target_sun)
        # Exact spherical average: higher harmonics integrate to zero. The
        # pooled control shares architecture and counterfactual supervision.
        if pooled:basis=torch.cat([basis[:,:1],torch.zeros_like(basis[:,1:])],-1)
        return F.normalize(torch.einsum('bn,bnd->bd',basis,coefficients),dim=-1,eps=1e-8)

    def forward(self,patches,condition,target_sun,pooled=False):
        return self.decode(self.encode(patches,condition),target_sun,pooled)


def counterfactual_loss(model,patches,condition,target_suns,teacher_features,available,positive_pairs,pooled=False):
    """Targets at unavailable/heldout lights are rejected, not multiplied by0."""
    if available.dtype!=torch.bool or available.shape!=target_suns.shape[:2]:raise ValueError('Explicit BxLtraining-light availability mask required')
    if teacher_features.shape!=target_suns.shape[:2]+(32,):raise ValueError('BxLx32teacher targets required')
    # Unavailable targets MUST be omitted from the data loader, represented as
    # NaN placeholders. Finite heldout targets reveal forbidden target access.
    if torch.isfinite(teacher_features[~available]).any():raise ValueError('Heldout teacher features must never be decoded or supplied')
    if not available.any():raise ValueError('At least one available training target required')
    coefficients=model.encode(patches,condition);b,l=target_suns.shape[:2]
    expanded=coefficients[:,None].expand(-1,l,-1,-1)[available]
    predicted=model.decode(expanded,target_suns[available],False)
    target=teacher_features[available]
    if not torch.isfinite(target).all():raise ValueError('Available teacher targets must be finite')
    reconstruction=(1-(predicted*F.normalize(target,dim=-1)).sum(-1)).mean()
    # Compare both observations under the SAME counterfactual illumination.
    # Contrasting each at its observed Sun would force invariance and conflict
    # with the response-prediction objective.
    pairs=torch.as_tensor(positive_pairs,device=patches.device,dtype=torch.long)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not len(pairs)or pairs.min()<0 or pairs.max()>=b:raise ValueError('Valid nonempty positive-pair indices required')
    shared=available[pairs[:,0]]&available[pairs[:,1]]
    if not shared.any(-1).all():raise ValueError('Each positive pair needs a common available lighting target')
    light=shared.to(torch.long).argmax(-1)
    sun_a=target_suns[pairs[:,0],light];sun_b=target_suns[pairs[:,1],light]
    if not torch.allclose(sun_a,sun_b,atol=1e-6,rtol=0):raise ValueError('Paired canonical patches must share the same target Sun frame')
    da=model.decode(coefficients[pairs[:,0]],sun_a,pooled)
    db=model.decode(coefficients[pairs[:,1]],sun_b,pooled)
    logits=da@db.T/.1;label=torch.arange(len(pairs),device=patches.device)
    discrimination=.5*(F.cross_entropy(logits,label)+F.cross_entropy(logits.T,label))
    # Penalize unsupported high-frequency oscillation without asserting an
    # exact illumination group action in the presence of cast shadows.
    degrees=torch.tensor([l*(l+1)for l in range(6)for _ in range(2*l+1)],device=patches.device,dtype=patches.dtype)
    regularizer=(coefficients.square()*degrees[None,:,None]).mean()
    total=reconstruction+discrimination+.001*regularizer
    if not torch.isfinite(total):raise ValueError('Nonfinite response objective')
    return total,dict(reconstruction=float(reconstruction.detach()),discrimination=float(discrimination.detach()),harmonic_regularizer=float(regularizer.detach()))
