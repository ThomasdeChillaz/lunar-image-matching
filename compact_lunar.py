"""Small structural descriptors, learned matchability and shared-scale inference."""
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contrastive_terrain import TerrainDescriptor, sample_descriptors, structural_channels, valid_centers
from dataset import sha256
from matchers import Match, fit_geometry

VARIANTS = ('expanded', 'reliability', 'hard_negatives', 'multiscale')


class CompactDescriptor(TerrainDescriptor):
    def __init__(self, reliability=True):
        super().__init__()
        self.reliability = reliability
        self.matchability = nn.Conv2d(32, 1, 1) if reliability else None
        if sum(p.numel() for p in self.parameters()) > 100000:
            raise ValueError('Compact model exceeds declared parameter budget')

    def forward(self, image):
        features = self.layers(structural_channels(image))
        descriptor = F.normalize(features, dim=1, eps=1e-8)
        logits = self.matchability(features) if self.matchability is not None else torch.zeros_like(features[:, :1])
        return descriptor, logits


def load_checkpoint(directory, device='cpu'):
    directory = Path(directory)
    metadata = json.loads((directory/'training.json').read_text())
    complete = json.loads((directory/'COMPLETE.json').read_text())
    if sha256(directory/'selected.pt') != complete['checkpoint_sha256'] or sha256(directory/'training.json') != complete['training_sha256']:
        raise ValueError('Model or training metadata changed after completion')
    if metadata['source_sha256']['compact_lunar.py'] != sha256(__file__):
        raise ValueError('Model source differs from trained implementation')
    model = CompactDescriptor(metadata['variant'] != 'expanded').to(device)
    model.load_state_dict(torch.load(directory/'selected.pt',map_location=device,weights_only=True),strict=True)
    return model.eval(), metadata


class CompactMatcher:
    def __init__(self, directory, device='cpu'):
        self.model,self.metadata=load_checkpoint(directory,device)
        self.device=device
        self.last_evidence={}
        self.all_points=(np.empty((0,2)),np.empty((0,2)))

    def points(self,a,b,ma=None,mb=None):
        ma=np.ones(a.shape,bool) if ma is None else np.asarray(ma,bool)
        mb=np.ones(b.shape,bool) if mb is None else np.asarray(mb,bool)
        self.last_evidence={}
        if a.ndim!=2 or a.shape!=b.shape or ma.shape!=a.shape or mb.shape!=b.shape:
            raise ValueError('Equal grayscale images and masks required')
        if not ma.any() or not mb.any() or min(a[ma].std(),b[mb].std())<1:
            return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
        multiscale=self.metadata['variant']=='multiscale'
        radius=26 if multiscale else 12
        yy,xx=np.mgrid[radius:a.shape[0]-radius:4,radius:a.shape[1]-radius:4]
        grid=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
        pa,pb=[grid[valid_centers(m,radius)[grid[:,1].astype(int),grid[:,0].astype(int)]] for m in (ma,mb)]
        if min(len(pa),len(pb))<6:
            return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
        inputs=torch.from_numpy(np.stack([np.where(ma,a,0),np.where(mb,b,0)])).to(self.device).float()[:,None]/255
        with torch.inference_mode():
            dense,logits=self.model(inputs)
            if multiscale:
                half,_=self.model(F.interpolate(inputs,scale_factor=.5,mode='bilinear',align_corners=False))
                half=F.interpolate(half,size=a.shape,mode='bilinear',align_corners=False)
                dense=F.normalize(dense+half,dim=1)
            pta,ptb=[torch.from_numpy(p).to(self.device)[None] for p in (pa,pb)]
            da=sample_descriptors(dense[:1],pta)[0]
            db=sample_descriptors(dense[1:],ptb)[0]
            scores=da@db.T
            top,index=scores.max(1)
            second=scores.topk(2,dim=1).values[:,1]
            reverse=scores.argmax(0)
            keep=(reverse[index]==torch.arange(len(pa),device=self.device)) & (top>=.5)
            ia=torch.where(keep)[0]
            ia=ia[torch.argsort(top[ia],descending=True)[:512]]
            ib=index[ia]
            # Scalar logits must be sampled without descriptor unit normalization.
            qa=F.grid_sample(logits[:1].sigmoid(),(2*(pta+.5)/pta.new_tensor(a.shape[::-1])-1)[:,None],align_corners=False)[0,0,0]
            qb=F.grid_sample(logits[1:].sigmoid(),(2*(ptb+.5)/ptb.new_tensor(b.shape[::-1])-1)[:,None],align_corners=False)[0,0,0]
            if self.model.reliability:
                reliable=(qa[ia]>=.25)&(qb[ib]>=.25)
                ia,ib=ia[reliable],ib[reliable]
            aa,bb=pa[ia.cpu().numpy()],pb[ib.cpu().numpy()]
            if multiscale and len(aa):
                offsets=torch.tensor([(x,y) for y in (-1.,0.,1.) for x in (-1.,0.,1.)],device=self.device)
                candidates=torch.from_numpy(bb).to(self.device)[:,None,:]+offsets[None]
                descriptors=sample_descriptors(dense[1:],candidates.reshape(1,-1,2))[0].reshape(len(bb),9,-1)
                sim=(descriptors*da[ia,None]).sum(-1)
                best=sim.argmax(1)
                # Quadratic subpixel offset at an interior local maximum. Edge
                # maxima remain discrete; no extrapolation outside the stencil.
                refined=candidates[torch.arange(len(bb),device=self.device),best].clone()
                middle=best==4
                for axis,left,right in ((0,3,5),(1,1,7)):
                    denom=sim[:,left]-2*sim[:,4]+sim[:,right]
                    delta=(.5*(sim[:,left]-sim[:,right])/denom.clamp(max=-1e-6)).clamp(-.5,.5)
                    refined[middle,axis]+=delta[middle]
                bb=refined.cpu().numpy()
            self.last_evidence=dict(descriptor_margin=float((top[ia]-second[ia]).median()) if len(ia) else 0.,
                matchability=float(torch.minimum(qa[ia],qb[ib]).median()) if len(ia) else 0.,
                potential_matches=len(pa),raw_matches=len(aa))
        return aa.astype(np.float32),bb.astype(np.float32)

    def __call__(self,a,b,ma=None,mb=None):
        aa,bb=self.points(a,b,ma,mb)
        self.all_points=(aa,bb)
        return fit_geometry(aa,bb)
