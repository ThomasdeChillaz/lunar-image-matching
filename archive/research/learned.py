"""Pinned DISK/LightGlue inference with explicit dependencies and masked features."""
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from fetch_models import verify_weights
from matchers import Match, fit_geometry


class LearnedMatcher:
    def __init__(self, cache='data/models', max_keypoints=512, device='cpu'):
        lock=json.loads(Path(__file__).with_name('model_lock.json').read_text())
        verify_weights(cache,lock)
        try:
            import torch
            import lightglue
            from lightglue import DISK,LightGlue
        except ImportError as exc:
            raise RuntimeError('Install requirements-learned.txt before learned matching') from exc
        distribution=importlib.metadata.distribution('lightglue')
        direct=json.loads(distribution.read_text('direct_url.json') or '{}')
        if direct.get('vcs_info',{}).get('commit_id')!=lock['lightglue_revision']:
            raise RuntimeError('LightGlue source revision differs from model_lock.json; reinstall requirements-learned.txt')
        self.torch=torch;self.device=device
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS requested but unavailable')
        # Constructors use torch hub filenames. All bytes are verified BEFORE loading.
        os.environ['TORCH_HOME']=str(Path(cache).resolve());torch.hub.set_dir(str(Path(cache).resolve()/'hub'))
        torch.set_num_threads(1)
        start=perf_counter()
        self.extractor=DISK(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher=LightGlue(features='disk',flash=False,depth_confidence=-1,width_confidence=-1).eval().to(device)
        self.sync();self.initialization_ms=(perf_counter()-start)*1000
        self.warmed=False

    def sync(self):
        if self.device=='cuda':self.torch.cuda.synchronize()
        elif self.device=='mps':self.torch.mps.synchronize()

    def features(self,image,mask):
        tensor=self.torch.from_numpy(np.ascontiguousarray(image)).float()[None]/255
        features=self.extractor.extract(tensor.to(self.device),resize=None)
        if mask is not None:
            points=features['keypoints'][0].round().long()
            x=points[:,0].clamp(0,mask.shape[1]-1);y=points[:,1].clamp(0,mask.shape[0]-1)
            valid=self.torch.from_numpy(mask.astype(bool)).to(self.device)[y,x]
            for key in ['keypoints','keypoint_scores','descriptors']:
                features[key]=features[key][:,valid]
        return features

    def __call__(self,reference,query,reference_mask=None,query_mask=None):
        if reference.std()<1 or query.std()<1:return Match(reason='low_texture')
        with self.torch.inference_mode():
            a=self.features(reference,reference_mask);b=self.features(query,query_mask)
            if min(a['keypoints'].shape[1],b['keypoints'].shape[1])<6:return Match(reason='insufficient_features')
            matches=self.matcher({'image0':a,'image1':b})['matches'][0]
            if len(matches)<6:return Match(reason='insufficient_matches',support=len(matches))
            aa=a['keypoints'][0][matches[:,0]].cpu().numpy()
            bb=b['keypoints'][0][matches[:,1]].cpu().numpy()
        return fit_geometry(aa,bb)

    def warmup(self,image):
        if not self.warmed:
            self(image,image);self.sync();self.warmed=True
