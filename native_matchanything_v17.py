"""Pinned MatchAnything ELoFTR on full native512px content, without a2D fit.

The historical128px integration remains unchanged. This interface exposes
raw mask-valid image correspondences to the common reference-geometry pose
estimator; an affine fit must not prefilter nonplanar3D correspondences.
"""
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from dataset import sha256
from eloftr_matcher import install_dependency_path
from loftr_matcher import valid_correspondences
from matchanything_matcher import LOCK,author_config,lower_config,verify_source


def to_native(points,internal_size,native_size=512):
    return (np.asarray(points,dtype=np.float64).reshape(-1,2)+.5)*native_size/internal_size-.5


class NativeMatchAnything:
    def __init__(self,root='.',dependency_dir=None,profile='native512',threads=2):
        if profile not in ('native512','training_size832') or threads not in (1,2):
            raise ValueError('Declared input profile and one/two CPU threads required')
        install_dependency_path(dependency_dir);lock=json.loads(LOCK.read_text())
        for package,want in lock['optional_dependencies'].items():
            if importlib.metadata.version(package)!=want:raise ValueError('Pinned dependency differs: '+package)
        import torch
        import kornia
        from safetensors.torch import load_file
        if torch.__version__.split('+')[0]!=lock['torch_version'] or kornia.__version__!=lock['kornia_version']:
            raise ValueError('Pinned PyTorch/Kornia runtime required')
        torch.set_num_threads(threads);self.torch=torch
        source=Path(root)/lock['source_directory'];verify_source(source,lock)
        weight=Path(root)/lock['safe_weight_path']
        if sha256(weight)!=lock['safe_weight_sha256']:raise ValueError('Pretrained weights changed')
        namespace='lunar_author_matchanything'
        if namespace not in sys.modules:
            package=(source/'src').resolve()
            spec=importlib.util.spec_from_file_location(namespace,package/'__init__.py',submodule_search_locations=[str(package)])
            module=importlib.util.module_from_spec(spec);sys.modules[namespace]=module;spec.loader.exec_module(module)
        implementation=importlib.import_module(namespace+'.loftr')
        self.internal_size=512 if profile=='native512' else 832
        config=lower_config(author_config(source,namespace,self.internal_size))['loftr']
        self.model=implementation.LoFTR(config)
        self.model.load_state_dict(load_file(str(weight)),strict=True);self.model.eval().requires_grad_(False)
        self.provenance=dict(method='MatchAnything-EfficientLoFTR',profile=profile,native_input_size=512,
            internal_input_size=self.internal_size,npe=[832,832,self.internal_size,self.internal_size],
            source_repository=lock['source_repository'],source_revision=lock['source_revision'],
            source_license=lock['source_license'],source_lock_sha256=sha256(LOCK),weight_sha256=sha256(weight),
            input='Full native512px grayscale content; optional832px bilinear resize preserves field of view.',
            filtering='Unchanged author confidence0.1 and fine matching; finite/native-mask checks only. No2D affine/homography prefilter.',
            pose='Separate common3-native-pixel AP3P with reference geometry; query geometry never enters this class.',
            profile_note='832 is the positional-encoding training-size control, not a claimed author-recommended lunar profile.',
            reparameterization=False,cpu_threads=threads,trainable_parameters=0,
            environment=dict(torch=torch.__version__,kornia=kornia.__version__,numpy=np.__version__,opencv=cv2.__version__))

    def sync(self):
        pass

    def points(self,reference,query,reference_mask=None,query_mask=None):
        if any(im.shape!=(512,512) or im.ndim!=2 or im.dtype!=np.uint8 for im in (reference,query)):
            raise ValueError('Equal native512px uint8 grayscale images required')
        masks=[np.ones(im.shape,bool) if mask is None else np.asarray(mask,bool)
            for im,mask in ((reference,reference_mask),(query,query_mask))]
        if any(mask.shape!=(512,512) for mask in masks):raise ValueError('Native mask shape differs')
        if any(not mask.any() for mask in masks) or min(im[mask].std() for im,mask in zip((reference,query),masks))<1:
            return np.empty((0,2)),np.empty((0,2))
        batch={}
        for i,(image,mask) in enumerate(zip((reference,query),masks)):
            values=np.where(mask,image,0)
            if self.internal_size!=512:values=cv2.resize(values,(self.internal_size,self.internal_size),interpolation=cv2.INTER_LINEAR)
            batch[f'image{i}']=self.torch.from_numpy(np.ascontiguousarray(values)).float()[None,None]/255
        with self.torch.inference_mode():
            self.model(batch)
            p=to_native(batch['mkpts0_f'].cpu().numpy(),self.internal_size)
            q=to_native(batch['mkpts1_f'].cpu().numpy(),self.internal_size)
        return valid_correspondences(p,q,*masks)
