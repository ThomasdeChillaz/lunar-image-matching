"""Pinned lunar MASt3R descriptor baseline, static architecture and safe tensors.

Prepared integration. Actual pretrained-forward/native control must pass before
any lunar comparison. Model-predicted3D is deliberately excluded from common
reference-depth camera localization. The exact training overlap is unknown.
"""
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from dataset import sha256
from eloftr_matcher import install_dependency_path
from loftr_matcher import valid_correspondences

REVISION='ed6ea257475abb8a5235dff3a4d6167214b6f697'
SAFE_SHA='f1f356da8f400d503f2c675623775e08f736bb24066e75f48301a1aff8cb3d5f'
WEIGHT_REVISION='f339a2c46f4e37ba7ed0ad3b7df51a1cb6aa8664'
LOCK=Path(__file__).with_name('moonst3r_model_lock_v19.json')


def verify_source(source):
    source=Path(source).resolve()
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD']).decode().strip()
    lock=json.loads(LOCK.read_text())
    if actual!=REVISION or actual!=lock['source_revision']:raise ValueError('MOONSt3R source revision differs')
    for name,digest in lock['source_files_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Pinned author source differs: '+name)
    subprocess.run(['git','-C',str(source),'diff','--exit-code',REVISION,'--'],check=True,stdout=subprocess.DEVNULL)
    untracked=subprocess.check_output(['git','-C',str(source),'ls-files','--others','--exclude-standard','-z']).split(b'\0')
    if any(p and Path(p.decode()).suffix in ('.py','.so','.pyd') for p in untracked):raise ValueError('Unpinned executable source in author tree')
    return source


def model_kwargs():
    # These are static reviewed kwargs, not evaluated checkpoint text. The
    # upstream loader makes exactly the same patch-embed/landscape substitution.
    return dict(pos_embed='RoPE100',patch_embed_cls='PatchEmbedDust3R',img_size=(512,512),
        head_type='catmlp+dpt',output_mode='pts3d+desc24',depth_mode=('exp',-float('inf'),float('inf')),
        conf_mode=('exp',1,float('inf')),enc_embed_dim=1024,enc_depth=24,enc_num_heads=16,
        dec_embed_dim=768,dec_depth=12,dec_num_heads=12,two_confs=True,desc_conf_mode=('exp',0,float('inf')),
        freeze='encoder',landscape_only=False)


def native_inputs(reference,query,reference_mask=None,query_mask=None):
    if any(im.ndim!=2 or im.dtype!=np.uint8 or im.shape!=(512,512) for im in (reference,query)):
        raise ValueError('Equal original512px uint8 grayscale inputs required')
    masks=[np.ones((512,512),bool) if m is None else np.asarray(m,bool) for m in (reference_mask,query_mask)]
    if any(m.shape!=(512,512) for m in masks):raise ValueError('Native mask shape differs')
    usable=all(m.any() for m in masks) and min(im[m].std() for im,m in zip((reference,query),masks))>=1
    return masks,usable


class NativeMOONSt3R:
    def __init__(self,source='data/models/moonst3r-upstream',weight='data/models/moonst3r/model.safetensors',dependency_dir=None,threads=2):
        if threads not in (1,2):raise ValueError('One/two CPU threads required')
        install_dependency_path(dependency_dir);source=verify_source(source);lock=json.loads(LOCK.read_text())
        for package,want in lock['dependencies'].items():
            if importlib.metadata.version(package)!=want:raise ValueError('Pinned runtime dependency differs: '+package)
        if sha256(weight)!=SAFE_SHA:raise ValueError('Safe pretrained tensors differ')
        import torch
        from safetensors.torch import load_file
        torch.set_num_threads(threads);self.torch=torch
        path=source/'mast3r'
        for prefix in ('mast3r','dust3r','models'):
            if prefix in sys.modules:
                location=getattr(sys.modules[prefix],'__file__',None)
                if location is None or source not in Path(location).resolve().parents:
                    raise ValueError('Another author implementation is already imported: '+prefix)
        sys.path.insert(0,str(path));implementation=importlib.import_module('mast3r.model')
        if path not in Path(implementation.__file__).resolve().parents:raise ValueError('Unpinned model implementation')
        self.match=importlib.import_module('mast3r.fast_nn').fast_reciprocal_NNs
        self.model=implementation.AsymmetricMASt3R(**model_kwargs())
        state=load_file(str(weight))
        if len(state)!=1017:raise ValueError('Official tensor count differs')
        self.model.load_state_dict(state,strict=True);del state
        self.model.eval().requires_grad_(False)
        self.provenance=dict(method='MOONSt3R',source_repository='clementinegrethen/StereoLunar',source_revision=REVISION,
            weight_repository='cgrethen/MOONSt3R',weight_revision=WEIGHT_REVISION,safe_weight_sha256=SAFE_SHA,
            architecture='Static checkpoint-reviewed kwargs; strict1017-key safe tensor load; no checkpoint eval or author pickle loader.',
            source_lock_sha256=sha256(LOCK),dependencies=lock['dependencies'],
            profile='common_native512_full_field_of_view',native_input_size=512,internal_input_size=512,
            preprocessing='Repeat original grayscale into three channels, normalize[-1,1], preserve square512px FOV. No crop/resize.',
            author_difference='Author load_images default crops square input to512x384; this principal common-content profile explicitly preserves512x512.',
            matching='Author fast_reciprocal_NNs, dot distance, stride8, block8192, max10iterations,3px border. No planar prefilter.',
            reference_geometry='Principal pose uses external common reference geometry/AP3P; predicted3D is unused.',
            query_geometry_in_interface=False,cpu_threads=threads,total_parameters=sum(p.numel() for p in self.model.parameters()),
            trainable_parameters=0,exact_training_pair_overlap='Unknown; checkpoint-declared train/test split files absent from pinned source.',
            license='Top-level Apache2; nested MASt3R/DUSt3R CC BY-NC-SA4. Weight-card license unspecified; no redistribution.',
            environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,
                safetensors=importlib.metadata.version('safetensors')))

    def sync(self):
        pass

    def points(self,reference,query,reference_mask=None,query_mask=None):
        masks,usable=native_inputs(reference,query,reference_mask,query_mask)
        if not usable:return np.empty((0,2)),np.empty((0,2))
        views=[]
        for name,image,mask in zip(('reference','query'),(reference,query),masks):
            values=np.where(mask,image,0)
            tensor=self.torch.from_numpy(np.ascontiguousarray(values)).float()[None,None].repeat(1,3,1,1)/127.5-1
            views.append(dict(img=tensor,true_shape=self.torch.tensor([[512,512]],dtype=self.torch.int32),instance=[name]))
        with self.torch.inference_mode():
            a,b=self.model(*views)
            da,db=[x['desc'][0].detach() for x in (a,b)]
            if da.shape!=(512,512,24) or db.shape!=da.shape:raise ValueError('Author native descriptor output differs')
            p,q=self.match(da,db,subsample_or_initxy1=8,device='cpu',dist='dot',block_size=8192)
        p,q=[np.asarray(x,dtype=np.float64).reshape(-1,2) for x in (p,q)]
        keep=((p>=3)&(p<509)&(q>=3)&(q<509)).all(1)
        return valid_correspondences(p[keep],q[keep],*masks)
