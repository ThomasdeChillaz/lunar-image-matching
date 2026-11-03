"""Lazy native patch refinement, independently encoded after selection.

Integer crops retain the historical radius12support. Canonicalized patch
resampling is a separate explicit operation, not claimed equal to full maps.
"""
from collections import OrderedDict
import hashlib
import numpy as np
import torch
from contrastive_terrain import sample_descriptors
from native_crop_alignment_v57 import integer_crop
from lunar_geometry_v5 import map_points_z


def state_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SparseNativeRefiner:
    def __init__(self,encoder,capacity=128):
        if type(capacity)is not int or capacity<1:raise ValueError('Positive sparse feature cache size required')
        self.encoder=encoder.eval();self.capacity=capacity;self.cache=OrderedDict();self.forward_calls=0

    def encode(self,image):
        a=np.asarray(image)
        if a.ndim!=2 or a.dtype!=np.uint8:raise ValueError('Grayscale uint8 native patch required')
        key=(state_digest(self.encoder),a.shape,hashlib.sha256(a.tobytes()).hexdigest())
        if key not in self.cache:
            with torch.inference_mode():
                result=self.encoder(torch.from_numpy(a.copy()).float()[None,None]/255)
                self.cache[key]=result[0]if isinstance(result,tuple)else result
            self.forward_calls+=1
        self.cache.move_to_end(key)
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return self.cache[key]

    def refine(self,reference,query,point,proposal):
        p,q=np.asarray(point,float),np.asarray(proposal,float)
        if p.shape!=(2,)or q.shape!=(2,)or not np.isfinite([p,q]).all():raise ValueError('Finite paired native xy coordinates required')
        ro=np.floor(p).astype(int)-13;center=np.rint(q).astype(int);qo=center-28
        try:a=integer_crop(reference,ro,27);b=integer_crop(query,qo,57)
        except ValueError:return None,dict(reason='incomplete_native_patch_support')
        da,db=self.encode(a),self.encode(b)
        local=torch.from_numpy((p-ro).astype(np.float32))[None,None]
        vector=sample_descriptors(da,local)[0,0].detach().cpu().numpy()
        window=db[0,:,12:45,12:45].detach().cpu().numpy()
        similarity=np.einsum('c,cyx->yx',vector,window,optimize=False)
        iy,ix=np.unravel_index(np.argmax(similarity),similarity.shape);offset=np.array([float(ix-16),float(iy-16)])
        for axis,at in ((0,ix),(1,iy)):
            if 0<at<32:
                left=similarity[iy,ix-1]if axis==0 else similarity[iy-1,ix]
                right=similarity[iy,ix+1]if axis==0 else similarity[iy+1,ix]
                curvature=left-2*similarity[iy,ix]+right
                if curvature< -1e-8:offset[axis]+=np.clip(.5*(left-right)/curvature,-.5,.5)
        return (center+offset).astype(np.float32),dict(reason=None,peak=float(similarity[iy,ix]),forward_calls=self.forward_calls)


def canonical_patch(reference,query,point,depth,k_reference,reference_pose,k_query,trial_pose,size=33,query_center=None):
    """Project an observed reference surface patch using a TRIAL query pose.

    No query depth/true pose. Output pixel maps explicitly retain native units.
    """
    import cv2
    for image in(reference,query):
        if np.asarray(image).shape!=(512,512)or np.asarray(image).dtype!=np.uint8:raise ValueError('Native512uint8 grayscale canonicalization inputs required')
    if np.asarray(depth).shape!=(512,512):raise ValueError('Native reference axial depth required')
    for value,shape in((k_reference,(3,3)),(k_query,(3,3)),(reference_pose,(4,4)),(trial_pose,(4,4))):
        if np.asarray(value).shape!=shape or not np.isfinite(value).all():raise ValueError('Finite calibrated reference/trial geometry required')
    p=np.asarray(point,float)
    if size!=33 or p.shape!=(2,)or not np.isfinite(p).all():raise ValueError('Native33px canonical patch and finite reference xy required')
    yy,xx=np.mgrid[-16:17,-16:17];source=np.stack([p[0]+xx,p[1]+yy],-1).reshape(-1,2)
    world,valid=map_points_z(source,depth,k_reference,reference_pose)
    target=np.full((len(source),2),np.nan);cam=(world-np.asarray(trial_pose)[:3,3])@np.asarray(trial_pose)[:3,:3]
    with np.errstate(divide='ignore',invalid='ignore'):
        h=cam@np.asarray(k_query).T;pixels=h[:,:2]/h[:,2,None]
    inside=(cam[:,2]>0)&np.isfinite(pixels).all(1)&((pixels>=0)&(pixels<=511)).all(1)
    ids=np.flatnonzero(valid);target[ids[inside]]=pixels[inside]
    if query_center is not None:
        centre=np.asarray(query_center,float)
        if centre.shape!=(2,)or not np.isfinite(centre).all():raise ValueError('Finite measured query centre required')
        if not np.isfinite(target[16*33+16]).all():return None,dict(reason='missing_canonical_centre')
        target+=centre-target[16*33+16]
    supported=np.isfinite(target).all(1)&((source>=0)&(source<=511)).all(1)&((target>=0)&(target<=511)).all(1)
    if not supported.all():return None,dict(reason='incomplete_canonical_surface_or_image_support')
    maps=[v.reshape(33,33,2).astype(np.float32)for v in (source,target)]
    patches=[cv2.remap(im,m[:,:,0],m[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)for im,m in zip((reference,query),maps)]
    return patches,dict(reason=None,reference_xy=maps[0],query_xy=maps[1],mapping='Reference surface coordinates projected to native query pixels under the trial pose.')
