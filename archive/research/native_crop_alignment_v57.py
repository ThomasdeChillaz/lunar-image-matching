"""Integer-origin image crops for native-grid descriptor supervision.

Draft diagnostic support, not an adopted model. Geometry/visibility remain
external labels. This never resamples input images or labels shadows occluded.
"""
import numpy as np


def integer_crop(image, origin, size):
    a=np.asarray(image);o=np.asarray(origin)
    if a.ndim!=2 or a.dtype!=np.uint8 or o.shape!=(2,) or not np.issubdtype(o.dtype,np.integer):
        raise ValueError('Native grayscale image and integer x,y origin required')
    if not isinstance(size,int) or size<1:
        raise ValueError('Positive integer crop size required')
    x,y=map(int,o)
    if x<0 or y<0 or x+size>a.shape[1] or y+size>a.shape[0]:
        raise ValueError('Complete native crop support required')
    return a[y:y+size,x:x+size].copy()


def native_example(reference,query,point,proposal,target,geometrically_visible):
    p,q,t=[np.asarray(v,dtype=float)for v in (point,proposal,target)]
    if any(v.shape!=(2,) for v in (p,q,t)) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Finite native reference/proposal coordinates required')
    if geometrically_visible and not np.isfinite(t).all():
        raise ValueError('A visible exact geometric target must be finite')
    # The reference remains fractional AFTER encoding, matching native inference.
    # Radius12 plus the bilinear neighbour fits inside the27px integer crop.
    reference_origin=np.floor(p).astype(int)-13
    center=np.rint(q).astype(int)
    query_origin=center-28
    a=integer_crop(reference,reference_origin,27)
    b=integer_crop(query,query_origin,57)
    local_reference=(p-reference_origin).astype(np.float32)
    if ((np.floor(local_reference)-12<0)|(np.ceil(local_reference)+12>=27)).any():
        raise ValueError('Reference feature lacks full radius12/bilinear support')
    in_window=bool(np.isfinite(t).all()and(np.abs(t-center)<=16).all())
    target_local=(t-center+16).astype(np.float32) if np.isfinite(t).all() else np.zeros(2,np.float32)
    return dict(reference_patch=a,query_patch=b,reference_xy=local_reference,
        target_xy=target_local,geometrically_visible=bool(geometrically_visible),
        target_in_search=in_window,matchable=bool(geometrically_visible and in_window),
        reference_origin=reference_origin,query_origin=query_origin,query_center=center)
