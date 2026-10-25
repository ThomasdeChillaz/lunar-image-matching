"""Validate and import independently annotated correspondences; never auto-approve."""
import argparse,json
from pathlib import Path
import numpy as np
from dataset import validate_manifest


def apply(manifest,annotations):
    if annotations.get('coordinate_system')!='512px common-grid pixel centers' or not annotations.get('reviewer'):
        raise ValueError('Reviewer and coordinate convention required')
    uncertainty=float(annotations['uncertainty_px'])
    if not np.isfinite(uncertainty) or uncertainty<=0: raise ValueError('Positive uncertainty required')
    pairs={p['pair_id']:p for p in manifest['pairs']}
    for name,points in annotations['pairs'].items():
        if name not in pairs: raise ValueError('Unknown pair')
        arr=np.array([[*p['reference'],*p['query']] for p in points],float)
        if arr.ndim!=2 or arr.shape[0]<10 or arr.shape[1]!=4 or not np.isfinite(arr).all() or (arr<0).any() or (arr>=512).any():
            raise ValueError('Ten finite, in-image independent point pairs required')
        if len(np.unique(arr[:,:2],axis=0))!=len(arr) or len(np.unique(arr[:,2:],axis=0))!=len(arr):
            raise ValueError('Repeated checkpoints')
        if min(np.ptp(arr[:,0]),np.ptp(arr[:,1]))<128: raise ValueError('Points must span the scene')
        pair=pairs[name];pair.update(reference_type='independent_checkpoints',checkpoints=points,
            reference_uncertainty_px=uncertainty,audit_status='independent_annotation_imported',
            reviewer=annotations['reviewer'],reviewed_utc=annotations.get('reviewed_utc'))
    validate_manifest(manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('annotations');p.add_argument('--manifest',default='lunar_manifest.json');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    data=apply(json.loads(Path(a.manifest).read_text()),json.loads(Path(a.annotations).read_text()))
    out.write_text(json.dumps(data,indent=2));print(out)
