"""Return an alignment only when the frozen deployment gate accepts it.

Inputs are native 128px grayscale uint8 images. The raw model proposal is hidden
by default; --include-proposal exposes it solely as an explicitly ungated record.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from reserve_evaluation import infer,validate_freeze


def read_image(path,mask_path=None):
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or image.shape!=(128,128) or image.dtype!=np.uint8:
        raise ValueError('Provide a native128x128 single-channel uint8 image; no implicit resizing or bit-depth conversion')
    mask=np.ones_like(image,bool)
    if mask_path:
        values=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)
        if values is None or values.shape!=image.shape or values.dtype!=np.uint8:
            raise ValueError('Provide a matching single-channel uint8 validity mask')
        mask=values>0
    return image,mask


def run(reference,query,protocol,output,reference_mask=None,query_mask=None,include_proposal=False):
    import torch
    from contrastive_terrain import ContrastiveTerrainMatcher
    from dataset import sha256
    from synthetic import texture
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    doc=validate_freeze(protocol)
    a,ma=read_image(reference,reference_mask)
    b,mb=read_image(query,query_mask)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model=ContrastiveTerrainMatcher(doc['training'])
    model.warmup(texture(20261030,128))
    prediction=infer(a,b,ma,mb,model,doc['policy'])
    matrix=prediction['deployed'].matrix
    result=dict(accepted=prediction['accepted'],transform=None if matrix is None else matrix.tolist(),
        transform_convention='Reference integer-centered x,y pixels to query x,y pixels.',
        reason=prediction['deployed'].reason,protocol_sha256=sha256(protocol),
        checkpoint_sha256=model.checkpoint_sha256,
        bidirectional_with_gate_ms=prediction['bidirectional_with_gate_ms'],
        warning='Frozen empirical rejection rule, not a correctness or navigation-accuracy guarantee.')
    if include_proposal:
        raw=prediction['raw'].matrix
        result['ungated_research_proposal']=dict(transform=None if raw is None else raw.tolist(),
            evidence=prediction['evidence'],warning='May be incorrect or match disjoint terrain. Never substitute for the accepted public transform.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True)
    p.add_argument('--query',required=True)
    p.add_argument('--reference-mask')
    p.add_argument('--query-mask')
    p.add_argument('--protocol',default='reports/reserve_protocol_v1.json')
    p.add_argument('--output',required=True)
    p.add_argument('--include-proposal',action='store_true')
    args=p.parse_args()
    run(args.reference,args.query,args.protocol,args.output,args.reference_mask,args.query_mask,args.include_proposal)
