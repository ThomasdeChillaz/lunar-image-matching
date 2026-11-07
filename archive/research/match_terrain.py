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


def run_v2(reference,query,protocol,output,model_dir,reference_mask=None,query_mask=None,
           include_proposal=False,correspondence_output=None):
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    from compact_lunar import CompactMatcher
    from compact_confidence import accepts
    from compact_research import predict
    from dataset import sha256
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if correspondence_output and Path(correspondence_output).exists():raise FileExistsError(correspondence_output)
    doc=json.loads(Path(protocol).read_text())
    for path,digest in {**doc['source_sha256'],**doc['policies']}.items():
        if sha256(path)!=digest:raise ValueError('Frozen compact source/policy changed')
    if model_dir not in doc['models'] or sha256(Path(model_dir)/'selected.pt')!=doc['models'][model_dir]:
        raise ValueError('Model is not bound by this compact protocol')
    policy=json.loads(Path(doc['policy_index'][model_dir]).read_text())
    a,ma=read_image(reference,reference_mask);b,mb=read_image(query,query_mask)
    matcher=CompactMatcher(model_dir)
    from synthetic import texture
    warm=texture(20261030,128);matcher(warm,warm)
    found,reverse,features,points,ms,_=predict(matcher,a,b,ma,mb,'compact')
    accepted=accepts(policy,{'features':features.tolist()})
    result=dict(model_version='compact-v2',accepted=accepted,
        transform=found.matrix.tolist() if accepted else None,
        reason='accepted_by_frozen_confidence' if accepted else 'frozen_confidence_rejected',
        protocol_sha256=sha256(protocol),checkpoint_sha256=doc['models'][model_dir],
        transform_convention='Reference integer-centered x,y pixels to query x,y pixels.',
        bidirectional_with_gate_ms=ms,warning='Empirical confidence, not certified navigation accuracy')
    if include_proposal:
        result['ungated_research_proposal']=dict(transform=None if found.matrix is None else found.matrix.tolist(),
            warning='Rejected raw proposals are not accepted alignments')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if correspondence_output:
        path=Path(correspondence_output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(accepted=accepted,reference_points=np.asarray(points[0]).tolist() if accepted else [],
            query_points=np.asarray(points[1]).tolist() if accepted else [],
            coordinate_convention='Native128px x,y centers; reference to query',
            warning='These are predictions, never independent reference correspondences'),indent=2)+'\n')
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True)
    p.add_argument('--query',required=True)
    p.add_argument('--reference-mask')
    p.add_argument('--query-mask')
    p.add_argument('--protocol',default='reports/reserve_protocol_v1.json')
    p.add_argument('--output',required=True)
    p.add_argument('--include-proposal',action='store_true')
    p.add_argument('--model-version',choices=['v1','compact-v2'],default='v1')
    p.add_argument('--model-dir')
    p.add_argument('--correspondence-output')
    args=p.parse_args()
    if args.model_version=='v1':
        if args.model_dir or args.correspondence_output:p.error('Compact model/correspondence options require --model-version compact-v2')
        run(args.reference,args.query,args.protocol,args.output,args.reference_mask,args.query_mask,args.include_proposal)
    else:
        if not args.model_dir:p.error('--model-dir is required for compact-v2')
        run_v2(args.reference,args.query,args.protocol,args.output,args.model_dir,args.reference_mask,args.query_mask,
               args.include_proposal,args.correspondence_output)
