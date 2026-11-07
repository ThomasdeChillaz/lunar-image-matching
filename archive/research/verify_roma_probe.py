"""Independently reconstruct the pretrained dense software control locally."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from geometry import endpoint_error, evaluation_points
from synthetic import texture, make_pair


def verify(root):
    root=Path(root)
    terminal=json.loads((root/'COMPLETE.json').read_text())
    if sha256(root/'result.json')!=terminal['result_sha256'] or sha256(root/'frozen_inputs.json')!=terminal['frozen_inputs_sha256']:
        raise ValueError('Dense pretrained control bytes changed')
    result=json.loads((root/'result.json').read_text())
    frozen=json.loads((root/'frozen_inputs.json').read_text())
    for name,digest in frozen['source_sha256'].items():
        if sha256(name)!=digest:
            raise ValueError('Current probe implementation differs from executed source')
    lock=json.loads(Path('roma_model_lock.json').read_text())
    if frozen['model_lock']!=lock or any(frozen['runtime'][p].split('+')[0]!=v for p,v in lock['runtime_versions'].items()):
        raise ValueError('Dense model/runtime differs from declared profile')
    image=texture(42,128)
    query,truth=make_pair(image,dict(angle=12,dx=6,dy=-4,gain=.7,gamma=1.2,noise=.005),43)
    ma=np.ones_like(image,bool)
    mb=cv2.warpAffine(ma.astype(np.float32),truth,(128,128),flags=cv2.INTER_LINEAR)>=1.-1e-6
    actual={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=image,query=query,reference_mask=ma,query_mask=mb).items()}
    error=endpoint_error(result['transform'],truth,evaluation_points(image.shape,truth))
    if actual!=result['input_array_sha256'] or not np.allclose(truth,result['truth'],atol=1e-12) or error is None or not np.isclose(error,result['error_px'],atol=1e-6):
        raise ValueError('Dense control native inputs or independent geometric error differ')
    if error>3 or not result['passed'] or not result['returned'] or result['support']<6 or result['blank_reason']!='low_texture' or result['invalid_reason']!='no_valid_support':
        raise ValueError('Dense pretrained control failed its acceptance checks')
    return dict(error_px=error,exact_native_inputs=True,independent_geometry=True,explicit_failure_controls=True,
        result_sha256=terminal['result_sha256'],frozen_inputs_sha256=terminal['frozen_inputs_sha256'],
        warning='Procedural software proof, not a lunar or fresh geographic accuracy result.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=verify(args.probe)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
