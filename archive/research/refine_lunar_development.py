"""Run a trained development refiner; uncalibrated research proposals are opt-in."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256


def load_candidate(directory, checkpoint='selected'):
    import torch
    from localization_refiner_v3 import LocalRefiner
    directory = Path(directory)
    complete = json.loads((directory/'COMPLETE.json').read_text())
    metadata = json.loads((directory/'training.json').read_text())
    if checkpoint not in ('selected','final'):
        raise ValueError('Explicit selected or final checkpoint required')
    weight_path=directory/(checkpoint+'.pt')
    digest=complete['checkpoint_sha256'] if checkpoint=='selected' else complete['final_sha256']
    if sha256(directory/'training.json') != complete['training_sha256'] or sha256(weight_path) != digest:
        raise ValueError('Completed candidate bytes differ')
    if sha256('localization_refiner_v3.py') != metadata['source_sha256']['localization_refiner_v3.py']:
        raise ValueError('Refiner implementation differs from its training')
    if metadata['final_test_enabled'] or metadata['calibration_or_test_pixels_loaded'] != 0:
        raise ValueError('Expected a development-only candidate')
    model = LocalRefiner(metadata['variant'])
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
    if sum(p.numel() for p in model.parameters())+28272 != metadata['total_parameters']:
        raise ValueError('Parameter budget differs')
    return model.eval(), metadata


def run(reference, query, directory, output, include_proposals=False, checkpoint='selected'):
    import torch
    from compact_lunar import CompactMatcher
    from localization_diagnostics_v3 import native_points
    from localization_refiner_v3 import refine_dense
    from lunargeo_dataset import write_new
    a = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(query), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != b.shape or min(a.shape) < 96:
        raise ValueError('Readable equal-size grayscale images of at least96px required')
    model, metadata = load_candidate(directory,checkpoint)
    manifest = json.loads(Path(metadata['manifest_path']).read_text())
    if sha256(metadata['manifest_path']) != metadata['manifest_sha256']:
        raise ValueError('Candidate provenance manifest differs')
    torch.set_num_threads(2); cv2.setNumThreads(1)
    result = dict(model_version='compact-v3-development', variant=metadata['variant'],
                  checkpoint=checkpoint,checkpoint_sha256=sha256(Path(directory)/(checkpoint+'.pt')),
                  transform=None, pose=None, accepted=False,
                  reason='development_candidate_has_no_calibrated_pair_acceptance_policy',
                  coordinate_convention='Reference pixel centers to query pixel centers; original image pixels.',
                  reference_points=[], query_points=[], point_scores=[],
                  proposals_included=include_proposals)
    if include_proposals:
        coarse = CompactMatcher(manifest['coarse_model'])
        if sha256(Path(manifest['coarse_model'])/'selected.pt') != manifest['coarse_checkpoint_sha256']:
            raise ValueError('Frozen coarse checkpoint differs')
        p, q = coarse.points(*[cv2.resize(v,(128,128),interpolation=cv2.INTER_AREA) for v in (a,b)])
        p, q = native_points(p,a.shape), native_points(q,b.shape)
        with torch.inference_mode():
            da, db = [model.encoder(torch.from_numpy(v.copy()).float()[None,None]/255) for v in (a,b)]
            p, q, confidence, _ = refine_dense(model,p,q,da,db)
        if model.variant == 'reliability':
            use = confidence >= .5; p,q,confidence = p[use],q[use],confidence[use]
        result.update(reference_points=p.tolist(),query_points=q.tolist(),point_scores=confidence.tolist())
    write_new(output,result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True); p.add_argument('--query',required=True)
    p.add_argument('--model-dir',required=True); p.add_argument('--output',required=True)
    p.add_argument('--include-proposals',action='store_true')
    p.add_argument('--checkpoint',choices=['selected','final'],default='selected')
    a=p.parse_args(); print(json.dumps(run(a.reference,a.query,a.model_dir,a.output,a.include_proposals,a.checkpoint),indent=2))
