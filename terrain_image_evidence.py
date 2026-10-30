"""Add image-only confidence evidence to completed structural-CNN predictions.

No transform is selected or changed. This standalone experiment intentionally
has no other-predictor agreement signal; that feature is fixed at zero. The
separate fourteen-predictor pool experiment measures agreement/complementarity.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from dataset import sha256
from moon_dataset import load_image
from research_geometry import prepare
from structural_refinement import AlignmentObjective
from verify_research_dev import verify


def evidence(a, b, ma, mb, row):
    quality = AlignmentObjective(a, b, ma, mb).quality(row['transform'])
    reverse = AlignmentObjective(b, a, mb, ma).quality(row['reverse_transform'])
    def values(q):
        return {} if q is None else dict(q, agreement_fraction=0., agreeing_other_families=[])
    return values(quality), values(reverse)


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    # Full native/geometry audit must pass before confidence calibration.
    upstream_audit = verify(source)
    frozen = json.loads((source/'frozen_inputs.json').read_text())
    rows = json.loads((source/'results.json').read_text())
    tiles = {t['tile_id']: t for t in frozen['manifest']['tiles'] if t['split']=='dev'}
    images, inputs, scored = {}, {}, []
    def image(tile, index):
        key = tile, index
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    for row in rows:
        pair = row['pair_id']
        if pair not in inputs:
            a, ma = image(row['tile_id'], row['reference_light_index'])
            base, mb = image(row.get('query_tile_id', row['tile_id']), row['query_light_index'])
            if row['geometry_family'] == 'nonoverlap_negative':
                b, qm = base, mb
            else:
                b, qm, _, _, _, _ = prepare(a, base, ma, mb, pair, row['geometry_family'])
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
            inputs[pair] = (a, b, ma, qm), hashes
        arrays, hashes = inputs[pair]
        if hashes != row['input_array_sha256']:
            raise ValueError('Image evidence inputs differ from original native prediction')
        start = perf_counter()
        q, reverse = evidence(*arrays, row)
        scored.append(dict(row, selection_quality=q, reverse_selection_quality=reverse,
                           image_evidence_ms=(perf_counter()-start)*1000))
    files = ['terrain_image_evidence.py', 'structural_refinement.py', 'structural_matchers.py', 'candidate_selection.py']
    design = dict(frozen, source_sha256={**frozen['source_sha256'], **{f: sha256(f) for f in files}},
        source_run=str(source), source_completion=json.loads((source/'COMPLETE.json').read_text()),
        status='Development-only image evidence on unchanged CNN transforms; predictor agreement unavailable and fixed at zero. No new alignment or reserve evaluation.')
    output.mkdir(parents=True)
    (output/'frozen_inputs.json').write_text(json.dumps(design, sort_keys=True, allow_nan=False))
    (output/'results.json').write_text(json.dumps(scored, indent=2, allow_nan=False))
    (output/'summary.json').write_bytes((source/'summary.json').read_bytes())
    (output/'upstream_audit.json').write_text(json.dumps(upstream_audit, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(scored), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    audit = verify(output)
    # Independent post-serialization replay, preserving every raw prediction.
    saved = json.loads((output/'results.json').read_text())
    for raw, derived in zip(rows, saved):
        if any(derived[k] != v for k, v in raw.items()):
            raise ValueError('Raw predictor outcome changed while adding image evidence')
        q, reverse = evidence(*inputs[raw['pair_id']][0], raw)
        if q != derived['selection_quality'] or reverse != derived['reverse_selection_quality']:
            raise ValueError('Replayed confidence evidence differs')
    audit.update(image_evidence_replayed=len(saved), unchanged_raw_predictions=True,
                 separate_reverse_evidence=True, agreement_fixed_at_zero=True)
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='runs/contrastive-terrain-dev-v1')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(args.source, args.output)
