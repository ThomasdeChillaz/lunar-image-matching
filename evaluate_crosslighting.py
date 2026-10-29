"""Evaluate every Sun pair with one fixed geometry and unchanged learned models.

Follow-up on already inspected terrain: freeze the complete all-pairs design
before new outcomes, keep all failures, and transport the original dev policies
without tuning. New Sun combinations are not a fresh geographic holdout.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from confidence import accepts, diagnostics, quality_counts
from dataset import sha256
from evaluate import endpoint_statistics
from evaluate_extended import CachedLearnedMatcher
from evaluate_moon import AdaptedMatcher
from geometry import evaluation_points, transform
from landmark_review import fingerprint
from matchers import ncc, preprocess, sift
from moon_dataset import load_image, validate
from physical_lighting_analysis import solar_separation, shared_support
from synthetic import make_pair

METHODS = ['ncc', 'sift', 'disk_lightglue', 'disk_lightglue_clahe', 'disk_lightglue_lunar']
WARP = dict(angle=0, dx=6, dy=-4, gain=1, gamma=1, noise=0)


def plan(manifest, split):
    return [dict(pair_id=tile['tile_id']+f'_sun_{i}_{j}', tile_id=tile['tile_id'], split=split,
        reference_light_index=i, query_light_index=j)
        for tile in manifest['tiles'] if tile['split'] == split
        for i,j in itertools.combinations(range(len(tile['images'])), 2)]


def prepare_pair(reference, query, reference_mask, query_mask):
    b, truth = make_pair(query, WARP, 20261026)
    mask = cv2.warpAffine(query_mask.astype(np.uint8), truth, (128,128), flags=cv2.INTER_NEAREST).astype(bool)
    b[~mask] = 0
    points = evaluation_points(reference.shape, truth)
    targets = transform(points, truth)
    keep = reference_mask[np.rint(points[:,1]).astype(int), np.rint(points[:,0]).astype(int)]
    keep &= mask[np.rint(targets[:,1]).astype(int), np.rint(targets[:,0]).astype(int)]
    if not keep.any():
        raise ValueError('No valid known-geometry checkpoints; cannot assign correctness')
    return b, mask, truth, points[keep], targets[keep]


def run(physical_run, output, split='test', resume=False):
    physical_run, output = Path(physical_run), Path(output)
    completed_source = json.loads((physical_run/'COMPLETE.json').read_text())
    if (sha256(physical_run/'frozen_policies.json') != completed_source['policies_sha256']
            or sha256(physical_run/'results.json') != completed_source['result_sha256']):
        raise ValueError('Original physical experiment hashes differ')
    manifest = json.loads((physical_run/'frozen_manifest.json').read_text())
    adapter = json.loads((physical_run/'frozen_adapter.json').read_text())
    policies = json.loads((physical_run/'frozen_policies.json').read_text())
    validate(manifest, '.')
    if adapter['manifest_sha256'] != fingerprint(manifest) or adapter['model_lock_sha256'] != sha256('model_lock.json'):
        raise ValueError('Frozen model provenance mismatch')
    for name, policy in policies.items():
        if 'policy' in policy and policy['selected_on_split'] != 'dev':
            raise ValueError('Only previously frozen development policies may be transported')
    planned = plan(manifest, split)
    sources = ['evaluate_crosslighting.py','evaluate_extended.py','evaluate_moon.py','moon_dataset.py',
        'physical_lighting_analysis.py','matchers.py','confidence.py','geometry.py','synthetic.py','learned.py','fetch_models.py','dataset.py','evaluate.py']
    frozen = dict(manifest=manifest, adapter=adapter, original_physical_policies=policies,
        physical_source_result_sha256=completed_source['result_sha256'], split=split, methods=METHODS,
        planned=planned, warp=WARP, seed=20261026, threshold_px=3, image_size=128,
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in sources},
        status='Exploratory follow-up on previously inspected terrain; all Sun combinations and fixed warp frozen before new outcomes. No retraining or threshold selection.',
        negative_cases='Original known non-overlap tests remain separately reported; this run contains same-terrain positives only.')
    canonical = json.dumps(frozen, sort_keys=True, allow_nan=False)
    if output.exists():
        if not resume or (output/'frozen_inputs.json').read_text() != canonical:
            raise ValueError('Resume requires identical complete design, models, policies and source')
    else:
        output.mkdir(parents=True)
        (output/'frozen_inputs.json').write_text(canonical)
        (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    rows_path = output/'results.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    identities = [(r['pair_id'],r['method']) for r in rows]
    expected = {(p['pair_id'],m) for p in planned for m in METHODS}
    completed = set(identities)
    if len(identities) != len(completed) or not completed <= expected:
        raise ValueError('Existing partial results contain duplicate or unplanned outcomes')
    cv2.setNumThreads(1)
    cached = CachedLearnedMatcher()
    adapted = AdaptedMatcher(cached, adapter)
    tiles = {t['tile_id']:t for t in manifest['tiles']}
    images = {}
    for number, case in enumerate(planned, 1):
        if all((case['pair_id'],m) in completed for m in METHODS):
            continue
        tile = tiles[case['tile_id']]
        def image(index):
            key = (tile['tile_id'],index)
            if key not in images:
                images[key] = load_image(tile['images'][index])
            return images[key]
        first, second = tile['images'][case['reference_light_index']], tile['images'][case['query_light_index']]
        a,ma,raw_a = image(case['reference_light_index'])
        base,mb,raw_b = image(case['query_light_index'])
        b,qm,truth,points,targets = prepare_pair(a,base,ma,mb)
        light = dict(solar_separation_deg=solar_separation(first,second),
            reference_solar_elevation_deg=first['elevation_deg'], query_solar_elevation_deg=second['elevation_deg'],
            shared_nondark_fraction=shared_support(raw_a,raw_b,ma,mb,.01),
            reference_zero_render=first['constant_zero_render'], query_zero_render=second['constant_zero_render'])
        hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=qm).items()}
        for method in METHODS:
            if (case['pair_id'],method) in completed:
                continue
            aa,bb = preprocess(a,method.endswith('_clahe')).copy(),preprocess(b,method.endswith('_clahe')).copy()
            aa[~ma],bb[~qm] = 0,0
            def match(x,y,mx,my):
                cv2.setRNGSeed(20261026)
                if method == 'ncc':
                    return ncc(x,y,mx,my)
                if method == 'sift':
                    return sift(x,y,mx,my)
                if method == 'disk_lightglue_lunar':
                    return adapted(x,y,mx,my)
                return cached(x,y,mx,my)[0]
            forward,reverse = match(aa,bb,ma,qm),match(bb,aa,qm,ma)
            error,rmse = endpoint_statistics(forward.matrix,points,targets)
            row = dict(**case,**light,method=method,observation_group=tile['observation_group'],
                dataset_kind='physical_render_fixed_translation',label_source='exact_geometry',
                estimated=forward.matrix is not None,correct=bool(error is not None and error <= 3),error_px=error,rmse_px=rmse,
                reason=forward.reason,support=forward.support,scored_points=len(points),input_array_sha256=hashes,
                transform=None if forward.matrix is None else forward.matrix.tolist(),
                reverse_transform=None if reverse.matrix is None else reverse.matrix.tolist(),
                reference_points=None if forward.reference_points is None else forward.reference_points.tolist(),
                query_points=None if forward.query_points is None else forward.query_points.tolist(),
                **diagnostics(forward,reverse,a.shape))
            row['accepted_by_original_development_policy'] = accepts(row,policies[method]['policy']) if 'policy' in policies.get(method,{}) else None
            with rows_path.open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
            rows.append(row);completed.add((case['pair_id'],method))
        print(number,'/',len(planned),case['pair_id'],flush=True)
    summaries = []
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        error = [r['error_px'] for r in selected if r['error_px'] is not None]
        summary = dict(method=method,pairs=len(selected),returned=sum(r['estimated'] for r in selected),correct=sum(r['correct'] for r in selected),
            median_error_returned_px=float(np.median(error)) if error else None,
            dark_input_pairs=sum(r['reference_zero_render'] or r['query_zero_render'] for r in selected))
        if 'policy' in policies.get(method,{}):
            summary['transported_rejection'] = quality_counts(selected,policies[method]['policy'])
        else:
            summary['policy_unavailable'] = 'No original physical development policy for this method; no test-based threshold fitted.'
        summaries.append(summary)
    (output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),pairs=len(planned),
        results_sha256=sha256(output/'results.json'),frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physical-run',default='runs/physical-adaptation-v1')
    p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['dev','test'],default='test')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    run(args.physical_run,args.output,args.split,args.resume)
