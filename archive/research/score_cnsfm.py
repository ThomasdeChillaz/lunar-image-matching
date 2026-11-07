"""Audit every bidirectional Windows shard before common-protocol scoring.

Missing jobs and execution errors stop scoring rather than disappearing from
the denominator. Crater predictions never become independent references.
"""
import argparse
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from cnsfm_outputs import load_prediction
from confidence import diagnostics, independent_correctness
from dataset import common_points, load_pair, sha256, validate_manifest
from evaluate import endpoint_statistics
from evaluate_extended import illumination_metrics, circular_difference
from landmark_review import fingerprint


def audit_shards(root, manifest):
    expected = {(p['reference'], p['query']):p for p in manifest['pairs']}
    executions, plans = {}, []
    for plan_path in sorted(Path(root).rglob('shard_plan.json')):
        folder = plan_path.parent
        plan = json.loads(plan_path.read_text())
        complete = json.loads((folder/'COMPLETE.json').read_text())
        outcomes = json.loads((folder/'outcomes.json').read_text())
        if not plan['bidirectional'] or complete['pairs'] != len(plan['planned_pairs']) or complete['executions'] != 2*len(plan['planned_pairs']):
            raise ValueError('Incomplete bidirectional CNSFM shard')
        allowed = {(p['index'], p['reference'], p['query'], direction)
            for p in plan['planned_pairs'] for direction in ['forward', 'reverse']}
        seen = set()
        for outcome in outcomes:
            key = tuple(outcome[k] for k in ['index', 'reference', 'query', 'direction'])
            if key not in allowed or key in seen or outcome['status'] != 'executed':
                raise ValueError('Missing, duplicated or failed CNSFM execution')
            seen.add(key)
            _, first, second, direction = key
            pair = expected.get((first, second)) or expected.get((second, first))
            if pair is None or pair['region_id'] != plan['region']:
                raise ValueError('Unplanned CNSFM pair')
            ordered = (first, second) if direction == 'forward' else (second, first)
            identity = (pair['pair_id'], ordered == (pair['reference'], pair['query']))
            if identity in executions:
                raise ValueError('Repeated pair direction across CNSFM shards')
            execution = folder/f'pair_{outcome["index"]:03}_{direction}'
            provenance = json.loads((execution/'provenance.json').read_text())
            if tuple(provenance['products']) != ordered:
                raise ValueError('CNSFM ordered input provenance differs from plan')
            executions[identity] = execution
        if seen != allowed:
            raise ValueError('CNSFM shard did not execute its entire plan')
        plans.append(dict(path=plan_path.relative_to(root).as_posix(), sha256=sha256(plan_path),
            outcomes_sha256=sha256(folder/'outcomes.json'), complete_sha256=sha256(folder/'COMPLETE.json')))
    if set(executions) != set(itertools.product([p['pair_id'] for p in manifest['pairs']], [True, False])):
        raise ValueError('CNSFM shards do not cover all frozen pairs in both directions')
    return executions, plans


def score(root, manifest, predictions, dataset, output):
    assets = validate_manifest(manifest, '.')
    executions, plans = audit_shards(root, manifest)
    prediction_assets = {a['product_id']:a for a in predictions['assets']}
    if predictions['source_manifest_sha256'] != fingerprint(manifest):
        raise ValueError('CNSFM predictions do not belong to this frozen manifest')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    rows = []
    inputs = {}
    for pair in manifest['pairs']:
        a, b, ma, mb, grid = load_pair(pair, assets, '.')
        fits = []
        for direction in [True, False]:
            execution = executions[pair['pair_id'], direction]
            provenance = json.loads((execution/'provenance.json').read_text())
            common = provenance.get('common_grid_provenance', {})
            if (common.get('source_manifest_sha256') != fingerprint(manifest)
                    or common.get('predictions_sha256') != fingerprint(predictions)):
                raise ValueError('CNSFM common-grid preparation provenance mismatch')
            for item in common['assets']:
                original = prediction_assets[item['product_id']]
                if any(item[key] != original[key] for key in ['image_bytes_sha256', 'mask_bytes_sha256']):
                    raise ValueError('CNSFM pixels differ from frozen common-grid detector inputs')
            cv2.setRNGSeed(20261026)
            fit, products = load_prediction(execution, Path(dataset)/pair['region_id'], grid,
                (ma, mb) if direction else (mb, ma), 'common_512_redetected')
            expected = [pair['reference'], pair['query']]
            if products != (expected if direction else expected[::-1]):
                raise ValueError('CNSFM input direction mismatch')
            fits.append(fit)
            inputs[execution.relative_to(root).as_posix()] = sha256(execution/'provenance.json')
        if pair['reference_type'] == 'independent_checkpoints':
            points = np.asarray([p['reference'] for p in pair['checkpoints']])
            targets = np.asarray([p['query'] for p in pair['checkpoints']])
            label = 'independent_checkpoints' if pair.get('reference_validation_status') == 'reviewed' else None
            kind = 'independent_checkpoint_error'
        else:
            points = common_points(ma, mb)
            targets = points.copy()
            kind, label = 'map_reference_disagreement', None
        forward, reverse = fits
        error, rmse = endpoint_statistics(forward.matrix, points, targets)
        first, second = assets[pair['reference']], assets[pair['query']]
        row = dict(pair_id=pair['pair_id'], region_id=pair['region_id'], split=pair['split'],
            reference=pair['reference'], query=pair['query'], method='cnsfm', preprocessing='none',
            variant='common_512_redetected', dataset_kind='real_repeat_observation', error_kind=kind,
            label_source=label, correct=independent_correctness(error, pair.get('reference_uncertainty_px'), 3) if label else None,
            estimated=forward.matrix is not None, error_px=error, rmse_px=rmse, scored_points=len(points),
            reference_uncertainty_px=pair.get('reference_uncertainty_px'), reason=forward.reason,
            support=forward.support, support_type='geometric_inliers',
            transform=None if forward.matrix is None else forward.matrix.tolist(),
            reverse_transform=None if reverse.matrix is None else reverse.matrix.tolist(),
            reference_points=None if forward.reference_points is None else forward.reference_points.tolist(),
            query_points=None if forward.query_points is None else forward.query_points.tolist(),
            **diagnostics(forward, reverse, a.shape), **illumination_metrics(a, b, ma, mb),
            solar_azimuth_difference_deg=circular_difference(first['solar_azimuth_deg'], second['solar_azimuth_deg']),
            reference_solar_elevation_deg=first['solar_elevation_deg'], query_solar_elevation_deg=second['solar_elevation_deg'])
        rows.append(row)
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'audit.json').write_text(json.dumps(dict(manifest_sha256=fingerprint(manifest),
        predictions_sha256=fingerprint(predictions), shard_plans=plans, execution_provenance_sha256=inputs,
        scoring_source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ['score_cnsfm.py', 'cnsfm_outputs.py', 'confidence.py', 'matchers.py', 'dataset.py', 'evaluate.py']},
        comparison='Same 512px pixels/masks, no CLAHE, one-to-one predictions, partial-affine RANSAC threshold 3px and minimum six inliers. Original engine with whole-image redetection; not author-native paper reproduction.',
        warning='Provisional map disagreement is not independently verified accuracy.'), indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), executions=len(executions),
        results_sha256=sha256(output/'results.json'), audit_sha256=sha256(output/'audit.json'))))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executions', required=True)
    p.add_argument('--manifest', default='expanded_solar_manifest.json')
    p.add_argument('--predictions', default='cnsfm_detector_outputs.json')
    p.add_argument('--dataset', default='data/cnsfm/common512/Dataset')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rows = score(Path(args.executions), json.loads(Path(args.manifest).read_text()),
        json.loads(Path(args.predictions).read_text()), args.dataset, args.output)
    print('Audited and scored', len(rows), 'CNSFM pairs')
