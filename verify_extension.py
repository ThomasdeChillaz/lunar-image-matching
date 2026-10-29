"""Audit saved extension measurements; this does not certify scientific completion."""
import argparse
import json
from pathlib import Path
import numpy as np
from confidence import calibrate, diagnostics
from dataset import load_pair, common_points, validate_manifest, sha256
from evaluate import endpoint_statistics
from evaluate_moon import cases, summarize
from landmark_review import fingerprint
from matchers import Match


def source_check(hashes):
    for name, expected in hashes.items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError('Measurement source changed: '+name)


def same(actual, expected, label):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError('Null/scored mismatch: '+label)
    elif not np.isclose(actual, expected, rtol=1e-8, atol=1e-8):
        raise ValueError('Recomputed value differs: '+label)


def verify_extended(folder):
    folder = Path(folder)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    if sha256(folder/'results.json') != complete['results_sha256']:
        raise ValueError('Extended result digest mismatch')
    frozen = json.loads((folder/'frozen_inputs.json').read_text())
    source_check(frozen['source_sha256'])
    manifest, config = frozen['manifest'], frozen['config']
    assets = validate_manifest(manifest, '.')
    rows = json.loads((folder/'results.json').read_text())
    expected = {(p['pair_id'], m, prep) for p in manifest['pairs'] for m in config['methods'] for prep in config['preprocessing']}
    found = [(r['pair_id'], r['method'], r['preprocessing']) for r in rows]
    if set(found) != expected or len(found) != len(expected) or complete['rows'] != len(rows):
        raise ValueError('Missing/duplicate extended evaluations')
    index = {p['pair_id']:p for p in manifest['pairs']}
    point_cache = {}
    for row in rows:
        pair = index[row['pair_id']]
        if row['split'] != pair['split'] or row['region_id'] != pair['region_id']:
            raise ValueError('Result split/region drift')
        if row['pair_id'] not in point_cache:
            _, _, ma, mb, _ = load_pair(pair, assets, '.')
            point_cache[row['pair_id']] = common_points(ma, mb)
        if row['error_kind'] == 'map_reference_disagreement':
            if row['correct'] is not None or row['label_source'] is not None:
                raise ValueError('Provisional map reference became an independent label')
            points = point_cache[row['pair_id']]
            error, rmse = endpoint_statistics(row['transform'], points, points)
            same(row['error_px'], error, 'map discrepancy')
            same(row['rmse_px'], rmse, 'map RMSE')
        forward = Match(matrix=row['transform'], reference_points=None if row['reference_points'] is None else np.asarray(row['reference_points']),
            query_points=None if row['query_points'] is None else np.asarray(row['query_points']))
        reverse = Match(matrix=row['reverse_transform'])
        values = diagnostics(forward, reverse, (512, 512))
        for name, value in values.items():
            same(row[name], value, name)
    return dict(evaluations=len(rows), pairs=len(manifest['pairs']), source_files_verified=len(frozen['source_sha256']),
        manifest_sha256=fingerprint(manifest), results_sha256=complete['results_sha256'],
        reference_status='Provisional map diagnostics; no independent real accuracy established')


def verify_physical(folder):
    folder = Path(folder)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    if sha256(folder/'results.json') != complete['result_sha256'] or sha256(folder/'frozen_policies.json') != complete['policies_sha256']:
        raise ValueError('Physical result/policy digest mismatch')
    config = json.loads((folder/'frozen_config.json').read_text())
    source_check(config['source_sha256'])
    manifest = json.loads((folder/'frozen_manifest.json').read_text())
    adapter = json.loads((folder/'frozen_adapter.json').read_text())
    if adapter['manifest_sha256'] != fingerprint(manifest) or adapter['training_source_sha256'] != sha256('train_lunar_adapter.py'):
        raise ValueError('Adapter source/manifest mismatch')
    selected = min(adapter['history'], key=lambda x:x['development_loss'])
    if adapter['selected_step'] != selected['step']:
        raise ValueError('Checkpoint not selected by frozen development criterion')
    rows = json.loads((folder/'results.json').read_text())
    lookups = {}
    for split in ['dev', 'test']:
        for case in cases(manifest, split):
            lookups[case['pair_id']] = case
    expected = {(name, method) for name in lookups for method in config['methods']}
    found = [(r['pair_id'], r['method']) for r in rows]
    if set(found) != expected or len(found) != len(expected) or complete['rows'] != len(rows):
        raise ValueError('Missing/duplicate physical evaluations')
    for row in rows:
        case = lookups[row['pair_id']]
        if row['split'] != case['split']:
            raise ValueError('Physical result split drift')
        if case['truth'] is not None:
            error, rmse = endpoint_statistics(row['transform'], case['points'], case['targets'])
            correct = bool(error is not None and error <= config['threshold_px'])
        else:
            error, rmse, correct = None, None, False
        same(row['error_px'], error, 'physical endpoint error')
        same(row['rmse_px'], rmse, 'physical RMSE')
        if row['correct'] != correct:
            raise ValueError('Incorrect physical correctness label')
    policies = json.loads((folder/'frozen_policies.json').read_text())
    for method, policy in policies.items():
        if 'policy' in policy:
            repeated = calibrate([r for r in rows if r['split'] == 'dev' and r['method'] == method], 0.)
            if repeated != policy:
                raise ValueError('Policy does not reproduce from development-only outcomes')
    summary = json.loads((folder/'summary.json').read_text())
    if summary['registration'] != summarize(rows):
        raise ValueError('Physical summary mismatch')
    return dict(evaluations=len(rows), cases=len(lookups), selected_adapter_step=adapter['selected_step'],
        results_sha256=complete['result_sha256'], policies_sha256=complete['policies_sha256'],
        reference_status='Exact rendered/known-warp supervision; real-image transfer remains unverified')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extended-run', required=True)
    p.add_argument('--physical-run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    result = dict(extended=verify_extended(args.extended_run), physical=verify_physical(args.physical_run),
        scope='Saved-measurement, geometry, split and development-selection audit only. Human reference review, CNSFM common-grid comparison and real adaptation transfer are separate open requirements.')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
