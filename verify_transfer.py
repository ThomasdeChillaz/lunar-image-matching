"""Recompute real-transfer diagnostics from saved fits and exact input bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from confidence import accepts, diagnostics
from dataset import common_points, load_pair as load_cnsfm, validate_manifest, sha256
from evaluate import endpoint_statistics
from landmark_review import fingerprint
from matchers import Match
from tycho_dataset import load_pair as load_tycho, validate as validate_tycho
from verify_extension import source_check, same


def verify(folder):
    folder = Path(folder)
    frozen = json.loads((folder/'frozen_inputs.json').read_text())
    complete = json.loads((folder/'COMPLETE.json').read_text())
    if (sha256(folder/'results.json') != complete['results_sha256']
            or sha256(folder/'frozen_inputs.json') != complete['frozen_inputs_sha256']):
        raise ValueError('Transfer completion hash mismatch')
    source_check(frozen['source_sha256'])
    physical = json.loads(Path('moon_manifest.json').read_text())
    if frozen['physical_manifest_sha256'] != fingerprint(physical) or frozen['adapter']['manifest_sha256'] != fingerprint(physical):
        raise ValueError('Transfer model/physical provenance mismatch')
    manifest = frozen['manifest']
    assets = validate_tycho(manifest, physical, '.') if frozen['dataset'] == 'tycho' else validate_manifest(manifest, '.')
    rows = json.loads((folder/'results.json').read_text())
    expected = {(p['pair_id'], method) for p in manifest['pairs'] for method in frozen['methods']}
    found = [(r['pair_id'], r['method']) for r in rows]
    if set(found) != expected or len(found) != len(expected) or complete['rows'] != len(rows):
        raise ValueError('Incomplete/duplicated transfer measurements')
    index = {p['pair_id']:p for p in manifest['pairs']}
    cache = {}
    for row in rows:
        pair = index[row['pair_id']]
        if row['split'] != pair['split'] or row['region_id'] != pair['region_id']:
            raise ValueError('Transfer split drift')
        if any(row[k] is not None for k in ['correct', 'label_source', 'reference_uncertainty_px']):
            raise ValueError('Unknown real alignment became an independent correctness label')
        if pair['pair_id'] not in cache:
            loaded = load_tycho(pair, assets) if frozen['dataset'] == 'tycho' else load_cnsfm(pair, assets, '.')
            a, b, ma, mb = loaded[:4]
            hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=mb).items()}
            size = a.shape[0]
            cache[pair['pair_id']] = (hashes, common_points(ma, mb, spacing=size//16, margin=size//16), a.shape)
        hashes, points, shape = cache[pair['pair_id']]
        if hashes != row['input_array_sha256']:
            raise ValueError('Real transfer pixel/mask bytes changed')
        error, rmse = endpoint_statistics(row['transform'], points, points)
        same(row['error_px'], error, 'transfer map discrepancy')
        same(row['rmse_px'], rmse, 'transfer map RMSE')
        forward = Match(matrix=row['transform'], reference_points=None if row['reference_points'] is None else np.asarray(row['reference_points']),
            query_points=None if row['query_points'] is None else np.asarray(row['query_points']))
        reverse = Match(matrix=row['reverse_transform'])
        for key, value in diagnostics(forward, reverse, shape).items():
            same(row[key], value, key)
        if 'accepted_by_transported_physical_policy' in row:
            policy = frozen['physical_development_policies'][row['method']]['policy']
            if row['accepted_by_transported_physical_policy'] != accepts(row, policy):
                raise ValueError('Transported physical policy changed')
    return dict(evaluations=len(rows), pairs=len(manifest['pairs']), dataset=frozen['dataset'],
        manifest_sha256=fingerprint(manifest), results_sha256=complete['results_sha256'],
        source_files_verified=len(frozen['source_sha256']),
        scope='Saved-fit geometry, exact input bytes, source splits and frozen policy transport. Independent real accuracy is still unknown.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    result = verify(args.run)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
