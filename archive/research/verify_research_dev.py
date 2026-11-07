"""Audit development coverage, native inputs, geometry, failures and summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

from dataset import sha256
from evaluate_research_dev import development_plan
from geometry import evaluation_points
from moon_dataset import load_image
from research_dataset import overlap, validate
from research_geometry import motion


def verify(run, source_ref=None, source_snapshot='reports/development_source_snapshot_v1.json'):
    run = Path(run)
    complete = json.loads((run/'COMPLETE.json').read_text())
    frozen = json.loads((run/'frozen_inputs.json').read_text())
    for filename, field in [('results.json', 'results_sha256'), ('frozen_inputs.json', 'frozen_inputs_sha256')]:
        if sha256(run/filename) != complete[field]:
            raise ValueError('Changed development artifact')
    snapshot = json.loads(Path(source_snapshot).read_text())['sources'] if source_snapshot and Path(source_snapshot).exists() else {}
    for name, digest in frozen['source_sha256'].items():
        content = Path(name).read_bytes() if source_ref is None else subprocess.check_output(['git', 'show', f'{source_ref}:{name}'])
        if source_ref is None and hashlib.sha256(content).hexdigest() != digest and snapshot.get(name, {}).get('sha256') == digest:
            content = snapshot[name]['text'].encode('utf-8')
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'Changed inference source: {name}; select its recorded historical source commit')
    manifest = validate(frozen['manifest'])
    cases = {c['pair_id']: c for c in development_plan(manifest, frozen['tile_limit'])}
    if list(cases.values()) != frozen['cases'] or frozen['split'] != 'dev':
        raise ValueError('Changed development plan')
    rows = json.loads((run/'results.json').read_text())
    expected = {(p, m) for p in cases for m in frozen['methods']}
    actual = [(r['pair_id'], r['method']) for r in rows]
    if set(actual) != expected or len(actual) != len(expected) or complete['rows'] != len(rows):
        raise ValueError('Incomplete/duplicate method coverage')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    images, case_cache = {}, {}
    def image(tile, index):
        key = (tile, index)
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    import cv2
    for row in rows:
        case = cases[row['pair_id']]
        if any(row[key] != value for key, value in case.items()):
            raise ValueError('Changed case identity')
        negative = case['geometry_family'] == 'nonoverlap_negative'
        if case['pair_id'] not in case_cache:
            a, ma = image(case['tile_id'], case['reference_light_index'])
            base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
            if negative:
                first, second = tiles[case['tile_id']], tiles[case['query_tile_id']]
                if first['observation_group'] == second['observation_group'] or overlap(first['bounds_lonlat_deg'], second['bounds_lonlat_deg'], .1):
                    raise ValueError('Negative is not disjoint')
                b, qm, points, targets = base, mb, None, None
            else:
                encoded, parameters = motion(case['pair_id'], case['geometry_family'])
                theta = np.deg2rad(parameters['angle_deg'])
                c, s = np.cos(theta)*parameters['scale'], np.sin(theta)*parameters['scale']
                linear = np.array([[c, s], [-s, c]])
                center = np.array([63.5, 63.5])
                truth = np.column_stack((linear, center-linear@center+[parameters['dx'], parameters['dy']]))
                if not np.allclose(truth, encoded):
                    raise ValueError('Analytic similarity convention differs')
                b = cv2.warpAffine(np.where(mb, base, 0), truth, (128, 128), flags=cv2.INTER_LINEAR)
                qm = cv2.warpAffine(mb.astype(np.float32), truth, (128, 128), flags=cv2.INTER_LINEAR) >= 1.-1e-6
                b[~qm] = 0
                points = evaluation_points(a.shape, truth)
                targets = points@linear.T+truth[:, 2]
                keep = ma[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
                keep &= qm[np.rint(targets[:, 1]).astype(int), np.rint(targets[:, 0]).astype(int)]
                points, targets = points[keep], targets[keep]
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
            case_cache[case['pair_id']] = hashes, points, targets
        hashes, points, targets = case_cache[case['pair_id']]
        if hashes != row['input_array_sha256']:
            raise ValueError('Native input or validity bytes differ')
        if negative:
            if row['correct'] or row['error_px'] is not None or row['rmse_px'] is not None or row['label_source'] != 'known_nonoverlap':
                raise ValueError('Invented negative alignment truth')
        elif row['transform'] is None:
            if row['estimated'] or row['correct'] or row['error_px'] is not None:
                raise ValueError('Failure scored as successful')
        else:
            matrix = np.asarray(row['transform'])
            residual = np.linalg.norm(np.column_stack((points, np.ones(len(points))))@matrix.T-targets, axis=1)
            if not np.isclose(np.median(residual), row['error_px']) or not np.isclose(np.sqrt(np.mean(residual**2)), row['rmse_px']):
                raise ValueError('Incorrect saved geometric errors')
            if row['correct'] != bool(np.median(residual) <= 3) or row['label_source'] != 'exact_geometry' or not row['estimated']:
                raise ValueError('Incorrect positive label')
    for entry in json.loads((run/'summary.json').read_text()):
        subset = [r for r in rows if r['method'] == entry['method'] and r['geometry_family'] == entry['geometry_family']]
        if entry['cases'] != len(subset) or entry['correct'] != sum(r['correct'] for r in subset) or entry['returned'] != sum(r['estimated'] for r in subset):
            raise ValueError('Incorrect family summary')
    return dict(rows=len(rows), cases=len(cases), split='dev', loaded_reserve_images=0,
        source_ref=source_ref, source_snapshot_sha256=sha256(source_snapshot) if source_snapshot and Path(source_snapshot).exists() else None,
        results_sha256=complete['results_sha256'], frozen_inputs_sha256=complete['frozen_inputs_sha256'],
        exact_input_reconstruction=True, analytic_similarity_geometry=True, failure_and_negative_coverage=True,
        warning='Development screening only, not a fresh held-out comparison.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--output', required=True)
    p.add_argument('--source-ref')
    p.add_argument('--source-snapshot', default='reports/development_source_snapshot_v1.json')
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = verify(args.run, args.source_ref, args.source_snapshot)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
