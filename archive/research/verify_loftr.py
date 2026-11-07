"""Reconstruct LoFTR benchmark pixels and errors independently of its runner."""
import argparse
import json
from pathlib import Path
import numpy as np

from dataset import sha256
from evaluate_loftr import array_hashes
from geometry import evaluation_points, transform
from moon_dataset import load_image
from verify_crosslighting import translated_input


def verify(run):
    run = Path(run)
    frozen = json.loads((run/'frozen_inputs.json').read_text())
    complete = json.loads((run/'COMPLETE.json').read_text())
    if sha256(run/'results.json') != complete['results_sha256'] or sha256(run/'frozen_inputs.json') != complete['frozen_inputs_sha256']:
        raise ValueError('Run hashes differ')
    for name, digest in frozen['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError(f'Inference source changed: {name}')
    source = Path(frozen['source_run'])
    if sha256(source/'results.json') != frozen['original_results_sha256'] or sha256(source/'frozen_inputs.json') != frozen['original_inputs_sha256']:
        raise ValueError('Original common benchmark changed')
    rows = json.loads((run/'results.json').read_text())
    expected = {p['pair_id']: p for p in frozen['planned']}
    if len(rows) != len(expected) or len({r['pair_id'] for r in rows}) != len(rows) or complete['rows'] != len(rows):
        raise ValueError('Incomplete or duplicate pair coverage')
    original_rows = {r['pair_id']: r for r in json.loads((source/'results.json').read_text())}
    tiles = {t['tile_id']: t for t in frozen['manifest']['tiles']}
    images = {}
    for row in rows:
        case = expected[row['pair_id']]
        if any(row[key] != value for key, value in case.items()) or row['method'] != 'loftr_outdoor':
            raise ValueError('Pair identity or method differs')
        tile = tiles[case['tile_id']]
        def image(index):
            key = (tile['tile_id'], index)
            if key not in images:
                images[key] = load_image(tile['images'][index])[:2]
            return images[key]
        a, ma = image(case['reference_light_index'])
        base, mb = image(case['query_light_index'])
        # Array slicing reconstructs the fixed shift without the runner's warp code.
        b, qm = translated_input(base, mb)
        hashes = array_hashes(reference=a, query=b, reference_mask=ma, query_mask=qm)
        if hashes != row['input_array_sha256'] or hashes != original_rows[row['pair_id']]['input_array_sha256']:
            raise ValueError('Actual input bytes differ from original benchmark')
        truth = np.array([[1., 0, 6], [0, 1, -4]])
        points = evaluation_points(a.shape, truth)
        targets = points + [6, -4]
        keep = ma[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
        keep &= qm[np.rint(targets[:, 1]).astype(int), np.rint(targets[:, 0]).astype(int)]
        points, targets = points[keep], targets[keep]
        if row['scored_points'] != len(points):
            raise ValueError('Scored support differs')
        if row['transform'] is None:
            if row['estimated'] or row['correct'] or row['error_px'] is not None or row['rmse_px'] is not None:
                raise ValueError('Failure incorrectly scored')
            continue
        h = np.asarray(row['transform'], float)
        predictions = np.column_stack((points, np.ones(len(points)))) @ h.T
        residuals = np.linalg.norm(predictions-targets, axis=1)
        if not np.isclose(np.median(residuals), row['error_px']) or not np.isclose(np.sqrt(np.mean(residuals**2)), row['rmse_px']):
            raise ValueError('Saved error differs from independent recomputation')
        if row['correct'] != bool(np.median(residuals) <= 3) or not row['estimated']:
            raise ValueError('Incorrect success label')
        aa, bb = np.asarray(row['reference_points']), np.asarray(row['query_points'])
        if row['support'] != len(aa) or aa.shape != bb.shape or row['support'] < 6:
            raise ValueError('Invalid match support')
        if not np.isfinite(h).all() or not np.isfinite(aa).all() or not np.isfinite(bb).all():
            raise ValueError('Nonfinite prediction')
        for p, mask in [(aa, ma), (bb, qm)]:
            if (p < 0).any() or (p > 127).any():
                raise ValueError('Out-of-bounds correspondence')
            xy = np.rint(p).astype(int)
            if not mask[xy[:, 1], xy[:, 0]].all():
                raise ValueError('Correspondence in invalid support')
    summary = json.loads((run/'summary.json').read_text())
    for name, count in [('pairs', len(rows)), ('correct', sum(r['correct'] for r in rows)), ('returned', sum(r['estimated'] for r in rows))]:
        if summary[name] != count:
            raise ValueError('Summary differs from exhaustive outcomes')
    return dict(rows=len(rows), pairs=len(expected), results_sha256=sha256(run/'results.json'),
        frozen_inputs_sha256=sha256(run/'frozen_inputs.json'), original_results_sha256=frozen['original_results_sha256'],
        exact_input_identity=True, independent_shift_reconstruction=True, error_and_failure_audit=True,
        warning='Previously inspected terrain; synthetic known-geometry accuracy, not real navigation accuracy.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = verify(args.run)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
