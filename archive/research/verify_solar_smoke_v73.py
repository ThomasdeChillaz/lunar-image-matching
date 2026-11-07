"""Independent arithmetic replay of translated/rotated procedural controls.

No matcher imports, lunar-performance claim, learning claim or timing replay.
Missing poses remain failures rather than being removed from the denominator.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, output, source_root='.'):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    config = summary['configuration']
    angle = np.deg2rad(float(config['rotation_deg']))
    co, si = np.cos(angle), np.sin(angle)
    image_rotation = np.array([[co, si], [-si, co]])
    centre = np.array([255.5, 255.5])
    affine = np.column_stack([image_rotation, centre-image_rotation@centre+[8., -4.]])
    truth = np.eye(4)
    truth[:2, :2] = image_rotation.T
    truth[:3, 3] = -truth[:3, :3]@np.array([2., -1., 0.])
    if not np.allclose(affine, config['affine_reference_to_query'], rtol=0, atol=1e-10):
        raise ValueError('Known procedural affine differs from independent camera construction')
    if not np.allclose(truth, config['scoring_query_cam2world'], rtol=0, atol=1e-10):
        raise ValueError('Known procedural pose differs from independent camera construction')
    rows = []
    for row in summary['rows']:
        name = ''.join(str(int(row[k])) for k in ('solar', 'response', 'active'))
        path = root/(name+'.json')
        saved = json.loads(path.read_text())
        raw = saved['raw_research']
        a = np.asarray(raw['reference_xy'], float).reshape(-1, 2)
        b = np.asarray(raw['query_xy'], float).reshape(-1, 2)
        if len(a) != len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid native correspondence arrays')
        correspondence = None if not len(a) else float(np.median(np.linalg.norm(b-np.column_stack([a, np.ones(len(a))])@affine.T, axis=1)))
        translation = rotation = None
        if raw['pose'] is not None:
            pose = np.asarray(raw['pose'], float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError('Invalid saved pose')
            translation = float(np.linalg.norm(pose[:3, 3]-truth[:3, 3]))
            rotation = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T@truth[:3, :3])-1)/2, -1, 1))))
        for actual, reported in ((translation, row['translation_error_m']), (correspondence, row['median_native_correspondence_error_px'])):
            if (actual is None) != (reported is None) or (actual is not None and not np.isclose(actual, reported, rtol=0, atol=1e-10)):
                raise ValueError('Saved metric differs from independent replay')
        passed = translation is not None and translation < 1 and correspondence is not None and correspondence < 1
        if passed != row['passed'] or saved['pose'] is not None or saved['accepted']:
            raise ValueError('Control outcome or uncalibrated public-null policy differs')
        if len(raw['measurements']) > 32 or raw['proposal_samples_requested'] > 4096:
            raise ValueError('Fixed procedural search budget exceeded')
        if raw.get('pose') is not None and raw.get('final_estimator_iterations') != 2000:
            raise ValueError('Common final estimator budget differs')
        rows.append(dict(toggle=name, passed=passed, translation_error_m=translation,
            rotation_error_deg=rotation, median_native_correspondence_error_px=correspondence,
            saved_result_sha256=sha(path)))
    expected = {''.join(map(str, x)) for x in itertools.product((0, 1), repeat=3)}
    if len(rows) != 8 or {r['toggle'] for r in rows} != expected:
        raise ValueError('Exactly eight distinct module combinations required')
    for path, digest in summary['source_sha256'].items():
        if sha(Path(source_root)/path) != digest:
            raise ValueError('Executed procedural source differs: '+path)
    all_pass = all(r['passed'] for r in rows)
    if bool(summary['passed']) != (all_pass and summary['blank_rejected']):
        raise ValueError('Combined software control outcome differs')
    result = dict(passed=all_pass and summary['blank_rejected'], metric_replay_consistent=True,
        rows=rows, source_sha256=sha(__file__), summary_sha256=sha(root/'summary.json'),
        scope='Independent arithmetic replay of eight PROCEDURAL image-to-pose wiring controls. Missing poses retained. Does not establish lunar accuracy, learned improvement, calibrated risk or replay runtime.')
    with Path(output).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-root', default='.')
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.output, args.source_root), indent=2))
