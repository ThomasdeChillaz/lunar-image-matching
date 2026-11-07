"""Replay new northern benchmark metrics/common pose fits without a network."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rigid_pose(value):
    pose = np.asarray(value, float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0., 0., 0., 1.], rtol=0, atol=1e-12)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., rtol=0, atol=1e-6)):
        raise ValueError('Finite rigid camera-to-world matrix required')
    return pose


def control_replay(checks):
    """Recompute saved analytical control evidence; never load a network."""
    from native_baseline_pose_v7 import propose_pose
    if set(checks) != {'pretrained_or_sift', 'compact'} or not all(c['passed'] for c in checks.values()):
        raise ValueError('Both actual control records are required')
    c = checks['pretrained_or_sift']
    p, q = [np.asarray(c[k], float).reshape(-1, 2)
            for k in ('reference_points_native', 'query_points_native')]
    if (p.shape != q.shape or len(p) < 6 or not all(np.isfinite(v).all() for v in (p, q))
            or any(((v < 0) | (v > 511)).any() for v in (p, q))):
        raise ValueError('Invalid native control points')
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    pose, info = propose_pose(p, q, np.full((512, 512), 10.), k, np.eye(4), k)
    saved = rigid_pose(c['estimated_cam2world'])
    if pose is None or not np.allclose(pose, saved, rtol=0, atol=1e-7):
        raise ValueError('Native control pose does not replay')
    error = float(np.median(np.linalg.norm(q - p - [8., -4.], axis=1)))
    translation = float(np.linalg.norm(saved[:3, 3] - [-.2, .1, 0.]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(saved[:3, :3]) - 1) / 2, -1, 1))))
    if (c['image_seed'] != 20261101 or c['known_translation_native'] != [8., -4.]
            or c['matches'] != len(p) or c['inliers'] != info['inliers']
            or not c['blank_empty'] or not c['invalid_mask_empty']
            or not np.isfinite(c['warm_match_seconds']) or c['warm_match_seconds'] < 0
            or error > 1 or translation > .05 or rotation > .5):
        raise ValueError('Native control identity/criterion differs')
    for key, actual in [('median_native_error_px', error), ('translation_error_m', translation),
                        ('rotation_error_deg', rotation)]:
        if not np.isclose(c[key], actual, rtol=0, atol=1e-9):
            raise ValueError('Saved native control error differs')
    c = checks['compact']; pose = rigid_pose(c['estimated_cam2world'])
    expected = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    rotation = float(np.linalg.norm(pose[:3, :3] - expected)); translation = float(np.linalg.norm(pose[:3, 3]))
    if (c['image_seed'] != 733 or not c['blank_pose_null'] or rotation >= .01 or translation >= .05
            or not np.isclose(c['rotation_matrix_error'], rotation, rtol=0, atol=1e-12)
            or not np.isclose(c['translation_error_m'], translation, rtol=0, atol=1e-12)):
        raise ValueError('Compact known-rotation control error differs')
    return dict(saved_native_control_pose_and_errors_replayed=True,
        saved_compact_known_rotation_errors_replayed=True,
        blank_and_invalid_model_forwards_repeated=False)


def audit(run, output):
    import cv2
    from benchmark_north_control_v38 import case_list, inference_inputs, score_case, totals
    from north_training_bundle_v38 import load_bundle
    from native_baseline_pose_v7 import propose_pose
    root = Path(run); out = Path(output)
    if out.exists(): raise FileExistsError('Preserve every independent replay')
    summary, protocol = read(root / 'summary.json'), read(root / 'protocol.json')
    expected_files = read(root / 'COMPLETE.json')['sha256']
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual_files != set(expected_files) | {'COMPLETE.json'}:
        raise ValueError('Sealed comparison file inventory differs')
    for name, digest in expected_files.items():
        if sha(root / name) != digest: raise ValueError('Saved comparison changed')
    if (not summary['complete'] or summary['protocol_sha256'] != sha(root / 'protocol.json')
            or summary['results_sha256'] != sha(root / 'per_case.json.gz')):
        raise ValueError('Complete comparison seal required')
    if protocol['manifest_sha256'] != sha('reports/north_training_data_v38.json'):
        raise ValueError('Benchmark input manifest changed')
    if protocol['software_check_sha256'] != sha('reports/north_modern_software_checks_v49.json'):
        raise ValueError('Comparison software-check receipt changed')
    for name, digest in protocol['upstream_check_sha256'].items():
        if sha(name) != digest or not read(name)['passed']:
            raise ValueError('Upstream independent check differs')
    for name, digest in protocol['source_sha256'].items():
        if (sha(name) != digest or subprocess.check_output([
                'git', 'show', protocol['publication_commit'] + ':' + name]) != Path(name).read_bytes()):
            raise ValueError('Executed/public source differs')
    checks = read(root / 'controls.json')
    replayed_controls = control_replay(checks)
    cv2.setNumThreads(1); bundle = load_bundle('reports/north_training_data_v38.json', selection_only=True)
    cases = case_list(bundle[0], protocol['full'])
    if (protocol['case_ids'] != [c['pair_id'] for c in cases]
            or summary['eligible_pairs_per_method'] != len(cases)):
        raise ValueError('Complete declared case denominator differs')
    with gzip.open(root / 'per_case.json.gz', 'rt') as stream: rows = json.load(stream)
    method = protocol['method']; fits = 0
    for name in (method, 'frozen_surface'):
        part = [r for r in rows if r['method'] == name]
        if [r['pair_id'] for r in part] != [c['pair_id'] for c in cases]:
            raise ValueError('Missing/reordered/duplicate method cases')
        for row, case in zip(part, cases):
            if any(row[k] != value for k, value in case.items()): raise ValueError('Case metadata changed')
            key = hashlib.sha256((case['pair_id'] + '/' + name).encode()).hexdigest() + '.json'
            if read(root / 'cases' / key) != row: raise ValueError('Case cache/result differs')
            p, q = [np.asarray(row[k], float).reshape(-1, 2)
                    for k in ('reference_points_native', 'query_points_native')]
            pose = None if row['estimated_cam2world'] is None else rigid_pose(row['estimated_cam2world'])
            if (p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all()
                    or any(((v < 0) | (v > 511)).any() for v in (p, q))):
                raise ValueError('Invalid native correspondence record')
            inputs = inference_inputs(bundle, case)
            replay_pose, info = propose_pose(p, q, *inputs[2:])
            if ((pose is None) != (replay_pose is None)
                    or (pose is not None and not np.allclose(pose, replay_pose, rtol=0, atol=1e-7))):
                raise ValueError('Common pose-estimator replay differs')
            # Failure records deliberately keep their explicit inference-error
            # reason instead of replacing it with the estimator's empty-input reason.
            if row['inference_error'] is None and row['pose_info'] != info:
                raise ValueError('Common pose-estimator evidence differs')
            actual = score_case(bundle, case, p, q, pose)
            if any(row[k] != value for k, value in actual.items()):
                raise ValueError('Native correspondence/pose metric replay differs')
            fits += 1
        if totals(part) != summary['results'][name]: raise ValueError('Aggregate denominator/metric differs')
    if len(rows) != 2 * len(cases): raise ValueError('Unexpected method rows')
    result = dict(passed=True, replayed_rows=len(rows), replayed_common_pose_fits=fits,
        summary_sha256=sha(root / 'summary.json'), protocol_sha256=sha(root / 'protocol.json'),
        source_sha256=sha(__file__), neural_forwards=0, optimizer_updates=0,
        replayed_controls=replayed_controls,
        calibration_or_final_pixels_loaded=0, actual_pretrained_forward_replayed=False,
        scope='Saved native metrics, common pose fits, fixed cases and paired totals independently replayed. No fresh-generalization, accepted-risk or SOTA claim.')
    out.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True)
    p.add_argument('--output', required=True); a = p.parse_args(); audit(a.run, a.output)
