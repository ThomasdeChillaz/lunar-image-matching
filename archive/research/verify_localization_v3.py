"""Independent replay of development diagnostics and full camera inventory."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def replay(run, gate_path, output):
    cv2.setNumThreads(1)
    run = Path(run)
    protocol = json.loads((run/'protocol.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    complete = json.loads((run/'COMPLETE.json').read_text())
    for name, digest in [('protocol.json', complete['protocol_sha256']),
                         ('summary.json', complete['summary_sha256']),
                         ('results.json.gz', complete['results_sha256'])]:
        if sha256(run/name) != digest:
            raise ValueError('Saved artifact changed: '+name)
    for name, digest in protocol['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Diagnostic implementation changed: '+name)
    if sha256(protocol['manifest_path']) != protocol['manifest_sha256']:
        raise ValueError('Development manifest changed')
    manifest = json.loads(Path(protocol['manifest_path']).read_text())
    tiles = {t['tile_id']: t for t in manifest['tiles']}
    with gzip.open(run/'results.json.gz', 'rt') as f:
        rows = json.load(f)
    expected = set()
    for tile_id in protocol['tiles']:
        expected.update((tile_id, name, None, None) for name in protocol['controls'])
        expected.update((tile_id, name, i, j) for name in protocol['matcher_experiments']
                        for i in range(3) for j in range(3))
    seen = set(); counts = {}; depths = {}
    for row in rows:
        key = (row['tile_id'], row['experiment'], row.get('light_a'), row.get('light_b'))
        if key in seen:
            raise ValueError('Repeated case')
        seen.add(key)
        tile = tiles[row['tile_id']]
        if tile['split'] != 'model_selection' or row['split'] != tile['split']:
            raise ValueError('Diagnostics accessed a forbidden split')
        if row['component'] != tile['component'] or row['mode'] != tile['mode']:
            raise ValueError('Case metadata differs')
        if tile['tile_id'] not in depths:
            if sha256(tile['cache']) != tile['cache_sha256']:
                raise ValueError('Cached depth changed')
            with np.load(tile['cache'], allow_pickle=False) as z:
                depths[tile['tile_id']] = z['depth_a'].copy(), z['depth_b'].copy()
        da, db = depths[tile['tile_id']]
        p, q = [np.array(row[k], float).reshape(-1, 2)
                for k in ('reference_points_native', 'query_points_native')]
        if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError('Invalid saved points')
        ka, ca = [np.array(tile['cameras'][0][k]) for k in ('intrinsics', 'cam2world')]
        kb, cb = [np.array(tile['cameras'][1][k]) for k in ('intrinsics', 'cam2world')]
        index = np.rint(p).astype(int)
        valid = ((index >= 0) & (index < np.array(da.shape[::-1]))).all(1)
        safe = index.clip(0, np.array(da.shape[::-1])-1)
        ranges = da[safe[:, 1], safe[:, 0]]
        valid &= np.isfinite(ranges) & (ranges > 0)
        # Inverting a3x3 matrix avoids an oversubscribed small-RHS BLAS solve.
        # This algebra remains independent of the experiment's world_points().
        rays = np.column_stack([p, np.ones(len(p))])@np.linalg.inv(ka).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        world = (rays*np.where(valid, ranges, 1)[:, None])@ca[:3, :3].T+ca[:3, 3]
        query = (world-cb[:3, 3])@cb[:3, :3]
        h = query@kb.T
        with np.errstate(divide='ignore', invalid='ignore'):
            truth = h[:, :2]/h[:, 2, None]
        valid &= (query[:, 2] > 0) & np.isfinite(truth).all(1)
        valid &= ((truth >= 0) & (truth < np.array(db.shape[::-1])-1)).all(1)
        if len(truth):
            safe_truth = np.where(np.isfinite(truth), truth, -1).astype(np.float32)
            sample = cv2.remap(np.where(np.isfinite(db), db, 0).astype(np.float32),
                safe_truth[:, 0, None], safe_truth[:, 1, None], cv2.INTER_LINEAR).ravel()
            support = cv2.remap((np.isfinite(db) & (db > 0)).astype(np.float32),
                safe_truth[:, 0, None], safe_truth[:, 1, None], cv2.INTER_LINEAR).ravel()
            distance = np.linalg.norm(query, axis=1)
            valid &= (support >= .9999) & (np.abs(sample-distance) <= .05+.01*distance)
        if valid.tolist() != row['visible']:
            raise ValueError('Independent visibility differs')
        target = np.array(row['target_points_native'], float).reshape(-1, 2)
        if not np.allclose(truth[valid], target, atol=1e-5):
            raise ValueError('Independent projection differs')
        errors = np.linalg.norm(q[valid]-truth[valid], axis=1)
        correct = {str(t): int((errors <= t).sum()) for t in (1, 2, 4, 8, 12)}
        median = None if not len(errors) else float(np.median(errors))
        if correct != row['correct_at_native_px'] or row['returned'] != len(p) or row['scorable'] != int(valid.sum()):
            raise ValueError('Correspondence counts differ')
        if (median is None) != (row['median_native_px'] is None) or (median is not None and not np.isclose(median, row['median_native_px'])):
            raise ValueError('Correspondence error differs')
        pose = row['estimated_cam2world']
        reference_range = float(np.median(da[np.isfinite(da) & (da > 0)]))
        if not np.isclose(reference_range, row['reference_range_m']):
            raise ValueError('Range scale differs')
        passed = strict = False
        if pose is not None:
            pose = np.array(pose)
            rotation = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T@cb[:3, :3])-1)/2, -1, 1))))
            translation = float(np.linalg.norm(pose[:3, 3]-cb[:3, 3]))
            if not np.allclose([rotation, translation], [row['pose_error']['rotation_deg'], row['pose_error']['translation_m']], atol=1e-6):
                raise ValueError('Pose error differs')
            passed = rotation <= 2 and translation <= .01*reference_range
            strict = rotation <= .5 and translation <= .001*reference_range
        if passed != row['pose_correct'] or strict != row['strict_pose_correct'] or (pose is not None) != row['pose_estimated']:
            raise ValueError('Pose classification differs')
        c = counts.setdefault(row['experiment'], dict(cases=0, correct_poses=0, strict_correct_poses=0,
                             poses_estimated=0, returned_matches=0, scorable_matches=0,
                             correct_correspondences_at_native_px=Counter()))
        c['cases'] += 1; c['correct_poses'] += int(passed); c['strict_correct_poses'] += int(strict)
        c['poses_estimated'] += pose is not None; c['returned_matches'] += len(p)
        c['scorable_matches'] += int(valid.sum()); c['correct_correspondences_at_native_px'].update(correct)
    if seen != expected:
        raise ValueError('Incomplete diagnostic coverage')
    for experiment, count in counts.items():
        for key, value in count.items():
            if value != summary['results'][experiment][key]:
                raise ValueError('Aggregate differs: '+experiment+' '+key)
    gate = json.loads(Path(gate_path).read_text())
    if sha256(gate['records_path']) != gate['records_sha256']:
        raise ValueError('Full camera inventory changed')
    inventory_counts = Counter(); groups = set(); fields = Counter()
    with gzip.open(gate['records_path'], 'rt') as f:
        for line in f:
            record = json.loads(line)
            inventory_counts[record['mode']] += 1; groups.add(record['fallback_group'])
            fields.update(record['metadata'].keys())
    if sum(inventory_counts.values()) != gate['camera_files']:
        raise ValueError('Full camera count differs')
    if dict(fields) != gate['metadata_field_counts']:
        raise ValueError('Released metadata fields differ')
    for mode, count in inventory_counts.items():
        if count != gate['archives'][mode]['camera_files']:
            raise ValueError('Per-archive inventory differs')
    old = json.loads(Path(gate['exposed_manifest']).read_text())
    exposed = {t['mode']+'/'+f['camera'].rsplit('/', 1)[0] for t in old['tiles'] for f in t['frames']}
    fresh = groups-exposed
    if len(fresh) != gate['gate']['fresh_group_count'] or (len(fresh) >= 12) != gate['gate']['passed']:
        raise ValueError('Freshness gate differs')
    result = dict(passed=True, diagnostic_rows=len(rows), complete_coverage=True,
                  independently_projected=True, aggregate_tables=len(counts),
                  inventory_camera_records=sum(inventory_counts.values()),
                  data_gate_passed=gate['gate']['passed'], source_sha256=sha256(__file__),
                  results_sha256=complete['results_sha256'], gate_sha256=sha256(gate_path))
    write_new(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='runs/localization-diagnostics-v3-validated')
    p.add_argument('--gate', default='reports/localization_data_gate_v3.json')
    p.add_argument('--output', default='reports/localization_replay_audit_v3.json')
    a = p.parse_args(); print(json.dumps(replay(a.run, a.gate, a.output), indent=2))
