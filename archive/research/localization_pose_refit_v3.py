"""Explicit development follow-up: EPNP initialization versus inlier LM refit.

Uses saved matches and reference geometry only for estimation. Ground-truth
query pose is read only after estimation. Does not modify the v2 estimator.
"""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from localization_diagnostics_v3 import map_points, SEED
from lunar_geometry_v2 import pose_error
from lunargeo_dataset import write_new


def estimate_refined_pose(world, pixels, intrinsics):
    world, pixels = np.asarray(world, np.float64), np.asarray(pixels, np.float64)
    k = np.asarray(intrinsics, np.float64)
    if len(world) < 6 or not np.isfinite(world).all() or not np.isfinite(pixels).all():
        return None, 0
    center = world.mean(0)
    cv2.setRNGSeed(SEED)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(world-center, pixels, k, None,
        iterationsCount=2000, reprojectionError=3., confidence=.99, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None, 0
    index = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(world[index]-center, pixels[index], k, None, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = center-rotation.T@tvec.ravel()
    return pose, len(index)


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((source/'protocol.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if sha256(source/'results.json.gz') != summary['records_sha256']:
        raise ValueError('Input correspondences changed')
    with gzip.open(source/'results.json.gz', 'rt') as f:
        rows = json.load(f)
    manifest = json.loads(Path(protocol['manifest_path']).read_text())
    if sha256(protocol['manifest_path']) != protocol['manifest_sha256']:
        raise ValueError('Input geometry changed')
    tiles = {t['tile_id']: t for t in manifest['tiles']}; cache = {}; records = []; counts = {}
    cv2.setNumThreads(1)
    for number, row in enumerate(rows):
        tile = tiles[row['tile_id']]
        if tile['split'] != 'model_selection':
            raise ValueError('Development-only follow-up')
        if tile['tile_id'] not in cache:
            if sha256(tile['cache']) != tile['cache_sha256']:
                raise ValueError('Reference depth changed')
            with np.load(tile['cache'], allow_pickle=False) as z:
                cache[tile['tile_id']] = z['depth_a'].copy()
        reference, query = [np.array(row[k], float).reshape(-1, 2)
                            for k in ('reference_points_native', 'query_points_native')]
        camera = tile['cameras'][0]
        world, valid = map_points(reference, cache[tile['tile_id']], camera['intrinsics'], camera['cam2world'])
        pose, inliers = estimate_refined_pose(world, query[valid], tile['cameras'][1]['intrinsics'])
        # Query ground truth is only consumed here, after inference.
        error = None if pose is None else pose_error(pose, np.array(tile['cameras'][1]['cam2world']))
        correct = bool(error and error['rotation_deg'] <= 2 and error['translation_m'] <= .01*row['reference_range_m'])
        strict = bool(error and error['rotation_deg'] <= .5 and error['translation_m'] <= .001*row['reference_range_m'])
        records.append(dict(source_row=number, tile_id=tile['tile_id'], experiment=row['experiment'],
                            inliers=inliers, pose_error=error, pose_correct=correct,
                            strict_pose_correct=strict,
                            estimated_cam2world=None if pose is None else pose.tolist()))
        c = counts.setdefault(row['experiment'], dict(cases=0, original_correct=0, refined_correct=0,
                                                    refined_strict_correct=0, refined_estimated=0))
        c['cases'] += 1; c['original_correct'] += row['pose_correct']; c['refined_correct'] += correct
        c['refined_strict_correct'] += strict; c['refined_estimated'] += pose is not None
    result = dict(schema_version=3, scope='Post-diagnostic development-only estimator ablation; not training or a final test.',
        source_run=str(source), source_results_sha256=sha256(source/'results.json.gz'),
        source_sha256=sha256(__file__), inference_inputs='Saved 2D correspondences, reference ray-depth and pose, query intrinsics only.',
        estimator='Identical EPNP/RANSAC initialization and3px threshold, followed by solvePnPRefineLM on its original inliers.',
        results=counts, rows=records)
    write_new(output, result)
    print(json.dumps(counts, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='runs/localization-diagnostics-v3-validated')
    p.add_argument('--output', default='reports/localization_pose_refit_v3.json')
    a = p.parse_args(); run(a.source, a.output)
