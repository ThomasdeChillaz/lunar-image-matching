"""Development-only native-pixel geometry, precision and refinement experiments.

This is diagnosis, not compact-v3 training or a new held-out result. The fixed
v2 pose estimator is used throughout. Query depth/pose enter scoring only.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import zipfile

import cv2
import numpy as np
from dataset import sha256
from lunar_geometry_v2 import correspondences, estimate_pose, pose_error, world_points
from lunargeo_dataset import write_new

THRESHOLDS = (1, 2, 4, 8, 12)
SEED = 20261101


def native_points(points, native_shape, input_shape=(128, 128)):
    factor = np.array(native_shape[::-1])/np.array(input_shape[::-1])
    return (np.asarray(points, float).reshape(-1, 2)+.5)*factor-.5


def point_metrics(predicted, target, visible):
    predicted = np.asarray(predicted, float).reshape(-1, 2)
    target = np.asarray(target, float).reshape(-1, 2)
    visible = np.asarray(visible, bool)
    error = np.linalg.norm(predicted[visible]-target, axis=1)
    return dict(returned=len(predicted), scorable=int(visible.sum()),
        median_native_px=None if not len(error) else float(np.median(error)),
        correct_at_native_px={str(t): int((error <= t).sum()) for t in THRESHOLDS})


def refine_native(reference_points, query_points, reference_features, query_features,
                  radius=16, support_radius=12):
    """Descriptor search, then bounded quadratic interpolation; no geometry input.

    Both images use the frozen encoder at native resolution. This is a simple
    deterministic baseline, not a trained multi-resolution method.
    """
    import torch
    from contrastive_terrain import sample_descriptors
    p = np.asarray(reference_points, np.float32).reshape(-1, 2)
    q = np.asarray(query_points, np.float32).reshape(-1, 2)
    if p.shape != q.shape:
        raise ValueError('Paired Nx2 coordinates required')
    if not len(p):
        return p, q, np.empty(0, int)
    h, w = query_features.shape[-2:]
    hr, wr = reference_features.shape[-2:]
    center = np.rint(q).astype(int)
    valid = ((p >= support_radius) & (p < [wr-support_radius, hr-support_radius])).all(1)
    valid &= ((center-radius >= support_radius) &
              (center+radius < [w-support_radius, h-support_radius])).all(1)
    keep = np.where(valid)[0]
    if not len(keep):
        return p[:0], q[:0], keep
    query = query_features[0].detach().cpu().numpy()
    ref = sample_descriptors(reference_features, torch.from_numpy(p[keep])[None])[0].detach().cpu().numpy()
    result = []
    for index, descriptor in zip(keep, ref):
        x, y = center[index]
        patch = query[:, y-radius:y+radius+1, x-radius:x+radius+1]
        similarity = np.einsum('c,cyx->yx', descriptor, patch, optimize=False)
        iy, ix = np.unravel_index(np.argmax(similarity), similarity.shape)
        offset = np.array([float(ix-radius), float(iy-radius)])
        for axis, position in ((0, ix), (1, iy)):
            if 0 < position < 2*radius:
                left = similarity[iy, ix-1] if axis == 0 else similarity[iy-1, ix]
                right = similarity[iy, ix+1] if axis == 0 else similarity[iy+1, ix]
                middle = similarity[iy, ix]
                curvature = left-2*middle+right
                if curvature < -1e-8:
                    offset[axis] += np.clip(.5*(left-right)/curvature, -.5, .5)
        result.append(center[index]+offset)
    return p[keep], np.asarray(result, np.float32), keep


def map_points(points, depth, k, pose):
    points = np.asarray(points, float).reshape(-1, 2)
    index = np.rint(points).astype(int)
    valid = ((index >= 0) & (index < np.array(depth.shape[::-1]))).all(1)
    index = index.clip(0, np.array(depth.shape[::-1])-1)
    ranges = depth[index[:, 1], index[:, 0]]
    valid &= np.isfinite(ranges) & (ranges > 0)
    return world_points(points[valid], ranges[valid], k, pose), valid


def result_row(case, experiment, points, predicted, da, db, cameras, elapsed_ms=0.):
    ka, ca = [np.array(cameras[0][key], float) for key in ('intrinsics', 'cam2world')]
    kb, cb = [np.array(cameras[1][key], float) for key in ('intrinsics', 'cam2world')]
    points, predicted = [np.asarray(a, float).reshape(-1, 2) for a in (points, predicted)]
    _, truth, _, visible = correspondences(da, db, ka, ca, kb, cb, points=points)
    world, reference_valid = map_points(points, da, ka, ca)
    cv2.setRNGSeed(SEED)
    pose = estimate_pose(world, predicted[reference_valid], kb)
    error = None if pose is None else pose_error(pose, cb)
    ranges = da[np.isfinite(da) & (da > 0)]
    reference_range = float(np.median(ranges))
    correct = bool(error and error['rotation_deg'] <= 2 and error['translation_m'] <= .01*reference_range)
    strict = bool(error and error['rotation_deg'] <= .5 and error['translation_m'] <= .001*reference_range)
    if len(points):
        cell = np.floor(points/np.array(da.shape[::-1])*4).astype(int).clip(0, 3)
        spatial_cells = len(set(map(tuple, cell)))
    else:
        spatial_cells = 0
    return dict(case, experiment=experiment, **point_metrics(predicted, truth, visible),
        pose_estimated=pose is not None, pose_error=error, pose_correct=correct,
        strict_pose_correct=strict, reference_range_m=reference_range,
        occupied_4x4_cells=spatial_cells, elapsed_ms=elapsed_ms,
        reference_points_native=points.tolist(), query_points_native=predicted.tolist(),
        target_points_native=truth.tolist(), visible=visible.tolist(),
        estimated_cam2world=None if pose is None else pose.tolist())


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['experiment']].append(row)
    result = {}
    for experiment, items in sorted(groups.items()):
        scorable = sum(r['scorable'] for r in items)
        correct = {str(t): sum(r['correct_at_native_px'][str(t)] for r in items) for t in THRESHOLDS}
        result[experiment] = dict(cases=len(items), geometries=len({r['tile_id'] for r in items}),
            spatial_components=len({r['component'] for r in items}),
            poses_estimated=sum(r['pose_estimated'] for r in items),
            correct_poses=sum(r['pose_correct'] for r in items),
            strict_correct_poses=sum(r['strict_pose_correct'] for r in items),
            returned_matches=sum(r['returned'] for r in items), scorable_matches=scorable,
            correct_correspondences_at_native_px=correct,
            precision_at_native_px={k: None if not scorable else v/scorable for k, v in correct.items()},
            median_case_ms=float(np.median([r['elapsed_ms'] for r in items])))
    return result


def run(manifest_path, model, output, limit=None):
    import torch
    from compact_lunar import CompactMatcher
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(manifest_path).read_text())
    tiles = [t for t in manifest['tiles'] if t['split'] == 'model_selection']
    if limit is not None:
        tiles = tiles[:limit]
    if not tiles:
        raise ValueError('No model-selection geometries')
    output.mkdir(parents=True)
    source_files = ['localization_diagnostics_v3.py', 'lunar_geometry_v2.py', 'compact_lunar.py',
                    'contrastive_terrain.py']
    protocol = dict(schema_version=3, recorded_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path),
        model=str(model), checkpoint_sha256=sha256(Path(model)/'selected.pt'),
        source_sha256={p: sha256(p) for p in source_files}, split='model_selection',
        tiles=[t['tile_id'] for t in tiles], seed=SEED, pose_estimator='unchanged v2 EPNP/RANSAC,3 native px',
        controls=['oracle']+['grid_'+str(s) for s in (1, 2, 4, 8, 16)]+
                 ['noise_'+str(s) for s in (.5, 1, 2, 4)]+['outliers_'+str(s) for s in (.1, .3)],
        matcher_experiments=['coarse', 'deterministic_native', 'coarse_on_refinement_support'],
        refinement_radius_native_px=16, maximum_oracle_points=512,
        input_validity='finite image pixels only; query depth/pose restricted to scoring',
        warning='Development diagnosis, no new test or learned compact-v3 result. Native descriptor costs included separately; no speed claim.')
    write_new(output/'protocol.json', protocol)
    matcher = CompactMatcher(model)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    archives = {}; rows = []; feature_times = []
    started = perf_counter()
    try:
        for tile in tiles:
            mode = tile['mode']
            if mode not in archives:
                archives[mode] = zipfile.ZipFile('data/lunargeo/'+mode+'.zip')
            archive = archives[mode]
            if sha256(tile['cache']) != tile['cache_sha256']:
                raise ValueError('Development cache changed')
            with np.load(tile['cache'], allow_pickle=False) as data:
                da, db = data['depth_a'].copy(), data['depth_b'].copy()
            ka, ca = [np.array(tile['cameras'][0][k]) for k in ('intrinsics', 'cam2world')]
            kb, cb = [np.array(tile['cameras'][1][k]) for k in ('intrinsics', 'cam2world')]
            p, truth, _, _ = correspondences(da, db, ka, ca, kb, cb, stride=16)
            indices = np.linspace(0, len(p)-1, min(512, len(p)), dtype=int)
            p, truth = p[indices], truth[indices]
            case = dict(tile_id=tile['tile_id'], component=tile['component'], mode=mode, split='model_selection')
            seed = int(hashlib.sha256(tile['tile_id'].encode()).hexdigest()[:8], 16)^SEED
            for experiment in protocol['controls']:
                rng = np.random.default_rng(seed)
                q = truth.copy()
                if experiment.startswith('grid_'):
                    step = float(experiment.split('_')[1]); q = np.rint(q/step)*step
                elif experiment.startswith('noise_'):
                    q += rng.normal(0, float(experiment.split('_')[1]), q.shape)
                elif experiment.startswith('outliers_'):
                    corrupt = rng.choice(len(q), int(len(q)*float(experiment.split('_')[1])), replace=False)
                    q[corrupt] = rng.uniform([0, 0], np.array(db.shape[::-1])-1, (len(corrupt), 2))
                rows.append(result_row(case, experiment, p, q, da, db, tile['cameras']))
            images = []; features = []
            for frame, expected_hash in zip(tile['frames'], tile['source_hashes']):
                content = archive.read(frame['image'])
                if hashlib.sha256(content).hexdigest() != expected_hash['image']:
                    raise ValueError('Released native image differs from v2 provenance')
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != da.shape:
                    raise ValueError('Native image dimensions differ')
                images.append(image)
                tick = perf_counter()
                with torch.inference_mode():
                    dense, _ = matcher.model(torch.from_numpy(image.copy()).float()[None, None]/255)
                features.append(dense); feature_times.append((perf_counter()-tick)*1000)
            for light_a in range(3):
                for light_b in range(3):
                    ia, ib = 2*light_a, 2*light_b+1
                    a, b = [cv2.resize(images[i], (128, 128), interpolation=cv2.INTER_AREA) for i in (ia, ib)]
                    tick = perf_counter(); pa, pb = matcher.points(a, b)
                    coarse_ms = (perf_counter()-tick)*1000
                    pa, pb = native_points(pa, da.shape), native_points(pb, db.shape)
                    current = dict(case, light_a=light_a, light_b=light_b)
                    coarse = result_row(current, 'coarse', pa, pb, da, db, tile['cameras'], coarse_ms)
                    rows.append(coarse)
                    tick = perf_counter()
                    rp, rq, keep = refine_native(pa, pb, features[ia], features[ib])
                    refined = result_row(current, 'deterministic_native', rp, rq, da, db,
                                         tile['cameras'], coarse_ms+(perf_counter()-tick)*1000)
                    rows.append(refined)
                    rows.append(result_row(current, 'coarse_on_refinement_support', pa[keep], pb[keep],
                                           da, db, tile['cameras'], coarse_ms))
                    target = np.asarray(coarse['target_points_native']).reshape(-1, 2)
                    visible = np.array(coarse['visible'], bool)
                    coarse['visible_within_refinement_window'] = int((np.abs(pb[visible]-target) <= 16).all(1).sum())
            print('development geometry', tile['tile_id'], 'rows', len(rows), flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    records = output/'results.json.gz'
    with records.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            compressed.write(json.dumps(rows, allow_nan=False).encode())
    summary = dict(schema_version=3, scope=protocol['warning'], results=summarize(rows),
        elapsed_seconds=perf_counter()-started, native_feature_images=len(feature_times),
        native_feature_total_ms=sum(feature_times),
        native_feature_median_ms=float(np.median(feature_times)),
        oracle_gate_passed=sum(r['pose_correct'] for r in rows if r['experiment']=='oracle') >=
                           .95*sum(r['experiment']=='oracle' for r in rows),
        records_path=str(records), records_sha256=sha256(records),
        protocol_sha256=sha256(output/'protocol.json'),
        latency_note='Refinement row timing excludes cached native feature extraction; totals above are required for end-to-end timing. No speed comparison claimed.')
    write_new(output/'summary.json', summary)
    write_new(output/'COMPLETE.json', dict(results_sha256=sha256(records),
                                         summary_sha256=sha256(output/'summary.json'),
                                         protocol_sha256=sha256(output/'protocol.json')))
    print(json.dumps({k: {a: b for a, b in v.items() if a in ('cases', 'correct_poses', 'poses_estimated')}
                      for k, v in summary['results'].items()}, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='reports/compact_dataset_v2.json')
    p.add_argument('--model', default='runs/compact-v2-expanded-20261030')
    p.add_argument('--output', default='runs/localization-diagnostics-v3')
    p.add_argument('--limit', type=int)
    a = p.parse_args(); run(a.manifest, a.model, a.output, a.limit)
