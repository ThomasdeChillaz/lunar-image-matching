"""Saved-only conditioning diagnostic for the completed 384-case v92 run."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np


NATIVE = 512
INLIER_PX = 3.0


def camera(k, pose):
    k, pose = np.asarray(k, float), np.asarray(pose, float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError('Finite native perspective intrinsics required')
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])):
        raise ValueError('Finite homogeneous camera-to-world pose required')
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError('Proper camera rotation required')
    return k, r, pose[:3, 3]


def reference_range(depth, k):
    y, x = np.where(np.isfinite(depth) & (depth > 0))
    if not len(x):
        raise ValueError('Finite positive reference surface required')
    rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(k).T
    return float(np.median(depth[y, x] * np.linalg.norm(rays, axis=1)))


def projection_jacobian(camera_points, k, scale):
    """dq/d[rotation radians, camera-centre translation divided by scale]."""
    camera_points = np.asarray(camera_points, float)
    h = camera_points @ k.T
    q = h[:, :2] / h[:, 2, None]
    dq_dc = (k[None, :2] - q[:, :, None] * k[None, None, 2]) / h[:, 2, None, None]
    x, y, z = camera_points.T
    cross = np.zeros((len(x), 3, 3))
    cross[:, 0, 1], cross[:, 0, 2] = -z, y
    cross[:, 1, 0], cross[:, 1, 2] = z, -x
    cross[:, 2, 0], cross[:, 2, 1] = -y, x
    # R' = R @ Exp(omega); t' = t + R @ (scale * tau), at the saved pose.
    return np.concatenate((dq_dc @ cross, -scale * dq_dc), axis=2).reshape(-1, 6)


def local_conditioning(jacobian, scale):
    n = len(jacobian) // 2
    if not n:
        return dict(singular_values_px_per_sqrt_inlier=[0.] * 6, numerical_rank=0,
                    condition_number=None, translation_rank=0, translation_sensitivity=None)
    values = np.linalg.svd(jacobian, compute_uv=False)
    tol = np.finfo(float).eps * max(jacobian.shape) * values[0]
    rank = int((values > tol).sum())
    normalized = np.pad(values / np.sqrt(n), (0, 6 - len(values)))

    # Profile out the infinitesimal rotations. A translation that can be
    # mimicked by rotation must not appear well constrained here.
    u, rotation_values, _ = np.linalg.svd(jacobian[:, :3], full_matrices=False)
    basis = u[:, rotation_values > tol]
    translation = jacobian[:, 3:]
    reduced = translation - basis @ (basis.T @ translation)
    _, t_values, right = np.linalg.svd(reduced, full_matrices=False)
    t_rank = int((t_values > tol).sum())
    sensitivity = None
    if t_rank == 3:
        factor = scale * right.T / t_values[None, :]
        sensitivity = dict(camera_axis_sd_m_per_px=np.linalg.norm(factor, axis=1).tolist(),
                           position_rms_m_per_px=float(np.linalg.norm(factor)),
                           worst_direction_sd_m_per_px=float(scale / t_values[-1]))
    return dict(singular_values_px_per_sqrt_inlier=normalized.tolist(), numerical_rank=rank,
                condition_number=float(values[0] / values[-1]) if rank == 6 else None,
                translation_rank=t_rank, translation_sensitivity=sensitivity)


def image_spread(points):
    if not len(points):
        return None
    centered = points - points.mean(axis=0)
    values = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(points))
    span = np.ptp(points, axis=0)
    cells = np.floor(points / (NATIVE / 4)).astype(int)
    return dict(span_xy_px=span.tolist(), bbox_area_fraction=float(np.prod(span / NATIVE)),
                principal_sd_px=np.pad(values, (0, 2 - len(values))).tolist(),
                occupied_4x4_cells=int(len(np.unique(cells[:, 0] + 4 * cells[:, 1]))))


def features(p, q, estimated_pose, depth, reference_k, reference_pose, query_k, scale=None):
    p, q, depth = np.asarray(p, float), np.asarray(q, float), np.asarray(depth, float)
    if (p.ndim != 2 or p.shape[1] != 2 or p.shape != q.shape or depth.shape != (NATIVE, NATIVE)
            or any(not np.isfinite(x).all() or (x < 0).any() or (x > NATIVE - 1).any() for x in (p, q))):
        raise ValueError('Paired finite native points and native reference axial depth required')
    reference_k, rr, tr = camera(reference_k, reference_pose)
    query_k, rq, tq = camera(query_k, estimated_pose)
    scale = reference_range(depth, reference_k) if scale is None else scale
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive reference normalization range required')
    pixels = np.rint(p).astype(int)
    axial = depth[pixels[:, 1], pixels[:, 0]]
    use = np.isfinite(axial) & (axial > 0)
    rays = np.column_stack((p[use], np.ones(int(use.sum())))) @ np.linalg.inv(reference_k).T
    world = (rays * axial[use, None]) @ rr.T + tr
    xyz = (world - tq) @ rq
    h = xyz @ query_k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals = np.linalg.norm(h[:, :2] / h[:, 2, None] - q[use], axis=1)
    inliers = (xyz[:, 2] > 0) & np.isfinite(residuals) & (residuals <= INLIER_PX)
    j = projection_jacobian(xyz[inliers], query_k, scale)
    return dict(returned_correspondences=len(p), finite_reference_depth_correspondences=int(use.sum()),
                reprojection_inlier_count=int(inliers.sum()),
                reprojection_inlier_indices=np.flatnonzero(use)[inliers].tolist(),
                inlier_median_residual_px=float(np.median(residuals[inliers])) if inliers.any() else None,
                reference_image_spread=image_spread(p[use][inliers]), query_image_spread=image_spread(q[use][inliers]),
                reference_normalization_range_m=scale, **local_conditioning(j, scale))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def diagnosis(protocol_path):
    protocol = read(protocol_path)
    root = Path(protocol['output'])
    summary_path, watchdog_path = root / 'summary.json', root / 'watchdog.json'
    summary, watchdog, marker = (read(path) for path in (summary_path, watchdog_path, root / 'COMPLETE.json'))
    protocol_digest, summary_digest, watchdog_digest = map(sha, (protocol_path, summary_path, watchdog_path))
    if (protocol['purpose'] != 'full_direct_render_development_v92' or protocol['split'] != 'model_selection'
            or protocol['full_case_count'] != 384 or protocol['calibration_or_final_access']
            or not summary['complete'] or not watchdog['complete'] or not marker['external_watchdog_passed']
            or watchdog['failure'] is not None or watchdog['child_returncode'] != 0
            or any(record['protocol_sha256'] != protocol_digest for record in (summary, watchdog, marker))
            or marker['summary_sha256'] != summary_digest or marker['watchdog_sha256'] != watchdog_digest):
        raise ValueError('Only the original completed direct384 development result is admitted')
    manifest_digest = sha(protocol['manifest'])
    if manifest_digest != protocol['input_sha256'][protocol['manifest']]:
        raise ValueError('Original case manifest changed')
    cases = read(protocol['manifest'])['cases']
    rows = summary['rows']
    case_ids = protocol['case_ids']
    if (len(case_ids) != 384 or len(set(case_ids)) != len(case_ids)
            or [case['case_id'] for case in cases] != case_ids or [row['case_id'] for row in rows] != case_ids):
        raise ValueError('Exact original 384-case order required')

    diagnostics, reference_cache, reference_hashes = [], {}, {}
    for case, row in zip(cases, rows, strict=True):
        array_path = Path(row['saved_arrays'])
        if array_path.resolve().parent != root.resolve() or sha(array_path) != row['saved_arrays_sha256']:
            raise ValueError(f'Original saved correspondence archive changed: {case["case_id"]}')
        with np.load(array_path, allow_pickle=False) as saved:
            p, q, pose = (saved[name] for name in ('reference_xy', 'query_xy', 'pose'))
        if pose.shape not in ((4, 4), (0, 4)):
            raise ValueError('Saved pose requires original explicit shape')
        item = dict(case_id=case['case_id'], saved_arrays_sha256=row['saved_arrays_sha256'], geometry=None)
        if pose.shape == (4, 4):
            path = case['inference_inputs']['reference_geometry']
            ref_digest = case['input_sha256'][path]
            if path not in reference_cache:
                if sha(path) != ref_digest:
                    raise ValueError(f'Original reference geometry changed: {path}')
                with np.load(path, allow_pickle=False) as ref:
                    depth, k, camera_pose = (ref[name] for name in ('depth', 'intrinsics', 'cam2world'))
                reference_cache[path] = (depth, k, camera_pose, reference_range(depth, k))
                reference_hashes[path] = ref_digest
            if reference_hashes[path] != ref_digest:
                raise ValueError(f'Inconsistent reference geometry identity: {path}')
            depth, k, camera_pose, scale = reference_cache[path]
            item['geometry'] = features(p, q, pose, depth, k, camera_pose,
                                        case['inference_inputs']['query_intrinsics'], scale)
        diagnostics.append(item)

    # The metric fields are only accessed after all features have been computed.
    encoded = json.dumps(diagnostics, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    features_sha256 = hashlib.sha256(encoded).hexdigest()
    grouped = {name: [] for name in ('raw_correct', 'wrong_positive', 'raw_nonoverlap', 'no_pose')}
    count_mismatch = []
    for item, row in zip(diagnostics, rows, strict=True):
        metric = row['metrics']
        geometry = item['geometry']
        if metric['pose_returned'] != (geometry is not None):
            raise ValueError(f'Saved pose state differs: {item["case_id"]}')
        item['saved_outcome'] = {key: metric[key] for key in (
            'positive', 'correct', 'strict', 'pose_returned', 'wrong_positive', 'raw_nonoverlap', 'pose_error',
        )}
        item['saved_outcome']['two_px_correspondence'] = metric['correspondence'].get('2')
        category = ('no_pose' if geometry is None else 'raw_correct' if metric['correct'] else
                    'wrong_positive' if metric['wrong_positive'] else 'raw_nonoverlap')
        grouped[category].append(item)
        if geometry is not None:
            reported = row['final_estimator']['inliers']
            recomputed = geometry['reprojection_inlier_count']
            item['saved_estimator_inlier_count'] = reported
            if reported != recomputed:
                count_mismatch.append(dict(case_id=item['case_id'], original=reported, recomputed=recomputed))

    def group_summary(items):
        values = [item['geometry'] for item in items if item['geometry'] is not None]
        translation = [entry['translation_sensitivity'] for entry in values if entry['translation_sensitivity'] is not None]
        spread = [entry['query_image_spread'] for entry in values if entry['query_image_spread'] is not None]
        condition = [entry['condition_number'] for entry in values if entry['condition_number'] is not None]
        return dict(cases=len(items), geometry_cases=len(values),
                    positive_cases=sum(item['saved_outcome']['positive'] for item in items),
                    median_inliers=median([entry['reprojection_inlier_count'] for entry in values]) if values else None,
                    full_rank_cases=sum(entry['numerical_rank'] == 6 for entry in values),
                    median_query_minor_axis_sd_px=median([entry['principal_sd_px'][1] for entry in spread]) if spread else None,
                    median_query_occupied_4x4_cells=median([entry['occupied_4x4_cells'] for entry in spread]) if spread else None,
                    median_normalized_condition_number=median(condition) if condition else None,
                    translation_sensitivity_cases=len(translation),
                    median_translation_rms_m_per_px=median([entry['position_rms_m_per_px'] for entry in translation])
                    if translation else None)

    return dict(protocol_sha256=protocol_digest, summary_sha256=summary_digest, watchdog_sha256=watchdog_digest,
                manifest_sha256=manifest_digest, reference_geometry_sha256=reference_hashes,
                diagnosis_source_sha256=sha(__file__), feature_only_sha256=features_sha256,
                original_cases=len(rows), fixed_native_reprojection_radius_px=INLIER_PX,
                inlier_count_disagreements=count_mismatch, outcomes={key: group_summary(items) for key, items in grouped.items()},
                rows=diagnostics,
                normalization='Camera-frame rotation in radians and translation in full-reference median Euclidean range units; singular values divided by sqrt(inlier count). Numerical rank uses machine precision only.',
                scope='Saved pose, native correspondences, known reference axial depth/calibration, and query intrinsics only for features. Saved correctness was joined afterward. No image decode, query pose/depth/Sun access, fit, inference, or acceptance rule.',
                limitations='Translation sensitivity is local linear response to unit independent pixel noise with exact fixed map/calibration and fixed inlier set. It is not a pose error bound, uncertainty calibration, or optical/DEM validation; it omits correlated matching error, wrong correspondences, DEM error, alternate pose modes, and simulator-to-optics shift.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = diagnosis(args.protocol)
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
