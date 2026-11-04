"""Procedural initialization diagnosis, not a lunar or learned-model result."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from solar_pose_proposals_v73 import propose
from localization_robust_pose_v4 import estimate


def fixture(seed, stride, outlier_fraction=0.):
    k = np.array([[430., 0, 255.5], [0, 430., 255.5], [0, 0, 1.]])
    y, x = np.mgrid[65:448:24, 65:448:24]
    p = np.column_stack([x.ravel(), y.ravel()])
    z = 1200 + 80*np.sin(p[:, 0]/90)*np.cos(p[:, 1]/70)
    world = np.column_stack([(p[:, 0]-255.5)*z/430, (p[:, 1]-255.5)*z/430, z])
    rng = np.random.default_rng(20261104+seed)
    truth = np.eye(4)
    truth[:3, :3] = cv2.Rodrigues(rng.uniform(-.08, .08, 3))[0]
    truth[:3, 3] = rng.uniform(-70, 70, 3)
    camera = (world-truth[:3, 3]) @ truth[:3, :3]
    homogeneous = camera @ k.T
    query = homogeneous[:, :2]/homogeneous[:, 2, None]
    use = (camera[:, 2] > 0) & ((query >= 49.5) & (query <= 449.5)).all(1)
    world, exact = world[use], query[use]
    pixels = exact.copy() if not stride else 49.5 + stride*np.rint((exact-49.5)/stride)
    corrupted = rng.choice(len(pixels), int(outlier_fraction*len(pixels)), replace=False)
    pixels[corrupted] = rng.uniform(49.5, 449.5, (len(corrupted), 2))
    return world, pixels, k, truth


def error(pose, truth):
    if pose is None:
        return None
    return dict(rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T@truth[:3, :3])-1)/2, -1, 1)))),
        translation_m=float(np.linalg.norm(pose[:3, 3]-truth[:3, 3])))


def run(output):
    cv2.setNumThreads(1)
    rows = []
    for seed in range(8):
        for stride, outlier in ((0, 0.), (4, 0.), (8, 0.), (16, 0.), (16, .2), (16, .4)):
            world, pixels, k, truth = fixture(seed, stride, outlier)
            for method in ('strict3_initializer', 'quantization_disc_initializer'):
                tick = time.perf_counter()
                if method == 'strict3_initializer':
                    pose, info = estimate(world, pixels, k, dict(solver='ap3p', iterations=128, reclassify=True))
                else:
                    pose, info = propose(world, pixels, k, 128)
                # Pose truth is consumed only after the proposal function returns.
                rows.append(dict(seed=seed, stride_native_px=stride, outlier_fraction=outlier,
                    method=method, raw_trial_pose=None if pose is None else pose.tolist(),
                    pose_error=error(pose, truth), points=len(world), proposal_info=info,
                    software_elapsed_seconds=time.perf_counter()-tick))
    sources = ('diagnose_pose_quantization_v73.py', 'solar_pose_proposals_v73.py',
        'localization_robust_pose_v4.py', 'localization_pose_refit_v3.py')
    result = dict(complete=True, rows=rows, seeds=list(range(8)),
        source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources},
        final_estimator_modified=False, model_training=False, lunar_images_read=0,
        scope='Procedural 3D geometry and synthetic quantization/outliers. Raw INITIAL trial poses only; neither final navigation results nor a learned/SOTA improvement. Software diagnostic timings are not CPU efficiency measurements.')
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = run(a.output)
    for method in ('strict3_initializer', 'quantization_disc_initializer'):
        part = [r for r in result['rows'] if r['method'] == method and r['stride_native_px'] == 16 and not r['outlier_fraction']]
        errors = [r['pose_error'] for r in part if r['pose_error']]
        print(method, len(errors), np.median([e['rotation_deg'] for e in errors]), np.median([e['translation_m'] for e in errors]))
