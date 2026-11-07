"""Separate, seeded translation/rotation/scale/combined research conditions.

The transform is evaluation supervision only. A matcher must receive just the
images and validity masks, never this case's transform or sampling parameters.
"""
import hashlib
import itertools
import numpy as np
import cv2

from geometry import evaluation_points, transform


FAMILIES = ('translation', 'rotation', 'scale', 'combined')
SEED = 'lunar-geometry-20261029-v1'


def motion(pair_id, family, shape=(128, 128)):
    if family not in FAMILIES:
        raise ValueError('Unknown geometry family')
    seed = int.from_bytes(hashlib.sha256((SEED+':'+pair_id+':'+family).encode()).digest()[:8], 'big')
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-30, 30)) if family in ('rotation', 'combined') else 0.
    scale = float(np.exp(rng.uniform(np.log(.75), np.log(1.3)))) if family in ('scale', 'combined') else 1.
    dx, dy = rng.uniform(-12, 12, 2) if family in ('translation', 'combined') else (0., 0.)
    h, w = shape
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, scale)
    matrix[:, 2] += [dx, dy]
    return matrix, dict(angle_deg=angle, scale=scale, dx=float(dx), dy=float(dy))


def plan(manifest, split):
    return [dict(pair_id=f'{tile["tile_id"]}_sun_{i}_{j}_{family}', tile_id=tile['tile_id'],
                 split=split, reference_light_index=i, query_light_index=j, geometry_family=family)
            for tile in manifest['tiles'] if tile['split'] == split
            for i, j in itertools.combinations(range(len(tile['images'])), 2)
            for family in FAMILIES]


def prepare(reference, base_query, reference_mask, query_mask, pair_id, family):
    if reference.shape != base_query.shape or reference_mask.shape != reference.shape or query_mask.shape != reference.shape:
        raise ValueError('Geometry suite needs equal image/mask shapes')
    h, w = reference.shape
    matrix, parameters = motion(pair_id, family, reference.shape)
    # Warp floating validity with the same interpolation. Keep only fully valid
    # bilinear support; invalid neighboring pixels cannot become scored terrain.
    query = cv2.warpAffine(np.where(query_mask, base_query, 0), matrix, (w, h), flags=cv2.INTER_LINEAR)
    valid_fraction = cv2.warpAffine(query_mask.astype(np.float32), matrix, (w, h), flags=cv2.INTER_LINEAR)
    mask = valid_fraction >= 1.-1e-6
    query[~mask] = 0
    points = evaluation_points(reference.shape, matrix)
    targets = transform(points, matrix)
    keep = reference_mask[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
    keep &= mask[np.rint(targets[:, 1]).astype(int), np.rint(targets[:, 0]).astype(int)]
    if not keep.any():
        raise ValueError('No valid evaluation support')
    return query, mask, matrix, points[keep], targets[keep], parameters
