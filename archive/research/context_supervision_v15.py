"""Train-only3D labels, image augmentations and capacity controls for proposal context."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from global_context_v15 import PairContext
from lunar_geometry_v5 import correspondences_z
from rotation_bank_pose_v4 import quarter_turn
from train_balanced_localization_v3 import appearance, nonoverlap


def make_context(variant):
    model = PairContext(variant)
    width = {'pointwise': 192, 'self_context': 128, 'cross_context': 64}[variant]
    if width != 64:
        for block in model.blocks:
            block.mlp = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, 32))
            nn.init.zeros_(block.mlp[-1].weight); nn.init.zeros_(block.mlp[-1].bias)
    model.feedforward_width = width
    if 28272+35233+sum(p.numel() for p in model.parameters()) >= 100000:
        raise ValueError('Combined capacity control exceeds100k')
    return model


def proposal_grid():
    y, x = np.mgrid[12:116:4, 12:116:4]
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)


def full_targets(depth_a, depth_b, a, b, points):
    _, target, _, visible = correspondences_z(depth_a, depth_b, a['intrinsics'], a['cam2world'],
        b['intrinsics'], b['cam2world'], points=points)
    full = np.zeros((len(points), 2), np.float32); full[visible] = (target+.5)/4-.5
    return full, visible


def label_masks(target, visible):
    # Edge bands are deliberately ignored. A true target slightly outside the
    # token grid may still be recoverable by the existing native33px search.
    positive = visible & ((target >= 12) & (target <= 112)).all(-1)
    negative = ~visible | ((target < 8) | (target > 116)).any(-1)
    return positive, negative


def prepare_supervision(tiles, data):
    grid = proposal_grid(); native = (grid+.5)*4-.5; prepared = {}; counts = {}
    for key, tile in tiles.items():
        if tile['split'] != 'train':
            continue
        z = data[key]; a, b = tile['cameras']
        q, visible = full_targets(z['depth_a'], z['depth_b'], a, b, native)
        rotations = {}
        for k in range(4):
            forward = quarter_turn(q, k, 128).astype(np.float32)
            original_query = quarter_turn(native, k, 512, inverse=True)
            reverse, reverse_visible = full_targets(z['depth_b'], z['depth_a'], b, a, original_query)
            fp, fn = label_masks(forward, visible); rp, rn = label_masks(reverse, reverse_visible)
            if min(fp.sum(), rp.sum()) < 8:
                raise ValueError('A declared train geometry/orientation lacks eight supervised token matches')
            rotations[k] = dict(forward=forward, forward_visible=visible.copy(), reverse=reverse,
                reverse_visible=reverse_visible, forward_positive=int(fp.sum()), reverse_positive=int(rp.sum()))
        prepared[key] = dict(images=np.stack([cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA) for im in z['images']]),
                             rotations=rotations)
        counts[key] = dict(component=tile['component'],
            forward_positive=[rotations[k]['forward_positive'] for k in range(4)],
            reverse_positive=[rotations[k]['reverse_positive'] for k in range(4)])
    return prepared, dict(tiles=len(prepared), groups=len({c['component'] for c in counts.values()}),
        points_per_image=len(grid), per_tile=counts)


def sample_training(tiles, prepared, rng, negative_pair=False):
    groups = {}
    for tile in tiles.values():
        if tile['split'] == 'train':
            groups.setdefault(tile['component'], []).append(tile['tile_id'])
    names = sorted(groups); component = names[int(rng.integers(len(names)))]; ids = groups[component]
    key = ids[int(rng.integers(len(ids)))]; other = key
    la, lb = rng.integers(0, 3, 2); rotation = int(rng.integers(4))
    if negative_pair:
        eligible = [t['tile_id'] for t in tiles.values() if t['split'] == 'train' and nonoverlap(tiles[key], t)]
        if not eligible:
            raise ValueError('Verified nonoverlapping train terrain required')
        other = eligible[int(rng.integers(len(eligible)))]; n = len(proposal_grid())
        targets = dict(forward=np.zeros((n, 2), np.float32), reverse=np.zeros((n, 2), np.float32),
            forward_visible=np.zeros(n, bool), reverse_visible=np.zeros(n, bool))
    else:
        targets = prepared[key]['rotations'][rotation]
    reference = prepared[key]['images'][2*la]
    query = np.rot90(prepared[other]['images'][2*lb+1], rotation).copy()
    images = np.stack([appearance(reference, rng), appearance(query, rng)])[:, None]
    result = {k: torch.from_numpy(targets[k].copy()) for k in ('forward', 'reverse', 'forward_visible', 'reverse_visible')}
    metadata = dict(tile_a=key, tile_b=other, component_a=component, component_b=tiles[other]['component'],
        light_a=int(la), light_b=int(lb), query_quarter_turn=rotation, verified_nonoverlap=bool(negative_pair))
    return torch.from_numpy(images), result, metadata


def directional_loss(scores, target, visible):
    positive, negative = label_masks(target, visible)
    loss = scores.sum()*0
    if positive.any():
        xy = (target[positive]-12)/4
        low = xy.floor().long(); high = (low+1).clamp(max=25); fraction = xy-low
        logp = (scores[positive]/.1).log_softmax(-1); nll = scores.new_zeros(len(xy)); expected = scores.new_zeros(len(xy))
        for x, y, weight in [(low[:, 0], low[:, 1], (1-fraction[:, 0])*(1-fraction[:, 1])),
            (high[:, 0], low[:, 1], fraction[:, 0]*(1-fraction[:, 1])),
            (low[:, 0], high[:, 1], (1-fraction[:, 0])*fraction[:, 1]),
            (high[:, 0], high[:, 1], fraction[:, 0]*fraction[:, 1])]:
            index = (y*26+x)[:, None]
            nll -= weight*logp.gather(1, index)[:, 0]
            expected += weight*scores[positive].gather(1, index)[:, 0]
        loss = loss+nll.mean()+.25*F.softplus(10*(.7-expected)).mean()/10
    if negative.any():
        loss = loss+.25*F.softplus(10*(scores[negative].amax(-1)-.5)).mean()/10
    return loss, dict(positive=int(positive.sum()), negative=int(negative.sum()), ignored=int((~positive & ~negative).sum()))


def objective(scores, targets):
    if scores.shape != (1, 676, 676):
        raise ValueError('One complete fixed-grid proposal pair required')
    forward, fs = directional_loss(scores[0], targets['forward'], targets['forward_visible'])
    reverse, rs = directional_loss(scores[0].T, targets['reverse'], targets['reverse_visible'])
    return .5*(forward+reverse), dict(forward=fs, reverse=rs)
