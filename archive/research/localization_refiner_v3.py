"""Small native-pixel refinement models; no camera or depth inference inputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contrastive_terrain import TerrainDescriptor, sample_descriptors

VARIANTS = ('descriptor_3d', 'refinement', 'reliability')
WINDOW = 33
PATCH = 57
REFERENCE_PATCH = 27


class LocalRefiner(nn.Module):
    def __init__(self, variant='reliability'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown refinement ablation')
        self.variant = variant
        self.encoder = TerrainDescriptor()
        self.pair = None
        self.matchability = None
        if variant != 'descriptor_3d':
            self.pair = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(16, 16, 3, padding=1), nn.SiLU())
            self.offset_logits = nn.Conv2d(16, 1, 1)
            nn.init.zeros_(self.offset_logits.weight); nn.init.zeros_(self.offset_logits.bias)
        if variant == 'reliability':
            self.matchability = nn.Sequential(nn.Linear(35, 16), nn.SiLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.matchability[-1].weight); nn.init.zeros_(self.matchability[-1].bias)
        # Include the separate, frozen 28,272-parameter coarse proposal network.
        if sum(p.numel() for p in self.parameters())+28272 >= 100000:
            raise ValueError('Combined matcher exceeds parameter budget')

    def initialize_encoder(self, coarse):
        self.encoder.layers.load_state_dict(coarse.model.layers.state_dict(), strict=True)

    def score(self, reference, query_window):
        product = reference[:, :, None, None]*query_window
        logits = 10*product.sum(1)
        match_logit = torch.zeros(len(reference), device=reference.device)
        if self.pair is not None:
            features = self.pair(product)
            logits = logits+self.offset_logits(features)[:, 0]
            if self.matchability is not None:
                top = logits.flatten(1).topk(2, dim=1).values
                prob = logits.flatten(1).softmax(1)
                entropy = -(prob*logits.flatten(1).log_softmax(1)).sum(1)/np.log(WINDOW**2)
                evidence = torch.cat([features.mean((2, 3)), features.amax((2, 3)),
                                      top[:, :1], top[:, :1]-top[:, 1:], entropy[:, None]], dim=1)
                match_logit = self.matchability(evidence)[:, 0]
        return logits, match_logit

    def forward(self, reference_patch, query_patch, reference_xy):
        reference_dense = self.encoder(reference_patch)
        query_dense = self.encoder(query_patch)
        ref = sample_descriptors(reference_dense, reference_xy[:, None])[:, 0]
        return self.score(ref, query_dense[:, :, 12:45, 12:45])


def local_loss(logits, match_logit, target_xy, matchable, reliability):
    """Bilinear exact-coordinate target; occlusion is not inferred from brightness."""
    valid = matchable.bool()
    spatial = logits.sum()*0
    if valid.any():
        xy = target_xy[valid].clamp(0, WINDOW-1)
        low = xy.floor().long(); high = (low+1).clamp(max=WINDOW-1)
        fraction = xy-low
        logp = logits[valid].flatten(1).log_softmax(1)
        spatial = logits.new_zeros(len(xy))
        for x, y, weight in [
            (low[:, 0], low[:, 1], (1-fraction[:, 0])*(1-fraction[:, 1])),
            (high[:, 0], low[:, 1], fraction[:, 0]*(1-fraction[:, 1])),
            (low[:, 0], high[:, 1], (1-fraction[:, 0])*fraction[:, 1]),
            (high[:, 0], high[:, 1], fraction[:, 0]*fraction[:, 1])]:
            spatial -= weight*logp.gather(1, (y*WINDOW+x)[:, None])[:, 0]
        spatial = spatial.mean()
    if reliability:
        spatial = spatial+.25*F.binary_cross_entropy_with_logits(match_logit, matchable.float())
    return spatial


def decode_peak(logits):
    """Refine the strongest mode only; never average separated competing peaks."""
    n = len(logits); flat = logits.flatten(1)
    index = flat.argmax(1); x = index % WINDOW; y = index//WINDOW
    xy = torch.stack([x, y], 1).to(logits.dtype)
    batch = torch.arange(n, device=logits.device)
    for axis, position in ((0, x), (1, y)):
        inside = (position > 0) & (position < WINDOW-1)
        if not inside.any():
            continue
        b, xx, yy = batch[inside], x[inside], y[inside]
        left = logits[b, yy, xx-1] if axis == 0 else logits[b, yy-1, xx]
        right = logits[b, yy, xx+1] if axis == 0 else logits[b, yy+1, xx]
        middle = logits[b, yy, xx]
        denominator = left-2*middle+right
        delta = (.5*(left-right)/denominator.clamp(max=-1e-8)).clamp(-.5, .5)
        delta = torch.where(denominator < -1e-8, delta, torch.zeros_like(delta))
        xy[inside, axis] += delta
    return xy-16


def refine_dense(model, reference_points, query_points, reference_dense, query_dense, batch_size=32):
    p, q = [np.asarray(v, np.float32).reshape(-1, 2) for v in (reference_points, query_points)]
    if p.shape != q.shape:
        raise ValueError('Paired Nx2 points required')
    h, w = query_dense.shape[-2:]; hr, wr = reference_dense.shape[-2:]
    center = np.rint(q).astype(int)
    valid = ((p >= 13) & (p < [wr-13, hr-13])).all(1)
    valid &= ((center >= 28) & (center < [w-28, h-28])).all(1)
    indices = np.where(valid)[0]; offsets = torch.arange(-16, 17, device=query_dense.device)
    predictions = []; scores = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        centers = torch.from_numpy(center[selected]).to(query_dense.device)
        ys = centers[:, 1, None, None]+offsets[None, :, None]
        xs = centers[:, 0, None, None]+offsets[None, None, :]
        windows = query_dense[0][:, ys, xs].permute(1, 0, 2, 3).contiguous()
        vectors = sample_descriptors(reference_dense, torch.from_numpy(p[selected]).to(query_dense.device)[None])[0]
        logits, match = model.score(vectors, windows)
        predictions.append((centers+decode_peak(logits)).cpu().numpy())
        scores.append(match.sigmoid().cpu().numpy())
    predicted = np.concatenate(predictions) if predictions else np.empty((0, 2), np.float32)
    reliability = np.concatenate(scores) if scores else np.empty(0, np.float32)
    return p[indices], predicted, reliability, indices
