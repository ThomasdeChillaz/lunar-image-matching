"""Small pair-context proposal experiment with exact frozen-score initialization.

Generic attention is not an architectural novelty claim. Its purpose is to
test the diagnosed global-correspondence failure under the100k total cap.
No query geometry or case identity enters this model.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contrastive_terrain import sample_descriptors, valid_centers

VARIANTS = ('pointwise', 'self_context', 'cross_context')


def position_features(points, image_size):
    """Native integer-center x,y coordinates in the128px proposal images."""
    size = points.new_tensor(image_size[::-1])
    xy = 2*(points+.5)/size-1
    frequencies = points.new_tensor([1, 2, 4, 8, 16, 32, 64, 128])
    phase = np.pi*xy[..., None]*frequencies
    return torch.stack([phase.sin(), phase.cos()], -1).flatten(-3)


class ContextBlock(nn.Module):
    def __init__(self, variant):
        super().__init__(); self.variant = variant
        self.mlp_norm = nn.LayerNorm(32)
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        if variant != 'pointwise':
            self.self_norm = nn.LayerNorm(32)
            self.self_attention = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
            nn.init.zeros_(self.self_attention.out_proj.weight); nn.init.zeros_(self.self_attention.out_proj.bias)
        if variant == 'cross_context':
            self.cross_norm = nn.LayerNorm(32)
            self.cross_attention = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
            nn.init.zeros_(self.cross_attention.out_proj.weight); nn.init.zeros_(self.cross_attention.out_proj.bias)

    def forward(self, a, b, pa, pb):
        if self.variant != 'pointwise':
            na, nb = self.self_norm(a), self.self_norm(b)
            a = a+self.self_attention(na+pa, na+pa, na, need_weights=False)[0]
            b = b+self.self_attention(nb+pb, nb+pb, nb, need_weights=False)[0]
        if self.variant == 'cross_context':
            na, nb = self.cross_norm(a), self.cross_norm(b)
            # Synchronous updates preserve swapping symmetry.
            da = self.cross_attention(na+pa, nb+pb, nb, need_weights=False)[0]
            db = self.cross_attention(nb+pb, na+pa, na, need_weights=False)[0]
            a, b = a+da, b+db
        return a+self.mlp(self.mlp_norm(a)), b+self.mlp(self.mlp_norm(b))


class PairContext(nn.Module):
    def __init__(self, variant='cross_context'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown proposal context ablation')
        self.variant = variant; self.blocks = nn.ModuleList([ContextBlock(variant) for _ in range(2)])
        # Allow the frozen coarse28,272 plus the existing learned fine35,233.
        if 28272+35233+sum(p.numel() for p in self.parameters()) >= 100000:
            raise ValueError('Context and strongest fine model exceed total parameter cap')

    def forward(self, a, b, points_a, points_b, image_size=(128, 128)):
        if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[-1] != 32 or b.shape[-1] != 32:
            raise ValueError('Paired BxNx32 descriptor matrices required')
        if points_a.shape != a.shape[:2]+(2,) or points_b.shape != b.shape[:2]+(2,) or min(a.shape[1], b.shape[1]) == 0:
            raise ValueError('Nonempty paired native descriptor coordinates required')
        if any(not torch.isfinite(v).all() for v in (a, b, points_a, points_b)):
            raise ValueError('Nonfinite descriptor/coordinate input')
        # Preserve the exact original mm operation and memory layout at step0.
        original = torch.stack([aa@bb.T for aa, bb in zip(a, b)])
        base_a, base_b = a.contiguous(), b.contiguous(); x, y = base_a, base_b
        pa, pb = position_features(points_a, image_size), position_features(points_b, image_size)
        for block in self.blocks:
            x, y = block(x, y, pa, pb)
        normalized_a, normalized_b = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        base_normal_a, base_normal_b = F.normalize(base_a, dim=-1), F.normalize(base_b, dim=-1)
        update = normalized_a@normalized_b.transpose(1, 2)-base_normal_a@base_normal_b.transpose(1, 2)
        # This residual score is exactly zero initially and remains trainable.
        # Frozen input descriptors already have unit norm; the anchor avoids a
        # new floating-point normalization changing the baseline's tied matches.
        return original+update


class ContextMatcher:
    def __init__(self, coarse, context):
        if coarse.metadata['variant'] != 'expanded' or coarse.model.reliability:
            raise ValueError('The declared frozen expanded proposal checkpoint is required')
        self.coarse = coarse; self.context = context.eval(); self.model = coarse.model
        self.device = coarse.device; self.metadata = coarse.metadata; self.last_evidence = {}

    def points(self, a, b, ma=None, mb=None):
        ma = np.ones(a.shape, bool) if ma is None else np.asarray(ma, bool)
        mb = np.ones(b.shape, bool) if mb is None else np.asarray(mb, bool)
        empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
        if a.shape != (128, 128) or b.shape != a.shape or ma.shape != a.shape or mb.shape != b.shape:
            raise ValueError('Equal128px grayscale proposal images/masks required')
        if not ma.any() or not mb.any() or min(a[ma].std(), b[mb].std()) < 1:
            return empty
        y, x = np.mgrid[12:116:4, 12:116:4]
        grid = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
        pa, pb = [grid[valid_centers(mask, 12)[grid[:, 1].astype(int), grid[:, 0].astype(int)]] for mask in (ma, mb)]
        if min(len(pa), len(pb)) < 6:
            return empty
        images = torch.from_numpy(np.stack([np.where(ma, a, 0), np.where(mb, b, 0)])).to(self.device).float()[:, None]/255
        with torch.inference_mode():
            dense, _ = self.model(images)
            pta, ptb = [torch.from_numpy(p).to(self.device)[None] for p in (pa, pb)]
            da, db = sample_descriptors(dense[:1], pta), sample_descriptors(dense[1:], ptb)
            scores = self.context(da, db, pta, ptb)[0]
            top, index = scores.max(1); reverse = scores.argmax(0)
            keep = (reverse[index] == torch.arange(len(pa), device=self.device)) & (top >= .5)
            ia = torch.where(keep)[0]; ia = ia[torch.argsort(top[ia], descending=True)[:512]]
            ib = index[ia]
            return pa[ia.cpu().numpy()], pb[ib.cpu().numpy()]
