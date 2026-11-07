"""Original dense structural descriptor hypothesis, not a CoMIR/D2-Net reproduction.

One shared CNN learns local descriptors from aligned changing-light renders.
It receives no coordinates or motion. Its finite radius (12 pixels) permits
explicit exclusion of descriptors whose source neighborhood contains nodata.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import sha256
from matchers import Match, fit_geometry


RADIUS = 12
OFFSETS = [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1),
           (0, 1), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def structural_channels(image):
    """Local contrast and self-difference channels, all with radius <=4.

    No image-wide normalization: an invalid distant pixel cannot affect a
    descriptor whose complete radius-12 neighborhood remains valid.
    """
    h, w = image.shape[-2:]
    padded = F.pad(image, (2, 2, 2, 2), mode='reflect')
    differences = torch.cat([(image-padded[..., 2+dy:2+dy+h, 2+dx:2+dx+w]).square()
                             for dy, dx in OFFSETS], dim=1)
    differences = F.avg_pool2d(differences, 5, stride=1, padding=2)
    variance = differences.mean(1, keepdim=True).clamp_min(1e-6)
    similarity = (-(differences-differences.amin(1, keepdim=True))/variance).exp()
    mean = F.avg_pool2d(image, 9, stride=1, padding=4)
    local_variance = (F.avg_pool2d(image.square(), 9, stride=1, padding=4)-mean.square()).clamp_min(1e-4)
    contrast = ((image-mean)/local_variance.sqrt()).clamp(-4, 4)/4
    return torch.cat([similarity, contrast], dim=1)


class TerrainDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        # Radius 4 input + convolution radii 1+2+4+1 =12.
        self.layers = nn.Sequential(nn.Conv2d(13, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1))

    def forward(self, image):
        return F.normalize(self.layers(structural_channels(image)), dim=1, eps=1e-8)


def sample_descriptors(dense, points):
    """Bilinear descriptors at native integer-center x,y coordinates.

    points: B,N,2; output B,N,C. align_corners=False is deliberate.
    """
    h, w = dense.shape[-2:]
    size = points.new_tensor([w, h])
    grid = 2*(points+.5)/size-1
    samples = F.grid_sample(dense, grid[:, None], align_corners=False,
                            padding_mode='zeros')[:, :, 0].transpose(1, 2)
    return F.normalize(samples, dim=-1, eps=1e-8)


def valid_centers(mask, radius=RADIUS):
    return cv2.erode(np.asarray(mask, np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def contrastive_loss(a, b, temperature=.1):
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError('At least two paired descriptors required')
    scores = F.normalize(a, dim=-1)@F.normalize(b, dim=-1).T/temperature
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(scores, labels)+F.cross_entropy(scores.T, labels))/2


class ContrastiveTerrainMatcher:
    def __init__(self, directory, checkpoint='selected', device='cpu'):
        directory = Path(directory)
        if checkpoint not in ('initial', 'selected'):
            raise ValueError('Explicit initial or selected checkpoint required')
        completed = json.loads((directory/'COMPLETE.json').read_text())
        metadata = json.loads((directory/'training.json').read_text())
        if completed['training_sha256'] != sha256(directory/'training.json'):
            raise ValueError('Training metadata changed')
        path = directory/f'{checkpoint}.pt'
        if completed[f'{checkpoint}_sha256'] != sha256(path):
            raise ValueError('Descriptor checkpoint changed')
        if metadata['source_sha256']['contrastive_terrain.py'] != sha256(__file__):
            raise ValueError('Descriptor implementation changed after training')
        self.device = device
        self.model = TerrainDescriptor().to(device).eval()
        self.model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        self.checkpoint_sha256 = sha256(path)
        self.warmed = False

    def sync(self):
        if self.device == 'mps':
            torch.mps.synchronize()
        elif self.device == 'cuda':
            torch.cuda.synchronize()

    def warmup(self, image):
        if not self.warmed:
            self(image, image)
            self.sync()
            self.warmed = True

    def __call__(self, reference, query, reference_mask=None, query_mask=None):
        if reference.ndim != 2 or query.shape != reference.shape:
            raise ValueError('Equal grayscale native images required')
        ma = np.ones_like(reference, bool) if reference_mask is None else np.asarray(reference_mask, bool)
        mb = np.ones_like(query, bool) if query_mask is None else np.asarray(query_mask, bool)
        if not ma.any() or not mb.any():
            return Match(reason='no_valid_support')
        if min(reference[ma].std(), query[mb].std()) < 1:
            return Match(reason='low_texture')
        h, w = reference.shape
        yy, xx = np.mgrid[RADIUS:h-RADIUS:4, RADIUS:w-RADIUS:4]
        grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        pa, pb = [grid[valid_centers(m)[grid[:, 1].astype(int), grid[:, 0].astype(int)]] for m in (ma, mb)]
        if min(len(pa), len(pb)) < 6:
            return Match(reason='insufficient_valid_descriptors')
        inputs = np.stack([np.where(ma, reference, 0), np.where(mb, query, 0)])
        with torch.inference_mode():
            dense = self.model(torch.from_numpy(inputs.copy()).to(self.device).float()[:, None]/255)
            a = sample_descriptors(dense[:1], torch.from_numpy(pa).to(self.device)[None])[0]
            b = sample_descriptors(dense[1:], torch.from_numpy(pb).to(self.device)[None])[0]
            similarity = a@b.T
            score, index = similarity.max(1)
            reverse = similarity.argmax(0)
            keep = (reverse[index] == torch.arange(len(a), device=self.device)) & (score >= .5)
            ia = torch.where(keep)[0]
            # Fixed budget. No image truth, motion, region or correctness input.
            ia = ia[torch.argsort(score[ia], descending=True)[:512]]
            ib = index[ia]
            aa, bb = pa[ia.cpu().numpy()], pb[ib.cpu().numpy()]
        return fit_geometry(aa, bb)
