"""Development hypotheses inspired by multimodal and medical registration.

These are small original 2D structural-correlation experiments, not a reproduction
of MIND or RIFT. Squared self-differences and unoriented gradients avoid reliance
on brightness polarity; neither can recover terrain absent from both images.
"""
from dataclasses import dataclass
from functools import lru_cache
import cv2
import numpy as np

from matchers import Match


@dataclass
class CorrelationMatch(Match):
    score: float = 0.
    peak_margin: float = 0.


def self_similarity_features(image):
    values = image.astype(np.float32)/255
    padded = np.pad(values, 2, mode='reflect')
    h, w = image.shape
    offsets = [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    distance = np.stack([cv2.GaussianBlur((values-padded[2+dy:2+dy+h, 2+dx:2+dx+w])**2, (5, 5), 1.)
                         for dy, dx in offsets], axis=-1)
    variance = distance.mean(axis=-1, keepdims=True)
    variance = np.maximum(variance, max(float(variance.mean())*.01, 1e-8))
    return np.exp(-(distance-distance.min(axis=-1, keepdims=True))/variance).astype(np.float32)


def orientation_features(image):
    values = cv2.GaussianBlur(image.astype(np.float32)/255, (5, 5), .8)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    channels = np.stack([(gx*np.cos(theta)+gy*np.sin(theta))**2 for theta in np.arange(8)*np.pi/8], axis=-1)
    energy = gx**2+gy**2
    denominator = cv2.GaussianBlur(energy, (9, 9), 2.) + max(float(energy.mean())*.05, 1e-8)
    return np.sqrt(np.maximum(channels, 0)/denominator[..., None]).astype(np.float32)


@lru_cache(maxsize=32)
def cached_features(shape, values, kind):
    image = np.frombuffer(values, dtype=np.uint8).reshape(shape)
    if kind == 'self_similarity':
        return self_similarity_features(image)
    if kind == 'orientation':
        return orientation_features(image)
    raise ValueError('Unknown structural representation')


def multichannel_scores(reference, query):
    """Exact channel-centered normalized cross correlation at every offset."""
    th, tw, channels = reference.shape
    shape = (query.shape[0]-th+1, query.shape[1]-tw+1)
    numerator, variance = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    template_energy = 0.
    window = np.ones((th, tw), np.float32)
    for index in range(channels):
        template = np.ascontiguousarray(reference[:, :, index])
        image = np.ascontiguousarray(query[:, :, index])
        centered = template-float(template.mean())
        template_energy += float(np.sum(centered.astype(np.float64)**2))
        numerator += cv2.matchTemplate(image, centered, cv2.TM_CCORR)
        local_sum = cv2.matchTemplate(image, window, cv2.TM_CCORR).astype(np.float64)
        local_square = cv2.matchTemplate(image**2, window, cv2.TM_CCORR).astype(np.float64)
        variance += np.maximum(local_square-local_sum**2/(th*tw), 0)
    denominator = np.sqrt(np.maximum(template_energy*variance, 1e-16))
    scores = numerator/denominator
    scores[variance < 1e-8] = -1
    return np.clip(scores, -1, 1).astype(np.float32)


def structural_ncc(reference, query, reference_mask=None, query_mask=None, kind='self_similarity'):
    if reference.shape != query.shape or reference.ndim != 2:
        raise ValueError('Structural translation matching needs equal grayscale shapes')
    ma = np.ones_like(reference, bool) if reference_mask is None else reference_mask.astype(bool)
    mb = np.ones_like(query, bool) if query_mask is None else query_mask.astype(bool)
    if not ma.any() or not mb.any():
        return CorrelationMatch(reason='no_valid_support')
    if min(reference[ma].std(), query[mb].std()) < 1:
        return CorrelationMatch(reason='low_texture')
    h, w = reference.shape
    x, y = w//4, h//4
    # Representation uses a local neighborhood; erode masks to exclude features
    # affected by zero padding or invalid source pixels.
    ma = cv2.erode(ma.astype(np.uint8), np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    mb = cv2.erode(mb.astype(np.uint8), np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if not ma[y:h-y, x:w-x].all():
        return CorrelationMatch(reason='invalid_template_support')
    a = cached_features(reference.shape, np.where(ma, reference, 0).astype(np.uint8).tobytes(), kind)
    b = cached_features(query.shape, np.where(mb, query, 0).astype(np.uint8).tobytes(), kind)
    template = a[y:h-y, x:w-x]
    scores = multichannel_scores(template, b)
    support = cv2.matchTemplate(mb.astype(np.float32), np.ones(template.shape[:2], np.float32), cv2.TM_CCORR)/np.prod(template.shape[:2])
    scores[support < .9999] = -1
    _, score, _, location = cv2.minMaxLoc(scores)
    if not np.isfinite(score) or score < .2:
        return CorrelationMatch(reason='low_structural_correlation', score=score)
    alternative = scores.copy()
    px, py = location
    alternative[max(0, py-3):py+4, max(0, px-3):px+4] = -1
    margin = float(score-np.max(alternative))
    shift = np.array(location)-[x, y]
    points = np.array([[xx, yy] for yy in np.linspace(y, h-y-1, 5) for xx in np.linspace(x, w-x-1, 5)], np.float32)
    return CorrelationMatch(matrix=np.array([[1., 0, shift[0]], [0, 1, shift[1]]]),
        support=len(points), reference_points=points, query_points=points+shift, score=score, peak_margin=margin)
