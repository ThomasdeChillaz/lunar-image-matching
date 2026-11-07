"""Procedural textures and known warps, NOT physically rendered lunar terrain."""
import cv2
import numpy as np


def texture(seed, size=256):
    if size < 96:
        raise ValueError('Image size must be at least 96')
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.normal(110, 30, (size, size)).astype(np.float32), (3, 3), 0)
    for _ in range(70):
        center = tuple(int(v) for v in rng.integers(12, size-12, 2))
        radius = int(rng.integers(3, max(4, size//16)))
        cv2.circle(image, center, radius, float(rng.uniform(30, 220)), -1)
        cv2.circle(image, center, radius, float(rng.uniform(30, 220)), 1)
    return np.clip(image, 0, 255).astype(np.uint8)


def make_pair(image, case, seed):
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected grayscale uint8 input')
    h, w = image.shape
    truth = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), case['angle'], 1.0)
    truth[:, 2] += [case['dx'], case['dy']]
    query = cv2.warpAffine(image, truth, (w, h), borderMode=cv2.BORDER_CONSTANT)
    query = case['gain'] * (query.astype(float)/255)**case['gamma']
    query += np.random.default_rng(seed).normal(0, case['noise'], query.shape)
    return np.clip(query*255, 0, 255).astype(np.uint8), truth
