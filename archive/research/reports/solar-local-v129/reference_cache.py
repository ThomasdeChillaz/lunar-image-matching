"""Reference-only caches; query features and correspondence searches stay fresh."""

from hashlib import sha256
from pathlib import Path
import sys
import time

import numpy as np

OLD = Path(__file__).resolve().parents[1] / "solar-local-v127"
sys.path.insert(0, str(OLD))
import pipeline

OriginalSift = pipeline.SiftLibraryMatcher


def array_key(*arrays):
    digest = sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class ReferenceProjectionCache:
    def __init__(self):
        self.entries = {}
        self.last = None

    def initialize(self, reference, query, depth, k, reference_pose, query_k):
        started = time.perf_counter()
        cloud = pipeline.coarse.build_reference_cloud(reference, depth, k, reference_pose)
        key = array_key(cloud.world, cloud.flat_world, reference_pose, query_k, np.array(reference.shape))
        hit = key in self.entries
        if not hit:
            bank = pipeline.coarse.reference_pose_bank(reference_pose)
            projected = pipeline.FastProjectionBank(cloud, query_k, bank, reference.shape, 4)
            self.entries[key] = bank, projected
        bank, projected = self.entries[key]
        # Validate all geometry and every camera before reusing reference-only
        # visibility. Rank photometry again from the current two images.
        projected.validate(cloud, query_k, bank, reference.shape, 4)
        ranked = pipeline.coarse.rank_pose_bank(cloud, query, query_k, bank,
                                                cell_pixels=32, projection_bank=projected)
        indices = ranked["orders"]["planar"][:8]
        self.last = dict(cache_hit=hit, seconds=time.perf_counter()-started,
                         projection_entries=len(projected.indices), key=key)
        return bank[indices], dict(indices=indices, candidate_count=len(bank), reason=ranked["reason"],
                                  reference_cache=self.last)


class FeatureStore:
    def __init__(self):
        self.values = {}
        self.hits = self.misses = 0
        self.miss_seconds = []


class CachedReferenceSift(OriginalSift):
    store = FeatureStore()

    def __init__(self, settings):
        super().__init__(settings)
        self.original_detector = self.detector
        self.settings_key = tuple(sorted(settings.items()))
        # Parent match() obtains reference descriptors through this proxy.
        # Its BF matches, ratio tests and ordering are unchanged.
        self.detector = self

    def query_features(self, image):
        return self.original_detector.detectAndCompute(image, None)

    def detectAndCompute(self, image, mask):
        if mask is not None:
            raise ValueError("Frozen SIFT reference configuration uses no feature mask")
        key = self.settings_key, array_key(image)
        if key not in self.store.values:
            start = time.perf_counter()
            self.store.values[key] = self.original_detector.detectAndCompute(image, None)
            self.store.miss_seconds.append(time.perf_counter()-start)
            self.store.misses += 1
        else:
            self.store.hits += 1
        return self.store.values[key]


def install():
    cache = ReferenceProjectionCache()
    CachedReferenceSift.store = FeatureStore()
    pipeline.initialize = cache.initialize
    pipeline.SiftLibraryMatcher = CachedReferenceSift
    return cache
