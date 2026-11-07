import cv2
import numpy as np

from reference_cache import CachedReferenceSift, FeatureStore, OriginalSift, ReferenceProjectionCache, pipeline


def test_reference_feature_cache_preserves_actual_sift_matches_and_keeps_queries_fresh():
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261101)
    image = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    query = np.roll(image, 4, axis=1)
    settings = dict(nfeatures=128, ratio_threshold=.75)
    expected, actual = OriginalSift(settings), CachedReferenceSift(settings)
    actual.store = FeatureStore()
    qfeatures = actual.query_features(query)
    original = expected.match(image, expected.query_features(query))
    for _ in range(2):
        result = actual.match(image, qfeatures)
        assert all(np.array_equal(a, b) for a, b in zip(original, result))
    assert actual.store.hits == actual.store.misses == 1
    actual.query_features(image)
    assert actual.store.hits == actual.store.misses == 1
    actual.match(image[::-1].copy(), qfeatures)
    assert actual.store.misses == 2


def test_projected_visibility_reuse_preserves_scores_and_never_reuses_query_evidence(monkeypatch):
    # Keep this a small exact control; the admitted pilot measures 2048 cameras.
    shape = (256, 256)
    k = np.array([[280., 0., 127.5], [0., 280., 127.5], [0., 0., 1.]])
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 2000.
    rng = np.random.default_rng(2)
    image = rng.uniform(.1, 1., shape)
    depth = np.full(shape, 2000.)
    full = pipeline.coarse.reference_pose_bank
    monkeypatch.setattr(pipeline.coarse, "reference_pose_bank", lambda p: full(p, count=8))
    cache = ReferenceProjectionCache()
    first, details = cache.initialize(image, image, depth, k, pose, k)
    assert not details["reference_cache"]["cache_hit"]
    second, details = cache.initialize(image, image, depth, k, pose, k)
    assert details["reference_cache"]["cache_hit"] and np.array_equal(first, second)
    changed, details = cache.initialize(image, np.ones_like(image), depth, k, pose, k)
    assert details["reference_cache"]["cache_hit"] and len(changed) == 0
    assert details["reason"] == "uniform_observation_on_fixed_cells"
    cache.initialize(image, image, depth+10., k, pose, k)
    assert not cache.last["cache_hit"]
