import json

import numpy as np
import pytest

import finite_diagnostics as finite


def test_isolated_supported_peak_stays_rejected_and_serializes():
    rng = np.random.default_rng(20261134)
    image = rng.random((512, 512)).astype(np.float32)
    valid = np.zeros_like(image, bool)
    for y in (224, 256):
        for x in (224, 256):
            choices = rng.permutation(32*32)[:512]
            region = np.zeros((32, 32), bool)
            region.flat[choices] = True
            valid[y:y+32, x:x+32] = region
    center = np.array([256., 256.])
    before = finite.original_direction(image, valid, image, valid, center, center, subpixel=True)
    assert before["query_xy"] is None and np.isposinf(before["margin"])
    with pytest.raises(ValueError):
        json.dumps(before, allow_nan=False)
    after = finite.direction(image, valid, image, valid, center, center, subpixel=True)
    assert after["query_xy"] is None and after["reason"] == before["reason"]
    assert after["score"] == before["score"] and after["margin"] is None
    assert after["unavailable_peak_comparison"]
    json.dumps(after, allow_nan=False)


def test_finite_measurement_diagnostics_remain_byte_equivalent(monkeypatch):
    before = dict(query_xy=[1.25, 2.5], reason=None, score=.9, margin=.1)
    monkeypatch.setattr(finite, "original_direction", lambda *a, **kw: before)
    assert finite.direction() is before


def test_unexpected_nonfinite_accepted_match_is_an_error(monkeypatch):
    monkeypatch.setattr(finite, "original_direction", lambda *a, **kw: dict(query_xy=[1., 2.], margin=float("inf")))
    with pytest.raises(ValueError):
        finite.direction()
