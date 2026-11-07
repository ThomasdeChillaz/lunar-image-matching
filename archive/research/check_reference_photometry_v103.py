"""Synthetic metric checks and source-only role inventory verification."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

import diagnose_reference_photometry_v103 as audit
import lunar_known_sun_composite_v102 as method


def sun(azimuth, elevation=30.):
    a, e = np.deg2rad([azimuth, elevation])
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])


def encode(linear):
    return np.rint(255 * np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)).astype(np.uint8)


def test_light_reconstruction():
    lights = {i: sun(-82.5 + i * 7.5) for i in method.EVEN_LIBRARY}
    target_sun = sun(0.)
    plan = method.plan_supplied_sun(lights, target_sun)
    assert len(plan.component_indices) == 2
    y, x = np.mgrid[:512, :512]
    angle = np.sin(x / 23.) * 4.0 + np.cos(y / 12.) * 4.0
    tilt = .23
    surface = np.stack((tilt*np.cos(angle), tilt*np.sin(angle), np.full_like(angle, np.sqrt(1. - tilt**2))), axis=2)
    raw = encode(surface @ target_sun)
    target = method.linear_grayscale(raw)
    even = {i: method.linear_grayscale(encode(surface @ lights[i])) for i in set((plan.nearest_index, *plan.component_indices))}
    prediction = method.synthesize_linear_contrast(plan, even)
    result = audit.evaluate_pair(even[plan.nearest_index], prediction, target, raw)
    assert result["target_signal"]["raw_informative_patches"] > 0
    assert result["composite"][audit.PRIMARY] < result["nearest"][audit.PRIMARY]
    assert result["composite"]["raw_linear_mae"] < result["nearest"]["raw_linear_mae"]
    assert audit.evaluate_pair(target, None, target, raw)["composite"] is None


def test_low_signal_and_denominators():
    y, x = np.mgrid[:512, :512]
    low_raw = ((x + y) % 2).astype(np.uint8)
    linear = method.linear_grayscale(low_raw)
    score = audit.evaluate_pair(linear, linear, linear, low_raw)
    assert score["target_signal"]["raw_informative_patches"] == 0
    assert score["target_signal"]["stretched_informative_patches"] == 256
    assert score["composite"][audit.PRIMARY] is None
    assert score["composite"]["raw_linear_mae"] == 0
    assert score["composite"]["all_patch_derivative_mae_median"] == 0
    assert score["composite"]["stretched_texture_on_raw_low_signal"] == 256
    by_group = audit.group_summary([score, dict(score, composite=None)])
    assert by_group["angular_bracket_available"] == 1 and by_group["primary"]["paired"] == 0
    assert by_group["low_signal_patches_total"] == 256
    assert by_group["texture_after_stretch_on_same_raw_low_signal_patches"]["target"] == 256

    # For a 25×25 patch there are exactly 1,200 directional intra-patch edges.
    dx, dy = np.zeros((2, 25, 24), dtype=np.int16), np.zeros((2, 24, 25), dtype=np.int16)
    dx[0].flat[:59] = 2
    dx[1].flat[:60] = -2
    np.testing.assert_array_equal(audit.edge_support((dx, dy)), [False, True])


def test_reference_only_freeze():
    with patch.object(audit.Image, "open", side_effect=AssertionError("Metadata freeze cannot decode pixels")):
        plan = audit.metadata_freeze()
    assert plan["cases"] == 384
    views = plan["reference_views"]
    assert len(views) == 32 and set(Counter(v["terrain"] for v in views).values()) == {4}
    assert all(Path(entry["path"]).parent.name == "reference" for v in views for entry in v["images"])
    assert all((c["target"] in audit.ODD and c["nearest_index"] in method.EVEN_LIBRARY
                and set(c["component_indices"]) <= set(method.EVEN_LIBRARY))
               for v in views for c in v["cases"])
    brackets = Counter(c["target"] for v in views for c in v["cases"] if c["component_indices"])
    print("Metadata bracket coverage", dict(sorted(brackets.items())))


if __name__ == "__main__":
    test_light_reconstruction()
    test_low_signal_and_denominators()
    test_reference_only_freeze()
    print("PASS: analytic composite, low-signal no-primary guard, exact patch threshold, metadata-only reference-role provenance")
