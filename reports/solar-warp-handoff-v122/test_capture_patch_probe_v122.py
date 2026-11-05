import numpy as np
import pytest

from capture_patch_probe_v122 import PATCH, fractional_mask, prepare_support, score_after_seal, support_digest


def one(centre, target, *, size=320, radius=128, reference_mask=None, query_mask=None):
    all_valid = np.ones((size, size), dtype=bool)
    reference_mask = all_valid if reference_mask is None else reference_mask
    query_mask = all_valid if query_mask is None else query_mask
    prepared = prepare_support([centre], [True], reference_mask, query_mask,
                               reference_xy=[[128, 128]], radius=radius)
    return score_after_seal(prepared, support_digest(prepared), lambda: ([target], [True]))


def test_frozen_fractional_mask_matches_v120_taps():
    mask = np.ones((64, 64), dtype=bool)
    mask[15, 18] = False
    radiance = np.ones(mask.shape)
    for centre in ([31., 32.], [31.25, 32.], [31., 32.75], [31.25, 32.75]):
        centre = np.asarray(centre)
        observed, anchor = fractional_mask(mask, centre)
        _, expected, expected_anchor = PATCH._lattice(radiance, mask, centre)
        np.testing.assert_array_equal(observed, expected)
        np.testing.assert_array_equal(anchor, expected_anchor)


def test_wide_geometry_witness_and_same_code_narrow_exclusion():
    centre = [144.25, 144.75]
    target = [217.35, 123.65]
    wide = one(centre, target)
    narrow = one(centre, target, radius=16)
    assert wide["photometric_feasibility_evaluated"] is False
    assert wide["reachable_anchors"] == 1
    witness = wide["rows"][0]["witness"]
    assert witness["possible_query_xy"] == [217.25, 123.75]
    assert witness["forward_native_xy"] == [73, -21]
    assert witness["truth_error_px"] == pytest.approx(np.sqrt(.02))
    assert witness["round_trip_px"] <= PATCH.ROUND_TRIP_MAX_PX
    assert narrow["rows"][0]["reason"] == "no_truth_lattice_offset_within_radius"


def test_centre_only_gate_counts_out_of_frame_patch_but_probe_excludes_it():
    centre = [132.25, 170.25]
    target = [8.25, 170.25]
    rounded = np.rint(centre)
    assert (rounded - 128 >= 0).all() and (rounded + 128 <= 319).all()
    assert np.max(np.abs(target - rounded)) < 128
    result = one(centre, target)
    assert result["reachable_anchors"] == 0
    assert result["rows"][0]["reason"] == "no_feasible_forward_geometry_path"


def test_reverse_requires_full_fixed_mask_and_supported_rivals():
    reference = np.zeros((320, 320), dtype=bool)
    reference[112:176, 112:176] = True
    result = one([144., 144.], [184., 144.], reference_mask=reference)
    assert result["reachable_anchors"] == 0
    assert result["rows"][0]["reason"] == "no_feasible_reciprocal_geometry_path"


def test_support_is_sealed_before_truth_and_counts_distributed_anchors():
    mask = np.ones((256, 256), dtype=bool)
    centres = np.array([[64., 64.], [96., 96.], [160., 64.], [64., 160.], [160., 160.], [192., 192.],
                        [96., 160.]])
    visible = np.array([True] * 6 + [False])
    prepared = prepare_support(centres, visible, mask, mask, reference_xy=centres, radius=16)
    sealed = support_digest(prepared)
    truth = centres.copy()
    calls = []

    def load():
        calls.append(True)
        return truth, np.ones(len(truth), dtype=bool)

    with pytest.raises(ValueError, match="pre-scoring seal"):
        score_after_seal(prepared, "wrong", load)
    assert not calls
    mask[:] = False
    centres[:] = 0
    report = score_after_seal(prepared, sealed, load)
    assert calls == [True]
    assert report["reachable_anchors"] == 6 and report["distributed_six_upper_bound"] is True
    assert report["rows"][-1]["reason"] == "unsupported_reference_projection"


def test_witness_can_cross_truth_quadrant_boundary():
    mask = np.ones((256, 256), dtype=bool)
    reference = [[64, 64], [96, 96], [160, 64], [64, 160], [160, 160], [192, 192]]
    centres = np.array([[64., 64.], [96., 96.], [128.2, 64.], [64., 160.], [96., 160.], [110., 192.]])
    truth = centres.copy()
    truth[2, 0] = 127.8
    prepared = prepare_support(centres, [True]*6, mask, mask, reference_xy=reference, radius=16)
    report = score_after_seal(prepared, support_digest(prepared), lambda: (truth, [True]*6))
    assert report["true_query_quadrants"] == [3, 0, 3, 0]
    assert report["rows"][2]["witness"]["possible_query_xy"] == [128.2, 64.]
    assert report["query_quadrant_union_upper_bound"] == [0, 1, 2]
    assert report["reachable_anchors"] == 6 and report["distributed_six_upper_bound"] is True
