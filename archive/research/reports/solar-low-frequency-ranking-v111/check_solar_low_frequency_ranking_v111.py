"""Tiny procedural verification of the blind low-frequency reranker."""

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from solar_low_frequency_ranking_v111 import rerank_lunar_candidates


def main(path):
    texture_grid = np.array([[0., 1., -2., 4.], [-3., 2., 1., -1.],
                             [1., -3., 4., 0.], [4., 1., 0., -2.]])
    texture = np.repeat(np.repeat(texture_grid, 32, axis=0), 32, axis=1)
    alternative = np.roll(texture, 32, axis=1)
    predictions = np.stack([texture, alternative, -texture])
    query = .003*texture + .02
    valid = np.ones_like(predictions, dtype=bool)
    order = [2, 1, 0]
    baseline = rerank_lunar_candidates(query, predictions, valid, order)
    assert baseline["retained_blocks"] == 16 and baseline["active_quadrants"] == [0, 1, 2, 3]
    assert baseline["primary"]["order"][0] == baseline["residual_after_plane"]["order"][0] == 0
    assert baseline["primary"]["scores"][0] < 1e-25

    # Apply unrelated positive gains and offsets to the observed scene and
    # each prediction without moving their normalized robust losses.
    affine_query = 300*query + 14.
    affine_predictions = predictions * np.array([.3, 7., 3.])[:, None, None] + np.array([11., -3., .1])[:, None, None]
    affine = rerank_lunar_candidates(affine_query, affine_predictions, valid, order)
    for mode in ("primary", "residual_after_plane"):
        assert baseline[mode]["order"] == affine[mode]["order"]
        np.testing.assert_allclose(baseline[mode]["scores"], affine[mode]["scores"], atol=1e-12, rtol=0)

    crop = valid.copy()
    crop[1, 64:, :] = False
    cropped = rerank_lunar_candidates(query, predictions, crop, order)
    for mode in ("primary", "residual_after_plane"):
        assert cropped[mode] == dict(order=order, reason="insufficient_shared_quadrant_support",
                                     scores=None, quadrant_losses=None)

    # Exactly three quadrants, with a half-covered block. Prediction values
    # and query pixels outside reference-defined support cannot leak into rank.
    partial = valid.copy()
    partial[:, 64:, 64:] = False
    partial[:, :16, :32] = False
    masked = rerank_lunar_candidates(query, predictions, partial, order)
    assert masked["active_quadrants"] == [0, 1, 2] and masked["quadrant_blocks"] == [4, 4, 4, 0]
    masked_query = np.where(partial.all(axis=0), query, np.nan)
    masked_predictions = np.where(partial, predictions, np.nan)
    assert rerank_lunar_candidates(masked_query, masked_predictions, partial, order) == masked
    masked_query[32, 32] = np.nan
    invalid = rerank_lunar_candidates(masked_query, masked_predictions, partial, order)
    assert invalid["primary"]["order"] == order
    assert invalid["primary"]["reason"] == "nonfinite_query_on_retained_support"
    assert invalid["common_pixels"] == masked["common_pixels"]

    tied_predictions = np.stack([texture, .7*texture + 3., -texture])
    tied = rerank_lunar_candidates(query, tied_predictions, valid, order)
    assert tied["primary"]["order"][:2] == tied["residual_after_plane"]["order"][:2] == [1, 0]

    y, x = np.indices(query.shape)
    plane = 2*x - 3*y + 300.
    gradient_predictions = np.stack([plane + texture, plane, -plane + texture])
    plane_only = rerank_lunar_candidates(plane, gradient_predictions, partial, order)
    assert plane_only["residual_after_plane"]["order"] == order
    assert plane_only["residual_after_plane"]["reason"] == "no_query_variation_after_plane"
    residual_discrimination = rerank_lunar_candidates(plane + 3*texture, gradient_predictions, partial, order)
    assert residual_discrimination["primary"]["order"][0] == 0
    assert residual_discrimination["residual_after_plane"]["order"][0] in (0, 2)
    assert residual_discrimination["residual_after_plane"]["order"].index(1) == 2

    flat = rerank_lunar_candidates(np.ones_like(query), predictions, valid, order)
    assert flat["primary"]["reason"] == "no_query_block_variation" and flat["primary"]["order"] == order
    assert tuple(inspect.signature(rerank_lunar_candidates).parameters) == (
        "query_luminance", "predictions", "reference_valid", "existing_order",
    )
    try:
        rerank_lunar_candidates(query, predictions, valid, order, query_depth=np.ones_like(query))
    except TypeError:
        pass
    else:
        raise AssertionError("Unexpected query truth input accepted")

    result = dict(
        schema="solar_low_frequency_ranking_fixture_v111", passed=True,
        procedural_only=True, archived_inputs_opened=0, query_truth_inputs=0, renderer_calls=0,
        calibration_or_final_access=False,
        source_sha256=sha256(Path(__file__).with_name("solar_low_frequency_ranking_v111.py").read_bytes()).hexdigest(),
        fixture_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        distinct_winner=baseline, affine_invariance=True, cropped_fallback=cropped,
        reference_mask_leakage_control=True, invalid_query_fallback=invalid, deterministic_tie=tied,
        gradient_only_control=plane_only, residual_discrimination=residual_discrimination,
        flat_query_fallback=flat,
    )
    with Path(path).open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
