import copy
import json
from pathlib import Path

import numpy as np
import pytest

import verify_warp_archive_v117 as verify


HERE = Path(__file__).resolve().parent


def camera():
    pose = np.eye(4)
    pose[1, 1] = pose[2, 2] = -1.
    pose[2, 3] = 1200.
    return pose


def test_real_roster_roles_without_opening_archived_arrays():
    read = lambda name: json.loads((HERE / name).read_bytes())
    inputs = read("inference-inputs.json")
    labels = read("scoring-inputs.json")
    verified = verify.inference_inputs(inputs)
    assert len(verify.scoring_inputs(labels, verified)) == 4
    leaked = copy.deepcopy(inputs)
    leaked["exposures"][0]["query_geometry"] = labels["exposures"][0]["query_geometry"]
    with pytest.raises(ValueError, match="reference geometry"):
        verify.inference_inputs(leaked)
    labels["exposures"][0]["query_geometry"] = verified[0]["reference_geometry"]
    with pytest.raises(ValueError, match="overlaps"):
        verify.scoring_inputs(labels, verified)


def test_rejects_incomplete_prediction_before_opening_scoring_files(tmp_path, monkeypatch):
    calls = []
    inputs_path = tmp_path / "inputs.json"
    predictions_path = tmp_path / "predictions.json"

    def pinned(path, *unused):
        calls.append(path)
        return None

    def reject(*unused):
        raise ValueError("bad prediction attachment")

    monkeypatch.setattr(verify, "pinned_json", pinned)
    monkeypatch.setattr(verify, "inference_inputs", lambda inputs: None)
    monkeypatch.setattr(verify, "read_bank", reject)
    with pytest.raises(ValueError, match="bad prediction attachment"):
        verify.verify(tmp_path, inputs_path, None, tmp_path / "labels.json", None,
                      predictions_path, None, tmp_path / "scores.json", None)
    assert calls == [inputs_path, predictions_path]


def test_independent_normalization_and_affine_gradient():
    y, x = np.indices((8, 8))
    plane = 0.012 + 2e-6 * x.ravel() + 7e-6 * y.ravel()
    np.testing.assert_allclose(verify.trend_projection(plane[None], 32)[0], plane, atol=1e-17, rtol=0)
    observed, usable = verify.centered_unit(plane[None])
    assert usable.tolist() == [True]
    assert verify.photometric_loss(observed, 5.5 * plane[None] + 1e-3, usable)[0] < 1e-22
    assert verify.photometric_loss(observed, 1. - plane[None], usable) == [None]


@pytest.mark.parametrize("size", (16, 32))
def test_procedural_ranker_saved_attachment_replays_and_tampering_fails(size):
    from warp_candidates import build_reference_cloud, rank_pose_bank

    pose = camera()
    intrinsics = np.array([[400., 0., 255.5], [0., 400., 255.5], [0., 0., 1.]])
    image = 1e-7 + 1e-6 * np.random.default_rng(116).random((32, 32)).repeat(16, 0).repeat(16, 1)
    depth = np.full_like(image, 1200.)
    depth[:256] -= 150.
    reference = build_reference_cloud(image, depth, intrinsics, pose)
    bank = np.repeat(pose[None], 4, axis=0)
    bank[:, 0, 3] = [0., 6., 12., 12.]
    evidence = {}
    row = rank_pose_bank(reference, image, intrinsics, bank, cell_pixels=size, evidence=evidence)
    replayed = verify.reconstruct_row(row, evidence)
    assert replayed["orders"]["geometry_3d"][0] == 0
    assert replayed["support_indices"] == list(range(4))
    assert replayed["loss_max_absolute_difference"] < 2e-12
    bad = copy.deepcopy(evidence)
    bad["counts"][0, 0] = 0
    with pytest.raises(ValueError, match="occupancy"):
        verify.reconstruct_row(row, bad)
    bad_row = copy.deepcopy(row)
    bad_row["orders"]["geometry_3d"].reverse()
    with pytest.raises(ValueError, match="sort"):
        verify.reconstruct_row(bad_row, evidence)


def test_npz_scope_and_hash(tmp_path):
    name = "cells-exposure-0-r0-q1-s32.npz"
    path = tmp_path / name
    names = ("query_fixed", "observed", "indices", "geometry_3d", "planar", "counts")
    np.savez(path, **{key: np.array([1]) for key in names})
    row = dict(tag="exposure-0", reference_seed=0, query_seed=1, cell_pixels=32,
               cell_evidence=dict(path=name, sha256=verify.sha(path)))
    assert set(verify.read_cells(tmp_path, row)) == set(names)
    row["cell_evidence"]["path"] = "../" + name
    with pytest.raises(ValueError, match="outside"):
        verify.read_cells(tmp_path, row)


def test_independent_reference_pose_bank_matches_procedural_implementation():
    from warp_candidates import reference_pose_bank

    pose = camera()
    actual = verify.reference_bank(pose)
    expected = reference_pose_bank(pose)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
    inside = np.eye(4)
    inside[1, 1] = inside[2, 2] = -1
    inside[:3, 3] = (119, 0, 1200)
    outside = inside.copy()
    outside[0, 3] = 121
    assert verify.basin_mask(np.array([inside, outside]), pose, 1200).tolist() == [True, False]


def test_primary_gate_uses_32_px_only_and_detects_control_explanation():
    rows = []
    for tag, r, q, size in verify.GRID:
        case = verify.CASES[verify.TAGS.index(tag)]
        raw_hit = 2 if size == 32 and tag in ("exposure-0", "exposure-1") else None
        rows.append(dict(case_id=case, cell_pixels=size, reference_seed=r, query_seed=q,
                         bank_contains_coarse_basin=True, support_contains_coarse_basin=True,
                         prior_first_top_eight_hit=None,
                         top_eight_first_hit=dict(geometry_3d=raw_hit, planar=None, gradient=None)))
    good = verify.stage_gate(rows)
    assert good["passed"]
    assert good["outcomes"]["16"][verify.CASES[0]]["top_eight_counts"]["geometry_3d"] == 0
    for row in rows:
        if row["cell_pixels"] == 32 and row["case_id"] == verify.CASES[0]:
            row["top_eight_first_hit"]["gradient"] = 8
    bad = verify.stage_gate(rows)
    assert not bad["passed"]
    assert not bad["checks"]["beats_both_photometric_controls"]
