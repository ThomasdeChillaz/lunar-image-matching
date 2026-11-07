import json

import numpy as np
import pytest

import capture_audit as audit


def synthetic(size=64):
    yy, xx = np.indices((size, size))
    intrinsics = np.array([[100., 0., (size - 1.) / 2], [0., 100., (size - 1.) / 2], [0., 0., 1.]])
    depth = np.full((size, size), 100.)
    position = np.stack((xx - intrinsics[0, 2], yy - intrinsics[1, 2], depth), axis=-1)
    view = dict(position=position, valid=np.ones((size, size), bool), axial_depth=depth,
                cam2world=np.eye(4), intrinsics=intrinsics)
    anchors = np.array([(x, y) for y in (8, 20, 43, 55) for x in (8, 20, 43, 55)])
    return view, anchors


def test_anchor_grid_is_immutable_y_major():
    assert audit.ANCHORS.tolist() == [[x, y] for y in (80, 197, 314, 431) for x in (80, 197, 314, 431)]
    assert audit.RADII == (16, 32, 64, 128)


def test_reference_arrays_cannot_change_pixel_centres_or_axial_depth():
    view, anchors = synthetic()
    world, valid = audit.native_frame(view, anchors, (64, 64))
    assert world.shape == (16, 3)
    assert valid.all()
    view["position"][8, 8, 0] += .5
    with pytest.raises(ValueError, match="native anchor centres"):
        audit.native_frame(view, anchors, (64, 64))


def test_visibility_retains_distinct_missing_occluded_discontinuity_states():
    view, anchors = synthetic()
    world, valid = audit.native_frame(view, anchors, (64, 64))
    valid[0] = False
    view["valid"][8, 20] = False
    view["axial_depth"][8:10, 43:45] = 90.
    view["axial_depth"][8:10, 55:57] = 120.
    view["axial_depth"][20, 8] = 90.
    target, reasons = audit.query_visibility(world, valid, view, (64, 64))
    np.testing.assert_allclose(target, anchors)
    assert reasons[:5].tolist() == [
        "invalid_reference", "query_depth_missing", "query_foreground_occludes",
        "query_depth_farther_than_reference_point", "query_depth_discontinuity_ambiguous",
    ]
    assert np.count_nonzero(reasons == "co_visible_depth_consistent") == 11


def test_euclidean_square_and_full_window_are_separate():
    view, anchors = synthetic()
    world, valid = audit.native_frame(view, anchors, (64, 64))
    target, reasons = audit.query_visibility(world, valid, view, (64, 64))
    shifted = np.eye(4)
    shifted[:2, 3] = 12.
    measured, _ = audit.proposal_capture(anchors, world, target, reasons, view["intrinsics"], shifted, (64, 64))
    r16 = measured["radii"]["16"]
    assert r16["euclidean_unrounded_center"]["count"] == 0
    assert r16["rounded_center_square"]["count"] == 9
    assert r16["rounded_center_square_full_window"]["count"] == 4
    assert measured["projection_states"]["proposal_center_outside_native_image"] == 7
    assert r16["rounded_center_square"]["of_fixed_anchors"] == 16
    assert r16["rounded_center_square"]["of_co_visible_anchors"] == 16


def test_distribution_excludes_oracle_union_and_reports_missing_ranked_slots():
    view, anchors = synthetic(96)
    anchors = anchors + 16
    world, valid = audit.native_frame(view, anchors, (96, 96))
    target, reasons = audit.query_visibility(world, valid, view, (96, 96))
    away = np.eye(4)
    away[0, 3] = 300.
    result = audit.arm_capture(anchors, world, target, reasons, view["intrinsics"],
                               np.array([away, np.eye(4)]), [0, 1], (96, 96))
    r16 = result["aggregate"]["16"]
    assert r16["first_distributed_six_rank"] == 2
    assert r16["oracle_union_across_proposals"]["count"] == 16
    assert result["missing_top_eight_slots"] == 6
    result = audit.arm_capture(anchors, world, target, reasons, view["intrinsics"],
                               np.array([away]), [], (96, 96))
    assert result["aggregate"]["16"]["first_distributed_six_rank"] is None
    assert result["aggregate"]["16"]["oracle_union_across_proposals"]["count"] == 0
    assert result["missing_top_eight_slots"] == 8


def test_invalid_completion_cannot_open_geometry(tmp_path, monkeypatch):
    called = []
    completion = tmp_path / "COMPLETE.json"
    completion.write_text(json.dumps({"complete": False}))
    monkeypatch.setattr(audit, "open_geometry", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="pinned bytes"):
        audit.audit(tmp_path, tmp_path, tmp_path / "protocol.json", tmp_path / "i.json",
                    tmp_path / "labels.json", "0" * 64)
    assert called == []


def test_old_cell_attachment_format_or_changed_order_is_rejected(tmp_path):
    row = dict(tag="exposure-0", reference_seed=0, query_seed=0, cell_pixels=32,
               candidate_count=3, common_supported_candidates=2)
    name = "cells-exposure-0-r0-q0-s32.npz"
    path = tmp_path / name
    arrays = {field: np.array([]) for field in ("query_fixed", "observed", "geometry_3d", "planar", "counts")}
    np.savez(path, indices=np.array([0, 2]), **arrays)
    pin = audit.sha(path)
    row["cell_evidence"] = {"path": name, "sha256": pin}
    assert audit.evidence_indices(tmp_path, row, {name: pin}, 3)[0] == [0, 2]
    row["scores"] = {arm: [0.5, None, 0.1] for arm in audit.ARMS}
    row["orders"] = {arm: [2, 0] for arm in audit.ARMS}
    audit.check_order(row, [0, 2], 3)
    row["orders"][audit.ARMS[0]].reverse()
    with pytest.raises(ValueError, match="proposal order"):
        audit.check_order(row, [0, 2], 3)
    np.savez_compressed(path, indices=np.array([0, 2]), **arrays)
    pin = audit.sha(path)
    row["cell_evidence"]["sha256"] = pin
    with pytest.raises(ValueError, match="array format"):
        audit.evidence_indices(tmp_path, row, {name: pin}, 3)
