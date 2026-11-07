import inspect
import hashlib
import os
import subprocess
import json
from dataclasses import replace
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beam_matcher_v143 as beam
from beam_policy_v143 import (choose_diverse_seeds, normalized_weights, predicted_observations,
    search_schedule, selection_scores, candidate_queue, common_field_budget)
from procedural_fixture_v138 import fixture


def pool_fixture():
    k = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    xy = beam.active.ANCHORS.copy().astype(float)
    world = 1000*np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(k).T
    old_policy = sys.modules[beam.active.acquire.__module__]
    pool = old_policy.Pool(xy, world, xy.copy(), np.zeros((len(xy), 9, 2)),
                          np.ones((len(xy), 9), bool), np.ones(len(xy), bool))
    return pool, k


def test_fourth_light_and_separated_camera_survive_fixed_seed_budget():
    pool, k = pool_fixture()
    cameras = np.repeat(np.eye(4)[None], 4, axis=0)
    cameras[:, 0, 3] = [0., 1., 40., -40.]
    scores = [[offset+v for v in (0., .001, .01, .02)] for offset in (0., .1, .2, .3, .4)]
    seeds = choose_diverse_seeds(scores, tuple(range(5)), cameras, pool.local_world, k, np.eye(4))
    assert len(seeds) == 8
    assert {r["light_id"] for r in seeds} == {0, 1, 2, 3}
    for light in range(4):
        assert [r["pose_index"] for r in seeds if r["light_id"] == light] == [0, 2]
    assert {r["light_id"] for r in choose_diverse_seeds(scores, tuple(range(5)), cameras,
        pool.local_world, k, np.eye(4), lights=2)} == {0, 1}


def test_total_patch_budget_counts_branches_and_never_grants_128_each():
    pool, _ = pool_fixture()
    for count, rounds in ((1, 128), (2, 64), (8, 16)):
        _, schedule = search_schedule(pool, 128, count, beam.frozen_cell_schedule)
        assert len(schedule) == rounds
        assert len(schedule)*count == 128
    _, schedule = search_schedule(pool, 31, 8, beam.frozen_cell_schedule)
    assert len(schedule)*8 == 24
    for budget in (-1, 129, 1.5, True):
        with pytest.raises(ValueError): search_schedule(pool, budget, 8, beam.frozen_cell_schedule)


def test_scarce_reference_coverage_keeps_one_common_attempt_budget():
    pool, _ = pool_fixture()
    masks = np.zeros((8, 256), bool)
    masks[:, :81] = True
    masks[0, :40] = False
    branches = [dict(pool=replace(pool, eligible=m)) for m in masks]
    queue = candidate_queue(branches)
    assert queue.eligible.sum() == 81
    common, capacity = common_field_budget(branches, 128)
    assert (common, capacity) == (80, 81)
    for h in (1, 8):
        _, rounds = search_schedule(queue, common, h, beam.frozen_cell_schedule)
        assert len(rounds)*h == common


def test_reference_light_disagreement_selects_patches_without_query_measurements():
    pool, k = pool_fixture()
    branch = dict(pool=pool, camera=np.eye(4), preview=np.zeros((256, 9)),
        preview_known=np.ones((256, 9), bool), extra=dict(available=np.ones((512, 512), bool)))
    other = dict(branch, preview=np.tile([1., -1., 0., 1., -1., 0., 1., -1., 0.], (256, 1)))
    scores, support = selection_scores([branch, other], k, np.eye(4), [0., 0., 1.], [.5, .5],
        "active", beam.frozen_separation)
    assert scores.max() > 0 and support.max() == 1
    missing = dict(other, extra=dict(available=np.zeros((512, 512), bool)))
    scores, support = selection_scores([branch, missing], k, np.eye(4), [0., 0., 1.], [.5, .5],
        "active", beam.frozen_separation)
    assert not scores.any() and not support.any()


def test_evidence_counts_missing_measurements_and_preserves_exploration():
    _, k = pool_fixture()
    center = np.array([255.5, 255.5]); point = np.array([0., 0., 1000.])
    side = 2*beam.joint.REACH+1
    fields = dict(records=[dict(local_world=point, center=center)],
        scores=np.full((1, side, side), .95), support=np.ones((1, side, side), bool))
    full = dict(fields=fields, attempted=1, camera=np.eye(4))
    mostly_missing = dict(full, attempted=10)
    assert beam.branch_evidence(mostly_missing, np.eye(4), k) < beam.branch_evidence(full, np.eye(4), k)
    weights = normalized_weights([0., -1000., -1000.])
    assert np.isclose(weights.sum(), 1.) and (weights > 0).all()


def test_failed_observations_keep_all_shared_light_branches_and_consume_budget(monkeypatch):
    pool, k = pool_fixture()
    branches = [dict(branch_id=i, light_id=i, pool=pool, extra={}, fields=dict(records=[]),
                     camera=np.eye(4), attempted=0, fit_calls=0) for i in range(3)]
    calls = []
    monkeypatch.setattr(beam.active, "acquire_field", lambda index, *args: (calls.append(index) or None, "failed"))
    monkeypatch.setattr(beam.joint, "optimize_camera", lambda *args: (None, dict(calls=0)))
    updated, events = beam.advance(branches, 7, None, None, np.eye(4), k)
    assert calls == [7, 7, 7]
    assert [(r["branch_id"], r["light_id"]) for r in updated] == [(0, 0), (1, 1), (2, 2)]
    assert [r["attempted"] for r in updated] == [1, 1, 1]
    assert all(not r["observation_present"] for r in events)


def test_invalid_controls_fail_before_catalogue_access_and_truth_is_not_an_input():
    for kwargs in (dict(field_budget=129), dict(light_budget=5), dict(policies=("other",))):
        with pytest.raises(ValueError): beam.run(None, None, None, None, None, None,
            reference_valid=None, height_direction_reference=[0., 0., 1.], **kwargs)
    assert not any(k in inspect.signature(beam.run).parameters for k in
                   ("query_depth", "true_query_pose", "true_query_sun", "query_sun"))


def test_generated_native_pose_uses_real_measurements_and_public_pose_stays_null():
    cv2.setNumThreads(1)
    cat, query, depth, k, poses = fixture()
    result = beam.run(cat, query, depth, k, np.eye(4), k,
        reference_valid=np.ones_like(depth, bool), height_direction_reference=[0., 0., -1.],
        policies=("single_uniform", "beam_active"), field_budget=64, poses=poses, research_output=True)
    raw = result["raw_research"]
    assert result["pose"] is None and raw["query_truth_used"] is False
    branch_ids = {h["branch_id"] for h in raw["initialization"]["photometric_hypotheses"]}
    assert 1 <= len(branch_ids) <= 8
    assert len(branch_ids)+len(raw["initialization"]["unavailable_before_acquisition"]) == len(raw["initialization"]["seeds"])
    for name, row in raw["methods"].items():
        assert row["acquisition"]["attempted_native_fields"] == (64//len(branch_ids))*len(branch_ids)
        assert row["pose"] is not None
        assert np.linalg.norm(np.asarray(row["pose"])[:3, 3]-[20., 0., 0.]) < 2.
        points = np.asarray(row["reference_xy"]); target = np.asarray(row["query_xy"])
        assert len(points) >= 6 and np.median(np.linalg.norm(target-(points-[8., 0.]), axis=1)) < .3
        assert row["light_id"] == 8
        if name == "beam_active":
            assert set(row["acquisition"]["retained_branch_ids"]) == branch_ids
            for step in row["acquisition"]["trace"]:
                assert {b["branch_id"] for b in step["branches"]} == branch_ids
                assert len(step["weights_after"]) == len(branch_ids)
    json.dumps(result, allow_nan=False)


def test_blank_query_does_not_create_camera_or_lighting_evidence():
    cat, query, depth, k, poses = fixture()
    result = beam.run(cat, np.zeros_like(query), depth, k, np.eye(4), k,
        reference_valid=np.ones_like(depth, bool), height_direction_reference=[0., 0., -1.],
        field_budget=64, poses=poses, research_output=True)
    assert all(row["pose"] is None and row["acquisition"]["attempted_native_fields"] == 0
               for row in result["raw_research"]["methods"].values())


def test_generated_lower_ranked_light_camera_can_win_after_measurement():
    from smoke_beam_v143 import control
    result = control()
    assert result["passed"]
    assert result["deliberately_injected_control_hypotheses"]
    assert not result["lunar_images_opened"]
    for name in ("beam_uniform", "beam_active"):
        assert result["rows"][name]["retained_branch_ids"] == [0, 1]
        assert result["rows"][name]["chosen_light_id"] == 8


def test_command_rejects_query_truth_and_checks_input_bytes(tmp_path):
    from localize_beam import checked_file, validate_case
    example = dict(id="generated", reference_geometry={}, catalogue=[dict(light_id=0, sun_world=[1, 0, 0], image={})],
                   query_image={}, query_intrinsics=np.eye(3).tolist())
    validate_case(example)
    for field in ("query_geometry", "query_pose", "query_sun", "scoring_labels"):
        with pytest.raises(ValueError): validate_case(dict(example, **{field: "forbidden"}))
    path = tmp_path/"payload"; path.write_bytes(b"controlled test")
    record = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bit_depth=16)
    assert checked_file(record) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError): checked_file(record)


def test_all_seven_experiment_arms_have_actual_generated_forward_passes():
    import engine_beam_v143 as engine
    catalogue, query, depth, k, poses = fixture()
    result = engine.models(catalogue, query, depth, k, np.eye(4), k,
        np.ones_like(depth, bool), engine.frozen.setup(), 0, control_poses=poses)
    assert tuple(result["methods"]) == engine.METHODS
    assert result["query_truth_used"] is False
    for name, row in result["methods"].items():
        assert row["pose"] is not None, name
        assert np.linalg.norm(np.asarray(row["pose"])[:3, 3]-[20., 0., 0.]) < 2., name
        if name.endswith("128"):
            assert row["acquisition"]["attempted_native_fields"] == result["common_effective_native_fields"]


def test_command_help_works_outside_repository_without_pythonpath(tmp_path):
    command = Path(__file__).with_name("localize_beam.py").resolve()
    environment = dict(os.environ); environment.pop("PYTHONPATH", None)
    output = subprocess.check_output([sys.executable, str(command), "--help"], cwd=tmp_path, env=environment, text=True)
    assert "--research-output" in output and "--field-budget 0..128" in output
