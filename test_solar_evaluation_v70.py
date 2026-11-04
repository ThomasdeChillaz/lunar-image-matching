"""Procedural contract/geometry tests only: no lunar pixels, neural forwards or fitting."""
import copy
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

from solar_evaluation_v70 import (Backend, CHECKPOINT_STEPS, InferenceInput, LIGHT_PAIRS, additional_profiles,
    aggregate_rows, checkpoint_ranking, common_pose, evaluate_case,
    paired_component_bootstrap, principal_profiles, score_arrays,
    validate_manifest, validate_profiles)
from run_solar_evaluation_v70 import cache_index, measured_cost


def procedural_manifest():
    units = [dict(unit_index=i, component=f"train{i//4}", split="training", eligible=True) for i in range(64)]
    units += [dict(unit_index=64+i, component=f"select{i//4}", split="model_selection", eligible=True) for i in range(32)]
    audit = dict(passed=True, complete_dataset=True, expected_units=96, completed_units=96,
        oracle_success=1., oracle_pose_directions=192, units=units)
    cases = []
    for u in units[64:]:
        i = u["unit_index"]; g = (i-64)//4; q = 64+((i-64+4) % 32)
        for a, b in LIGHT_PAIRS:
            c = dict(case_id=f"{i}/{a}/{b}", split="model_selection", positive=True,
                unit_index=i, query_unit_index=i, component=u["component"], query_component=u["component"], light_a=a, light_b=b)
            cases.append(c)
            qg = (q-64)//4
            cases.append(dict(c, case_id=c["case_id"]+"/negative", positive=False, query_unit_index=q,
                query_component=f"select{qg}", matched_positive_case=c["case_id"],
                nonoverlap_certificate=dict(frame="body_xy", full_image_footprints=True,
                    reference_bounds=[g*1000, 0, g*1000+100, 100], query_bounds=[qg*1000, 0, qg*1000+100, 100])))
    return dict(schema_version=70, split="model_selection", calibration_or_final_access=False,
        terrain_groups=[f"select{i}" for i in range(8)], footprint_frame="body_xy", cases=cases), audit


def profiles_for_selection():
    return [dict(name=f"step_{i}", backend="solar", checkpoint=f"{i}.pt", training_step=i, training_seed=20261101,
        configuration=dict(model_version="solar-experimental-v63", checkpoint_sha256=str(i), solar_mode="shared",
            response=True, active=True, pooled=False, budget=128, policy="active")) for i in CHECKPOINT_STEPS]


def test_complete_original_group_camera_lighting_and_negative_grid():
    m, a = procedural_manifest()
    assert len(validate_manifest(m, a)) == 384
    broken = copy.deepcopy(m); broken["cases"] = broken["cases"][2:]
    with pytest.raises(ValueError, match="All eligible cameras"):
        validate_manifest(broken, a)
    # Deliberately failed oracle direction remains denominator, still eligible.
    a["oracle_success"] = 191/192
    assert len(validate_manifest(m, a)) == 384
    a["oracle_pose_directions"] -= 1
    with pytest.raises(ValueError, match="denominator"):
        validate_manifest(m, a)


def test_negative_requires_both_endpoints_and_conservative_full_footprints():
    m, a = procedural_manifest()
    m["cases"][1]["query_component"] = m["cases"][1]["component"]
    with pytest.raises(ValueError, match="endpoints"):
        validate_manifest(m, a)
    m, a = procedural_manifest(); m["cases"][1]["nonoverlap_certificate"]["query_bounds"] = [50, 0, 150, 100]
    with pytest.raises(ValueError, match="Nonoverlap"):
        validate_manifest(m, a)


def test_matched_eight_ablations_cannot_change_weights_or_budget():
    ps = principal_profiles("sh.pt", "pool.pt", "s", "p")
    validate_profiles(ps, "principal_ablations")
    assert len(ps) == 9
    ps[0]["checkpoint"] = "different.pt"
    with pytest.raises(ValueError, match="share the exact checkpoint"):
        validate_profiles(ps, "principal_ablations")


def test_independent_sun_and_all_policy_budgets_are_explicit_controls():
    ps = additional_profiles("fixed.pt", "fixed")
    validate_profiles(ps, "additional_controls")
    assert ps[0]["configuration"]["solar_mode"] == "independent"
    assert {(p["configuration"]["policy"], p["configuration"]["budget"]) for p in ps[1:]} == {
        (policy, budget) for policy in ("active", "uniform", "appearance", "geometry") for budget in (32, 64, 128)}
    ps = principal_profiles("sh.pt", "pool.pt", "s", "p"); ps[2]["configuration"]["budget"] = 64
    with pytest.raises(ValueError, match="share the exact checkpoint"):
        validate_profiles(ps, "principal_ablations")


def test_checkpoint_selection_uses_raw_poses_then_precision_then_earlier():
    ps = profiles_for_selection(); rows = []
    for p in ps:
        s = p["training_step"]
        rows.append(dict(profile=p["name"], case_id="positive", track="estimated_sun", positive=True,
            metrics=dict(correct=s in (500, 1000, 1500), correspondence={"2": dict(correct=9 if s in (1000, 1500) else 2)})))
        rows.append(dict(profile=p["name"], case_id="negative", track="estimated_sun", positive=False,
            metrics=dict(correct=False, correspondence={})))
    selected = checkpoint_ranking(rows, ps, ["positive", "negative"])
    assert selected["selected"]["step"] == 1000
    assert not selected["automatically_promoted"]
    with pytest.raises(ValueError, match="complete, identical"):
        checkpoint_ranking(rows[:-1], ps, ["positive", "negative"])


def test_failed_poses_occlusion_and_wrong_negative_outputs_count():
    p = np.array([[10., 10.], [20., 20.], [30., 30.]])
    truth = dict(positive=True, query_cam2world=np.eye(4), query_xy=p.copy(), visible=np.array([True, False, True]))
    q = p+np.array([[.9, 0], [0, 0], [2.1, 0]])
    score = score_arrays(p, q, None, truth, 100.)
    assert not score["correct"] and not score["pose_returned"]
    assert score["correspondence"]["1"] == dict(correct=1, co_visible=2, returned=3)
    assert score["correspondence"]["2"]["correct"] == 1
    assert score["correspondence"]["4"]["correct"] == 2
    neg = score_arrays(p, q, np.eye(4), dict(positive=False), 100.)
    assert neg["raw_nonoverlap"] and not neg["correct"] and neg["correspondence"] == {}
    with pytest.raises(ValueError, match="rigid transform"):
        score_arrays(p, q, np.ones((4, 4)), truth, 100.)


def test_primary_inference_boundary_has_no_query_truth_fields():
    assert {f.name for f in fields(InferenceInput)} == {"reference", "query", "geometry", "query_intrinsics", "reference_sun", "known_query_sun", "solar_lookup"}
    forbidden = {"query_pose", "query_depth", "scoring_labels", "truth", "positive"}
    assert not forbidden & {f.name for f in fields(InferenceInput)}


def test_solar_backend_passes_reference_sun_common_solver_and_lazy_lookup_only():
    inp = procedural_input(); callback = object()
    inp = InferenceInput(inp.reference, inp.query, inp.geometry, inp.query_intrinsics, np.array([0., 1., 0.]), None, callback)
    observed = {}
    class FakeSolar:
        def localize(self, reference, query, geometry, k, **kwargs):
            observed.update(kwargs)
            pose, info = kwargs["final_estimator"](np.empty((0, 2)), np.empty((0, 2)))
            assert pose is None and info["iterations_requested"] == 2000
            return dict(pose=None, reference_xy=[], query_xy=[])
    backend = Backend.__new__(Backend)
    backend.profile = principal_profiles("sh.pt", "pool.pt", "s", "p")[-2]
    backend.model = FakeSolar(); backend.infer(inp)
    assert observed["known_sun"] is None and observed["solar_lookup"] is callback
    assert np.array_equal(observed["reference_sun"], [0, 1, 0])
    assert not {"query_pose", "query_depth", "query_truth"} & set(observed)


def procedural_input():
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    y, x = np.mgrid[:512, :512]; depth = 10.+.3*np.sin(x/40)*np.cos(y/37)
    geometry = SimpleNamespace(depth=depth, intrinsics=k, cam2world=np.eye(4))
    return InferenceInput(np.zeros((512, 512), np.uint8), np.zeros((512, 512), np.uint8), geometry, k, np.array([1., 0, 0]))


def test_actual_common_pose_recovers_procedural_native_geometry():
    from lunar_geometry_v5 import map_points_z
    inp = procedural_input()
    yy, xx = np.meshgrid(np.linspace(80, 430, 5), np.linspace(70, 440, 5)); p = np.column_stack([xx.ravel(), yy.ravel()])
    world, use = map_points_z(p, inp.geometry.depth, inp.geometry.intrinsics, inp.geometry.cam2world)
    truth = np.eye(4); truth[:3, 3] = [.15, -.05, .02]
    camera = world-truth[:3, 3]; h = camera@inp.query_intrinsics.T; q = h[:, :2]/h[:, 2, None]
    pose, info = common_pose(p, q, inp)
    assert use.all() and pose is not None and np.max(np.abs(pose-truth)) < 1e-6
    assert info["iterations_requested"] == 2000 and info["reprojection_threshold_native_px"] == 3


def test_scoring_labels_are_not_decoded_before_backend_returns_and_failures_retained():
    inp = procedural_input(); order = []; pose = np.eye(4); p = np.array([[100., 100.], [200., 200.]])
    class Fake:
        def infer(self, data):
            assert isinstance(data, InferenceInput); order.append("inference")
            return dict(reference_xy=p, query_xy=p, pose=pose)
    def labels(case, points):
        assert order == ["inference"]; order.append("score")
        return dict(positive=True, query_cam2world=pose, query_xy=points.copy(), visible=np.ones(len(points), bool)), "procedural"
    row, arrays = evaluate_case(Fake(), inp, dict(positive=True), labels, lambda p, q, i: (pose, dict(reason=None)))
    assert row["metrics"]["correct"] and order == ["inference", "score"]
    class Broken:
        def infer(self, data):
            order.append("inference"); raise RuntimeError("procedural unavailable model")
    order.clear()
    row, arrays = evaluate_case(Broken(), inp, dict(positive=True), labels)
    assert row["inference_failure"]["type"] == "RuntimeError" and not row["metrics"]["correct"] and arrays["pose"].shape == (0, 4)


def test_common_estimator_disagreement_is_fatal_not_hidden_as_a_model_miss():
    inp = procedural_input()
    class Fake:
        def infer(self, data): return dict(reference_xy=[], query_xy=[], pose=np.eye(4))
    with pytest.raises(ValueError, match="differs from common estimator"):
        evaluate_case(Fake(), inp, dict(positive=False), estimator=lambda p, q, i: (None, {}))


def test_pilot_cost_requires_every_profile_and_counts_complete_case_time():
    profiles = [dict(name="a"), dict(name="b")]
    rows = [dict(profile=name, track="estimated_sun", case_calendar_seconds=2., profile_load_seconds=4., inference_failure=None) for name in ("a", "b") for _ in range(4)]
    cost = measured_cost(rows, 100, ["estimated_sun"], profiles)
    assert cost["conservative_total_seconds"] == 612
    with pytest.raises(ValueError, match="four successful measured"):
        measured_cost(rows[:-1], 100, ["estimated_sun"], profiles)


def test_completed_case_cache_requires_immutable_row_and_arrays(tmp_path):
    from solar_evaluation_v70 import digest, sha
    arrays = tmp_path / "arrays.npz"; np.savez(arrays, reference_xy=np.zeros((0, 2)))
    row_path = tmp_path / "row.json"; row = dict(saved_arrays=str(arrays), saved_arrays_sha256=sha(arrays))
    row_path.write_text(json.dumps(row))
    identity = digest(dict(checkpoint="one", case="same", track="estimated_sun"))
    assert identity != digest(dict(checkpoint="two", case="same", track="estimated_sun"))
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(dict(complete=True, identity=identity, row_path=str(row_path), row_sha256=sha(row_path), arrays_path=str(arrays), arrays_sha256=sha(arrays))))
    protocol = dict(case_cache_seals={str(seal_path): sha(seal_path)})
    assert identity in cache_index(protocol)
    row_path.write_text("{}")
    with pytest.raises(ValueError, match="payload changed"):
        cache_index(protocol)


def test_paired_bootstrap_keeps_negative_endpoint_dependence_and_replays():
    rows = []
    for profile, better in (("candidate", True), ("control", False)):
        for i in range(4):
            for positive in (True, False):
                rows.append(dict(profile=profile, case_id=f"{i}/{positive}", track="estimated_sun", component=str(i),
                    query_component=str(i if positive else (i+1) % 4), positive=positive,
                    metrics=dict(correct=positive and better, wrong_positive=False, raw_nonoverlap=not positive and not better)))
    a = paired_component_bootstrap(rows, "candidate", "control", repetitions=50)
    b = paired_component_bootstrap(rows, "candidate", "control", repetitions=50)
    assert a == b and a["raw_correct_pose_difference"] == [1., 1., 1.]
    assert a["raw_nonoverlap_difference"] == [-1., -1., -1.]
    assert a["negative_resampling"] == "product of both endpoint terrain multiplicities"


def test_independent_saved_array_replay_detects_changed_metrics(tmp_path):
    from verify_solar_evaluation_v70 import replay
    from solar_evaluation_v70 import sha
    from lunar_geometry_v5 import median_reference_range_z
    inp = procedural_input(); pxy = np.array([[100., 100.], [200., 200.]])
    geom = tmp_path / "geometry.npz"; labels = tmp_path / "labels.npz"
    np.savez(geom, depth=inp.geometry.depth, intrinsics=inp.geometry.intrinsics)
    np.savez(labels, reference_xy=pxy, query_xy=pxy, visible=np.array([True, True]), query_cam2world=np.eye(4))
    rows = []; cases = []
    for positive in (True, False):
        case = dict(case_id=str(positive), positive=positive, component="a", query_component="a" if positive else "b",
            reference_geometry=str(geom), scoring_labels=str(labels), input_sha256={str(geom): sha(geom), str(labels): sha(labels)})
        cases.append(case)
        truth = dict(positive=positive, query_cam2world=np.eye(4), query_xy=pxy, visible=np.ones(2, bool))
        metrics = score_arrays(pxy, pxy, np.eye(4), truth, median_reference_range_z(inp.geometry.depth, inp.geometry.intrinsics))
        path = tmp_path / (str(positive)+".npz")
        np.savez(path, reference_xy=pxy, query_xy=pxy, pose=np.eye(4), scoring_query_xy=pxy if positive else np.empty((0, 2)),
            scoring_visible=np.ones(2, bool) if positive else np.empty(0, bool), scoring_query_cam2world=np.eye(4) if positive else np.empty((0, 4)))
        rows.append(dict(case_id=str(positive), profile="dummy", track="estimated_sun", component=case["component"], query_component=case["query_component"],
            positive=positive, saved_arrays=str(path), saved_arrays_sha256=sha(path), metrics=metrics,
            label_method="exact_frozen_projected_labels" if positive else "nonoverlap_full_footprint_certificate"))
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps(dict(cases=cases)))
    protocol = tmp_path / "protocol.json"; protocol.write_text(json.dumps(dict(manifest=str(manifest), purpose="additional_controls", case_ids=["True", "False"],
        profiles=[dict(name="dummy")], tracks=["estimated_sun"], source_sha256={name: sha(Path(__file__).with_name(name)) for name in ("verify_solar_evaluation_v70.py", "solar_evaluation_v70.py")})))
    summary = dict(complete=True, protocol_sha256=sha(protocol), manifest_sha256=sha(manifest), rows=rows)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "COMPLETE.json").write_text(json.dumps(dict(external_watchdog_passed=True, summary_sha256=sha(tmp_path / "summary.json"))))
    result = replay(tmp_path, protocol, tmp_path / "replayed.json")
    assert result["passed"] and result["replayed_rows"] == 2
    summary["rows"][0]["metrics"]["correspondence"]["1"]["correct"] -= 1
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "COMPLETE.json").write_text(json.dumps(dict(external_watchdog_passed=True, summary_sha256=sha(tmp_path / "summary.json"))))
    with pytest.raises(ValueError, match="Correspondence arithmetic"):
        replay(tmp_path, protocol, tmp_path / "replayed_bad.json")


def test_external_watchdog_terminates_worker_instead_of_accepting_overrun(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    import solar_evaluation_protocol_v74 as freeze
    import signal
    import run_solar_evaluation_v70 as runner
    from solar_evaluation_v70 import sha
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(dict(accounting_complete=True, remaining_baseline_wall_seconds=1, remaining_training_wall_seconds=0)))
    protocol = tmp_path / "protocol.json"; protocol.write_text("{}")
    p = dict(output=str(tmp_path / "run"), compute_ledger=str(ledger), compute_ledger_sha256=sha(ledger),
        reservation_journal=str(tmp_path / "reservations.json"), max_calendar_seconds=1,
        absolute_deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    monkeypatch.setattr(runner, "admission", lambda *args: (p, {}, []))
    # This constructed process checks timeout cleanup; schema74 source/budget
    # admission has dedicated tests and must not hide the watchdog assertion.
    monkeypatch.setattr(freeze, "validate_runtime", lambda *args: dict())
    calls = []
    class FakeChild:
        pid = 99999999
        returncode = None
        def __init__(self, *args, **kwargs): self.args = args
        def poll(self): return self.returncode
        def wait(self, timeout=None): self.returncode = -15; return self.returncode
    clock = iter([0., 2., 2.])
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: next(clock, 2.), sleep=lambda x: None))
    monkeypatch.setattr(runner.subprocess, "Popen", FakeChild)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(TimeoutError, match="External evaluation watchdog"):
        runner.run(str(protocol), "procedural", sys.executable)
    assert calls == [(99999999, signal.SIGTERM)]
    assert json.loads((tmp_path / "run" / "FAILED.json").read_text())["automatic_restart"] is False
