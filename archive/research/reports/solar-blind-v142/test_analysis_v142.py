from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import numpy as np

import pytest

spec = spec_from_file_location("saved_blind_analysis_v142", Path(__file__).with_name("analyze_saved.py"))
analysis = module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_component_bootstrap_keeps_repeated_terrain_together():
    labels = [dict(positive=True, component=g) for g in ("A", "B", "B", "C")]
    scores = [dict(methods=dict(active=dict(correct=i == 0), direct=dict(correct=False))) for i in range(4)]
    interval = analysis.positive_interval(labels, scores, "active", "direct")
    assert interval["enumerated_draws"] == 27
    assert interval["groups"] == ["A", "B", "C"]
    assert interval["interval_pp"] == pytest.approx([0, 78.33333333333329])


def test_negative_bootstrap_requires_both_terrain_endpoints():
    labels = [dict(positive=False, component="A", query_component="B")]
    scores = [dict(methods=dict(active=dict(raw_nonoverlap=True)))]
    result = analysis.negative_endpoint_interval(labels, scores, "active")
    assert result["enumerated_draws"] == 4
    assert result["undefined_no_edge_draws"] == 2
    assert result["interval_percent"] == [100, 100]
    scores[0]["methods"]["active"]["raw_nonoverlap"] = False
    assert analysis.negative_endpoint_interval(labels, scores, "active")["interval_percent"] == [0, 0]


def fixture():
    labels, raw, scores = [], dict(rows=[], query_truth_used=False, accepted_results=False), []
    for i in range(8):
        positive = i % 2 == 0
        labels.append(dict(id=str(i), source_case="fixture", positive=positive,
                           component="A", query_component="A" if positive else "B"))
        raw["rows"].append(dict(id=str(i), result=dict(methods={
            m: dict(online_seconds=float(i + 1)) for m in analysis.METHODS})))
        scores.append(dict(id=str(i), methods={m: dict(correct=positive, strict=False,
            wrong_positive=False, raw_nonoverlap=False, sun_proposal_error_deg=None,
            correspondence={str(t): dict(correct=1, visible=2, returned=3)
                            for t in (1, 2, 4, 8, 12)} if positive else {})
            for m in analysis.METHODS}))
    return raw, scores, labels


def test_summary_includes_failures_and_native_denominators():
    raw, scores, labels = fixture()
    scores[0]["methods"]["active"]["correct"] = False
    result = analysis.summarize(raw, scores, labels)
    assert result["methods"]["active"]["correct"] == 3
    assert result["methods"]["active"]["positive_pairs"] == 4
    assert result["methods"]["active"]["median_positive_online_seconds"] == 4
    assert result["methods"]["active"]["native_correspondences"]["1"] == dict(correct=4, visible=8, returned=12)
    assert result["active_minus_control"]["direct"]["paired_gain_pp"] == -25
    for field, value in (("query_truth_used", True), ("accepted_results", True)):
        changed = deepcopy(raw); changed[field] = value
        with pytest.raises(ValueError): analysis.summarize(changed, scores, labels)
    with pytest.raises(ValueError): analysis.summarize(raw, scores[:-1], labels)
    changed = deepcopy(scores); changed[0]["id"] = "other"
    with pytest.raises(ValueError): analysis.summarize(raw, changed, labels)
    changed = deepcopy(raw)
    changed["rows"][0]["result"]["methods"].pop("direct")
    with pytest.raises(ValueError): analysis.summarize(changed, scores, labels)


def test_empirical_quantile_handles_endpoints_and_singletons():
    assert analysis.quantile([4], .5) == 4
    assert analysis.quantile([4, 2], 0) == 2
    assert analysis.quantile([4, 2], 1) == 4
    assert analysis.quantile([4, 2], .25) == 2.5
    with pytest.raises(ValueError): analysis.quantile([], .5)


def test_independent_replay_rejects_changed_pose_and_correspondence_scores(monkeypatch):
    monkeypatch.setitem(sys.modules, "analyze_saved", analysis)
    replay_spec = spec_from_file_location("public_replay_v142_test", Path(__file__).with_name("replay_saved.py"))
    replay = module_from_spec(replay_spec); replay_spec.loader.exec_module(replay)
    record = dict(correct=False, strict=False, pose_returned=False, wrong_positive=False,
                  raw_nonoverlap=False, correspondence={}, pose_error=None)
    replay.compare_score(record, deepcopy(record))
    for changed in (dict(correct=True), dict(raw_nonoverlap=True),
                    dict(correspondence={"1": dict(correct=4)}),
                    dict(pose_error=dict(translation_m=1))):
        with pytest.raises(ValueError): replay.compare_score(record, dict(record, **changed))
    record["pose_error"] = dict(translation_m=2., rotation_deg=.3)
    changed = deepcopy(record); changed["pose_error"]["translation_m"] += .01
    with pytest.raises(ValueError): replay.compare_score(record, changed)


def test_posthoc_camera_diagnosis_keeps_absolute_pose_criterion(monkeypatch):
    monkeypatch.setitem(sys.modules, "analyze_saved", analysis)
    module_spec = spec_from_file_location("diagnose_saved_v142_test", Path(__file__).with_name("diagnose_saved.py"))
    diagnosis = module_from_spec(module_spec); module_spec.loader.exec_module(diagnosis)
    truth = np.eye(4); camera = truth.copy(); camera[0, 3] = 9.
    assert diagnosis.pose_error(camera, truth, 1000.)["correct"]
    camera[0, 3] = 11.
    assert not diagnosis.pose_error(camera, truth, 1000.)["correct"]
    angle = np.deg2rad(2.1); camera = truth.copy()
    camera[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    assert diagnosis.pose_error(camera, truth, 1000.)["rotation_deg"] == pytest.approx(2.1)
    assert not diagnosis.pose_error(camera, truth, 1000.)["correct"]
