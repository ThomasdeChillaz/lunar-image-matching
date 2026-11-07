from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest

spec = spec_from_file_location("saved_beam_analysis_v144", Path(__file__).with_name("analyze_saved.py"))
analysis = module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    labels, raw, scores = [], dict(rows=[], query_truth_used=False, accepted_results=False), []
    for i in range(8):
        positive = i % 2 == 0
        labels.append(dict(id=str(i), source_case="generated", positive=positive,
                           component="A", query_component="A" if positive else "B"))
        acquisition = dict(attempted_native_fields=128, rounds=16, retained_branch_ids=list(range(8)), pruning=False)
        raw["rows"].append(dict(id=str(i), result=dict(
            methods={m: dict(online_seconds=float(i+1), acquisition=deepcopy(acquisition)) for m in analysis.METHODS},
            initialization=dict(photometric_hypotheses=[{}]*8), common_effective_native_fields=128,
            shared_reference_candidate_capacity=200)))
        scores.append(dict(id=str(i), methods={m: dict(correct=positive, strict=False,
            pose_returned=positive, wrong_positive=False, raw_nonoverlap=False, sun_proposal_error_deg=None,
            correspondence={str(t): dict(correct=1, visible=2, returned=3)
                            for t in (1, 2, 4, 8, 12)} if positive else {})
            for m in analysis.METHODS}))
    return raw, scores, labels


def test_recovery_denominator_includes_failed_poses_and_strong_single_control():
    raw, scores, labels = fixture()
    scores[0]["methods"]["beam_active128"].update(correct=False, wrong_positive=True)
    result = analysis.summarize(raw, scores, labels)
    assert result["methods"]["beam_active128"]["correct"] == 3
    assert result["methods"]["beam_active128"]["positive_pairs"] == 4
    assert result["methods"]["beam_active128"]["raw_recovery_percent"] == 75
    assert result["methods"]["beam_active128"]["wrong_positive"] == 1
    assert result["active_gain_over_strongest_nonbeam_pp"] == -25
    assert "single_uniform128" in result["strongest_nonbeam_controls_by_raw_count"]
    assert result["active_minus_control"]["single_uniform128"]["paired_gain_pp"] == -25
    assert result["methods"]["beam_active128"]["native_correspondences"]["2"] == dict(correct=4, visible=8, returned=12)
    assert result["accepted_pose_study"] is False


def test_seven_arm_summary_rejects_changed_case_roster_and_inference_scope():
    raw, scores, labels = fixture()
    for field in ("accepted_results", "query_truth_used"):
        changed = deepcopy(raw)
        changed[field] = True
        with pytest.raises(ValueError):
            analysis.summarize(changed, scores, labels)
    with pytest.raises(ValueError):
        analysis.summarize(raw, scores[:-1], labels)
    changed = deepcopy(scores)
    changed[0]["id"] = "different"
    with pytest.raises(ValueError):
        analysis.summarize(raw, changed, labels)
    changed = deepcopy(raw)
    changed["rows"][0]["result"]["methods"].pop("single_uniform128")
    with pytest.raises(ValueError):
        analysis.summarize(changed, scores, labels)


def test_branch_oracle_diagnosis_uses_absolute_pose_criterion(monkeypatch):
    monkeypatch.setitem(sys.modules, "analyze_saved", analysis)
    spec = spec_from_file_location("branch_diagnosis_v144_test", Path(__file__).with_name("diagnose_saved.py"))
    diagnosis = module_from_spec(spec)
    spec.loader.exec_module(diagnosis)
    truth = np.eye(4)
    pose = truth.copy()
    pose[0, 3] = 9
    assert diagnosis.pose_error(pose, truth, 1000)["correct"]
    pose[0, 3] = 11
    assert not diagnosis.pose_error(pose, truth, 1000)["correct"]
    angle = np.deg2rad(2.1)
    pose = truth.copy()
    pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    assert diagnosis.pose_error(pose, truth, 1000)["rotation_deg"] == pytest.approx(2.1)
    assert not diagnosis.pose_error(pose, truth, 1000)["correct"]
    assert diagnosis.pose_error(None, truth, 1000) is None
