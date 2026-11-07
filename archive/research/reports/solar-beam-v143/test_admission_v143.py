from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_beam_v143 as study
from replay_beam_v143 import verify_accounting


def grant():
    return dict(schema="explicit_user_authorized_beam_comparison_v143", user_approved=True, accounting_complete=True,
        remaining_baseline_wall_seconds=1800, remaining_training_wall_seconds=0, paid_compute=False,
        threads_per_scientific_job=2, calibration_or_final_access=False, protocol_sha256="b"*64,
        parent_journals={"previous": "a"*64}, approval_text="generated fixture, not user approval",
        user_approval_source="generated fixture only")


def test_old_or_absent_approval_cannot_fund_new_comparison(monkeypatch):
    monkeypatch.setattr(study, "sha", lambda path: "b"*64)
    protocol = dict(parent_journals={"previous": "a"*64})
    study.validate_approval(grant(), protocol)
    for change in (dict(user_approved=False), dict(schema="explicit_user_authorized_blind_continuation_v140"),
        dict(remaining_baseline_wall_seconds=900), dict(remaining_training_wall_seconds=1),
        dict(paid_compute=True), dict(calibration_or_final_access=True), dict(threads_per_scientific_job=3),
        dict(parent_journals={}), dict(protocol_sha256="c"*64), dict(approval_text="")):
        with pytest.raises(ValueError): study.validate_approval(dict(grant(), **change), protocol)


def test_failure_or_polling_cannot_reset_fixed_shared_deadline():
    start = datetime.now(timezone.utc)-timedelta(seconds=1801)
    row = dict(category="baseline", seconds=1800, refund_on_failure=False, reserved_utc=start.isoformat())
    assert study.deadline(row) == start+timedelta(seconds=1800)
    assert study.deadline(row) < datetime.now(timezone.utc)
    for change in (dict(seconds=1801), dict(refund_on_failure=True), dict(category="training"),
                   dict(reserved_utc="2026-11-06T12:00:00")):
        with pytest.raises(ValueError): study.deadline(dict(row, **change))


def test_full_case_forecast_cannot_shrink_methods_camera_bank_or_fields():
    samples = [dict(layered=l, elapsed_seconds=t, all_seven_pose_controls_pass=True, lights=24,
                   cameras=2048, maximum_branches=8, requested_native_field_cap=128,
                   common_effective_native_fields=128) for l, t in ((False, 60.), (True, 70.))]
    timing = dict(samples=samples, comparison_forecast_seconds=885.)
    assert study.forecast_fits(timing, 1000.)
    assert not study.forecast_fits(timing, 800.)
    changed = deepcopy(timing); changed["samples"][0]["requested_native_field_cap"] = 32
    with pytest.raises(ValueError): study.forecast_fits(changed, 1000.)
    changed = deepcopy(timing); changed["samples"][0]["cameras"] = 3
    with pytest.raises(ValueError): study.forecast_fits(changed, 1000.)


def test_timing_and_prediction_cannot_read_query_truth_or_historical_outputs(monkeypatch):
    monkeypatch.setattr(study, "sha", lambda path: "a"*64)
    protocol = dict(inference_inputs={"query-image": "a"*64},
                    inputs={"query-image": "a"*64, "query-truth": "a"*64, "old-predictions": "a"*64})
    assert study.checked_input(protocol, "predict", "query-image") == Path("query-image")
    for phase, name in (("timing", "query-image"), ("predict", "query-truth"), ("predict", "old-predictions")):
        with pytest.raises(ValueError): study.checked_input(protocol, phase, name)
    assert study.checked_input(protocol, "score", "query-truth") == Path("query-truth")


def test_all_eight_predictions_must_be_sealed_before_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "RUN", tmp_path)
    raw = dict(rows=[dict(id=str(i)) for i in range(8)], query_truth_used=False, accepted_results=False)
    protocol = dict(case_ids=[str(i) for i in range(8)])
    study.write(tmp_path/"predictions.json", raw)
    with pytest.raises(FileNotFoundError): study.sealed(protocol)
    study.write(tmp_path/"PREDICTIONS_SEALED.json", dict(rows=8, sha256=study.sha(tmp_path/"predictions.json")))
    study.sealed(protocol)
    raw["rows"].pop(); (tmp_path/"predictions.json").write_text(__import__("json").dumps(raw))
    with pytest.raises(ValueError): study.sealed(protocol)


def accounting_fixture():
    hypotheses = [dict(branch_id=i, light_id=i) for i in range(2)]
    candidates = [dict(branch_id=i, light_id=i, reference_xy=[], query_xy=[], pose=None, estimator={}) for i in range(2)]
    methods = {}
    for name in ("single_uniform128", "beam_uniform128", "beam_geometry128", "beam_active128"):
        ids = [0] if name.startswith("single") else [0, 1]
        rounds = 128//len(ids)
        trace = [dict(index=j, branches=[dict(branch_id=i, light_id=i, attempted_fields=j+1) for i in ids],
                      weights_before=[1/len(ids)]*len(ids), weights_after=[1/len(ids)]*len(ids)) for j in range(rounds)]
        methods[name] = dict(candidates[0], branch_candidates=candidates[:len(ids)], acquisition=dict(
            retained_branch_ids=ids, pruning=False, requested_native_fields=128, query_fields_before_selection=0,
            attempted_native_fields=128, selected_indices=list(range(rounds)), trace=trace, rounds=rounds))
    return dict(initialization=dict(photometric_hypotheses=hypotheses), common_effective_native_fields=128,
                shared_reference_candidate_capacity=256, methods=methods)


def test_independent_replay_detects_branch_loss_light_switch_and_uncharged_searches():
    good = accounting_fixture(); verify_accounting(good)
    for operation in ("remove", "sun", "charge", "mix"):
        changed = deepcopy(good); row = changed["methods"]["beam_active128"]
        if operation == "remove": row["acquisition"]["trace"][0]["branches"].pop()
        elif operation == "sun": row["acquisition"]["trace"][0]["branches"][0]["light_id"] = 99
        elif operation == "charge": row["acquisition"]["attempted_native_fields"] = 64
        else: row["light_id"] = 99
        with pytest.raises(ValueError): verify_accounting(changed)
