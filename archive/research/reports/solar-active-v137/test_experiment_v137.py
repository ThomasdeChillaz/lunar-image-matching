from datetime import datetime, timedelta, timezone
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("supplement_admission_v137_test", HERE/"admission.py")
budget = module_from_spec(spec); spec.loader.exec_module(budget)


def fixture_budget(tmp_path):
    ledger = dict(schema="explicit_user_authorized_supplement_v137", user_approved=True,
        accounting_complete=True, remaining_baseline_wall_seconds=2400, remaining_training_wall_seconds=0,
        original_ledger_sha256="a"*64, original_journal_sha256="b"*64,
        original_remaining_seconds=dict(training=13493., baseline=0.),
        paid_compute=False, calibration_or_final_access=False)
    path = tmp_path/"ledger.json"; path.write_text(json.dumps(ledger))
    summary = dict(ledger, allowance_seconds=2400, stage_caps_seconds=dict(timing=120, comparison=1800, recovery=480),
                   threads_per_scientific_job=2, private_ledger_sha256=budget.sha(path))
    return path, tmp_path/"journal.json", summary


def test_explicit_supplement_keeps_failed_and_duplicate_charges(tmp_path):
    ledger, journal, summary = fixture_budget(tmp_path)
    calls = []
    guard = lambda: calls.append("original_checked")
    reservation = budget.charge(ledger, journal, summary, "c"*64, "timing", guard)
    assert reservation["seconds"] == 120 and reservation["refund_on_failure"] is False
    assert budget.state(ledger, journal, summary, guard)[1] == 2280
    with pytest.raises(ValueError, match="already charged"):
        budget.charge(ledger, journal, summary, "c"*64, "timing", guard)
    assert budget.state(ledger, journal, summary, guard)[1] == 2280
    budget.charge(ledger, journal, summary, "c"*64, "comparison", guard)
    assert budget.state(ledger, journal, summary, guard)[1] == 480
    with pytest.raises(ValueError, match="exhausted"):
        budget.charge(ledger, journal, summary, "d"*64, "comparison", guard)
    assert calls


def test_changed_original_or_private_authority_fails_before_reservation(tmp_path):
    ledger, journal, summary = fixture_budget(tmp_path)
    def changed_original():
        raise ValueError("Original canonical accounting changed")
    with pytest.raises(ValueError, match="Original"):
        budget.charge(ledger, journal, summary, "c"*64, "timing", changed_original)
    assert not journal.exists()
    ledger.write_text(ledger.read_text()+" ")
    with pytest.raises(ValueError, match="ledger changed"):
        budget.charge(ledger, journal, summary, "c"*64, "timing", lambda: None)
    assert not journal.exists()


def test_cap_scope_and_unapproved_recovery_cannot_expand(tmp_path):
    ledger, journal, summary = fixture_budget(tmp_path)
    values = json.loads(ledger.read_text())
    for changes in (dict(user_approved=False), dict(allowance_seconds=2401), dict(paid_compute=True),
                    dict(calibration_or_final_access=True), dict(threads_per_scientific_job=3)):
        with pytest.raises(ValueError): budget.validate(values, dict(summary, **changes))
    with pytest.raises(ValueError, match="separately published"):
        budget.charge(ledger, journal, summary, "c"*64, "recovery", lambda: None)
    assert not journal.exists()


def test_original_reservation_deadline_is_not_reset():
    start = datetime.now(timezone.utc)-timedelta(seconds=121)
    reservation = dict(reserved_utc=start.isoformat(), seconds=120)
    assert budget.deadline(reservation) == start+timedelta(seconds=120)
    assert not budget.before_deadline(reservation)
    with pytest.raises(ValueError): budget.deadline(dict(reserved_utc="2026-11-05T10:00:00", seconds=120))


def test_scientific_lease_prevents_overlapping_controllers(tmp_path):
    path = tmp_path/"scientific.lock"
    with budget.serial_lease(path):
        with pytest.raises(BlockingIOError):
            with budget.serial_lease(path):
                pass
    with budget.serial_lease(path):
        pass


def test_control_identity_checks_coordinates_and_camera_without_timing():
    spec = spec_from_file_location("sequential_engine_v137_test", HERE/"engine.py")
    engine = module_from_spec(spec); spec.loader.exec_module(engine)
    row = dict(reference_xy=[[1., 2.]], query_xy=[[3., 4.]], pose=None, estimator=dict(reason="few"))
    engine.exact_core(row, dict(row, online_seconds=999.), "fixture")
    for key, changed in (("query_xy", [[3., 5.]]), ("pose", [[1.]]), ("estimator", dict(reason="different"))):
        with pytest.raises(ValueError, match=key): engine.exact_core(row, dict(row, **{key:changed}), "fixture")


def test_new_protocol_names_preserve_every_control():
    spec = spec_from_file_location("sequential_engine_v137_names", HERE/"engine.py")
    engine = module_from_spec(spec); spec.loader.exec_module(engine)
    assert engine.FRESH_NAMES == ("direct", "joint_no_extra", "uniform", "appearance", "geometry", "active")
    assert len(set(engine.FRESH_NAMES)) == 6


def test_scoring_refuses_partial_unsealed_or_truth_exposed_predictions(tmp_path, monkeypatch):
    spec = spec_from_file_location("sequential_controller_seal_v137_test", HERE/"experiment.py")
    controller = module_from_spec(spec); spec.loader.exec_module(controller)
    monkeypatch.setattr(controller, "HOME", tmp_path)
    protocol = dict(cases=[dict(case=i) for i in range(32)])
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    raw = dict(protocol_sha256=controller.base.sha(tmp_path/"protocol.json"),
               query_truth_used=False, rows=[dict(case=c) for c in protocol["cases"]])
    output = tmp_path/"run"; output.mkdir()
    def save(value, sealed=True):
        (output/"predictions.json").write_text(json.dumps(value))
        if sealed:
            (output/"PREDICTIONS_SEALED.json").write_text(json.dumps(dict(rows=32,
                sha256=controller.base.sha(output/"predictions.json"))))
    save(raw, sealed=False)
    with pytest.raises(FileNotFoundError): controller.complete_predictions(protocol, output)
    save(raw)
    assert controller.complete_predictions(protocol, output) == raw
    for changed in (dict(raw, rows=raw["rows"][:-1]), dict(raw, query_truth_used=True)):
        save(changed)
        with pytest.raises(ValueError, match="sealed"):
            controller.complete_predictions(protocol, output)
