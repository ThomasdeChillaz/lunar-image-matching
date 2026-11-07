from copy import deepcopy
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

spec = spec_from_file_location('blind_continuation_v140_test', Path(__file__).with_name('continuation_v140.py'))
study = module_from_spec(spec)
spec.loader.exec_module(study)


def approved():
    return dict(schema='explicit_user_authorized_blind_continuation_v140', user_approved=True,
        accounting_complete=True, remaining_baseline_wall_seconds=900, remaining_training_wall_seconds=0,
        paid_compute=False, calibration_or_final_access=False, threads_per_scientific_job=2,
        parent_journal_sha256='a'*64, protocol_sha256='b'*64,
        user_approval_source='generated test fixture only', approval_text='test fixture, not a real user decision')


def test_preparation_and_old_grant_cannot_authorize_continuation(monkeypatch):
    monkeypatch.setattr(study, 'sha', lambda p: 'b'*64)
    protocol = dict(parent_journal_sha256='a'*64)
    study.validate_approval(approved(), protocol)
    for change in (dict(user_approved=False), dict(schema='explicit_user_authorized_supplement_v137'),
                   dict(remaining_baseline_wall_seconds=2400), dict(remaining_training_wall_seconds=1),
                   dict(paid_compute=True), dict(calibration_or_final_access=True),
                   dict(threads_per_scientific_job=3), dict(parent_journal_sha256='c'*64),
                   dict(protocol_sha256='d'*64), dict(approval_text=''), dict(user_approval_source='')):
        with pytest.raises(ValueError): study.validate_approval(dict(approved(), **change), protocol)


def test_original_fixed_deadline_and_failed_charge_are_not_reset():
    start = datetime.now(timezone.utc)-timedelta(seconds=901)
    reservation = dict(seconds=900, category='baseline', refund_on_failure=False, reserved_utc=start.isoformat())
    assert study.deadline(reservation) == start+timedelta(seconds=900)
    assert study.deadline(reservation) < datetime.now(timezone.utc)
    for change in (dict(seconds=901), dict(refund_on_failure=True), dict(category='training'),
                   dict(reserved_utc='2026-11-06T00:00:00')):
        with pytest.raises(ValueError): study.deadline(dict(reservation, **change))


def test_full_timing_gate_uses_measured_remaining_time():
    timing = dict(samples=[dict(layered=k, cameras=2048, lights=24, extra_searches=32,
                               all_six_pose_controls_pass=True, elapsed_seconds=t)
                           for k,t in ((False,46.),(True,52.))], comparison_forecast_seconds=654.)
    assert study.old.forecast_fits(timing, 790.)
    assert not study.old.forecast_fits(timing, 380.)
    short = deepcopy(timing); short['samples'][0]['extra_searches'] = 2
    with pytest.raises(ValueError): study.old.forecast_fits(short, 790.)


def test_incomplete_or_unsealed_predictions_cannot_be_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'WORK', tmp_path)
    protocol = dict(case_ids=[str(i) for i in range(8)])
    raw = dict(rows=[dict(id=str(i)) for i in range(8)], query_truth_used=False)
    study.write(tmp_path/'predictions.json', raw)
    with pytest.raises(FileNotFoundError): study.sealed(protocol)
    study.write(tmp_path/'PREDICTIONS_SEALED.json', dict(rows=8,sha256=study.sha(tmp_path/'predictions.json')))
    study.sealed(protocol)
    raw['rows'].pop()
    (tmp_path/'predictions.json').write_text(__import__('json').dumps(raw))
    with pytest.raises(ValueError): study.sealed(protocol)
