from datetime import datetime, timezone
import hashlib
import json

import pytest

from north_completion_v52 import (snapshot_tree, copy_original, verify_copies,
                                  completed_training, missing_cases, completion_budget)


def test_copy_preserves_bytes_without_hard_links_and_never_changes_original(tmp_path):
    original = tmp_path / 'original'; original.mkdir(); (original / 'weights').write_bytes(b'original-weight-bytes')
    before = snapshot_tree(original); copied = tmp_path / 'new'; copy_original(original, copied, before)
    assert snapshot_tree(original) == before == snapshot_tree(copied)
    assert (original / 'weights').stat().st_ino != (copied / 'weights').stat().st_ino
    verify_copies(original, copied, before)


def test_copy_refuses_different_existing_output_without_overwriting_it(tmp_path):
    original = tmp_path / 'original'; original.mkdir(); (original / 'x').write_bytes(b'old')
    copied = tmp_path / 'new'; copied.mkdir(); (copied / 'x').write_bytes(b'different')
    with pytest.raises(ValueError): copy_original(original, copied, snapshot_tree(original))
    assert (original / 'x').read_bytes() == b'old' and (copied / 'x').read_bytes() == b'different'


def test_copy_cannot_write_inside_original(tmp_path):
    original = tmp_path / 'original'; original.mkdir(); (original / 'x').write_bytes(b'old')
    with pytest.raises(ValueError): copy_original(original, original / 'child', snapshot_tree(original))
    assert not (original / 'child').exists()


def test_incomplete_training_state_cannot_trigger_refitting():
    state = dict(step=1999, history=[{'step': i} for i in range(1, 2000)], optimizer={'state': {1: {'step': 1999}}})
    with pytest.raises(ValueError): completed_training(state)


def test_all_original_optimizer_updates_must_be_verified():
    state = dict(step=2000, history=[{'step': i} for i in range(1, 2001)], optimizer={'state': {1: {'step': 2000}}})
    completed_training(state); state['optimizer']['state'][1]['step'] = 1999
    with pytest.raises(ValueError): completed_training(state)


def test_missing_cases_reuse_every_valid_original_case(tmp_path):
    cases = [{'pair_id': 'a', 'positive': True}, {'pair_id': 'b', 'positive': True}]
    name = hashlib.sha256(b'a').hexdigest() + '.json'; (tmp_path / name).write_text(json.dumps(cases[0]))
    assert missing_cases(tmp_path, cases) == [cases[1]]
    assert json.loads((tmp_path / name).read_text()) == cases[0]


def test_changed_cached_case_is_not_silently_recomputed(tmp_path):
    name = hashlib.sha256(b'a').hexdigest() + '.json'; (tmp_path / name).write_text('{"pair_id":"a","positive":false}')
    with pytest.raises(ValueError): missing_cases(tmp_path, [{'pair_id': 'a', 'positive': True}])


def test_new_inference_budget_never_rewrites_original_deadline():
    original = dict(root='original', created_utc='2026-11-02T08:24:00+00:00',
                    absolute_deadline_utc='2026-11-02T14:24:00+00:00', source_sha256={'kernel': 'unchanged'})
    before = json.dumps(original, sort_keys=True); now = datetime(2026, 11, 2, 17, tzinfo=timezone.utc)
    budget = completion_budget(original, 'separate', now)
    assert json.dumps(original, sort_keys=True) == before
    assert budget['original_absolute_deadline_utc'] == original['absolute_deadline_utc']
    assert budget['absolute_deadline_utc'] == '2026-11-02T17:30:00+00:00'
    assert budget['additional_fitting_updates_allowed'] == 0
    assert not budget['original_time_protocol_completed']


@pytest.mark.parametrize('field,value', [('original_time_protocol_completed', True),
    ('additional_fitting_updates', 1), ('recovery_calendar_seconds', 1801), ('total_updates', 4000)])
def test_amended_audit_cannot_claim_original_protocol_or_extra_training(field, value):
    from verify_north_completion_v52 import check_amended_summary
    summary = dict(complete=True, total_updates=6000, original_time_protocol_completed=False,
        completion_only_amendment=True, additional_fitting_updates=0, calibration_or_test_pixels_loaded=0,
        recovery_calendar_seconds=500)
    check_amended_summary(summary); summary[field] = value
    with pytest.raises(ValueError): check_amended_summary(summary)
