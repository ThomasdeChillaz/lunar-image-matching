"""Clock and case-identity controls without model loading or imagery."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from recover_native_moonst3r_v28 import check_deadline, inventory


def test_utc_deadline_detects_host_clock_jump_without_waiting_for_alarm():
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    check_deadline(start, 600, start+timedelta(seconds=599))
    with pytest.raises(TimeoutError):check_deadline(start, 600, start+timedelta(hours=11))


def test_missing_cases_follow_original_order_and_duplicates_are_rejected(tmp_path):
    (tmp_path/'cases').mkdir(); (tmp_path/'protocol.json').write_text(json.dumps(dict(case_ids=['b', 'a', 'c'])))
    (tmp_path/'cases'/'one.json').write_text(json.dumps(dict(pair_id='a')))
    protocol, rows, files, missing = inventory(tmp_path)
    assert missing == ['b', 'c'] and set(rows) == {'a'} and len(files) == 1
    (tmp_path/'cases'/'two.json').write_text(json.dumps(dict(pair_id='a')))
    with pytest.raises(ValueError, match='duplicate'):inventory(tmp_path)
