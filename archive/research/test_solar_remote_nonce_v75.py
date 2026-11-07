"""Pure metadata regression: another nonce cannot reuse a canonical charge."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import solar_remote_relay_v75 as relay
from test_solar_remote_relay_v75 import authorized


def test_nonce_matches_independent_exact_full_entry_serialization(authorized):
    auth = authorized["auth"]
    entry = auth["reservation"]["reservation"]
    expected_payload = dict(protocol_sha256=relay.sha(authorized["protocol_path"]), canonical_original_entry=entry)
    expected = hashlib.sha256(b"solar-remote-charge-v75:" + json.dumps(expected_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    assert auth["nonce"] == expected
    assert relay.charge_nonce(auth["protocol_sha256"], dict(reversed(list(entry.items())))) == expected
    relay.validate_authorization(auth, authorized["protocol_path"], True)


def test_another_nonce_cannot_reuse_same_original_canonical_charge(authorized):
    changed = copy.deepcopy(authorized["auth"])
    changed["nonce"] = "0" * 64 if changed["nonce"] != "0" * 64 else "1" * 64
    with pytest.raises(ValueError, match="complete original canonical charge"):
        relay.validate_authorization(changed, authorized["protocol_path"], True)


@pytest.mark.parametrize("field,value", [
    ("job_id", "other/evaluation"), ("category", "training"), ("seconds", 11),
    ("reserved_utc", "2026-01-01T00:00:00+00:00"), ("refund_on_failure", True),
    ("extra_original_entry_field", "all fields count"),
])
def test_every_original_entry_field_is_bound(authorized, field, value):
    auth = authorized["auth"]; entry = copy.deepcopy(auth["reservation"]["reservation"]); entry[field] = value
    assert relay.charge_nonce(auth["protocol_sha256"], entry) != auth["nonce"]


def test_protocol_sha_is_also_bound(authorized):
    auth = authorized["auth"]; entry = auth["reservation"]["reservation"]
    assert relay.charge_nonce("f" * 64, entry) != auth["nonce"]


def test_an_identical_protocol_file_alias_cannot_create_another_run_identity(authorized):
    original = authorized["auth"]
    alias = "reports/identical-protocol-alias.json"
    Path(alias).write_bytes(Path(authorized["protocol_path"]).read_bytes())
    changed = copy.deepcopy(original); changed["protocol_path"] = alias
    # Path-specific publication checks still apply independently. The one-use
    # identity is charge/SHA bound, so an identical file alias cannot evade the
    # workflow history/concurrency check by generating a different run title.
    relay.validate_authorization(changed, alias, True)
    assert changed["nonce"] == original["nonce"]
    assert relay.charge_nonce(relay.sha(alias), original["reservation"]["reservation"]) == original["nonce"]
