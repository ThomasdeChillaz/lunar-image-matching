"""An explicitly approved supplement; never alter or refund the original ledger."""

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from solar_compute_budget_v67 import reserve

CAPS = dict(timing=120, comparison=1800)
TOTAL = 2400
RECOVERY = 480


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate(ledger, summary):
    check(ledger.get("schema") == "explicit_user_authorized_supplement_v137"
          and ledger.get("user_approved") is True and summary.get("user_approved") is True,
          "Explicit supplementary approval required")
    check(ledger.get("accounting_complete") is True and summary.get("allowance_seconds") == TOTAL
          and ledger.get("remaining_baseline_wall_seconds") == TOTAL
          and ledger.get("remaining_training_wall_seconds") == 0
          and summary.get("stage_caps_seconds") == dict(**CAPS, recovery=RECOVERY),
          "Supplement must preserve its approved40-minute cap and stage reservations")
    for name in ("original_ledger_sha256", "original_journal_sha256", "original_remaining_seconds"):
        check(ledger[name] == summary[name], "Original accounting identity changed: "+name)
    check(summary["paid_compute"] is False and summary["calibration_or_final_access"] is False
          and summary["threads_per_scientific_job"] == 2
          and ledger["paid_compute"] is False and ledger["calibration_or_final_access"] is False,
          "Supplement cannot broaden compute or test-data scope")


def state(ledger_path, journal_path, summary, original_guard):
    original_guard()
    check(sha(ledger_path) == summary["private_ledger_sha256"], "Private supplementary ledger changed")
    ledger = read(ledger_path); validate(ledger, summary)
    journal = read(journal_path) if Path(journal_path).exists() else dict(ledger_sha256=sha(ledger_path), reservations=[])
    check(journal["ledger_sha256"] == sha(ledger_path), "Supplement journal identity changed")
    seen = set(); charged = 0
    for row in journal["reservations"]:
        check(row["job_id"] not in seen and row["category"] == "baseline"
              and row["refund_on_failure"] is False and type(row["seconds"]) is int
              and row["seconds"] > 0, "Invalid or duplicated supplement reservation")
        seen.add(row["job_id"]); charged += row["seconds"]
    check(charged <= TOTAL, "Supplement exceeded its approved cap")
    return journal, TOTAL-charged


def charge(ledger_path, journal_path, summary, protocol_sha, stage, original_guard):
    check(stage in CAPS, "Recovery requires a separately published prospective protocol")
    _, remaining = state(ledger_path, journal_path, summary, original_guard)
    check(remaining >= CAPS[stage], "Supplementary allowance exhausted")
    result = reserve(ledger_path, summary["private_ledger_sha256"], journal_path,
                     protocol_sha+"/active-v137-"+stage, "baseline", CAPS[stage])
    return result["reservation"]


def deadline(reservation):
    start = datetime.fromisoformat(reservation["reserved_utc"])
    check(start.tzinfo is not None, "Timezone-aware original reservation required")
    return start+timedelta(seconds=reservation["seconds"])


def before_deadline(reservation):
    return datetime.now(timezone.utc) < deadline(reservation)


@contextmanager
def serial_lease(path):
    """Hold one scientific-job lease across controller and child lifetime."""
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield handle.fileno()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
