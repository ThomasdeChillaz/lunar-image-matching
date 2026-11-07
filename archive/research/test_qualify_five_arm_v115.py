"""Guard the measured admission and exact canonical prior debits."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

import qualify_five_arm_v115 as qualification


ROOT = Path(__file__).resolve().parent


@pytest.fixture
def before_five_arm():
    direct = qualification.pinned(ROOT, qualification.DIRECT_PROTOCOL)
    with ZipFile(qualification.pin(ROOT, *qualification.DIRECT_ARCHIVE)) as archive:
        receipt = json.loads(qualification._archive_member(archive, *qualification.DIRECT_RECEIPT))
        watchdog = receipt["files"]["watchdog"]
        original = json.loads(qualification._archive_member(archive, watchdog["path"], watchdog["sha256"]))
    return dict(ledger_sha256=direct["compute_ledger_sha256"],
                reservations=[*direct["baseline_admission"]["initial_reservations"],
                              original["original_reservation"]["reservation"]])


def test_original_witnesses_admit_96_within_the_original_cap(before_five_arm):
    entries, forecast = qualification._verified_timing(ROOT, before_five_arm)
    assert tuple(entry["seconds"] for entry in entries) == (1800, 1800)
    assert forecast["largest_measured_budget_fit"] == 96
    assert forecast["cohorts"]["96"]["five_arm_cells"] == 480
    assert forecast["cohorts"]["96"]["five_arm_conservative_seconds"] == pytest.approx(17321.65147815151)
    assert forecast["effective_larger_run_cap_seconds"] == 18000
    assert forecast["execution_authorized"] is False


def test_an_altered_direct_debit_cannot_admit_the_comparison(before_five_arm):
    changed = deepcopy(before_five_arm)
    changed["reservations"][-1]["seconds"] -= 1
    with pytest.raises(ValueError, match="same canonical budget"):
        qualification.measured_forecast(ROOT, changed)


def test_an_ambiguous_archive_member_never_becomes_timing_evidence():
    raw = BytesIO()
    with ZipFile(raw, "w") as archive:
        archive.writestr("summary.json", "first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("summary.json", "second")
    with ZipFile(raw) as archive, pytest.raises(ValueError, match="ambiguous"):
        qualification._archive_member(archive, "summary.json", "0" * 64)
