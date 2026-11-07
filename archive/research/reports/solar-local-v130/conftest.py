"""Source-only CI cannot audit an unbundled local compute reservation journal."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    root=Path(__file__).resolve().parents[2]
    required=(root/"runs/solar-local-v129/timing/timing.json",root/"runs/solar-compute-reservations-v67.json")
    if all(file.is_file() for file in required):
        return
    names=("test_only_per_job_allocation_changes_and_all_cases_remain",
           "test_incomplete_geometry_control_cannot_admit_the_job")
    for item in items:
        if "solar-local-v130/test_admission.py::" in item.nodeid and item.name in names:
            item.add_marker(pytest.mark.skip(reason="Requires the released timing evidence and original local reservation journal; generated camera tests still run"))
