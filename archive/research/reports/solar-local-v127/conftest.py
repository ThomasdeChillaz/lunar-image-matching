"""Keep the real-manifest integration check optional in source-only checkouts."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    root=Path(__file__).resolve().parents[2]
    required=root/"runs/solar-reference-v80-full/case-assets/manifest.json"
    if required.is_file():
        return
    for item in items:
        if item.nodeid.endswith("test_experiment.py::test_same_case_scope_includes_all_seed_pairs_and_certified_negatives"):
            item.add_marker(pytest.mark.skip(reason="Requires the separately released lunar manifest; generated matching tests still run"))
