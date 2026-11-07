"""Supply exact small JSON fixtures, without acquiring lunar rasters or compute authority."""

from pathlib import Path

import pytest

from research_test_metadata_v131 import SNAPSHOT_SHA256,restore


def pytest_sessionstart(session):
    root=Path(__file__).resolve().parent
    archive=root/"reports/test-metadata-v131/metadata.zip"
    restore(root,archive.read_bytes(),SNAPSHOT_SHA256)


@pytest.fixture(autouse=True)
def historical_native_metadata_identity(request,monkeypatch):
    """Historical dispatch unit tests use their declared software identity.

    This fixture only isolates metadata construction. Scientific scripts never
    import conftest, and a separate test verifies the real native-runtime refusal.
    """
    if "reports/solar-warp-v125/test_control_warp_archive_v125.py::" not in request.node.nodeid:
        return
    import control_warp_archive_v125 as control
    original=control.legacy.immutable
    software=control.legacy.read(control.OLD_DISPATCH)["software"]
    def frozen_identity(*args,**kwargs):
        metadata=original(*args,**kwargs)
        metadata["software"]=software.copy()
        return metadata
    monkeypatch.setattr(control.legacy,"immutable",frozen_identity)
