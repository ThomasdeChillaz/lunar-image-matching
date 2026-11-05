"""Supply exact small JSON fixtures, without acquiring lunar rasters or compute authority."""

from pathlib import Path

from research_test_metadata_v131 import SNAPSHOT_SHA256,restore


def pytest_sessionstart(session):
    root=Path(__file__).resolve().parent
    archive=root/"reports/test-metadata-v131/metadata.zip"
    restore(root,archive.read_bytes(),SNAPSHOT_SHA256)
