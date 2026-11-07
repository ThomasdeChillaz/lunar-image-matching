from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import pytest

from research_test_metadata_v131 import restore


def capsule(name="runs/example/summary.json",content=b'{"complete": true}'):
    buffer=BytesIO()
    with ZipFile(buffer,"w") as z:
        z.writestr("inventory.json",json.dumps({name:sha256(content).hexdigest()}))
        z.writestr(name,content)
    data=buffer.getvalue()
    return data,sha256(data).hexdigest()


def test_fixtures_restore_missing_records_and_never_overwrite_existing_history(tmp_path):
    data,digest=capsule()
    assert restore(tmp_path,data,digest)==1
    assert restore(tmp_path,data,digest)==0
    file=tmp_path/"runs/example/summary.json"
    file.write_text('{"different": true}')
    with pytest.raises(ValueError,match="never overwrite"):
        restore(tmp_path,data,digest)
    assert json.loads(file.read_text())=={"different":True}


@pytest.mark.parametrize("name",["runs/../private.json","runs/solar-compute-reservations-v67.json","runs/data/image.npz"])
def test_fixture_cannot_restore_outside_paths_compute_journals_or_imagery(tmp_path,name):
    data,digest=capsule(name)
    with pytest.raises(ValueError,match="allowlisted"):
        restore(tmp_path,data,digest)
    assert not list(tmp_path.iterdir())


def test_archive_digest_is_checked_before_any_write(tmp_path):
    data,_=capsule()
    with pytest.raises(ValueError,match="archive digest"):
        restore(tmp_path,data,"0"*64)
    assert not list(tmp_path.iterdir())


def test_production_controller_still_refuses_a_different_native_runtime(monkeypatch):
    from importlib.util import module_from_spec,spec_from_file_location
    from pathlib import Path
    source=Path(__file__).parent/"reports/solar-warp-v125/control_warp_archive_v125.py"
    spec=spec_from_file_location("unpatched_native_runtime_guard",source)
    control=module_from_spec(spec);spec.loader.exec_module(control)
    monkeypatch.setattr(control.legacy.platform,"system",lambda:"unsupported-runtime")
    with pytest.raises(ValueError,match="Original sources, native runtime"):
        control.immutable("archive")
