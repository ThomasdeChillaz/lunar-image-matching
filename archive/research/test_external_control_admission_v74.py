"""Optional outer receipt admission tests; metadata stubs, no model/images."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from test_control_historical_native_v74 import control,HERE,REPO


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def patched_adapter():return load('external_control_patched_native_adapters',REPO/'native_pretrained_adapters_v73.py')


def test_optional_external_profile_gate_fails_before_inner_files_or_model(patched_adapter,monkeypatch):
    profile=dict(backend='native_pretrained',method='matchanything',author_profile='native512',cpu_threads=2,external_control={'TEST_ONLY':'not evidence'})
    def failed(*args):raise ValueError('outer deadline receipt failed')
    monkeypatch.setattr(control,'validate_external_pretrained',failed)
    monkeypatch.setattr(patched_adapter,'_evidence_paths',lambda *args:pytest.fail('Read inner evidence after failed outer gate'))
    with pytest.raises(ValueError,match='outer deadline'):patched_adapter.validate_native_profile(profile)


def test_comparison_requires_outer_source_and_input_pins(patched_adapter,monkeypatch):
    profile=dict(external_control={'TEST_ONLY':'not evidence'})
    gate=dict(source_sha256={'control_historical_native_v74.py':'a'*64},input_sha256={'outer.json':'b'*64,'receipt.json':'c'*64})
    monkeypatch.setattr(control,'validate_external_pretrained',lambda *args:gate)
    p=dict(source_sha256={},input_sha256={})
    with pytest.raises(ValueError,match='source must be pinned'):patched_adapter.validate_external_comparison(profile,p)
    p['source_sha256']=gate['source_sha256'].copy()
    with pytest.raises(ValueError,match='protocol/receipt must be pinned'):patched_adapter.validate_external_comparison(profile,p)
    p['input_sha256']=gate['input_sha256'].copy()
    assert patched_adapter.validate_external_comparison(profile,p)==gate
    assert patched_adapter.validate_external_comparison({},p)is None


def test_actual_comparison_admission_invokes_outer_gate(patched_adapter,monkeypatch,tmp_path):
    runner=load('external_control_patched_evaluation_runner',REPO/'run_solar_evaluation_v70.py')
    monkeypatch.setitem(sys.modules,'native_pretrained_adapters_v73',patched_adapter)
    import solar_evaluation_protocol_v74 as freeze
    monkeypatch.setattr(freeze,'validate_runtime',lambda *args:dict())
    monkeypatch.setattr(runner,'validate_profiles',lambda *args:None);monkeypatch.setattr(runner,'validate_profile_training',lambda *args:None);monkeypatch.setattr(runner,'REQUIRED_SOURCES',set())
    monkeypatch.setattr(patched_adapter,'validate_native_profile',lambda *args:dict(TEST_ONLY=True))
    def failed(*args):raise ValueError('actual comparison blocked failed outer receipt')
    monkeypatch.setattr(control,'validate_external_pretrained',failed)
    profile=dict(name='test_only',backend='native_pretrained',method='matchanything',author_profile='native512',cpu_threads=2,external_control={'TEST_ONLY':'not evidence'})
    protocol=dict(schema_version=70,builder_schema_version=74,split='model_selection',calibration_or_final_access=False,cpu_threads=2,tracks=['estimated_sun'],max_calendar_seconds=1,purpose='modern_comparison',profiles=[profile],source_sha256={},input_sha256={})
    path=tmp_path/'TEST_ONLY_PROTOCOL.json';path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='actual comparison blocked'):runner.admission(str(path),'f'*40)
