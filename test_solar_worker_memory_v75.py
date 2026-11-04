"""Synthetic resource readings and stubbed runner flow; no model/data jobs."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).with_name("run_solar_evaluation_v70.py")
    spec = importlib.util.spec_from_file_location("rss_candidate_runner_v75", path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("platform,raw,expected,unit", [("darwin", 123, 123, "bytes"), ("linux", 123, 125952, "KiB"), ("linux", 0, 0, "KiB")])
def test_native_resource_units_normalize_to_bytes_without_child_usage(runner, monkeypatch, platform, raw, expected, unit):
    calls = []; self_usage = object()
    resource = SimpleNamespace(RUSAGE_SELF=self_usage, getrusage=lambda who: calls.append(who) or SimpleNamespace(ru_maxrss=raw))
    monkeypatch.setitem(sys.modules, "resource", resource); monkeypatch.setattr(sys, "platform", platform)
    result = runner.worker_peak_rss_snapshot()
    assert result["available"] is True and result["peak_rss_bytes"] == expected
    assert result["native_maxrss_unit"] == unit and result["worker_pid"] == os.getpid()
    assert calls == [self_usage] and result["child_processes_included"] is False
    assert result["newly_allocated_memory"] is False and "LIFETIME" in result["scope"]


@pytest.mark.parametrize("raw", [-1, float("nan"), float("inf"), True, "unknown"])
def test_invalid_resource_reading_is_explicitly_unavailable(runner, monkeypatch, raw):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=raw)))
    result = runner.worker_peak_rss_snapshot()
    assert result["available"] is False and result["peak_rss_bytes"] is None
    assert result["unavailable_reason"] == "resource_snapshot_error" and result["error"]["type"] == "ValueError"


def test_unsupported_platform_and_resource_error_do_not_claim_zero_or_raise(runner, monkeypatch):
    calls = []
    def fail(who): calls.append(who); raise OSError("procedural resource failure")
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=fail))
    monkeypatch.setattr(sys, "platform", "win32")
    result = runner.worker_peak_rss_snapshot()
    assert result["unavailable_reason"] == "unsupported_platform" and result["peak_rss_bytes"] is None and not calls
    monkeypatch.setattr(sys, "platform", "darwin")
    result = runner.worker_peak_rss_snapshot()
    assert result["unavailable_reason"] == "resource_snapshot_error" and result["error"]["type"] == "OSError"


def test_standalone_profile_is_worker_job_scope_not_model_memory(runner):
    one = runner.worker_memory_scope([dict(name="only")])
    many = runner.worker_memory_scope([dict(name="first"), dict(name="second")])
    assert one["job_scope"] == "standalone_profile_worker_process"
    assert many["job_scope"] == "shared_multi_profile_worker_process" and many["profile_order"] == ["first", "second"]
    for value in (one, many):
        assert value["per_model_memory_ranking_supported"] is False
        assert value["isolated_model_memory_measured"] is False and value["per_case_memory_measured"] is False
        assert value["child_processes_included"] is False


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_stub_worker_profiles_and_caches_keep_timings_and_measure_current_process_only(runner, monkeypatch, tmp_path):
    # Admission and inference are stubs. All paths/arrays below are synthetic;
    # no actual canonical journal, image, checkpoint or geometry is accessed.
    output = tmp_path / "worker"; output.mkdir()
    profiles = [dict(name="fresh", backend="sift"), dict(name="cached", backend="sift")]
    cases = [dict(case_id=f"fixture-{i}", component="a", query_component="b", positive=bool(i), unit_index=1, query_unit_index=2,
        light_a=0, light_b=0, reference_geometry="synthetic-reference", input_sha256={}) for i in range(2)]
    prep = tmp_path / "prep.json"; _write(prep, dict(calendar_seconds=.5))
    manifest = dict(cases=cases, reference_precompute_receipts={"synthetic-reference":dict(receipt_path=str(prep), sha256=runner.sha(prep))})
    manifest_path = tmp_path / "manifest.json"; _write(manifest_path, manifest)
    protocol = dict(output=str(output), max_calendar_seconds=300, profiles=profiles, tracks=["estimated_sun"], purpose="modern_comparison",
        source_sha256={}, manifest=str(manifest_path), reference_preparation_costs=dict(outer_calendar_seconds_by_archive={"synthetic-reference":1.}))
    protocol_path = tmp_path / "protocol.json"; _write(protocol_path, protocol)
    _write(output / "reservation.json", dict(reservation=dict(job_id=runner.sha(protocol_path)+"/evaluation", seconds=300)))
    _write(output / "watchdog_admission.json", dict(protocol_sha256=runner.sha(protocol_path), reservation_sha256=runner.sha(output / "reservation.json"),
        start_monotonic=time.monotonic(), absolute_deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=300)).isoformat()))
    monkeypatch.setattr(runner, "admission", lambda *a: (protocol, manifest, cases))
    monkeypatch.setattr(runner, "software_identity", lambda: dict(procedural=True))
    monkeypatch.setattr(runner, "aggregate_rows", lambda rows: [])
    loaded = []; snapshots = []
    class FakeBackend:
        def __init__(self, profile):
            loaded.append(profile["name"]); self.provenance = dict(historical_control_peak_rss_bytes=999999999)
    class FakeLoader:
        def __init__(self, *a, **k): self.last_setup = dict(image_decode_seconds=.25, reference_geometry_load_seconds=.5, reference_physics_setup_seconds=.75)
        def get(self, *a): return None
        def close(self): pass
    monkeypatch.setattr(runner, "Backend", FakeBackend); monkeypatch.setattr(runner, "InputLoader", FakeLoader)
    monkeypatch.setattr(runner, "evaluate_case", lambda *a: (dict(inference_seconds=1.), dict(procedural=np.zeros(1))))
    def snapshot():
        result = dict(available=True, peak_rss_bytes=100+len(snapshots), worker_pid=os.getpid(), scope=runner.MEMORY_SCOPE)
        snapshots.append(result); return result
    monkeypatch.setattr(runner, "worker_peak_rss_snapshot", snapshot)
    cached_npz = tmp_path / "historical.npz"; np.savez_compressed(cached_npz, procedural=np.zeros(1))
    cache = {}
    for case in cases:
        key = runner.digest(dict(case=case, profile=profiles[1], track="estimated_sun", sources={}))
        cache[key] = dict(case_calendar_seconds=99., inference_seconds=10., complete_localization_cold_seconds=20., profile_load_seconds=4.,
            reference_setup=dict(image_decode_seconds=1., reference_geometry_load_seconds=2., reference_physics_setup_seconds=7.),
            saved_arrays=str(cached_npz), worker_peak_rss_bytes=888888888)
    monkeypatch.setattr(runner, "cache_index", lambda p: cache)
    # No torch/cv model import or forward is needed to exercise runner flow.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(setNumThreads=lambda n: None))
    runner.worker(str(protocol_path), "a"*40)
    summary = runner.read(output / "summary.json"); execution = runner.read(output / "execution.json")
    assert loaded == ["fresh"] and len(snapshots) == 2+2*len(profiles)  # Never sampled per case/batch.
    assert execution["worker_peak_rss"]["peak_rss_bytes"] == 100 and summary["worker_peak_rss"]["peak_rss_bytes"] == 105
    records = summary["memory_measurement"]["profile_snapshots"]
    assert [r["profile"] for r in records] == ["fresh", "cached"]
    assert [(r["fresh_inference_cases"], r["cached_cases"]) for r in records] == [(2,0),(0,2)]
    assert [r["before"]["peak_rss_bytes"] for r in records] == [101,103]
    assert [r["after"]["peak_rss_bytes"] for r in records] == [102,104]
    assert summary["memory_measurement"]["per_model_memory_ranking_supported"] is False
    fresh = [r for r in summary["rows"] if not r["reused_case_cache"]]
    replayed = [r for r in summary["rows"] if r["reused_case_cache"]]
    assert all(r["inference_seconds"] == 1. and r["complete_localization_cold_seconds"] == 2.5 for r in fresh)
    assert all("ACTUALLY incurred" in r["timing_scope"] for r in fresh)
    assert all(r["inference_seconds"] == 10. and r["complete_localization_cold_seconds"] == 20. and r["profile_load_seconds"] == 4. for r in replayed)
    assert all("replayed" in r["timing_scope"] and "historical" in r["memory_measurement_scope"] for r in replayed)
    assert all(r["original_case_calendar_seconds"] == 99. for r in replayed)
    assert all(r["worker_peak_rss_bytes"] == 888888888 for r in replayed)  # Preserved as history, never sampled as fresh.
    assert summary["worker_peak_rss"]["peak_rss_bytes"] not in (888888888,999999999)
