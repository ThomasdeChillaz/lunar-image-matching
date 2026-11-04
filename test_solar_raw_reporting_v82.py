"""Single-call raw reporting and exact JSON/NPZ roundtrip; no neural jobs."""
import copy
from datetime import datetime, timedelta, timezone
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).with_name("run_solar_evaluation_v70.py")
    spec = importlib.util.spec_from_file_location("raw_capture_runner_v82", path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def restore(value, arrays):
    """Test-only independent inverse of typed storage, not a policy replay."""
    if isinstance(value, list): return [restore(v, arrays) for v in value]
    if not isinstance(value, dict): return value
    kind = value.get("$raw")
    if kind in ("ndarray", "numpy_scalar", "ndarray_sequence"):
        a = arrays[value["npz_key"]]
        assert a.dtype.str == value["dtype"] and list(a.shape) == value["shape"]
        if kind == "numpy_scalar": return a[()]
        if kind == "ndarray_sequence":
            result = [np.array(a[i], dtype=a.dtype, copy=True).reshape(a.shape[1:]) for i in range(len(a))]
            return tuple(result) if value["container"] == "tuple" else result
        return a.copy()
    if kind == "nonfinite_float": return {"NaN":float("nan"), "+Infinity":float("inf"), "-Infinity":float("-inf")}[value["value"]]
    if kind == "tuple": return tuple(restore(v, arrays) for v in value["items"])
    if kind == "escaped_mapping": return {k:restore(v,arrays) for k,v in value["items"].items()}
    return {k:restore(v,arrays) for k,v in value.items()}


def exact(a, b):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b); return
    if isinstance(a, np.generic):
        assert isinstance(b, np.generic) and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b); return
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): exact(x,y)
    elif type(a) is float and np.isnan(a): assert np.isnan(b)
    else: assert a == b


def trace(updates=3):
    return dict(pose=None, solar_supported=False, estimated_sun_parameters=np.array([12.5,.25]),
        measurements=[dict(anchor=i,candidate=i%8,trial_sun_index=4,response_used=bool(i%2)) for i in range(updates)],
        incremental_updates=[dict(after_search=i+1,weights=[.125]*8,
            candidate_assignments=[(np.arange(256,dtype=np.int64)+i+j)%8 for j in range(8)]) for i in range(updates)],
        hypotheses=[dict(pose=np.eye(4),native_pose=True,assignment=dict(candidate=np.arange(256,dtype=np.int32)%8,informative_anchors=40))],
        response_measurements=2,response_refresh_calls=5,native_encoder_forwards=6,
        proposal_samples_requested=4096,final_estimator_iterations=2000,final_only_rescue=False)


def test_proxy_one_call_exact_return_and_no_retained_model(runner):
    class Model: pass
    class Backend:
        profile = dict(configuration=dict(response=True))
        provenance = dict(fixture=True)
        def __init__(self): self.calls=0; self.model=Model()
        def infer(self, inp): self.calls+=1; return dict(pose=None, array=np.arange(3), unsupported_model=self.model)
    delegate = Backend(); model = weakref.ref(delegate.model); proxy = runner.InferenceResultCapture(delegate)
    result = proxy.infer(None)
    assert delegate.calls == 1 and proxy.profile is delegate.profile and proxy.provenance is delegate.provenance
    assert proxy._result is result
    arrays = {}; encoded, info = runner.capture_case_report(proxy, arrays)
    assert info["backend_infer_calls"] == 1 and not info["encoding_complete"]
    assert encoded["unsupported_model"]["$raw"] == "unsupported"
    assert proxy._result is runner._NO_INFERENCE_RESULT and proxy._calls == 0
    del result, proxy, delegate; gc.collect()
    assert model() is None  # JSON/NPZ storage cannot keep the previous model alive.


def test_large_original_trace_roundtrips_compactly_in_npz(runner, tmp_path):
    original = trace(128); arrays = dict(reference_xy=np.zeros((0,2)))
    encoded, info = runner.pack_raw_inference(original, arrays)
    assert info["encoding_complete"] and info["effective_C_policy"] == "unrecorded_in_frozen_solver"
    assert info["C_fallback_reason"] == "unrecorded_in_frozen_solver"
    text = json.dumps(encoded, indent=2, allow_nan=False)
    assert len(text.encode()) < 250000 and 128 <= info["serialized_array_count"] < 150
    assert info["serialized_array_bytes"] < 3*1024*1024
    path = tmp_path/"case.npz"; np.savez_compressed(path, **arrays)
    with np.load(path,allow_pickle=False) as archive:
        loaded = {key:archive[key].copy() for key in archive.files}
    assert not any(a.dtype.hasobject for a in loaded.values())
    exact(original, restore(json.loads(text), loaded))
    assert encoded["solar_supported"] is False and encoded["pose"] is None


def test_nonfinite_dtype_tuple_and_marker_values_are_explicit(runner, tmp_path):
    a = np.array([1.,np.nan,np.inf,-np.inf],dtype=">f8")
    original = dict(pose=None,scalar_nan=float("nan"),positive=float("inf"),negative=float("-inf"),
        array=a,scalar=np.float32(np.nan),tuple_arrays=(a,a.copy()),literal={"$raw":"user value","npz_key":"ordinary string"})
    arrays = {}; encoded, info = runner.pack_raw_inference(original, arrays)
    encoded = json.loads(json.dumps(encoded,allow_nan=False))
    assert encoded["pose"] is None and encoded["scalar_nan"] == {"$raw":"nonfinite_float","value":"NaN"}
    assert encoded["array"]["dtype"] == ">f8" and encoded["array"]["nonfinite_count"] == 3
    assert len(info["nonfinite_scalar_paths"]) == 3 and info["encoding_complete"]
    path=tmp_path/"typed.npz"; np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as z: exact(original,restore(encoded,{k:z[k] for k in z.files}))


def test_unsupported_is_not_null_silently_dropped_or_pickled(runner):
    class NeverInspect:
        def __repr__(self): raise AssertionError("Do not inspect model/provenance objects")
    arrays = {}; raw=dict(pose=None,object=NeverInspect(),object_array=np.array([NeverInspect()],dtype=object))
    encoded, info = runner.pack_raw_inference(raw,arrays)
    assert encoded["pose"] is None and encoded["object"]["value_preserved"] is False
    assert encoded["object_array"]["$raw"] == "unsupported" and not info["encoding_complete"]
    assert len(info["unsupported_values"]) == 2 and arrays == {}
    json.dumps(encoded,allow_nan=False)


def test_bounds_and_reserved_prefix_fail_without_partial_array_insertion(runner, monkeypatch):
    arrays = dict(reference_xy=np.ones((2,2))); before=arrays["reference_xy"].copy()
    monkeypatch.setattr(runner,"RAW_MAX_ARRAY_BYTES",1)
    with pytest.raises(ValueError,match="numeric-byte bound"): runner.pack_raw_inference(dict(array=np.ones(2)),arrays)
    assert list(arrays) == ["reference_xy"]; np.testing.assert_array_equal(arrays["reference_xy"],before)
    with pytest.raises(ValueError,match="namespace collision"):
        runner.pack_raw_inference({}, {runner.RAW_ARRAY_PREFIX+"00000":np.zeros(1)})
    monkeypatch.setattr(runner,"RAW_MAX_JSON_BYTES",2)
    with pytest.raises(ValueError,match="JSON-byte bound"): runner.pack_raw_inference(dict(text="long"),arrays)
    assert list(arrays) == ["reference_xy"]


def test_capture_distinguishes_backend_exception_from_a_null_pose(runner):
    class Backend:
        def infer(self, inp): raise RuntimeError("procedural failure")
    proxy=runner.InferenceResultCapture(Backend())
    with pytest.raises(RuntimeError): proxy.infer(None)
    encoded, info=runner.capture_case_report(proxy,{})
    assert encoded is None and info["capture_status"] == "backend_did_not_return" and not info["encoding_complete"]
    assert proxy._result is runner._NO_INFERENCE_RESULT


def test_original_evaluator_metrics_coordinates_pose_and_call_order_are_unchanged(runner):
    from solar_evaluation_v70 import evaluate_case
    p=np.column_stack([np.linspace(100,400,8),np.linspace(120,380,8)])
    raw=dict(reference_xy=p.copy(),query_xy=p.copy(),pose=np.eye(4),solar_supported=False,measurements=[dict(anchor=2,candidate=1,response_used=True)])
    class Backend:
        profile=dict(configuration=dict(response=True))
        def __init__(self): self.calls=0
        def infer(self, inp): self.calls+=1; return raw
    geometry=SimpleNamespace(depth=np.full((512,512),10.),intrinsics=np.array([[400.,0.,255.5],[0.,400.,255.5],[0.,0.,1.]]))
    inp=SimpleNamespace(solar_lookup=None,geometry=geometry);case=dict(positive=True)
    def truth(case, points): return dict(positive=True,query_xy=p.copy(),visible=np.ones(8,bool),query_cam2world=np.eye(4)),"procedural"
    estimator_calls=[]
    def estimator(a,b,inp): estimator_calls.append((a.copy(),b.copy())); return np.eye(4),dict(reason=None,inliers=8)
    baseline=Backend(); before_row,before_arrays=evaluate_case(baseline,inp,case,truth,estimator)
    delegated=Backend(); proxy=runner.InferenceResultCapture(delegated)
    row,arrays=evaluate_case(proxy,inp,case,truth,estimator)
    assert baseline.calls == delegated.calls == 1 and len(estimator_calls) == 2
    assert before_row["metrics"] == row["metrics"]
    for key in before_arrays: np.testing.assert_array_equal(before_arrays[key],arrays[key])
    encoded,info=runner.capture_case_report(proxy,arrays)
    for key in before_arrays: np.testing.assert_array_equal(before_arrays[key],arrays[key])
    np.testing.assert_array_equal(restore(encoded,arrays)["pose"],raw["pose"])
    assert info["backend_infer_calls"] == 1 and "inside inference_seconds" in info["scope"]


def test_trace_survives_existing_case_cache_json_npz_and_hash_seals(runner, tmp_path):
    raw=trace(); arrays=dict(reference_xy=np.zeros((0,2)),query_xy=np.zeros((0,2)),pose=np.empty((0,4)))
    encoded,info=runner.pack_raw_inference(raw,arrays)
    original=tmp_path/"original.npz"; np.savez_compressed(original,**arrays)
    row=dict(saved_arrays=str(original),saved_arrays_sha256=runner.sha(original),raw_inference_result=encoded,raw_inference_reporting=info)
    row_path=tmp_path/"original.json"; row_path.write_text(json.dumps(row,allow_nan=False))
    seal=tmp_path/"original.COMPLETE.json"; seal.write_text(json.dumps(dict(complete=True,identity="fixture-key",row_path=str(row_path),row_sha256=runner.sha(row_path),arrays_path=str(original),arrays_sha256=runner.sha(original))))
    cached=runner.cache_index(dict(case_cache_seals={str(seal):runner.sha(seal)}))["fixture-key"]
    copied=tmp_path/"copied.npz"
    with copied.open("xb") as destination,original.open("rb") as source: runner.shutil.copyfileobj(source,destination)
    replay=dict(cached,reused_case_cache=True,saved_arrays=str(copied),raw_inference_execution_scope="Historical pinned prediction/trace from case cache; no fresh backend inference or module-use observation.")
    assert runner.sha(copied) == runner.sha(original) and "Historical" in replay["raw_inference_execution_scope"]
    with np.load(copied,allow_pickle=False) as z: exact(raw,restore(replay["raw_inference_result"],{k:z[k] for k in z.files}))
    assert replay["raw_inference_reporting"]["effective_C_policy"] == "unrecorded_in_frozen_solver"
