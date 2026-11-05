from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

spec = spec_from_file_location("measured_admission_v130", Path(__file__).with_name("experiment.py"))
admission = module_from_spec(spec); spec.loader.exec_module(admission)


def test_only_per_job_allocation_changes_and_all_cases_remain():
    p = admission.build()
    previous = admission.base.read(admission.OLD / "protocol.json")
    assert p["cases"] == previous["cases"] and len(p["cases"]) == 32
    assert p["caps"]["comparison"] == 900
    assert p["admission"]["forecast_seconds"] < 900
    assert p["admission"]["previous_refusal_preserved"]
    assert p["matching_scoring_and_roster_unchanged_from_v129"]


def test_incomplete_geometry_control_cannot_admit_the_job(monkeypatch):
    original = admission.base.read
    def changed(file):
        value = deepcopy(original(file))
        if str(file).endswith("timing/timing.json"):
            value["samples"][0]["methods"]["pooled_parallax"] = False
        return value
    monkeypatch.setattr(admission.base, "read", changed)
    with pytest.raises(ValueError, match="Both complete native controls"):
        admission.timing_proof()


def test_corrected_native_matches_recover_pose_in_the_lunar_coordinate_frame(monkeypatch):
    import cv2
    import numpy as np
    from reference_cache import CachedReferenceSift, FeatureStore, pipeline
    from smoke_v126 import generated
    cv2.setNumThreads(1)
    args, valid, truth = generated(False)
    reference, depth, k, old_pose, query, qk, _ = args
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 2000.
    expected = pose @ np.linalg.inv(old_pose) @ truth
    CachedReferenceSift.store = FeatureStore()
    monkeypatch.setattr(pipeline, "SiftLibraryMatcher", CachedReferenceSift)
    image, _, _ = pipeline.common_images(reference, query, np.ones_like(valid))
    measured = pipeline.run(reference, query, depth, k, pose, qk, pose[None], [image],
                            reference_valid=np.ones_like(valid))
    for name in ("pooled_uniform", "pooled_parallax", "direct"):
        result = np.asarray(measured["methods"][name]["pose"])
        assert result.shape == (4, 4)
        assert np.linalg.norm(result[:3, 3]-expected[:3, 3]) < .5
        assert np.linalg.norm(result[:3, :3]-expected[:3, :3]) < .005
