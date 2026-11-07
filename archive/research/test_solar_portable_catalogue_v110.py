"""Independent procedural controls for the versioned archive validator."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import solar_eval_data_v69 as v69
import solar_portable_catalogue_v110 as candidate


@pytest.fixture
def data():
    latitude = -86.0
    rows = v69.sun_search_catalogue(latitude)
    s = len(rows)
    receipt = {
        "complete": True, "precompute": "full", "query_truth_used": False,
        "neural_forwards": 0, "uncomputed_solar_columns": 0,
        "computed_solar_columns": s, "sun_catalogue": rows,
    }
    geometry = {
        "sun_vectors": np.array([row["sun_enu"] for row in rows]),
        "sun_parameters": np.array([[row["hour_angle_deg"], row["subsolar_latitude_deg"]] for row in rows]),
        "coarse_sun_mask": np.array([row["coarse"] for row in rows], dtype=bool),
        "solar_materialized_mask": np.ones(s, dtype=bool),
    }
    return geometry, receipt, latitude


def test_roundoff_is_only_in_recomputed_vectors(data, monkeypatch):
    geometry, receipt, latitude = data
    altered = deepcopy(receipt["sun_catalogue"])
    x = altered[0]["sun_enu"][0]
    altered[0]["sun_enu"][0] = np.nextafter(x, 2.0).item()
    monkeypatch.setattr(candidate, "sun_search_catalogue", lambda _: altered)
    result = candidate.validate_frozen_generic_catalogue(*data)
    assert 0.0 < result["max_component_roundoff"] <= candidate.MAX_SUN_COMPONENT_ROUNDOFF

    geometry["sun_vectors"][0, 0] = altered[0]["sun_enu"][0]
    with pytest.raises(ValueError, match="pinned precompute receipt"):
        candidate.validate_frozen_generic_catalogue(geometry, receipt, latitude)


def test_exact_receipt_agreement_does_not_cover_large_mismatch(data):
    geometry, receipt, latitude = data
    geometry["sun_vectors"][0, 0] += 1e-9
    geometry["sun_vectors"][0] /= np.linalg.norm(geometry["sun_vectors"][0])
    receipt["sun_catalogue"][0]["sun_enu"] = geometry["sun_vectors"][0].tolist()
    with pytest.raises(ValueError, match="beyond libm roundoff"):
        candidate.validate_frozen_generic_catalogue(geometry, receipt, latitude)


@pytest.mark.parametrize("fault", ("missing", "false", "integer", "short", "extra"))
def test_full_availability_is_exact(data, fault):
    geometry, _, _ = data
    mask = geometry["solar_materialized_mask"]
    if fault == "missing":
        del geometry["solar_materialized_mask"]
    elif fault == "false":
        mask[0] = False
    elif fault == "integer":
        geometry["solar_materialized_mask"] = mask.astype(np.uint8)
    elif fault == "short":
        geometry["solar_materialized_mask"] = mask[:-1]
    else:
        geometry["solar_materialized_mask"] = np.r_[mask, True]
    with pytest.raises(ValueError, match="availability"):
        candidate.validate_frozen_generic_catalogue(*data)


@pytest.mark.parametrize("fault", ("reorder", "extra", "missing", "parameters", "coarse", "nan"))
def test_generic_catalogue_and_coarse_selection_remain_exact(data, fault):
    geometry, receipt, _ = data
    rows = receipt["sun_catalogue"]
    if fault == "reorder":
        rows.reverse()
        for name, array in geometry.items():
            geometry[name] = array[::-1]
    elif fault in ("extra", "missing"):
        rows[:] = rows + rows[-1:] if fault == "extra" else rows[:-1]
        receipt["computed_solar_columns"] = len(rows)
        for name, array in geometry.items():
            geometry[name] = np.concatenate((array, array[-1:])) if fault == "extra" else array[:-1]
    elif fault == "parameters":
        rows[0]["hour_angle_deg"] += 0.001
        geometry["sun_parameters"][0, 0] += 0.001
    elif fault == "coarse":
        rows[0]["coarse"] = not rows[0]["coarse"]
        geometry["coarse_sun_mask"][0] = not geometry["coarse_sun_mask"][0]
    else:
        rows[0]["sun_enu"][0] = float("nan")
        geometry["sun_vectors"][0, 0] = np.nan
    with pytest.raises(ValueError, match="lattice|finite"):
        candidate.validate_frozen_generic_catalogue(*data)


@pytest.mark.parametrize("field,value", (
    ("query_truth_used", True), ("precompute", "coarse"),
    ("neural_forwards", True), ("uncomputed_solar_columns", 1),
))
def test_ineligible_receipt_rejected(data, field, value):
    data[1][field] = value
    with pytest.raises(ValueError, match="reference-only"):
        candidate.validate_frozen_generic_catalogue(*data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_archive_and_original_receipt_digests(data, tmp_path):
    geometry, receipt, latitude = data
    s = len(receipt["sun_catalogue"])
    geometry.update(
        depth=np.ones((512, 512)), intrinsics=np.eye(3), cam2world=np.eye(4),
        normals=np.tile([0.0, 0.0, 1.0], (256, 1)),
        solar_states=np.zeros((256, s), np.uint8), solar_variants=np.zeros((256, s, 7), np.uint8),
        reference_sun=np.array([0.0, 0.0, 1.0]), world_variants=np.zeros((256, 7, 3)),
    )
    archive = tmp_path / "reference.npz"
    sidecar = tmp_path / "reference.npz.json"
    np.savez(archive, **geometry)
    receipt["output_sha256"] = digest(archive)
    write_json(sidecar, receipt)
    pins = {archive.name: digest(archive), sidecar.name: digest(sidecar)}
    assert candidate.validate_archived_full_physics(tmp_path, archive.name, latitude, pins)["sun_columns"] == s
    sidecar.write_text(sidecar.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        candidate.validate_archived_full_physics(tmp_path, archive.name, latitude, pins)

    receipt["output_sha256"] = "0" * 64
    write_json(sidecar, receipt)
    pins[sidecar.name] = digest(sidecar)
    with pytest.raises(ValueError, match="exact reference archive"):
        candidate.validate_archived_full_physics(tmp_path, archive.name, latitude, pins)
    pins.pop(sidecar.name)
    with pytest.raises(ValueError, match="omits"):
        candidate.validate_archived_full_physics(tmp_path, archive.name, latitude, pins)


@pytest.mark.parametrize("throws", (False, True))
def test_historical_callback_restored_and_root_scoped(tmp_path, monkeypatch, throws):
    original = candidate.historical.full_physics
    monkeypatch.setattr(candidate, "validate_archived_full_physics", lambda *args: {"ok": True})
    observations = {}
    with pytest.raises(RuntimeError) if throws else _no_exception_expected():
        with candidate._isolated_historical_replay(tmp_path, {}, observations):
            assert candidate.historical.full_physics(tmp_path, "reference.npz", -85.0)
            with pytest.raises(ValueError, match="duplicate"):
                candidate.historical.full_physics(tmp_path, "reference.npz", -85.0)
            with pytest.raises(ValueError, match="Unexpected root"):
                candidate.historical.full_physics(Path("/"), "other.npz", -85.0)
            with pytest.raises(RuntimeError, match="isolated"):
                with candidate._isolated_historical_replay(tmp_path, {}, {}):
                    pytest.fail("nested replay must fail before replacing the historical callback")
            if throws:
                raise RuntimeError("forced installer failure")
    assert candidate.historical.full_physics is original
    assert observations == {"reference.npz": {"ok": True}}


def _no_exception_expected():
    from contextlib import nullcontext
    return nullcontext()
