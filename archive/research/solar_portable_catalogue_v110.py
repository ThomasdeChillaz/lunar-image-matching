"""Strict v110 replay of frozen reference Sun archives on the ARM worker.

Only analytic recomputation uses a tolerance. Historical receipt and archive
identities, original ray availability and the generic lattice remain exact.
"""

from contextlib import contextmanager
from pathlib import Path
import threading

import numpy as np

import solar_capsule_v75 as capsule
import solar_portable_evidence_v75 as historical
from solar_eval_data_v69 import validate_reference_archive
from solar_sun_search_v68 import sun_search_catalogue

MAX_SUN_COMPONENT_ROUNDOFF = 2e-15
HISTORICAL_FULL_PHYSICS = historical.full_physics


def _exact_floats(actual, expected):
    return (
        isinstance(actual, np.ndarray)
        and actual.dtype == np.float64
        and actual.shape == expected.shape
        and np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    )


def _exact_mask(actual, expected):
    return (
        isinstance(actual, np.ndarray)
        and actual.dtype == np.bool_
        and np.array_equal(actual, expected)
    )


def validate_frozen_generic_catalogue(geometry, receipt, latitude_deg):
    """Validate a full primary archive against its pinned precompute receipt."""
    observed = receipt.get("sun_catalogue")
    if (
        receipt.get("complete") is not True
        or receipt.get("precompute") != "full"
        or receipt.get("query_truth_used") is not False
        or type(receipt.get("neural_forwards")) is not int
        or receipt["neural_forwards"] != 0
        or type(receipt.get("uncomputed_solar_columns")) is not int
        or receipt["uncomputed_solar_columns"] != 0
        or not isinstance(observed, list)
        or not observed
        or type(receipt.get("computed_solar_columns")) is not int
        or receipt["computed_solar_columns"] != len(observed)
    ):
        raise ValueError("Complete reference-only precompute receipt required")
    if any(
        not isinstance(row, dict)
        or set(row) != {"hour_angle_deg", "subsolar_latitude_deg", "sun_enu", "coarse"}
        or type(row["hour_angle_deg"]) is not float
        or type(row["subsolar_latitude_deg"]) is not float
        or type(row["coarse"]) is not bool
        or not isinstance(row["sun_enu"], list)
        or len(row["sun_enu"]) != 3
        or any(type(value) is not float for value in row["sun_enu"])
        for row in observed
    ):
        raise ValueError("Malformed frozen Sun catalogue receipt")

    expected = sun_search_catalogue(latitude_deg)
    receipt_parameters = np.array([
        [row["hour_angle_deg"], row["subsolar_latitude_deg"]] for row in observed
    ])
    expected_parameters = np.array([
        [row["hour_angle_deg"], row["subsolar_latitude_deg"]] for row in expected
    ])
    receipt_coarse = np.array([row["coarse"] for row in observed], dtype=bool)
    expected_coarse = np.array([row["coarse"] for row in expected], dtype=bool)
    if not (
        _exact_floats(geometry.get("sun_parameters"), receipt_parameters)
        and _exact_floats(receipt_parameters, expected_parameters)
        and _exact_mask(geometry.get("coarse_sun_mask"), receipt_coarse)
        and _exact_mask(receipt_coarse, expected_coarse)
        and expected_coarse.any()
    ):
        raise ValueError("Generic Sun lattice membership, order or coarse mask changed")
    if not _exact_mask(geometry.get("solar_materialized_mask"), np.ones(len(expected), bool)):
        raise ValueError("Full primary availability must be an exact all-true boolean mask")

    archived_suns = np.array([row["sun_enu"] for row in observed])
    if not _exact_floats(geometry.get("sun_vectors"), archived_suns):
        raise ValueError("Archive Sun directions differ from the pinned precompute receipt")
    if (
        not np.isfinite(archived_suns).all()
        or not (archived_suns[:, 2] > 0).all()
        or not np.allclose(
            np.linalg.norm(archived_suns, axis=1), 1.0, rtol=0.0, atol=MAX_SUN_COMPONENT_ROUNDOFF
        )
    ):
        raise ValueError("Frozen Sun directions must be finite, unit and above the horizon")
    computed_suns = np.array([row["sun_enu"] for row in expected])
    difference = float(np.max(np.abs(archived_suns - computed_suns)))
    if not np.isfinite(difference) or difference > MAX_SUN_COMPONENT_ROUNDOFF:
        raise ValueError("Recomputed Sun differs from frozen direction beyond libm roundoff")
    return {"sun_columns": len(expected), "max_component_roundoff": difference}


def validate_archived_full_physics(root, archive, latitude, pins):
    """Bind replay to the exact historical archive and its exact adjacent receipt."""
    receipt_path = archive + ".json"
    if archive not in pins or receipt_path not in pins:
        raise ValueError("Portable evidence omits the exact archive or its original receipt")
    archive_file = historical.pin(root, archive, pins[archive])
    receipt = historical.read(historical.pin(root, receipt_path, pins[receipt_path]))
    if receipt.get("output_sha256") != pins[archive]:
        raise ValueError("Original receipt does not identify the exact reference archive")
    geometry = validate_reference_archive(archive_file)
    return validate_frozen_generic_catalogue(geometry, receipt, latitude)


@contextmanager
def _isolated_historical_replay(payload, pins, observations):
    """For one dedicated, single-Python-thread worker installer only."""
    if (
        threading.current_thread() is not threading.main_thread()
        or threading.active_count() != 1
        or historical.full_physics is not HISTORICAL_FULL_PHYSICS
    ):
        raise RuntimeError("Versioned portable install requires an isolated historical replay")
    expected_root = Path(payload).resolve()

    def check_archive(root, archive, latitude):
        if Path(root).resolve() != expected_root or archive in observations:
            raise ValueError("Unexpected root or duplicate archive during scoped replay")
        observations[archive] = validate_archived_full_physics(root, archive, latitude, pins)
        return True

    historical.full_physics = check_archive
    try:
        yield
    finally:
        historical.full_physics = HISTORICAL_FULL_PHYSICS


def install_verified_capsule(capsule_dir, destination, expected_manifest_sha256, *, install=True):
    """Replay all original capsule gates with only the full-physics callback isolated.

    Before invoking this prototype, the new controller must validate the signed
    fresh binding and verify public transport. This helper grants no authority.
    The child workflow should record its per-archive roundoff diagnostics.
    """
    root = Path(capsule_dir)
    payload = root / "payload"
    manifest = historical.read(historical.pin(root, "manifest.json", expected_manifest_sha256))
    evidence_ref = manifest["portable_reference_evidence"]
    evidence = historical.read(historical.pin(payload, evidence_ref["path"], evidence_ref["sha256"]))
    pins = evidence["pinned_inputs"]
    archives = evidence["result"]["reference_archives"]
    if not archives or len(archives) != len(set(archives)):
        raise ValueError("Unique historical physical archives required")
    observations = {}
    with _isolated_historical_replay(payload, pins, observations):
        # Run the unchanged validator first so wrong coverage cannot install anything.
        capsule.validate_capsule(root)
        if set(observations) != set(archives):
            raise ValueError("Portable replay did not validate every frozen physical archive")
        if install:
            # The unchanged installer validates again before writing, including hashes.
            observations.clear()
            result = capsule.install_capsule(root, destination)
        else:
            result = {"complete": True, "installed_files": 0, "scientific_execution": False}
    if set(observations) != set(archives):
        raise ValueError("Portable replay archive coverage changed")
    return {"installer": result, "archives": observations}
