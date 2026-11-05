"""Procedural v117 checks and native timing callable for an authorized controller."""

from time import perf_counter

_IMPORT_START = perf_counter()

import argparse
from hashlib import file_digest
import json
import math
import os
from pathlib import Path

import numpy as np

from north_render_plan_v31 import camera_intrinsics, camera_to_world
from warp_candidates import ProjectionBank, build_reference_cloud, rank_pose_bank, reference_pose_bank


_IMPORT_SECONDS = perf_counter() - _IMPORT_START
POSES = 32
SCALES = (32, 16)
FULL_PAIRINGS = 16
FULL_BANK = 2048
FULL_REFERENCE_BANKS = 4
FULL_REFERENCE_CLOUDS = 8
CAP_SECONDS = 180.
STORAGE_CAP_BYTES = 512 * 1024**2
MARGIN = 1.5
ARCHIVE_IO_AND_SEAL_ALLOWANCE_SECONDS = 20.
REPETITIONS = 2
TRUE_INDEX = 17


def _generated(size, count):
    rng = np.random.default_rng(20261104)
    reference = (1e-7 + 1e-6 * rng.random((size//16, size//16))).repeat(16, 0).repeat(16, 1)
    k = np.asarray(camera_intrinsics(size))
    ref_pose = np.asarray(camera_to_world([0., 0., 1250.], [0., 0., 0.]))
    depth = np.full_like(reference, 1250.)
    depth[:size//2] = 1000.

    # Ground truth for the fixture is an independently calculated integer shift
    # on each horizontal layer. No prediction from the matcher makes the query.
    query = np.full_like(reference, 1e-7)
    half = size // 2
    query[:half, :size-40] = reference[:half, 40:]
    query[half:, :size-32] = reference[half:, 32:]
    # Multiples of 32 far-plane pixels align both layers with the sampled
    # subdetector. Small distinct offsets within those bins exercise scoring
    # for every pose without tying an artificial near-true duplicate for first.
    wrong = [base + delta for base in (-96, -64, -32, 0, 64, 96)
             for delta in (-.24, -.16, -.08, 0., .08, .16)]
    if size < 512:
        wrong = [0., -32., -.16]
    translations = wrong[:count-1]
    true_index = min(TRUE_INDEX, count-1)
    translations.insert(true_index, 32.)
    poses = np.repeat(ref_pose[None], count, axis=0)
    poses[:, 0, 3] = np.asarray(translations) * depth[-1, 0] / k[0, 0]
    return reference, depth, k, ref_pose, query, poses, true_index


def _known_transform(result, index, count):
    primary = result["orders"]["geometry_3d"]
    losses = {name: scores[index] for name, scores in result["scores"].items()}
    all_arms_scored = all(loss is not None and math.isfinite(loss) for loss in losses.values())
    passed = (result["reason"] is None and result["candidate_count"] == count and bool(primary)
              and primary[0] == index and all_arms_scored
              and losses["geometry_3d"] <= 1e-12
              and losses["planar"] > losses["geometry_3d"]
              and losses["gradient"] > losses["geometry_3d"])
    return dict(passed=passed, expected_index=index, order=result["orders"], true_candidate_losses=losses,
                common_supported_candidates=result["common_supported_candidates"], reason=result["reason"])


def tiny_functional():
    reference, depth, k, pose, query, bank, index = _generated(128, 4)
    cloud = build_reference_cloud(reference, depth, k, pose)
    return {str(scale): _known_transform(rank_pose_bank(cloud, query, k, bank, cell_pixels=scale,
                                                       min_cells_per_quadrant=1), index, len(bank))
            for scale in SCALES}


def _check_arrays(evidence, fixed_cells, candidate_count):
    shapes = {"query_fixed": (fixed_cells,), "indices": (candidate_count,)}
    for name in ("observed", "geometry_3d", "planar", "counts"):
        shapes[name] = (candidate_count, fixed_cells)
    if set(evidence) != set(shapes) or any(np.asarray(evidence[k]).shape != shape for k, shape in shapes.items()):
        raise ValueError("Every procedural pose must exercise full fixed-cell evidence at both scales")
    if (not np.array_equal(evidence["indices"], np.arange(candidate_count))
            or evidence["counts"].dtype != np.int16 or np.any(evidence["counts"] <= 0)
            or any(not np.isfinite(evidence[k]).all() for k in ("query_fixed", "observed", "geometry_3d", "planar"))):
        raise ValueError("Uncompressed evidence must retain every finite observation, arm and common-bin count")


def _write_evidence(path, evidence):
    with path.open("xb") as stream:
        np.savez(stream, **evidence)
        stream.flush()
        os.fsync(stream.fileno())
    return path.stat().st_size


def forecast(samples, *, startup, fixture, bank, cloud, cold):
    """Scale both measured rank and evidence-write costs to the complete scope."""
    factor = FULL_PAIRINGS * FULL_BANK / POSES
    scales = {}
    for scale in SCALES:
        part = [row for row in samples if row["cell_pixels"] == scale]
        if len(part) != REPETITIONS:
            raise ValueError("Both native scales need every timing repeat")
        writes = part + ([cold] if scale == cold["cell_pixels"] else [])
        scales[str(scale)] = dict(
            rank_seconds=max(row["rank_seconds"] for row in part) * factor,
            uncompressed_evidence_write_seconds=max(row["evidence_write_seconds"] for row in writes) * factor,
            uncompressed_evidence_bytes=math.ceil(max(row["evidence_bytes"] for row in writes) * factor),
        )
    setup = dict(python_import_seconds=startup, fixture_conservative_allowance_seconds=fixture,
                 reference_pose_banks_seconds=FULL_REFERENCE_BANKS * bank * (FULL_BANK - 1) / (POSES - 1),
                 reference_clouds_seconds=FULL_REFERENCE_CLOUDS * cloud,
                 full_cold_projection_seconds=FULL_REFERENCE_BANKS * cold["rank_seconds"] * FULL_BANK / POSES,
                 archive_read_verify_and_seal_allowance_seconds=ARCHIVE_IO_AND_SEAL_ALLOWANCE_SECONDS)
    rank = sum(row["rank_seconds"] for row in scales.values())
    writes = sum(row["uncompressed_evidence_write_seconds"] for row in scales.values())
    projected_seconds = MARGIN * (sum(setup.values()) + rank + writes)
    projected_bytes = sum(row["uncompressed_evidence_bytes"] for row in scales.values())
    return dict(projected_seconds=projected_seconds, projected_uncompressed_evidence_bytes=projected_bytes,
                admitted=projected_seconds <= CAP_SECONDS and projected_bytes <= STORAGE_CAP_BYTES,
                cap_seconds=CAP_SECONDS, evidence_storage_cap_bytes=STORAGE_CAP_BYTES,
                full_pairings=FULL_PAIRINGS, theoretical_candidates_per_pairing=FULL_BANK,
                all_candidates_retained_for_write_forecast=True, native_scales=list(SCALES),
                case_scale_factor=factor, timing_margin=MARGIN, by_scale=scales, setup=setup,
                projection_cache_geometry_bytes_per_case=math.ceil(cold["cache_bytes"] * FULL_BANK / POSES))


def _sample(check, directory, tag, cloud, query, k, bank, true_index, scale, cache=None):
    check()
    evidence = {}
    tick = perf_counter()
    if cache is None:
        cache = ProjectionBank(cloud, k, bank, query.shape, 4)
    result = rank_pose_bank(cloud, query, k, bank, cell_pixels=scale, evidence=evidence, projection_bank=cache)
    rank_seconds = perf_counter() - tick
    control = _known_transform(result, true_index, POSES)
    _check_arrays(evidence, result["fixed_cells"], POSES)
    check()
    path = directory / f"cell-{scale}-{tag}.npz"
    tick = perf_counter()
    size = _write_evidence(path, evidence)
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    write_seconds = perf_counter() - tick
    check()
    return cache, dict(cell_pixels=scale, repeat=tag, rank_seconds=rank_seconds, control=control,
                       evidence_write_seconds=write_seconds, evidence_bytes=size, evidence_file=path.name,
                       evidence_sha256=digest)


def measure_512(check, evidence_dir):
    """Run the fixed native timing from a controller with live budget/deadline checks."""
    if not callable(check):
        raise TypeError("A controlling guard must check the terminal predecessor and native timing budget")
    check()
    for key, expected in {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"Set {key}={expected} before loading NumPy")
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(exist_ok=False)

    tick = perf_counter()
    reference, depth, k, pose, query, bank, true_index = _generated(512, POSES)
    fixture_seconds = perf_counter() - tick
    check()
    tick = perf_counter()
    generated_prior = reference_pose_bank(pose, count=POSES)
    bank_seconds = perf_counter() - tick
    if len(generated_prior) != POSES:
        raise ValueError("Procedural pose-bank timing must include all 32 camera constructions")
    tick = perf_counter()
    cloud = build_reference_cloud(reference, depth, k, pose)
    cloud_seconds = perf_counter() - tick

    cache, cold = _sample(check, evidence_dir, "cold", cloud, query, k, bank, true_index, SCALES[0])
    if len(cache.indices) != POSES:
        raise ValueError("Cold timing must construct the complete reference projection cache")
    cold["cache_bytes"] = sum(array.nbytes for indices in cache.indices.values() for array in indices)
    samples = []
    for repeat in range(REPETITIONS):
        for scale in SCALES:
            _, sample = _sample(check, evidence_dir, f"warm-{repeat}", cloud, query, k, bank, true_index, scale, cache)
            samples.append(sample)

    projection = forecast(samples, startup=_IMPORT_SECONDS, fixture=fixture_seconds,
                          bank=bank_seconds, cloud=cloud_seconds, cold=cold)
    receipt = dict(schema="solar_warp_procedural_timing_v117", procedural_only=True,
                   archived_lunar_images_opened=False, native_shape=[512, 512], posed_candidates=POSES,
                   fixture_translation_native_pixels=[40, 32], reference_samples=len(cloud.radiance),
                   passed=projection["admitted"] and all(row["control"]["passed"] for row in [cold, *samples]),
                   cold_projection_sample=cold, warm_samples=samples, full_scope_forecast=projection,
                   timing_scope="Measured generated fixture. Full-scope time and storage are forecasts, not archive runtime.")
    check()
    with (evidence_dir / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not receipt["passed"]:
        raise RuntimeError("Procedural accuracy or the fixed 180-second/512-MiB admission gate failed")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-functional", action="store_true", help="Run only a generated 128-pixel functional fixture")
    args = parser.parse_args()
    if not args.tiny_functional:
        parser.error("Only --tiny-functional is callable directly. Native timing requires the guarded controller.")
    result = tiny_functional()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in result.values()) else 1)
