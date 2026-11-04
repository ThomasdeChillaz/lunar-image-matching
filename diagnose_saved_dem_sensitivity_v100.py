"""Exact whole-map height-shift refits of the sealed v99 direct observations."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time

import numpy as np


DISPATCH = "reports/solar-full-crosslight-rescue-dispatch-v99.json"
DISPATCH_SHA = "5ca6a4a5de7c7dcd3987db92007d7485074485bb746ebcecf5efef8b08612817"
REPLAY = "reports/solar_full_crosslight_rescue_replay_v99.json"
HEIGHTS = (0., -5., 5., -10., 10., -20., 20.)
MAX_SECONDS = 900


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require(path, expected):
    if sha(path) != expected:
        raise ValueError("Frozen input changed: " + str(path))


def shifted_intersections(pixels, intrinsics, reference_pose, raycast):
    """As in solar_lookup_worker_v70, but at the arbitrary archived pixels."""
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsics).T @ reference_pose[:3, :3].T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    world = np.full((len(rays), len(HEIGHTS), 3), np.nan)
    for column, dz in enumerate(HEIGHTS):
        offset = np.array([0., 0., dz])
        for row, ray in enumerate(rays):
            hit = raycast(reference_pose[:3, 3] - offset, ray, 1e8)
            if hit is not None:
                world[row, column] = np.asarray(hit, float) + offset
    return world


def mesh_worker(request, output):
    """Read only reference mesh, reference cameras and fixed reference pixels."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    data = json.loads(Path(request).read_text())
    if set(data) != {"meshes", "records"}:
        raise ValueError("Only frozen reference-side ray requests are accepted")
    bvh_by_mesh, arrays = {}, {}
    for record in data["records"]:
        if set(record) != {"key", "mesh", "reference_xy", "reference_intrinsics", "reference_pose"}:
            raise ValueError("Worker must receive only reference-side data")
        p = np.asarray(record["reference_xy"], float).reshape(-1, 2)
        if not len(p):
            arrays[record["key"]] = np.empty((0, len(HEIGHTS), 3))
            continue
        mesh = record["mesh"]
        if mesh not in bvh_by_mesh:
            require(mesh, data["meshes"][mesh])
            with np.load(mesh, allow_pickle=False) as handle:
                vertices, faces = handle["vertices"], handle["faces"]
            bvh_by_mesh[mesh] = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True, epsilon=0.)
        bvh = bvh_by_mesh[mesh]

        def raycast(origin, direction, limit):
            return bvh.ray_cast(Vector(origin), Vector(direction), float(limit))[0]

        arrays[record["key"]] = shifted_intersections(p, np.asarray(record["reference_intrinsics"], float),
                                                      np.asarray(record["reference_pose"], float), raycast)
    with Path(output).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def fixed_data():
    """Reopen the exact full run only after its watchdog and independent replay."""
    from run_solar_distributional_evaluation_v90 import fixed_v86_cases
    from run_solar_full_crosslight_rescue_v99 import DIRECT, completion, verify_replayed
    from solar_active_v63 import HEIGHT_VARIANTS_M
    from solar_eval_data_v69 import HEIGHT_VARIANTS

    if HEIGHTS != HEIGHT_VARIANTS_M or HEIGHTS != HEIGHT_VARIANTS:
        raise ValueError("Exact original v63/v69 height offsets required")
    require(DISPATCH, DISPATCH_SHA)
    protocol = json.loads(Path(DISPATCH).read_text())
    _, manifest, cases = fixed_v86_cases(protocol["basis"], full=True)
    if (protocol["split"] != "model_selection" or protocol["calibration_or_final_access"] is not False
            or protocol["tracks"] != ["estimated_sun"] or len(manifest["terrain_groups"]) != 8
            or len(cases) != 384 or sum(case["positive"] for case in cases) != 192):
        raise ValueError("Exact 192/192 primary development grid required")
    summary, _, seal, rows, _ = completion(DISPATCH, protocol)
    verify_replayed(REPLAY, DISPATCH, summary, seal, rows, two_arms=True)
    direct = [row for row in rows if row["profile"] == DIRECT]
    if [row["case_id"] for row in direct] != [case["case_id"] for case in cases]:
        raise ValueError("Fresh original direct rows must match every case")
    return protocol, cases, direct, seal


def get_geometry(case, cache):
    reference, lookup_path = case["reference_geometry"], case["solar_lookup_protocol"]
    pins = case["input_sha256"]
    if reference not in cache:
        require(reference, pins[reference])
        require(lookup_path, pins[lookup_path])
        lookup = json.loads(Path(lookup_path).read_text())
        mesh = lookup["mesh"]
        if (lookup["query_truth_used"] is not False or lookup["mesh_sha256"] != pins[mesh]
                or tuple(case["reference_asset_metadata"]["height_variants_metres"]) != HEIGHTS):
            raise ValueError("Pinned reference-only map and original height variants required")
        with np.load(reference, allow_pickle=False) as handle:
            depth, k, pose = (handle[name].copy() for name in ("depth", "intrinsics", "cam2world"))
        cache[reference] = (lookup_path, depth, k, pose, mesh, pins[mesh])
    found, *values = cache[reference]
    if found != lookup_path:
        raise ValueError("Reference lookup identity changed within the fixed case grid")
    return values


def build_requests(cases, rows, check):
    from lunar_geometry_v5 import map_points_z

    geometry, meshes, requests, records = {}, {}, [], []
    for index, (case, row) in enumerate(zip(cases, rows, strict=True)):
        check()
        depth, k, pose, mesh, mesh_sha = get_geometry(case, geometry)
        if meshes.setdefault(str(Path(mesh).resolve()), mesh_sha) != mesh_sha:
            raise ValueError("One physical mesh path declared with multiple identities")
        with np.load(row["saved_arrays"], allow_pickle=False) as handle:
            p, q, archived = [handle[name].copy() for name in ("reference_xy", "query_xy", "pose")]
        if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("Finite archived reference/query native correspondences required")
        nominal, mapped = map_points_z(p, depth, k, pose)
        key = f"c{index:03d}"
        requests.append(dict(key=key, mesh=str(Path(mesh).resolve()), reference_xy=p[mapped].tolist(),
                             reference_intrinsics=k.tolist(), reference_pose=pose.tolist()))
        records.append(dict(key=key, nominal=nominal, query_xy=q[mapped], reference_xy=p[mapped], k=k,
                            reference_pose=pose, archived_pose=archived, all_pairs=len(p)))
    return dict(meshes=meshes, records=requests), records


def exact_bvh(protocol, request, remaining):
    if remaining <= 0:
        raise TimeoutError("Original height-refit diagnostic allowance exhausted")
    with tempfile.TemporaryDirectory(prefix="lunar-dem-fixed-") as directory:
        source, target = (Path(directory) / name for name in ("reference-requests.json", "world-rays.npz"))
        source.write_text(json.dumps(request, separators=(",", ":"), allow_nan=False))
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   VECLIB_MAXIMUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run([protocol["blender_executable"], "--background", "--factory-startup", "--disable-autoexec",
                                  "--threads", "2", "--python-exit-code", "93", "--python", str(Path(__file__).resolve()),
                                  "--", "mesh-worker", str(source), str(target)], env=env, capture_output=True,
                                 text=True, timeout=remaining)
        if process.returncode != 0 or not target.is_file():
            raise RuntimeError("Reference-only BVH worker failed: " + (process.stderr or process.stdout)[-1500:])
        with np.load(target, allow_pickle=False) as arrays:
            return {key: arrays[key].copy() for key in arrays.files}


def pose_delta(a, b):
    angle = np.rad2deg(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.) / 2., -1., 1.)))
    return dict(rotation_deg=float(angle), translation_m=float(np.linalg.norm(a[:3, 3] - b[:3, 3])))


def refit(world, pixels, query_intrinsics):
    from localization_robust_pose_v4 import CONFIGS, estimate

    config = next(c for c in CONFIGS if c["name"] == "ap3p_reclassify")
    return estimate(world, pixels, query_intrinsics, config)


def score_pose(pose, info, truth, scale, base):
    error = pose_delta(pose, truth) if pose is not None and truth is not None else None
    correct = error is not None and error["rotation_deg"] <= 2. and error["translation_m"] <= .01 * scale
    strict = error is not None and error["rotation_deg"] <= .5 and error["translation_m"] <= .001 * scale
    return dict(pose_returned=pose is not None, reason=info["reason"], inliers=info["inliers"],
                pose_error=error, correct=correct, strict=strict, wrong_positive=error is not None and not correct,
                raw_nonoverlap=truth is None and pose is not None,
                pose_drift_from_matched_zero=None if pose is None or base is None else pose_delta(pose, base))


def case_refits(case, row, rec, worlds):
    q = rec["query_xy"]
    k = np.asarray(case["query_intrinsics"], float)
    if worlds.shape != (len(q), len(HEIGHTS), 3):
        raise ValueError("Ray results do not match the frozen saved observations")
    hits = np.isfinite(worlds).all(2)
    common = hits.all(1)
    archival = rec["archived_pose"]
    if archival.shape not in ((4, 4), (0, 4)):
        raise ValueError("Frozen archival pose shape differs")
    archived_pose = archival if len(archival) else None
    if row["inference_failure"] is not None and (rec["all_pairs"] != 0 or archived_pose is not None):
        raise ValueError("Failed archived inference must not supply invented observations")
    if row["inference_failure"] is None:
        nominal, _ = refit(rec["nominal"], q, k)
        if ((nominal is None) != (archived_pose is None)
                or nominal is not None and not np.allclose(nominal, archived_pose, rtol=0., atol=1e-7)):
            raise ValueError("Original zero-height nominal solver did not reproduce the sealed pose")
    matched = [refit(worlds[common, v], q[common], k) for v in range(len(HEIGHTS))]
    # Read the saved scoring-only target only after all poses have been fit.
    if case["positive"]:
        with np.load(row["saved_arrays"], allow_pickle=False) as scoring:
            truth = scoring["scoring_query_cam2world"].copy()
    else:
        truth = None
    scale = row["metrics"]["reference_range_m"]
    base = matched[0][0]
    scores = [dict(height_offset_m=height, per_offset_supported_pairs=int(hits[:, v].sum()),
                   **score_pose(pose, info, truth, scale, base))
              for v, (height, (pose, info)) in enumerate(zip(HEIGHTS, matched, strict=True))]
    bvh_zero_error = np.linalg.norm(worlds[hits[:, 0], 0] - rec["nominal"][hits[:, 0]], axis=1)
    return dict(case_id=case["case_id"], positive=case["positive"], component=case["component"],
                query_component=case["query_component"], light_a=case["light_a"], light_b=case["light_b"],
                inference_failure=row["inference_failure"], archived_correct=row["metrics"]["correct"],
                archived_pose_returned=row["metrics"]["pose_returned"], raw_pairs=rec["all_pairs"],
                nominal_mapped_pairs=len(q), all_offsets_supported_pairs=int(common.sum()), reference_range_m=scale,
                exact_zero_bvh_vs_nominal_surface_median_m=None if not len(bvh_zero_error) else float(np.median(bvh_zero_error)),
                exact_zero_bvh_vs_archived_pose_drift=None if base is None or archived_pose is None else pose_delta(base, archived_pose),
                variants=scores)


def summarize(cases, check):
    from solar_evaluation_v70 import paired_component_bootstrap

    def paired_rows(items, variant, name):
        return [dict(case_id=r["case_id"], component=r["component"], query_component=r["query_component"],
                     positive=r["positive"], track="estimated_sun", profile=name, metrics=r["variants"][variant]) for r in items]

    def median_or_none(values):
        return None if not values else float(median(values))

    groups = defaultdict(list)
    for case in cases:
        groups["all"].append(case)
        light = "same_light" if case["light_a"] == case["light_b"] else "changed_light"
        groups[light].append(case)
    output = {}
    for name, group in groups.items():
        baseline = paired_rows(group, 0, "zero")
        positive = [case for case in group if case["positive"]]
        negative = [case for case in group if not case["positive"]]
        by_offset = []
        for index, height in enumerate(HEIGHTS):
            check()
            pos, neg = ([case["variants"][index] for case in side] for side in (positive, negative))
            drift = [v["pose_drift_from_matched_zero"] for v in pos if v["pose_drift_from_matched_zero"] is not None]
            interval = None if index == 0 else paired_component_bootstrap(
                paired_rows(group, index, "height") + baseline, "height", "zero",
            )
            by_offset.append(dict(height_offset_m=height, correct_positive=sum(v["correct"] for v in pos),
                                  strict_positive=sum(v["strict"] for v in pos), wrong_positive=sum(v["wrong_positive"] for v in pos),
                                  positive_pose_outputs=sum(v["pose_returned"] for v in pos),
                                  raw_nonoverlap_pose_outputs=sum(v["raw_nonoverlap"] for v in neg),
                                  bvh_supported_pairs=sum(v["per_offset_supported_pairs"] for v in pos + neg),
                                  positive_drift_pairs=len(drift),
                                  median_positive_translation_drift_m=median_or_none([d["translation_m"] for d in drift]),
                                  median_positive_rotation_drift_deg=median_or_none([d["rotation_deg"] for d in drift]),
                                  paired_terrain_difference_interval_vs_matched_zero=interval))
        output[name] = dict(positive_cases=len(positive), nonoverlap_cases=len(negative),
                            archive_raw_correct=sum(r["archived_correct"] for r in positive),
                            archive_raw_nonoverlap=sum(r["archived_pose_returned"] for r in negative),
                            unavailable_original_inferences=sum(r["inference_failure"] is not None for r in group),
                            median_zero_bvh_vs_archived_map_discrepancy_m=median_or_none([
                                r["exact_zero_bvh_vs_nominal_surface_median_m"] for r in group
                                if r["exact_zero_bvh_vs_nominal_surface_median_m"] is not None
                            ]),
                            common_support_pairs=sum(r["all_offsets_supported_pairs"] for r in group), offsets=by_offset)
    return output


def run(output, max_seconds):
    if type(max_seconds) is not int or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("Set an explicit bounded allowance of at most 900 seconds")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve earlier diagnostic outputs")
    started = time.monotonic()
    deadline = started + max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Original height-refit diagnostic allowance exhausted")

    protocol, cases, rows, seal = fixed_data()
    check()
    import cv2

    cv2.setNumThreads(1)
    if cv2.__version__ != protocol["expected_software"]["opencv"] or np.__version__ != protocol["expected_software"]["numpy"]:
        raise ValueError("Use the original OpenCV/NumPy solver stratum")
    request, records = build_requests(cases, rows, check)
    worlds = exact_bvh(protocol, request, deadline - time.monotonic())
    if set(worlds) != {record["key"] for record in records}:
        raise ValueError("Exact complete physical ray case grid required")
    results = []
    for case, row, record in zip(cases, rows, records, strict=True):
        check()
        results.append(case_refits(case, row, record, worlds[record["key"]]))
    strata = summarize(results, check)
    check()
    report = dict(complete=True, cases=len(results), positive_cases=192, nonoverlap_cases=192, terrain_groups=8,
                  split="model_selection", accepted_results=False, calibration_or_final_access=False,
                  input_dispatch_sha256=DISPATCH_SHA, input_replay_sha256=sha(REPLAY),
                  input_summary_sha256=seal["summary_sha256"], input_watchdog_sha256=seal["watchdog_sha256"],
                  source_sha256=sha(Path(__file__)), cpu_threads=2, opencv=cv2.__version__, numpy=np.__version__,
                  max_calendar_seconds=max_seconds, elapsed_seconds=time.monotonic() - started,
                  pose_solver=dict(name="ap3p_reclassify", seed=20261101, iterations=2000,
                                   reprojection_threshold_native_px=3., confidence=.99),
                  height_offsets_m=HEIGHTS, source_map="Pinned v69 reference mesh, exact BVH ray reintersection",
                  common_support="Archived nominal-depth-valid pairs intersecting all seven shifted surfaces. No-hit support is reported by offset.",
                  metric_scale="Original nominal median Euclidean reference range, held fixed across offsets.",
                  zero_control="The original rounded axial-depth AP3P solution must reproduce the archived pose. Height effects are paired against exact-BVH zero using identical support.",
                  scope="Fixed-correspondence pose sensitivity only. No matching, active acquisition, new images, lighting changes, confidence fitting or reserve access.",
                  perturbation_limit="Uniform whole-map offsets are the original convention. No heterogeneous DEM errors are tested. A fronto-parallel planar map can show a scale gauge without rotation or residual degradation.",
                  strata=strata, rows=results)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    if "--" in sys.argv:
        mode, source, output = sys.argv[sys.argv.index("--") + 1:]
        if mode != "mesh-worker":
            raise ValueError("Unknown reference-only child action")
        mesh_worker(source, output)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.output, args.max_seconds)


if __name__ == "__main__":
    main()
