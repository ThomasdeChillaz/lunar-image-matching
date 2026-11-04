"""Bounded v112 geometry, independent pose gate and six raw sentry exposures."""

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_solar_raw_acquisition_v109 import (
    BLENDER_BUILD, BLENDER_SHA, SIZE, add_outputs, check_exr_header, digest,
    read, relative, render_settings, setup_camera, utc, write,
)


SOURCE = "solar_secondary_acquire_v112.py"
ROSTER = "reports/solar-secondary-ranking-roster-v112.json"
ROSTER_SHA256 = "59ea302a5bd9f924dd03366ebe66b92b5458ea29a4abb47371ae205ae6e3826f"
ROOT = "runs/solar-secondary-acquisition-v112"
ROLES = ("reference", "query")
ENDPOINTS = ("np31-r0-c7", "np31-r3-c2", "np31-r3-c5", "np31-r4-c1")
SOURCE_PATHS = (
    SOURCE, "render_solar_raw_acquisition_v109.py", "render_solar_clean_radiance_v106.py",
    "render_solar_data_v65.py", "solar_data_v65.py", "solar_physics_v63.py",
    "north_camera_scene_v59.py", "blender_camera_control_v31.py",
    "north_camera_correspondences_v59.py", "north_development_plan_v35.py",
    "north_camera_variation_v59.py", "north_render_plan_v31.py", "north_terrain_pilot_v32.py",
    "verify_solar_development_v68.py", "verify_north_camera_qualified_v64.py",
    "north_camera_visibility_v59.py", "verify_north_development_v35.py", "renderer_depth_v34.py",
    "solar_eval_data_v69.py", "solar_sun_search_v68.py",
    "native_baseline_pose_v7.py", "localization_robust_pose_v4.py",
    "localization_pose_refit_v3.py", "localization_diagnostics_v3.py",
    "lunar_geometry_v5.py", "lunar_geometry_v2.py", "lunargeo_dataset.py", "dataset.py", "remote_archive.py",
)


def keys(row):
    return row["view_id"], row["replicate"], row["seed"]


def approved_lights(endpoint):
    from solar_physics_v63 import lighting_schedule

    schedule = lighting_schedule(endpoint["site"]["centre_latitude_deg"])
    chosen = [schedule[index] for index in (8, 23)]
    if endpoint["lighting"] != chosen:
        raise ValueError("Frozen physical Sun schedule differs")
    return {row["index"]: row for row in chosen}


def frozen_sentry(roster):
    if (roster["schema"] != "solar_secondary_ranking_admission_reconciled_v112"
            or roster["calibration_or_final_access"] is not False
            or tuple(row["endpoint"] for row in roster["endpoints"]) != ENDPOINTS
            or len(roster["render_jobs"]) != 24 or len(roster["cases"]) != 16
            or roster["compute_design"]["cpu_threads_per_job"] != 2
            or roster["compute_design"]["stage_caps_seconds"]["geometry_and_sentry_renders"] != 500):
        raise ValueError("Require the fixed four-endpoint model-selection roster")
    endpoints = {row["endpoint"]: row for row in roster["endpoints"]}
    for row in endpoints.values():
        if (row["split"] != "model_selection" or row["site"]["split"] != "model_selection"
                or row["site"]["component"] != row["endpoint"]
                or row["camera_pair"]["split"] != "model_selection"
                or row["camera_pair"]["component"] != row["endpoint"]):
            raise ValueError("Refuse a camera or terrain outside model selection")
        approved_lights(row)
    all_jobs = {keys(row): row for row in roster["render_jobs"]}
    if len(all_jobs) != 24 or len({seed for _, _, seed in all_jobs}) != 24:
        raise ValueError("Frozen render schedule requires distinct identities and seeds")
    sentry = [all_jobs[keys(row)] for row in roster["early_coverage_stop"]["acquire_first_view_seed_pairs"]]
    expected = {
        ("v112/np31-r0-c7/reference/light-08", 0), ("v112/np31-r0-c7/reference/light-08", 1),
        ("v112/np31-r0-c7/query/light-23", 0), ("v112/np31-r0-c7/query/light-23", 1),
        ("v112/np31-r3-c2/query/light-23", 0), ("v112/np31-r3-c2/query/light-23", 1),
    }
    if len(sentry) != 6 or {(row["view_id"], row["replicate"]) for row in sentry} != expected:
        raise ValueError("Exactly six fixed sentry exposures are required")
    for row in sentry:
        if (row["view_id"] != f'v112/{row["endpoint"]}/{row["role"]}/light-{row["light"]:02d}'
                or row["source"] != "black_space_float32_combined" or row["samples_per_pixel"] != 64):
            raise ValueError("Sentry source, native view or fixed sampling differs")
    return endpoints, sentry


def admission(roster_path, root, commit, deadline_value):
    roster_file, output = relative(roster_path), relative(root)
    if (roster_path != ROSTER or root != ROOT or not output.is_dir()
            or digest(roster_file) != ROSTER_SHA256
            or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("Require the published reconciled roster and the controller-owned acquisition root")
    deadline = utc(deadline_value)

    def clock():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Original acquisition deadline elapsed")

    if (deadline - datetime.now(timezone.utc)).total_seconds() > 500:
        raise ValueError("Worker may not expand the original 500-second stage allowance")
    clock()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    for name in (roster_path, *SOURCE_PATHS):
        clock()
        public = subprocess.check_output(["git", "show", f"{commit}:{name}"], timeout=5)
        if public != relative(name).read_bytes():
            raise ValueError("Published source or roster differs: " + name)
    roster = read(roster_file)
    endpoints, sentry = frozen_sentry(roster)
    for name in SOURCE_PATHS:
        expected = roster["reused_source_sha256"].get(name)
        if expected is not None and digest(name) != expected:
            raise ValueError("Historical source identity differs: " + name)
    return roster, endpoints, sentry, output, clock


def stamp(commit, deadline):
    return dict(roster_sha256=ROSTER_SHA256, publication_commit=commit, deadline_utc=deadline,
                cpu_threads=2, split="model_selection", calibration_or_final_access=False)


def check_blender(bpy):
    if (tuple(bpy.app.version[:3]) != (4, 5, 9)
            or bpy.app.build_hash.decode() != BLENDER_BUILD
            or digest(bpy.app.binary_path) != BLENDER_SHA):
        raise ValueError("Require the original pinned v109 Blender renderer")


def verified_mesh(endpoint):
    path = relative(endpoint["map_mesh"])
    if digest(path) != endpoint["map_mesh_sha256"]:
        raise ValueError("Frozen reference terrain mesh differs")
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in ("vertices", "faces", "body_origin", "enu_rows")}


def new_array(path, **values):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return {"path": str(path), "sha256": digest(path)}


def read_array(row):
    path = relative(row["path"])
    if digest(path) != row["sha256"]:
        raise ValueError("Sealed geometry payload differs: " + str(path))
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def acquisition_geometry(endpoints, root, provenance, clock):
    import bpy
    from north_camera_correspondences_v59 import mesh_correspondences
    from north_camera_scene_v59 import setup_scene
    from north_development_plan_v35 import proposal_grid_native
    from render_solar_data_v65 import ray_geometry

    check_blender(bpy)
    output = root / "geometry"
    output.mkdir()
    cameras, directions = [], []
    started = time.monotonic()
    try:
        for endpoint, spec in endpoints.items():
            clock()
            folder = output / endpoint
            folder.mkdir()
            mesh = verified_mesh(spec)
            pair = spec["camera_pair"]
            intrinsics = {role: np.asarray(pair[role]["intrinsics"]) for role in ROLES}
            poses = {role: np.asarray(pair[role]["cam2world"]) for role in ROLES}
            _, _, _, _, bvh = setup_scene(folder, mesh["vertices"], mesh["faces"], intrinsics["reference"])
            for role in ROLES:
                before = time.monotonic()
                geometry = ray_geometry(bvh, poses[role], intrinsics[role], clock)
                geometry.update(body_origin=mesh["body_origin"], enu_rows=mesh["enu_rows"])
                saved = new_array(folder / f"{role}.npz", **geometry)
                cameras.append(dict(endpoint=endpoint, role=role, **saved,
                                    valid_pixels=int(geometry["valid"].sum()), elapsed_seconds=time.monotonic() - before))
            for role, other in (ROLES, ROLES[::-1]):
                clock()
                label = mesh_correspondences(bvh, poses[role], poses[other], intrinsics[role],
                                             intrinsics[other], proposal_grid_native())
                saved = new_array(folder / f"{role}-to-{other}.npz", **label)
                directions.append(dict(endpoint=endpoint, role=role, target_role=other, **saved,
                                       visible_anchors=int(label["visible"].sum())))
            print("v112 geometry", endpoint, flush=True)
        clock()
        write(output / "COMPLETE.json", dict(**provenance, schema="v112_geometry", complete=True,
              cameras=cameras, directions=directions, renders=0, matcher_calls=0,
              elapsed_seconds=time.monotonic() - started))
    except BaseException as error:
        write(output / "FAILED.json", dict(**provenance, complete=False, cameras=cameras, directions=directions,
                                          error_type=type(error).__name__, error=str(error)))
        raise


def geometry_inventory(root, provenance):
    path = root / "geometry" / "COMPLETE.json"
    receipt = read(relative(str(path)))
    if (receipt["schema"] != "v112_geometry" or receipt["complete"] is not True
            or any(receipt.get(key) != value for key, value in provenance.items())
            or receipt["renders"] != 0 or receipt["matcher_calls"] != 0):
        raise ValueError("Require the sealed geometry-only acquisition in the same allocation")
    expected = [(name, role) for name in ENDPOINTS for role in ROLES]
    inventory = []
    for field in ("cameras", "directions"):
        rows = receipt[field]
        if [(row["endpoint"], row["role"]) for row in rows] != expected:
            raise ValueError("Exact eight-camera and eight-direction geometry inventory required")
        by_key = {}
        for row in rows:
            name, role = row["endpoint"], row["role"]
            other = "query" if role == "reference" else "reference"
            filename = f"{role}.npz" if field == "cameras" else f"{role}-to-{other}.npz"
            wanted = str(path.parent / name / filename)
            if (row["path"] != wanted or (field == "directions" and row["target_role"] != other)):
                raise ValueError("Unexpected protected geometry artifact path")
            by_key[name, role] = row
        inventory.append(by_key)
    return path, *inventory


def audit_camera(row, pair, mesh, clock):
    from north_camera_visibility_v59 import first_mesh_hit
    from verify_north_camera_qualified_v64 import project, qualify_projected_pixels

    role = row["role"]
    geometry = read_array(row)
    valid, position, axial = (geometry[key] for key in ("valid", "position", "axial_depth"))
    pose, k = geometry["cam2world"], geometry["intrinsics"]
    if (valid.shape != (SIZE, SIZE) or valid.dtype != bool or position.shape != (SIZE, SIZE, 3)
            or axial.shape != valid.shape or geometry["normals"].shape != position.shape
            or int(valid.sum()) != row["valid_pixels"]
            or not np.array_equal(pose, np.asarray(pair[role]["cam2world"]))
            or not np.array_equal(k, np.asarray(pair[role]["intrinsics"]))
            or not np.array_equal(geometry["body_origin"], mesh["body_origin"])
            or not np.array_equal(geometry["enu_rows"], mesh["enu_rows"])):
        raise ValueError("Stored native camera, mask or body/ENU frame differs")
    vertices, faces = mesh["vertices"], mesh["faces"]
    reasons, qualification, mesh_error, depth_closed = [], None, None, False
    if not valid.all():
        reasons.append("incomplete_finite_surface")
    else:
        if not np.isfinite(position).all() or not np.isfinite(geometry["normals"]).all():
            raise ValueError("A valid surface contains nonfinite 3D coordinates or normals")
        y, x = np.indices(valid.shape)
        projected, projected_depth = project(position.reshape(-1, 3), k, pose)
        qualification = qualify_projected_pixels(projected, np.column_stack((x.ravel(), y.ravel())), projected_depth)
        depth_closed = bool(np.allclose(projected_depth.reshape(valid.shape), axial, rtol=1e-10, atol=1e-6))
        if not qualification["qualified"]:
            reasons.append("native_pixel_centre_qualification_failed")
        if not depth_closed:
            reasons.append("axial_depth_closure_failed")
        independent = []
        for point in position[32:-32:128, 32:-32:128].reshape(-1, 3):
            clock()
            ray = point - pose[:3, 3]
            distance = float(np.linalg.norm(ray))
            hit = first_mesh_hit(pose[:3, 3], ray / distance, vertices, faces)
            independent.append(abs(hit - distance))
        error = max(independent)
        mesh_error = float(error) if np.isfinite(error) else None
        if mesh_error is None or mesh_error > .05:
            reasons.append("independent_native_ray_hit_failed")
    clock()
    clearance = first_mesh_hit(pose[:3, 3], [0, 0, -1], vertices, faces)
    free = bool(np.isfinite(clearance) and clearance > 10.)
    if not free:
        reasons.append("camera_not_certified_free_space")
    return geometry, dict(role=role, complete_surface=bool(valid.all()), body_local_frame_verified=True,
                          pixel_qualification=qualification, axial_depth_closed=depth_closed,
                          maximum_independent_mesh_error_m=mesh_error, free_space=free,
                          camera_clearance_m=float(clearance) if np.isfinite(clearance) else None,
                          qualification_reasons=reasons)


def audit_labels(rows, geometries, mesh, clock):
    from north_camera_visibility_v59 import visibility_check
    from verify_north_camera_qualified_v64 import check_labels

    audit, labels, reasons = [], {}, []
    for row in rows:
        clock()
        role, other = row["role"], row["target_role"]
        a, b = geometries[role], geometries[other]
        exact = read_array(row)
        visible_fraction = check_labels(exact, a["cam2world"], b["cam2world"], a["intrinsics"], b["intrinsics"])
        if int(exact["visible"].sum()) != row["visible_anchors"]:
            raise ValueError("Sealed visibility count differs")
        if visible_fraction < .2:
            reasons.append(role + ":insufficient_visible_overlap")
        independent = []
        for state in ("visible", "geometrically_occluded", "visibility_uncertain"):
            ids = np.flatnonzero(exact[state])
            if len(ids):
                clock()
                index = int(ids[0])
                check = visibility_check(b["cam2world"][:3, 3], exact["world"][index],
                                         mesh["vertices"], mesh["faces"])
                independent.append(dict(state=state, anchor=index, **check))
                if not check[state]:
                    reasons.append(role + ":independent_visibility_disagrees:" + state)
        audit.append(dict(source=role, visible_fraction=visible_fraction,
                          visible_anchors=row["visible_anchors"], independent_visibility=independent,
                          occluded_anchors=int(exact["geometrically_occluded"].sum()),
                          uncertain_anchors=int(exact["visibility_uncertain"].sum())))
        labels[role] = exact
    return labels, audit, reasons


def oracle_results(geometries, labels, eligible, clock):
    import cv2
    from lunar_geometry_v5 import median_reference_range_z
    from native_baseline_pose_v7 import propose_pose
    from verify_north_camera_qualified_v64 import pose_errors

    outcomes = []
    for role, other in (ROLES, ROLES[::-1]):
        clock()
        if not eligible:
            outcomes.append(dict(source=role, eligible=False, correct=False, reason="geometry_ineligible"))
            continue
        a, b, label = geometries[role], geometries[other], labels[role]
        visible = label["visible"]
        try:
            estimated, info = propose_pose(label["source_points"][visible], label["target_points"][visible],
                                           a["axial_depth"], a["intrinsics"], a["cam2world"], b["intrinsics"])
        except (cv2.error, ValueError, RuntimeError) as error:
            estimated, info = None, {"reason": "oracle_solver_exception:" + type(error).__name__}
        returned = estimated is not None and estimated.shape == (4, 4) and bool(np.isfinite(estimated).all())
        error = pose_errors(estimated if returned else None, b["cam2world"])
        reference_range = median_reference_range_z(a["axial_depth"], a["intrinsics"])
        correct = error is not None and error["rotation_deg"] <= 2 and error["translation_m"] <= .01 * reference_range
        outcomes.append(dict(source=role, eligible=True, correct=bool(correct), pose_returned=returned,
                             pose_error=error, reference_range_m=float(reference_range),
                             reason=info.get("reason")))
    return outcomes


def gate_decision(rows, nonoverlap):
    outcomes = [outcome for row in rows for outcome in row["oracle_outcomes"] if outcome["eligible"]]
    eligible = sum(row["eligible"] for row in rows)
    if len(outcomes) != 2 * eligible:
        raise ValueError("All eligible exact pose directions must remain in the denominator")
    correct = sum(outcome["correct"] for outcome in outcomes)
    rate = correct / len(outcomes) if outcomes else 0.
    return dict(passed=(eligible == len(ENDPOINTS) and len(rows) == len(ENDPOINTS)
                        and rate >= .95 and nonoverlap),
                eligible_endpoints=eligible, expected_directions=2 * len(ENDPOINTS),
                eligible_directions=len(outcomes), correct_oracle_poses=correct, oracle_success=rate,
                all_nonoverlap_certified=nonoverlap)


def independent_gate(roster, endpoints, root, provenance, clock):
    import cv2
    from solar_eval_data_v69 import body_scene_bounds, bounds_disjoint

    cv2.setNumThreads(1)
    path = root / "gate.json"
    failure = root / "gate.FAILED.json"
    if path.exists() or failure.exists():
        raise FileExistsError("Preserve the original geometry gate attempt")
    rows = []
    started = time.monotonic()
    try:
        source, cameras, directions = geometry_inventory(root, provenance)
        bounds = {}
        for endpoint, spec in endpoints.items():
            clock()
            mesh = verified_mesh(spec)
            box, ceiling, _, _ = body_scene_bounds(mesh)
            bounds[endpoint] = box
            reasons = []
            if box != spec["body_fixed_xy_bounds"]:
                reasons.append("body_fixed_bounds_mismatch")
            geometries, camera_reports = {}, []
            for role in ROLES:
                geometry, report = audit_camera(cameras[endpoint, role], spec["camera_pair"], mesh, clock)
                geometries[role] = geometry
                camera_reports.append(report)
                reasons.extend(role + ":" + reason for reason in report["qualification_reasons"])
            labels, label_reports, label_reasons = audit_labels(
                [directions[endpoint, role] for role in ROLES], geometries, mesh, clock)
            reasons.extend(label_reasons)
            eligible = not reasons
            rows.append(dict(endpoint=endpoint, eligible=eligible, qualification_reasons=reasons,
                             map_mesh_sha256=spec["map_mesh_sha256"], scene_maximum_height_m=ceiling,
                             body_fixed_xy_bounds=box, cameras=camera_reports, directions=label_reports,
                             oracle_outcomes=oracle_results(geometries, labels, eligible, clock)))
            print("v112 geometry gate", endpoint, eligible, flush=True)
        disjoint = [{"left": left, "right": right, "disjoint": bounds_disjoint(bounds[left], bounds[right])}
                    for left, right in combinations(endpoints, 2)]
        edges = {(row["reference_endpoint"], row["query_endpoint"]) for row in roster["cases"] if not row["positive"]}
        expected_edges = set(zip(ENDPOINTS, ENDPOINTS[1:] + ENDPOINTS[:1]))
        if edges != expected_edges:
            raise ValueError("Original cyclic negative roster differs")
        negatives = [{"reference_endpoint": left, "query_endpoint": right,
                      "disjoint": bounds_disjoint(bounds[left], bounds[right])}
                     for left, right in sorted(edges)]
        decision = gate_decision(rows, all(row["disjoint"] for row in disjoint + negatives))
        clock()
        write(path, dict(**provenance, schema="v112_geometry_gate", **decision, geometry_complete_sha256=digest(source),
                         endpoints=rows, disjoint_scene_pairs=disjoint, cyclic_negatives=negatives,
                         image_decodes=0, matcher_calls=0, elapsed_seconds=time.monotonic() - started))
        return decision["passed"]
    except BaseException as error:
        write(failure, dict(**provenance, passed=False, endpoints=rows,
                            error_type=type(error).__name__, error=str(error)))
        raise


def require_gate(root, provenance, expected_sha):
    path = relative(str(root / "gate.json"))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or digest(path) != expected_sha:
        raise ValueError("Sentry requires the exact sealed geometry gate SHA")
    gate = read(path)
    source, cameras, directions = geometry_inventory(root, provenance)
    if (gate["schema"] != "v112_geometry_gate" or gate["passed"] is not True
            or gate["eligible_endpoints"] != 4 or gate["eligible_directions"] != 8
            or gate["correct_oracle_poses"] != 8 or gate["all_nonoverlap_certified"] is not True
            or gate["geometry_complete_sha256"] != digest(source)
            or any(gate.get(key) != value for key, value in provenance.items())):
        raise ValueError("Geometry gate did not qualify the complete original sentry design")
    for row in (*cameras.values(), *directions.values()):
        if digest(relative(row["path"])) != row["sha256"]:
            raise ValueError("A protected camera or direction artifact changed after the gate")


def acquire_sentry(endpoints, jobs, root, provenance, gate_sha, clock):
    import bpy
    from mathutils import Vector
    from north_camera_scene_v59 import setup_scene

    check_blender(bpy)
    require_gate(root, provenance, gate_sha)
    output = root / "sentry"
    output.mkdir()
    rows = []
    started = time.monotonic()
    try:
        for endpoint, spec in endpoints.items():
            subset = [job for job in jobs if job["endpoint"] == endpoint]
            if not subset:
                continue
            clock()
            mesh = verified_mesh(spec)
            pair = spec["camera_pair"]
            lights = approved_lights(spec)
            scene, camera, sun, _, _ = setup_scene(
                output, mesh["vertices"], mesh["faces"], np.asarray(pair["reference"]["intrinsics"]))
            for job in subset:
                clock()
                setup_camera(camera, pair, job["role"])
                sun.rotation_euler = (-Vector(lights[job["light"]]["sun_enu"])).to_track_quat("-Z", "Y").to_euler()
                render_settings(scene, dict(scene="black_space", samples=64, seed=job["seed"]))
                folder = output / job["view_id"] / f'r{job["replicate"]}'
                folder.mkdir(parents=True)
                add_outputs(scene, folder, include_light_passes=False)
                bpy.context.view_layer.update()
                before = time.monotonic()
                if bpy.ops.render.render(write_still=False) != {"FINISHED"}:
                    raise ValueError("Pinned renderer failed a frozen sentry exposure")
                clock()
                path = folder / "combined_0001.exr"
                if set(folder.iterdir()) != {path}:
                    raise ValueError("Sentry must write exactly one native Combined EXR")
                check_exr_header(path, SIZE)
                row = dict(**job, path=str(path), sha256=digest(path), rgb_float_bits=32,
                           elapsed_seconds=time.monotonic() - before)
                write(folder / "receipt.json", row)
                rows.append(row)
                print("v112 sentry", job["view_id"], job["replicate"], flush=True)
        if [keys(row) for row in rows] != [keys(job) for job in jobs]:
            raise ValueError("Incomplete or reordered six-exposure sentry inventory")
        clock()
        write(output / "COMPLETE.json", dict(**provenance, schema="v112_sentry", complete=True,
              gate_sha256=gate_sha, renderer_build=BLENDER_BUILD, renderer_sha256=BLENDER_SHA,
              renders=rows, optical_pixels_decoded=0, matcher_calls=0, elapsed_seconds=time.monotonic() - started))
    except BaseException as error:
        write(output / "FAILED.json", dict(**provenance, complete=False, renders=rows,
                                          gate_sha256=gate_sha, error_type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("geometry", "gate", "sentry"):
        command = sub.add_parser(stage)
        command.add_argument("--roster", required=True)
        command.add_argument("--root", required=True)
        command.add_argument("--publication-commit", required=True)
        command.add_argument("--deadline-utc", required=True)
        if stage == "sentry":
            command.add_argument("--gate-sha256", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    roster, endpoints, jobs, output, clock = admission(args.roster, args.root, args.publication_commit, args.deadline_utc)
    provenance = stamp(args.publication_commit, args.deadline_utc)
    if args.stage == "geometry":
        acquisition_geometry(endpoints, output, provenance, clock)
    elif args.stage == "gate":
        if not independent_gate(roster, endpoints, output, provenance, clock):
            raise SystemExit(2)
    else:
        acquire_sentry(endpoints, jobs, output, provenance, args.gate_sha256, clock)


if __name__ == "__main__":
    main()
