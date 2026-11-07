"""Bounded native direct-render control on an analytic textured plane."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import monotonic, perf_counter
from types import SimpleNamespace


METHOD = "reference_camera_render_match"
PURPOSE = "direct_render_procedural_control_v90"
LEDGER = "reports/solar_compute_admission_v67.json"
LEDGER_SHA256 = "51680600269e37fe745ded5181d04ef61de8713bf01b3c1792415c1a13802a5c"
JOURNAL = "runs/solar-compute-reservations-v67.json"
CASES = ("known_transform", "blank", "invalid_support")
CONFIGURATION = dict(
    viewpoint_search="reference_camera_only", descriptor="SIFT", ratio_threshold=0.75,
    nfeatures=2048, top_pose_candidates=8, known_sun_policy="exact_render_or_unavailable",
)
GEOMETRY = dict(
    image_seed=20261101, native_shape=[512, 512], illumination_count=24, target_light_index=7,
    solar_elevation_deg=35.0, reference_depth_m=10.0,
    intrinsics=[[400.0, 0.0, 255.5], [0.0, 400.0, 255.5], [0.0, 0.0, 1.0]],
    reference_to_query_affine=[[1.0, 0.0, 8.0], [0.0, 1.0, -4.0]],
    expected_query_cam2world=[
        [1.0, 0.0, 0.0, -0.2], [0.0, 1.0, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ],
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Use a canonical repository-relative path")
    path = Path(value)
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != value
            or path == Path(".") or not path.resolve().is_relative_to(Path.cwd())):
        raise ValueError("Use a canonical repository-relative path")
    return path


def source_pins():
    from control_historical_native_v74 import source_closure
    from direct_render_adapter_v73 import SOURCES

    return source_closure(SOURCES | {
        Path(__file__).name, "control_historical_native_v74.py", "solar_compute_budget_v67.py",
        "solar_learning_pipeline_v73.py", "run_solar_distributional_evaluation_v90.py", "synthetic.py",
    })


def definition(settings):
    keys = {"output", "max_calendar_seconds", "publication_dispatch_lease_seconds"}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError("Output, original calendar cap and original publication lease required")
    output = relative(settings["output"])
    if output == Path("runs") or not output.is_relative_to("runs"):
        raise ValueError("Use a new output directory below runs")
    cap = settings["max_calendar_seconds"]
    lease = settings["publication_dispatch_lease_seconds"]
    if type(cap) is not int or not 5 < cap <= 3600 or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Explicit finite control cap and lease required")
    return dict(
        **settings, settings=settings, purpose=PURPOSE, split="model_selection",
        execution_data="procedural_only", profiles=[dict(name=METHOD, configuration=CONFIGURATION)],
        case_ids=list(CASES), tracks=["estimated_sun"], cpu_threads=2,
        query_truth_used=False, calibration_or_final_access=False, lunar_images_loaded=0,
        geometry=GEOMETRY, compute_ledger=LEDGER, compute_ledger_sha256=LEDGER_SHA256,
        reservation_journal=JOURNAL, source_sha256=source_pins(),
    )


def budget():
    from solar_learning_pipeline_v73 import journal_state

    ledger, journal, charges = journal_state(LEDGER, LEDGER_SHA256, JOURNAL)
    return journal["reservations"], ledger["remaining_baseline_wall_seconds"] - charges["baseline"]


def prepare(settings_path, protocol_path):
    from run_solar_distributional_evaluation_v90 import write

    relative(protocol_path)
    protocol = definition(read(settings_path))
    if Path(protocol["output"]).exists():
        raise FileExistsError("Preserve the existing control attempt")
    entries, remaining = budget()
    if protocol["max_calendar_seconds"] > remaining:
        raise ValueError("Control cap exceeds the canonical remaining baseline allowance")
    protocol.update(prepared_only=True, initial_reservations=entries,
                    created_utc=None, latest_start_utc=None, absolute_deadline_utc=None)
    write(protocol_path, protocol)
    return protocol


def admission(path, commit, started):
    from run_solar_distributional_evaluation_v90 import JOB_SUFFIX, check_publication, utc_now, validate_dispatch

    relative(path)
    protocol = read(path)
    deadline = validate_dispatch(protocol, started, 3600)

    def check():
        if utc_now() >= deadline:
            raise TimeoutError("Original native control allowance exhausted")

    check()
    expected = definition(protocol["settings"])
    if protocol.get("prepared_only") is not False or any(protocol.get(key) != val for key, val in expected.items()):
        raise ValueError("Published procedural-only direct-render settings changed")
    check_publication(path, protocol["source_sha256"], {}, commit, check)
    entries, remaining = budget()
    initial = protocol["initial_reservations"]
    if entries[:len(initial)] != initial:
        raise ValueError("Canonical reservation history changed")
    job = sha(path) + JOB_SUFFIX
    own = [row for row in entries if row["job_id"] == job]
    cap = protocol["max_calendar_seconds"]
    if len(own) > 1 or any(row["category"] != "baseline" or row["seconds"] != cap for row in own):
        raise ValueError("Existing control reservation differs from original cap")
    if not own and cap > remaining:
        raise ValueError("Control cap exceeds remaining baseline allowance")
    check()
    return protocol, [dict(case_id=case) for case in CASES], deadline


def evidence_profile(protocol):
    return dict(method=METHOD, configuration=CONFIGURATION, artifact_sha256={},
                source_sha256=protocol["source_sha256"])


def native_geometry(valid=True):
    import numpy as np

    k = np.asarray(GEOMETRY["intrinsics"])
    depth = np.full((512, 512), GEOMETRY["reference_depth_m"] if valid else 0.0)
    return SimpleNamespace(depth=depth, intrinsics=k, cam2world=np.eye(4))


def render_library(root, geometry, check):
    """Rasterize one fixed plane with a procedural normal map under 24 Suns."""
    import numpy as np
    from PIL import Image
    from direct_render_adapter_v73 import geometry_digest
    from synthetic import texture

    started = perf_counter()
    albedo = texture(GEOMETRY["image_seed"], 512).astype(np.float64) / 255.0
    y, x = np.mgrid[:512, :512]
    nx = 0.44 * np.sin(x / 17.0) * np.cos(y / 31.0)
    ny = 0.44 * np.cos(y / 19.0) * np.sin(x / 29.0)
    inverse_length = 1.0 / np.sqrt(1.0 + nx * nx + ny * ny)
    elevation = math.radians(GEOMETRY["solar_elevation_deg"])
    entries = []
    views = []
    for index in range(GEOMETRY["illumination_count"]):
        check()
        azimuth = 2.0 * math.pi * index / GEOMETRY["illumination_count"]
        sun = [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth),
               math.sin(elevation)]
        cosine = np.maximum(0.0, (nx * sun[0] + ny * sun[1] + sun[2]) * inverse_length)
        view = np.rint(255.0 * albedo * (0.18 + 0.82 * cosine)).astype(np.uint8)
        path = root / f"render-{index:02d}.png"
        with path.open("xb") as stream:
            Image.fromarray(view).save(stream, format="PNG")
        entries.append(dict(image=str(path), image_sha256=sha(path), sun_world=sun,
                            light_index=index, primary_generic=True))
        views.append(view)
    digest = geometry_digest(geometry.depth, geometry.intrinsics, geometry.cam2world)
    # The source identity is the exact analytic plane used for every render.
    # No physical lunar mesh, precomputed lighting table or query map exists.
    lookup = SimpleNamespace(protocol=dict(reference_geometry_sha256=digest, mesh_sha256=digest,
                                           scene_mesh_sha256=digest))
    library = dict(entries=entries, reference_geometry_array_sha256=digest, map_mesh_sha256=digest,
                   scene_mesh_sha256=digest, original_reference_preparation_seconds=perf_counter() - started,
                   preparation_cost_scope="Actual procedural plane library construction")
    return lookup, library, views


def replay_controls(controls, protocol):
    """Replay analytic pixel/pose correctness from the saved native outputs."""
    import numpy as np
    from control_historical_native_v74 import pose_arithmetic
    from historical_compact_adapters_v73 import validate_control_outputs
    from solar_evaluation_v70 import common_pose

    if (controls.get("actual_render_forward") is not True or controls["lunar_images_loaded"] != 0
            or controls["calibration_or_final_access"] is not False):
        raise ValueError("Actual native procedural render searches required")
    basic = validate_control_outputs(dict(controls, actual_pretrained_forward=True), evidence_profile(protocol))
    known = controls["known_transform"]
    expected_search = dict(library_entries_considered=24, common_pnp_candidates=8, final_common_pnp_calls=1,
                           exact_known_sun=False, reference_viewpoint_only=True)
    if any(known["render_search"][field] != value for field, value in expected_search.items()):
        raise ValueError("Expected full primary render-library search and common-PnP reranking")
    if [controls[k]["render_reads"] for k in CASES] != [24, 24, 0]:
        raise ValueError("The real render loader must exercise valid/blank cases and reject invalid geometry")
    rejection = controls["invalid_support"]["rejection"]
    if rejection != "Common valid native reference geometry required":
        raise ValueError("Invalid reference depth did not fail at the production geometry guard")
    geometry = native_geometry()
    inp = SimpleNamespace(geometry=geometry, query_intrinsics=geometry.intrinsics)
    pose, info = common_pose(np.asarray(known["reference_xy"]), np.asarray(known["query_xy"]), inp)
    saved = known["estimated_cam2world"]
    if pose is None or saved is None or not np.allclose(saved, pose, rtol=0.0, atol=1e-7):
        raise ValueError("Saved direct-render pose differs from common geometry replay")
    errors = pose_arithmetic(pose)
    if info["inliers"] < 6 or errors["rotation_error_deg"] > 0.5 or errors["translation_error_m"] > 0.05:
        raise ValueError("Analytic direct-render pose exceeded the native control gate")
    return dict(basic, **errors, common_pose_replayed=True, inliers=info["inliers"])


def worker(path, commit, controller_pid):
    from run_solar_distributional_evaluation_v90 import JOB_SUFFIX, utc, utc_now, write

    relative(path)
    protocol = read(path)
    root = relative(protocol["output"])
    execution = read(root / "execution.json")
    if (os.getppid() != controller_pid or execution["controller_pid"] != controller_pid
            or execution["publication_commit"] != commit or execution["protocol_sha256"] != sha(path)):
        raise ValueError("Original live controller identity required")
    receipt = read(root / "reservation.json")["reservation"]
    entries, _ = budget()
    if (receipt not in entries or receipt["job_id"] != sha(path) + JOB_SUFFIX
            or receipt["category"] != "baseline" or receipt["seconds"] != protocol["max_calendar_seconds"]):
        raise ValueError("Exact canonical baseline reservation required")
    deadline = utc(execution["work_deadline_utc"])

    def check():
        if utc_now() >= deadline or monotonic() - execution["start_monotonic"] >= protocol["max_calendar_seconds"] - 5:
            raise TimeoutError("Original native control work allowance exhausted")

    check()
    _, _, hard = admission(path, commit, utc(execution["started_utc"]))
    if hard != utc(execution["hard_deadline_utc"]):
        raise ValueError("Worker cannot reset original control deadline")
    with (root / "WORKER_STARTED").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    import cv2
    import numpy as np
    from PIL import Image
    from direct_render_adapter_v73 import DirectRenderAdapter, SiftLibraryMatcher, load_native_render
    from historical_compact_adapters_v73 import value_digest
    from solar_evaluation_v70 import common_pose

    cv2.setNumThreads(1)
    geometry = native_geometry()
    lookup, library, views = render_library(root, geometry, check)
    target = views[GEOMETRY["target_light_index"]]
    query = cv2.warpAffine(target, np.asarray(GEOMETRY["reference_to_query_affine"]), (512, 512))
    query_path = root / "query.png"
    with query_path.open("xb") as stream:
        Image.fromarray(query).save(stream, format="PNG")
    # Procedural bootstrap only. The production constructor normally requires
    # the completed native-control artifact being measured here.
    adapter = DirectRenderAdapter.__new__(DirectRenderAdapter)
    profile = evidence_profile(protocol)
    adapter.profile = profile
    adapter.libraries = {lookup.protocol["reference_geometry_sha256"]: library}
    adapter.matcher = SiftLibraryMatcher(CONFIGURATION)
    adapter.estimator = common_pose
    reads = 0

    def measured_loader(entry):
        nonlocal reads
        check()
        image = load_native_render(entry)
        reads += 1
        return image

    adapter.image_loader = measured_loader
    controls = dict(
        method=METHOD, native_shape=[512, 512], cpu_threads=2, configuration_sha256=value_digest(CONFIGURATION),
        artifact_sha256={}, source_sha256=protocol["source_sha256"], query_truth_used=False,
        lunar_images_loaded=0, calibration_or_final_access=False, opencv_threads=cv2.getNumThreads(),
        rendered_inputs={**{item["image"]: item["image_sha256"] for item in library["entries"]},
                         str(query_path): sha(query_path)},
        render_scope="Deterministic Lambertian normal map on an analytic plane under 24 synthetic Sun vectors.",
    )
    rows = []
    for case in CASES:
        check()
        start = perf_counter()
        before = reads
        blank = np.zeros_like(query)
        inp = SimpleNamespace(
            reference=blank if case == "blank" else views[0], query=blank if case == "blank" else query,
            geometry=native_geometry(False) if case == "invalid_support" else geometry,
            query_intrinsics=geometry.intrinsics, reference_sun=library["entries"][0]["sun_world"],
            known_query_sun=None, solar_lookup=lookup,
        )
        rejection = None
        try:
            result = adapter.infer(inp)
        except ValueError as error:
            if case != "invalid_support" or str(error) != "Common valid native reference geometry required":
                raise
            result, rejection = {}, str(error)
        pose = result.get("pose")
        detail = dict(
            reference_xy=np.asarray(result.get("reference_xy", [])).reshape(-1, 2).tolist(),
            query_xy=np.asarray(result.get("query_xy", [])).reshape(-1, 2).tolist(),
            pose_returned=pose is not None, estimated_cam2world=None if pose is None else pose.tolist(),
            render_search=result.get("render_search"), rejection=rejection, render_reads=reads - before,
            elapsed_seconds=perf_counter() - start,
        )
        if case == "known_transform":
            detail["reference_to_query_affine"] = GEOMETRY["reference_to_query_affine"]
        controls[case] = detail
        rows.append(dict(case_id=case, profile=METHOD, track="estimated_sun",
                         render_reads=detail["render_reads"], pose_returned=detail["pose_returned"]))
    controls["actual_render_forward"] = len(library["entries"]) == 24 and controls["known_transform"]["render_reads"] == 24
    write(root / "controls.json", controls)
    replay = replay_controls(controls, protocol)
    check()
    write(root / "summary.json", dict(
        complete=True, protocol_sha256=sha(path), publication_commit=commit, rows=rows, purpose=PURPOSE,
        controls_sha256=sha(root / "controls.json"), replay=replay, execution_data="procedural_only",
        lunar_images_loaded=0, calibration_or_final_access=False,
    ))


def verify_completed_control(path):
    """Saved procedural evidence only. Does not reopen a dispatch lease."""
    from run_solar_distributional_evaluation_v90 import JOB_SUFFIX, check_publication, utc, validate_dispatch

    relative(path)
    protocol = read(path)
    expected = definition(protocol["settings"])
    if protocol.get("prepared_only") is not False or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen native control definition changed")
    root = relative(protocol["output"])
    watchdog, summary, seal = (read(root / filename) for filename in ("watchdog.json", "summary.json", "COMPLETE.json"))
    if (not watchdog["complete"] or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or not summary["complete"] or not seal["external_watchdog_passed"]
            or any(record["protocol_sha256"] != sha(path) for record in (watchdog, summary, seal))
            or seal["watchdog_sha256"] != sha(root / "watchdog.json")
            or seal["summary_sha256"] != sha(root / "summary.json")):
        raise ValueError("Complete original external control proof required")
    started = utc(watchdog["started_utc"])
    deadline = validate_dispatch(protocol, started, 3600)
    cap = protocol["max_calendar_seconds"]
    if (utc(watchdog["hard_deadline_utc"]) != deadline or not started <= utc(watchdog["ended_utc"]) < deadline
            or watchdog["cap_seconds"] != cap or not 0.0 < watchdog["calendar_seconds"] <= cap):
        raise ValueError("Control violated the original bounded allowance")
    entries, _ = budget()
    reservation = watchdog["compute_reservation"]["reservation"]
    if (reservation not in entries or reservation["job_id"] != sha(path) + JOB_SUFFIX
            or reservation["category"] != "baseline" or reservation["seconds"] != cap):
        raise ValueError("Actual procedural control charge is absent")
    commit = watchdog["publication_commit"]
    if summary["publication_commit"] != commit:
        raise ValueError("Original control publication identity changed")
    check_publication(path, protocol["source_sha256"], {}, commit, lambda: None)
    controls_path = root / "controls.json"
    controls = read(controls_path)
    if (summary["controls_sha256"] != sha(controls_path) or [row["case_id"] for row in summary["rows"]] != list(CASES)
            or summary["execution_data"] != "procedural_only"):
        raise ValueError("Native control artifact or exact case inventory changed")
    expected_inputs = {str(root / f"render-{i:02d}.png") for i in range(24)} | {str(root / "query.png")}
    if set(controls["rendered_inputs"]) != expected_inputs or any(sha(p) != h for p, h in controls["rendered_inputs"].items()):
        raise ValueError("Saved procedural render bytes changed")
    replay = replay_controls(controls, protocol)
    if replay != summary["replay"]:
        raise ValueError("Saved independent native control metrics differ")
    metadata = {path: sha(path)}
    metadata.update({str(root / name): sha(root / name)
                     for name in ("controls.json", "summary.json", "watchdog.json", "COMPLETE.json")})
    return dict(passed=True, controls=str(controls_path), controls_sha256=sha(controls_path),
                publication_commit=commit, canonical_baseline_charge_seconds=cap, replay=replay,
                source_sha256=protocol["source_sha256"], metadata_sha256=metadata)


def main():
    from run_solar_distributional_evaluation_v90 import guarded_run

    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "verify"):
        parser = sub.add_parser(action)
        parser.add_argument("--protocol", required=True)
        if action == "prepare":
            parser.add_argument("--settings", required=True)
        if action in ("run", "worker"):
            parser.add_argument("--publication-commit", required=True)
        if action == "run":
            parser.add_argument("--python", default=sys.executable)
        if action == "worker":
            parser.add_argument("--controller-pid", required=True, type=int)
    args = cli.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, Path(__file__).name)
    elif args.action == "worker":
        worker(args.protocol, args.publication_commit, args.controller_pid)
    else:
        print(json.dumps(verify_completed_control(args.protocol), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
