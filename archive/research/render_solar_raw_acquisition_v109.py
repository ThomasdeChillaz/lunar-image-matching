"""Freeze and render the bounded scene-linear lunar acquisition diagnostic."""

import argparse
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time


REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from render_solar_clean_radiance_v106 import (
    AUDIT, BLENDER_BUILD, BLENDER_SHA, DEPENDENCIES, MASTER, OUTPUT as V106_OUTPUT,
    SIZE, audit_geometry, digest, metadata, read, relative, rgb8_pixels, write,
)


SOURCE = "render_solar_raw_acquisition_v109.py"
COHORT = "reports/solar-clean-rerender-cohort-v106.json"
SIGNAL = "reports/solar-clean-signal-protocol-v107.json"
LEDGER = "reports/solar_compute_admission_v67.json"
JOURNAL = "runs/solar-compute-reservations-v67.json"
SUFFIX = "/raw_acquisition_v109"
STEM = "solar-raw-acquisition-v109"
CAPS = {"timing": 240, "study": 900}
ROOTS = {phase: f"runs/{STEM}-{phase}" for phase in CAPS}
PROTOCOLS = {phase: f"reports/{STEM}-{phase}.json" for phase in CAPS}
TIMING_RECEIPT = ROOTS["timing"] + "/COMPLETE.json"
LEGACY_SEED = 20261101
ANCHORS = ((13, 0), (25, 0), (27, 1), (61, 17))
ROLE_ORDER = ("reference", "query")
AOV_SOCKETS = {
    "combined": ("Image",),
    "direct": ("DiffDir", "Diffuse Direct"),
    "indirect": ("DiffInd", "Diffuse Indirect"),
    "albedo": ("DiffCol", "Diffuse Color"),
}
FIXTURE_NAMES = ("black_sun_off", "blocked_with_reflector", "unblocked", "unblocked_half")
OLD_SOURCE = ("render_solar_clean_radiance_v106.py", "solar_compute_budget_v67.py",
              "control_solar_png_dither_v105.py", *DEPENDENCIES)
METADATA = (COHORT, SIGNAL, MASTER, AUDIT, LEDGER, "reports/solar_reference_contract_v80.json")


def utc(value):
    moment = datetime.fromisoformat(value)
    if moment.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return moment.astimezone(timezone.utc)


def native_id(index, role, light):
    return f"u{index:03d}/{role}/light-{light}"


def select_rows(cohort, signal):
    """Join only the fixed model-selection optical identities."""
    rows = {row["key"]: row for row in cohort["renders"]}
    selected = [row for row in signal["items"]
                if row["purpose"] in ("query", "physical_query_sun_reference")]
    expected = {native_id(index, role, light) for index, light in ANCHORS for role in ROLE_ORDER}
    if len(selected) != len(expected) or {row["key"] for row in selected} != expected:
        raise ValueError("Require exactly the eight v107 same-physical-light roles")
    if len(rows) != len(cohort["renders"]):
        raise ValueError("Duplicate historical optical identity")
    for item in selected:
        row = rows[item["key"]]
        index, role, light = (row[k] for k in ("unit_index", "role", "light_index"))
        purpose = "query" if role == "query" else "physical_query_sun_reference"
        if (item["purpose"] != purpose or item["light_index"] != light
                or (index, light) not in ANCHORS or item["key"] != native_id(index, role, light)):
            raise ValueError("Frozen case/role/light join differs")
    return [rows[native_id(index, role, light)] for index, light in ANCHORS for role in ROLE_ORDER]


def schedule(rows, phase, extended=False):
    """Use disjoint black-world Cycles seeds across cameras and sample counts."""
    if phase == "timing" and extended:
        raise ValueError("The timing phase has no optional study extension")
    work = []
    for position, row in enumerate(rows):
        index = row["unit_index"]
        if phase == "timing" and index != 61:
            continue
        if phase == "study":
            work.append(dict(key=row["key"], scene="legacy_white", samples=16,
                             replicate=0, seed=LEGACY_SEED))
        levels = ((16, 1), (64, 1)) if phase == "timing" else ((64, 2),)
        if extended and index == 61:
            levels += ((256, 2),)
        for samples, count in levels:
            for replicate in range(count):
                seed = 20261104 + position * 100 + (samples // 16) * 3 + replicate
                if phase == "timing":
                    seed += 1000
                work.append(dict(key=row["key"], scene="black_space", samples=samples,
                                 replicate=replicate, seed=seed))
    return work


def timing_basis(timing, jobs):
    """Admit only a complete four-render timing pilot and a bounded extrapolation."""
    if (timing["schema"] != 1 or timing["phase"] != "timing"
            or timing["complete"] is not True or timing["calibration_or_final_access"] is not False
            or timing["cpu_threads"] != 2 or timing["renderer_sha256"] != BLENDER_SHA
            or timing["fixtures"]["passed"] is not True):
        raise ValueError("Require completed timing and physical export controls")
    keys = {(row["key"], row["samples"]) for row in timing["rows"]}
    expected = {(native_id(61, role, 17), spp) for role in ROLE_ORDER for spp in (16, 64)}
    if keys != expected or len(timing["rows"]) != len(expected):
        raise ValueError("Timing receipt does not contain the exact native render schedule")
    measured = {}
    for samples in (16, 64):
        durations = [row["elapsed_seconds"] for row in timing["rows"] if row["samples"] == samples]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in durations):
            raise ValueError("Require finite positive measured render times")
        measured[samples] = max(durations)
    # The legacy RGB archive comparison has additional decode I/O and eight geometry checks.
    estimate = 90 + 1.5 * sum(
        measured[16] if row["samples"] == 16 else measured[64] * row["samples"] / 64
        for row in jobs
    )
    if estimate >= CAPS["study"]:
        raise ValueError("Measured timing does not admit this study under its fixed cap")
    return {"max_elapsed_16_seconds": measured[16], "max_elapsed_64_seconds": measured[64],
            "conservative_study_seconds": estimate}


def build(phase, extended=False):
    source = {name: digest(name) for name in OLD_SOURCE}
    source[SOURCE] = digest(__file__)
    pins = {name: digest(name) for name in METADATA}
    cohort, master, _ = metadata(COHORT, str(V106_OUTPUT))
    audit = read(AUDIT)
    if audit["oracle_success"] < .95:
        raise ValueError("Historical exact-correspondence pose gate has not passed")
    signal = read(SIGNAL)
    if signal["evidence"][COHORT] != pins[COHORT]:
        raise ValueError("The signal audit does not identify the sealed optical cohort")
    rows = select_rows(cohort, signal)
    jobs = schedule(rows, phase, extended)
    basis = None
    if phase == "study":
        timing = read(relative(TIMING_RECEIPT))
        timing_protocol = PROTOCOLS["timing"]
        pins[timing_protocol] = digest(relative(timing_protocol))
        if timing["protocol_sha256"] != pins[timing_protocol]:
            raise ValueError("Timing protocol does not match the completed receipt")
        basis = timing_basis(timing, jobs)
        basis.update(receipt=TIMING_RECEIPT, receipt_sha256=digest(TIMING_RECEIPT))
    relevant = {job["key"] for job in jobs}
    return dict(
        schema=1, phase=phase, output=ROOTS[phase], cap_seconds=CAPS[phase],
        split="model_selection", calibration_or_final_access=False,
        purpose="same-physical-Sun scene-linear transport and renderer repeatability only",
        query_truth_in_inference=False, native_shape=[SIZE, SIZE], cpu_threads=2,
        renderer_sha256=BLENDER_SHA, renderer_build=BLENDER_BUILD, cpu_only=True,
        radiometric_units="Blender relative scene-linear RGB", exr_float_bits=32,
        fixed_black_space_spp=True, denoising=False, max_bounces=2,
        extended_256_spp=extended, timing_basis=basis,
        source_sha256=source, metadata_sha256=pins,
        fixtures=list(FIXTURE_NAMES) if phase == "timing" else [],
        camera_mesh_identity=[
            {k: row[k] for k in ("key", "mesh_path", "mesh_sha256", "archive_path", "archive_sha256",
                                 "archive_geometry_path", "archive_geometry_sha256", "sun_enu")}
            for row in rows if row["key"] in relevant
        ],
        render_jobs=jobs,
    ), master, {row["key"]: row for row in rows}


def validate(path):
    protocol = read(relative(path))
    phase = protocol["phase"]
    if (phase not in PROTOCOLS or path != PROTOCOLS[phase]
            or type(protocol["extended_256_spp"]) is not bool):
        raise ValueError("Use the declared immutable protocol path and phase")
    expected, master, rows = build(phase, protocol["extended_256_spp"])
    if protocol != expected:
        raise ValueError("Prospective protocol differs from its exact frozen inputs")
    return protocol, master, rows


def publication(protocol_path, protocol, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the full published commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    for name in (protocol_path, *protocol["source_sha256"], *protocol["metadata_sha256"]):
        published = subprocess.check_output(["git", "show", f"{commit}:{name}"], timeout=5)
        if published != relative(name).read_bytes():
            raise ValueError("Published source or metadata differs: " + name)


def active_deadline(protocol_path, protocol, value):
    """Require an existing canonical baseline allocation and enforce its UTC bound."""
    ledger_sha = protocol["metadata_sha256"][LEDGER]
    journal = read(relative(JOURNAL))
    if journal["ledger_sha256"] != ledger_sha or digest(LEDGER) != ledger_sha:
        raise ValueError("Canonical compute ledger differs")
    own = [entry for entry in journal["reservations"]
           if entry["job_id"] == digest(protocol_path) + SUFFIX]
    if (len(own) != 1 or own[0]["category"] != "baseline"
            or own[0]["seconds"] != protocol["cap_seconds"] or own[0]["refund_on_failure"] is not False):
        raise ValueError("Require the exact canonical baseline reservation")
    now = datetime.now(timezone.utc)
    reserved = utc(own[0]["reserved_utc"])
    deadline = utc(value)
    if not reserved <= now < deadline <= reserved + timedelta(seconds=protocol["cap_seconds"]):
        raise ValueError("Use the active reservation's absolute UTC deadline")
    return deadline


def render_settings(scene, job):
    legacy = job["scene"] == "legacy_white"
    scene.cycles.samples = job["samples"]
    scene.cycles.seed = job["seed"]
    if not legacy:
        scene.cycles.use_adaptive_sampling = False
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .001 if legacy else 0.
    if scene.cycles.use_denoising or scene.cycles.max_bounces != 2:
        raise ValueError("Require the original bounce model and no denoiser")


def add_outputs(scene, output, include_light_passes):
    from blender_camera_control_v31 import output_node

    tree = scene.node_tree
    layer = next(node for node in tree.nodes if node.bl_idname == "CompositorNodeRLayers")
    for node in list(tree.nodes):
        if node.bl_idname == "CompositorNodeOutputFile":
            tree.nodes.remove(node)
    view = scene.view_layers[0]
    if include_light_passes:
        view.use_pass_diffuse_direct = True
        view.use_pass_diffuse_indirect = True
        view.use_pass_diffuse_color = True
    for name, candidates in AOV_SOCKETS.items():
        if name != "combined" and not include_light_passes:
            continue
        choices = [layer.outputs.get(key) for key in candidates if layer.outputs.get(key) is not None]
        if len(choices) != 1:
            raise ValueError("Missing or ambiguous raw Cycles compositor output: " + name)
        output_node(tree, output.resolve(), name, choices[0])


def check_components(planes):
    import numpy as np

    if any(not np.isfinite(plane).all() for plane in planes.values()):
        raise ValueError("Nonfinite EXR signal")
    combined = planes["combined"]
    if len(planes) == 1:
        return None
    recovered = planes["albedo"] * (planes["direct"] + planes["indirect"])
    residual = float(np.max(np.abs(combined - recovered)))
    if residual > 2e-6 + 4e-5 * float(np.max(np.abs(combined))):
        raise ValueError("Combined EXR differs from diffuse-color-weighted direct and indirect")
    return residual


def check_fixture(values):
    """Black, reflected-only, bright and half-radiance tests exercise the exact EXR writer."""
    black, blocked, bright, half = (values[name] for name in FIXTURE_NAMES)
    if (black["combined_max"] > 1e-8 or blocked["direct_max"] > 1e-6
            or blocked["indirect_mean"] <= 1e-6 or bright["direct_mean"] <= .001
            or abs(half["combined_mean"] / bright["combined_mean"] - .5) > .01):
        raise ValueError("Raw EXR black, bounce, direct or linear-radiance fixture failed")


def check_exr_header(path, size):
    """Check on-disk float32 RGB and a full, uncropped native data window."""
    def string(stream):
        data = bytearray()
        while len(data) < 256:
            byte = stream.read(1)
            if byte == b"\0":
                return bytes(data)
            if not byte:
                break
            data.extend(byte)
        raise ValueError("Truncated or overlong EXR attribute name")

    attributes = {}
    with Path(path).open("rb") as stream:
        if stream.read(4) != b"\x76\x2f\x31\x01" or len(stream.read(4)) != 4:
            raise ValueError("Require actual OpenEXR file")
        while name := string(stream):
            kind = string(stream)
            length = struct.unpack("<I", stream.read(4))[0]
            if length > (1 << 20) or name in attributes:
                raise ValueError("Invalid EXR attribute length or repeated name")
            attributes[name] = (kind, stream.read(length))
    if attributes.get(b"dataWindow") != (b"box2i", struct.pack("<4i", 0, 0, size - 1, size - 1)):
        raise ValueError("Unexpected EXR native data window")
    kind, channels = attributes[b"channels"]
    if kind != b"chlist":
        raise ValueError("Missing EXR channel list")
    names = set()
    while channels != b"\0":
        name, entry = channels.split(b"\0", 1)
        pixel_type, _, x_sampling, y_sampling = struct.unpack("<iB3xii", entry[:16])
        if pixel_type != 2 or x_sampling != 1 or y_sampling != 1 or name in names:
            raise ValueError("Expected native-resolution float32 EXR channels")
        names.add(name)
        channels = entry[16:]
    if names != {b"R", b"G", b"B"}:
        raise ValueError("Expected three raw RGB channels")


def exr_pixels(output, size, include_light_passes):
    from blender_camera_control_v31 import read_exr

    names = tuple(AOV_SOCKETS) if include_light_passes else ("combined",)
    files = {}
    for name in names:
        found = list(output.glob(name + "_*.exr"))
        if len(found) != 1:
            raise ValueError("Each output must have exactly one native EXR file: " + name)
        check_exr_header(found[0], size)
        files[name] = found[0]
    return {name: read_exr(path.resolve(), size) for name, path in files.items()}, files


def render_once(bpy, scene, output, clock, *, legacy=False, size=SIZE):
    include_light_passes = not legacy
    add_outputs(scene, output, include_light_passes)
    scene.render.filepath = str((output / "legacy_rgb8.png").resolve())
    bpy.context.view_layer.update()
    clock()
    start = time.monotonic()
    if bpy.ops.render.render(write_still=legacy) != {"FINISHED"}:
        raise ValueError("Blender failed to render the frozen EXR exposure")
    clock()
    planes, files = exr_pixels(output, size, include_light_passes)
    residual = check_components(planes)
    receipt = {name: {"path": str(path), "sha256": digest(path)} for name, path in files.items()}
    return planes, receipt, residual, time.monotonic() - start


def fixture_mesh(blocked):
    """The centre sees the receiver; the off-axis slab blocks its direct Sun ray."""
    vertices, faces = [], []

    def quad(points):
        first = len(vertices)
        vertices.extend(points)
        faces.extend(((first, first + 1, first + 2), (first, first + 2, first + 3)))

    quad(((-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)))
    # Vertical reflector facing the receiver and the off-axis Sun.
    quad(((2, -2, .2), (2, -2, 2.4), (2, 2, 2.4), (2, 2, .2)))
    if blocked:
        quad(((-2.6, -2, 1.5), (-.6, -2, 1.5), (-.6, 2, 1.5), (-2.6, 2, 1.5)))
    return vertices, faces


def run_fixtures(bpy, root, clock):
    from mathutils import Vector
    import numpy as np
    from north_camera_scene_v59 import setup_scene

    values = {}
    size = 64
    for name in FIXTURE_NAMES:
        output = root / "fixtures" / name
        output.mkdir(parents=True)
        blocked = name in ("black_sun_off", "blocked_with_reflector")
        vertices, faces = fixture_mesh(blocked)
        scene, camera, sun, _, _ = setup_scene(
            output, np.asarray(vertices), np.asarray(faces), np.eye(3) * size, size)
        render_settings(scene, {"scene": "black_space", "samples": 64, "seed": LEGACY_SEED})
        camera.location = (0., 0., 6.)
        camera.rotation_euler = (0., 0., 0.)
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = .5
        sun.rotation_euler = (-Vector((-.6, 0., .8))).to_track_quat("-Z", "Y").to_euler()
        sun.data.energy = {"black_sun_off": 0., "unblocked_half": 1.5}.get(name, 3.)
        planes, receipts, residual, seconds = render_once(bpy, scene, output, clock, size=size)
        crop = np.s_[16:48, 16:48]
        values[name] = dict(
            combined_max=float(np.max(np.abs(planes["combined"][crop]))),
            combined_mean=float(np.mean(planes["combined"][crop])),
            direct_max=float(np.max(np.abs(planes["direct"][crop]))),
            direct_mean=float(np.mean(planes["direct"][crop])),
            indirect_mean=float(np.mean(planes["indirect"][crop])),
            recon_max_abs_error=residual, exr=receipts, elapsed_seconds=seconds,
        )
    check_fixture(values)
    return {"passed": True, "controls": values}


def setup_camera(camera, pair, role):
    from mathutils import Matrix
    import numpy as np

    pose, intrinsics = (np.asarray(pair[role][name]) for name in ("cam2world", "intrinsics"))
    camera.data.lens = 36 * intrinsics[0, 0] / SIZE
    world = pose.copy()
    world[:3, :3] = pose[:3, :3] @ np.diag([1., -1., -1.])
    camera.matrix_world = Matrix(world.tolist())
    return pose, intrinsics


def compare_seed_planes(first, second):
    """Measure repeatable contrast and Monte Carlo residual on fixed spatial support."""
    import numpy as np

    if first.shape != second.shape or first.shape != (SIZE, SIZE, 3):
        raise ValueError("Seed comparison requires two full native RGB captures")
    values = {}
    for size in (1, 32):
        def averaged(array):
            luminance = array @ np.array([.2126, .7152, .0722])
            return luminance.reshape(SIZE // size, size, SIZE // size, size).mean(axis=(1, 3))

        left, right = averaged(first), averaged(second)
        mean = (left + right) * .5
        left_centered, right_centered = left - left.mean(), right - right.mean()
        low, high = np.percentile(mean, [1, 99.5])
        values[str(size)] = dict(
            aggregate_p01=float(low), aggregate_p995=float(high), aggregate_span=float(high - low),
            seed_difference_single_render_rms=float(np.sqrt(np.mean((left - right) ** 2) / 2)),
            across_seed_spatial_covariance=float(np.mean(left_centered * right_centered)),
            positive_support_fraction=float(np.mean(mean > 0)),
        )
    return values


def run(protocol_path, commit, deadline_value):
    import bpy
    from mathutils import Vector
    import numpy as np
    from north_camera_scene_v59 import setup_scene
    from solar_physics_v63 import lighting_schedule

    protocol, master, sources = validate(protocol_path)
    publication(protocol_path, protocol, commit)
    deadline = active_deadline(protocol_path, protocol, deadline_value)
    if (tuple(bpy.app.version[:3]) != (4, 5, 9)
            or bpy.app.build_hash.decode() != BLENDER_BUILD
            or digest(bpy.app.binary_path) != BLENDER_SHA):
        raise ValueError("Require the exact original Blender executable")
    output = relative(protocol["output"])
    output.mkdir()
    started = time.monotonic()
    rows = []
    repeatability = {}

    def clock():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Original raw acquisition allocation elapsed")

    try:
        fixtures = run_fixtures(bpy, output, clock) if protocol["fixtures"] else None
        for index, _ in ANCHORS:
            jobs = [job for job in protocol["render_jobs"] if sources[job["key"]]["unit_index"] == index]
            if not jobs:
                continue
            clock()
            unit = master["units"][index]
            mesh = relative(unit["mesh"])
            if digest(mesh) != unit["mesh_sha256"]:
                raise ValueError("Frozen terrain mesh differs")
            with np.load(mesh, allow_pickle=False) as data:
                vertices, faces, origin, enu = (data[name].copy() for name in
                                                ("vertices", "faces", "body_origin", "enu_rows"))
            pair = unit["camera_pairs"][0]
            scene, camera, sun, _, bvh = setup_scene(
                output, vertices, faces, np.asarray(pair["reference"]["intrinsics"]))
            # Preserve the historical white controls before changing adaptive sampling or pass flags.
            legacy_adaptive = scene.cycles.use_adaptive_sampling
            lights = {light["index"]: light for light in lighting_schedule(unit["site"]["centre_latitude_deg"])}
            for role in ROLE_ORDER:
                pose, intrinsics = setup_camera(camera, pair, role)
                role_jobs = [job for job in jobs if sources[job["key"]]["role"] == role]
                geometry = audit_geometry(bvh, pose, intrinsics, origin, enu,
                                          sources[role_jobs[0]["key"]], clock)
                levels = {}
                for job in role_jobs:
                    source = sources[job["key"]]
                    light = lights[source["light_index"]]["sun_enu"]
                    if light != source["sun_enu"]:
                        raise ValueError("Frozen Sun vector differs")
                    sun.rotation_euler = (-Vector(light)).to_track_quat("-Z", "Y").to_euler()
                    legacy = job["scene"] == "legacy_white"
                    if legacy:
                        scene.cycles.use_adaptive_sampling = legacy_adaptive
                        layer = scene.view_layers[0]
                        layer.use_pass_diffuse_direct = False
                        layer.use_pass_diffuse_indirect = False
                        layer.use_pass_diffuse_color = False
                    render_settings(scene, job)
                    folder = output / job["key"] / job["scene"] / f'{job["samples"]}spp-r{job["replicate"]}'
                    folder.mkdir(parents=True)
                    planes, artifacts, residual, elapsed = render_once(bpy, scene, folder, clock, legacy=legacy)
                    legacy_equal = None
                    if legacy:
                        archived = relative(source["archive_path"])
                        if digest(archived) != source["archive_sha256"]:
                            raise ValueError("Legacy RGB8 hash differs")
                        legacy_equal = (rgb8_pixels((folder / "legacy_rgb8.png").read_bytes()) ==
                                        rgb8_pixels(archived.read_bytes()))
                        if not legacy_equal:
                            raise ValueError("Historical white-world RGB8 pixel parity failed")
                    elif protocol["phase"] == "study":
                        levels.setdefault(job["samples"], []).append(planes)
                    row = dict(**job, exr=artifacts, elapsed_seconds=elapsed,
                               recon_max_abs_error=residual, exact_legacy_rgb8=legacy_equal,
                               geometry=geometry, adaptive=scene.cycles.use_adaptive_sampling)
                    write(folder / "receipt.json", row)
                    rows.append(row)
                    print("raw acquisition", job["key"], job["scene"], job["samples"], flush=True)
                if levels:
                    repeatability[role_jobs[0]["key"]] = {}
                    for level, (a, b) in levels.items():
                        repeatability[role_jobs[0]["key"]][str(level)] = {
                            "combined": compare_seed_planes(a["combined"], b["combined"]),
                            "direct": compare_seed_planes(a["direct"] * a["albedo"], b["direct"] * b["albedo"]),
                            "indirect": compare_seed_planes(a["indirect"] * a["albedo"], b["indirect"] * b["albedo"]),
                        }
        clock()
        if len(rows) != len(protocol["render_jobs"]):
            raise ValueError("Incomplete frozen raw acquisition run")
        write(output / "COMPLETE.json", dict(
            schema=1, phase=protocol["phase"], complete=True, calibration_or_final_access=False,
            protocol_sha256=digest(protocol_path), renderer_sha256=BLENDER_SHA, cpu_threads=2,
            publication_commit=commit, deadline_utc=deadline.isoformat(), fixtures=fixtures,
            rows=rows, same_camera_repeatability=repeatability, elapsed_seconds=time.monotonic() - started,
        ))
    except BaseException as error:
        write(output / "FAILED.json", dict(complete=False, completed_rows=len(rows),
                                          error_type=type(error).__name__, error=str(error)))
        raise


def launch(protocol_path, commit):
    """Use the original ledger and an outer wall-time watchdog."""
    from control_solar_png_dither_v105 import renderer_path
    from solar_compute_budget_v67 import reserve

    protocol, _, _ = validate(protocol_path)
    publication(protocol_path, protocol, commit)
    blender = renderer_path()
    output = relative(protocol["output"])
    watchdog = output.with_name(output.name + "-watchdog.json")
    if output.exists() or watchdog.exists():
        raise FileExistsError("Preserve the original attempt")
    charge = reserve(LEDGER, protocol["metadata_sha256"][LEDGER], JOURNAL,
                     digest(protocol_path) + SUFFIX, "baseline", protocol["cap_seconds"])
    deadline = utc(charge["reservation"]["reserved_utc"]) + timedelta(seconds=protocol["cap_seconds"])
    worker = None
    error = None
    try:
        args = [str(blender), "--background", "--factory-startup", "--disable-autoexec", "--threads", "2",
                "--python-exit-code", "2", "--python", SOURCE, "--", "render", "--protocol", protocol_path,
                "--publication-commit", commit, "--deadline-utc", deadline.isoformat()]
        worker = subprocess.Popen(args, start_new_session=True, env=dict(os.environ, OPENBLAS_NUM_THREADS="1"))
        code = worker.wait(timeout=max(0., (deadline - datetime.now(timezone.utc)).total_seconds()))
        if code != 0 or not (output / "COMPLETE.json").is_file():
            raise RuntimeError("Raw acquisition worker did not finish successfully")
    except BaseException as exc:
        error = {"type": type(exc).__name__, "detail": str(exc)}
        raise
    finally:
        if worker is not None and worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait(timeout=5)
        write(watchdog, dict(complete=error is None, protocol_sha256=digest(protocol_path),
                             absolute_deadline_utc=deadline.isoformat(), error=error, **charge))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--phase", choices=tuple(CAPS), required=True)
    freeze.add_argument("--extend-256", action="store_true")
    freeze.add_argument("--destination", type=Path)
    sub.add_parser("validate").add_argument("--protocol", required=True)
    for action in ("launch", "render"):
        command = sub.add_parser(action)
        command.add_argument("--protocol", required=True)
        command.add_argument("--publication-commit", required=True)
        if action == "render":
            command.add_argument("--deadline-utc", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if args.action == "freeze":
        value, _, _ = build(args.phase, args.extend_256)
        write(args.destination or PROTOCOLS[args.phase], value)
    elif args.action == "validate":
        protocol, _, _ = validate(args.protocol)
        print("Frozen native render jobs:", len(protocol["render_jobs"]))
    elif args.action == "launch":
        launch(args.protocol, args.publication_commit)
    else:
        run(args.protocol, args.publication_commit, args.deadline_utc)


if __name__ == "__main__":
    main()
