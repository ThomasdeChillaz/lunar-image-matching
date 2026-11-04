"""Causal procedural control of Blender's native 8-bit PNG dither."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time


BLENDER_SHA = "7492b7f776bbd1d27a33f1912247f325f3f6314f0305417736a6f36c15cbbed5"
BLENDER_BUILD = "8bf95cbd38d1"
RENDER_VERSION = [4, 5, 9]
CONTRACT = "reports/solar_reference_contract_v80.json"
CONTRACT_SHA = "918e4a77630c995c4448d9440b3d09fccaf74fc6cea0d52e245696705321be98"
DEPENDENCIES = (
    "north_camera_scene_v59.py", "blender_camera_control_v31.py",
    "render_solar_development_v67.py", "render_solar_completion_v68.py",
    "solar_radiometric_control_v101.py", "direct_render_adapter_v73.py",
    "historical_compact_adapters_v73.py",
)
MODES = (("default", None), ("zero", 0.0), ("one", 1.0))
STIMULI = (
    ("flat_low", 1.5), ("flat_high", 2.5), ("flat_low_repeat", 1.5), ("ramp", None),
)
SHIFTS = ((0, 0), (17, 29), (31, -13), (-19, 23), (43, 11))
RAMP_CODES = (.2, 3.8)
RAMP_PLANE_X = (.34, .66)
RENDER_WALL_SECONDS = 120
SOURCE = Path(__file__).resolve()


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError("Frozen input hash differs: " + str(path))
    return json.loads(Path(path).read_text())


def renderer_path():
    contract = checked_json(CONTRACT, CONTRACT_SHA)
    if contract["blender_executable_sha256"] != BLENDER_SHA:
        raise ValueError("Unexpected archived renderer binary identity")
    binary = Path(contract["blender_executable"])
    if not binary.is_file() or digest(binary) != BLENDER_SHA:
        raise ValueError("The exact original official renderer binary is required")
    return binary


def freeze(output):
    renderer_path()
    write(output, dict(
        schema="procedural_png_dither_v105", scope="Synthetic emission planes only. Causal dither intervention; no images or data from any project cohort.",
        source_sha256=digest(SOURCE), dependency_sha256={name: digest(name) for name in DEPENDENCIES},
        renderer_sha256=BLENDER_SHA, renderer_build=BLENDER_BUILD, renderer_version=RENDER_VERSION,
        historical_contract_sha256=CONTRACT_SHA,
        native_shape=[512, 512], cpu_threads=2, limit_wall_seconds=RENDER_WALL_SECONDS,
        stimuli=[dict(name=name, nominal_srgb_8bit_code=code) for name, code in STIMULI],
        modes=[dict(name=name, dither_intensity=value) for name, value in MODES],
        ramp_srgb_8bit_end_codes=RAMP_CODES, ramp_visible_plane_x=RAMP_PLANE_X,
        residual_xy_shifts_px=SHIFTS, sift_fixed_coordinate_limit_px=1.,
        optical_or_pose_claim=False, calibration_or_final_access=False,
        archived_images_access=False, created_utc=datetime.now(timezone.utc).isoformat(),
    ))


def verify(protocol, expected):
    q = checked_json(protocol, expected)
    if (q["schema"] != "procedural_png_dither_v105" or q["source_sha256"] != digest(SOURCE)
            or q["dependency_sha256"] != {name: digest(name) for name in DEPENDENCIES}
            or q["renderer_sha256"] != BLENDER_SHA or q["renderer_build"] != BLENDER_BUILD
            or q["renderer_version"] != RENDER_VERSION or q["historical_contract_sha256"] != CONTRACT_SHA
            or q["native_shape"] != [512, 512] or q["cpu_threads"] != 2
            or q["limit_wall_seconds"] != RENDER_WALL_SECONDS
            or q["stimuli"] != [dict(name=n, nominal_srgb_8bit_code=v) for n,v in STIMULI]
            or q["modes"] != [dict(name=n, dither_intensity=v) for n,v in MODES]
            or q["ramp_srgb_8bit_end_codes"] != list(RAMP_CODES) or q["ramp_visible_plane_x"] != list(RAMP_PLANE_X)
            or q["residual_xy_shifts_px"] != [list(shift) for shift in SHIFTS]
            or q["sift_fixed_coordinate_limit_px"] != 1.
            or any(q[name] is not False for name in ("optical_or_pose_claim", "calibration_or_final_access", "archived_images_access"))):
        raise ValueError("Source, freeze or procedural controls changed")
    return q


def run(protocol, expected, output):
    verify(protocol, expected)
    binary = renderer_path()
    output.mkdir()
    start = time.monotonic()
    try:
        with (output / "blender.log").open("xb") as log:
            subprocess.run(
                [str(binary), "--background", "--factory-startup", "--disable-autoexec", "--threads", "2",
                 "--python-exit-code", "2", "--python", str(SOURCE), "--", "render-worker",
                 "--protocol", str(Path(protocol).resolve()), "--protocol-sha256", expected,
                 "--output", str(output.resolve())],
                stdout=log, stderr=subprocess.STDOUT, timeout=RENDER_WALL_SECONDS, check=True,
            )
        if not (output / "rendered.json").is_file():
            raise ValueError("Missing procedural renderer receipt")
        write(output / "complete.json", dict(complete=True, protocol_sha256=expected,
              rendered_sha256=digest(output / "rendered.json"), outer_calendar_seconds=time.monotonic() - start))
    except BaseException as exc:
        write(output / "FAILED.json", dict(complete=False, protocol_sha256=expected,
              error_type=type(exc).__name__, error=str(exc), outer_calendar_seconds=time.monotonic() - start))
        raise


def setup_plane(output):
    import bpy
    import numpy as np
    from north_camera_scene_v59 import setup_scene

    vertices = np.array([[-10., -10., 0.], [10., -10., 0.], [10., 10., 0.], [-10., 10., 0.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    intrinsics = np.array([[400., 0., 255.5], [0., 400., 255.5], [0., 0., 1.]])
    scene, camera, sun, nodes, _ = setup_scene(output, vertices, triangles, intrinsics)
    for node in nodes:
        scene.node_tree.nodes.remove(node)
    camera.location = (0., 0., 5.)
    camera.rotation_euler = (0., 0., 0.)
    bpy.data.objects.remove(sun, do_unlink=True)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.
    plane = bpy.data.objects["Finite terrain support"]
    material = plane.data.materials[0]
    tree = material.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.
    surface = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], surface.inputs["Surface"])
    return scene, tree, emission


def set_emission(tree, emission, code):
    # The entire intensity design is inside the linear toe of sRGB.
    scale = 1. / (255. * 12.92)
    color = emission.inputs["Color"]
    if code is not None:
        color.default_value = (code * scale, code * scale, code * scale, 1.)
        return
    coords = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    mapping = tree.nodes.new("ShaderNodeMapRange")
    mapping.inputs["From Min"].default_value, mapping.inputs["From Max"].default_value = RAMP_PLANE_X
    mapping.inputs["To Min"].default_value, mapping.inputs["To Max"].default_value = [x * scale for x in RAMP_CODES]
    tree.links.new(coords.outputs["Generated"], separate.inputs[0])
    tree.links.new(separate.outputs["X"], mapping.inputs["Value"])
    tree.links.new(mapping.outputs["Result"], color)


def render_worker(protocol, expected, output):
    import bpy

    verify(protocol, expected)
    if list(bpy.app.version[:3]) != RENDER_VERSION or bpy.app.build_hash.decode() != BLENDER_BUILD:
        raise ValueError("Pinned official Blender build required")
    if digest(Path(bpy.app.binary_path)) != BLENDER_SHA:
        raise ValueError("Pinned original Blender executable required")
    sys.path.insert(0, str(Path.cwd()))
    records, configuration = [], {}
    for mode, strength in MODES:
        scene, tree, emission = setup_plane(output)
        if strength is not None:
            scene.render.dither_intensity = strength
        image_settings = scene.render.image_settings
        configuration[mode] = dict(
            engine=scene.render.engine, device=scene.cycles.device, samples=scene.cycles.samples,
            cpu_threads=scene.render.threads, dither_intensity=scene.render.dither_intensity,
            view=scene.view_settings.view_transform, look=scene.view_settings.look,
            display=scene.display_settings.display_device, exposure=scene.view_settings.exposure,
            gamma=scene.view_settings.gamma, color_mode=image_settings.color_mode,
            color_depth=image_settings.color_depth, file_format=image_settings.file_format,
        )
        active = configuration[mode]
        if (active["engine"], active["device"], active["cpu_threads"], active["view"], active["display"],
                active["exposure"], active["gamma"], active["color_mode"], active["color_depth"], active["file_format"]) != (
                "CYCLES", "CPU", 2, "Standard", "sRGB", 0., 1., "RGB", "8", "PNG"):
            raise ValueError("Original output path or color transform changed")
        for stimulus, code in STIMULI:
            set_emission(tree, emission, code)
            name = f"{mode}_{stimulus}.png"
            destination = output / name
            if destination.exists():
                raise FileExistsError("Preserve previously rendered procedural output")
            scene.render.filepath = str(destination)
            bpy.context.view_layer.update()
            tick = time.monotonic()
            bpy.ops.render.render(write_still=True)
            records.append(dict(mode=mode, stimulus=stimulus, filename=name, sha256=digest(destination),
                                seconds=time.monotonic() - tick))
    write(output / "rendered.json", dict(
        protocol_sha256=expected, build=BLENDER_BUILD, version=RENDER_VERSION, configuration=configuration,
        archived_images_access=False, records=records,
    ))


def score(protocol, expected, folder, output):
    import cv2
    import numpy as np
    from PIL import Image
    from direct_render_adapter_v73 import SiftLibraryMatcher
    from solar_radiometric_control_v101 import normalize_exposure

    verify(protocol, expected)
    seal = json.loads((folder / "complete.json").read_text())
    if ((folder / "FAILED.json").exists() or not seal["complete"] or seal["protocol_sha256"] != expected
            or not 0 < seal["outer_calendar_seconds"] <= RENDER_WALL_SECONDS + 1):
        raise ValueError("Require a complete bounded original procedural invocation")
    run_meta = checked_json(folder / "rendered.json", seal["rendered_sha256"])
    expected_names = [(m, s, f"{m}_{s}.png") for m, _ in MODES for s, _ in STIMULI]
    if (run_meta["protocol_sha256"] != expected or run_meta["archived_images_access"] is not False
            or run_meta["build"] != BLENDER_BUILD or run_meta["version"] != RENDER_VERSION
            or [(r["mode"], r["stimulus"], r["filename"]) for r in run_meta["records"]] != expected_names):
        raise ValueError("The original complete procedural render matrix is required")

    def load(record):
        path = folder / record["filename"]
        if digest(path) != record["sha256"]:
            raise ValueError("A procedural render changed after the worker sealed it")
        with Image.open(path) as img:
            if img.format != "PNG" or img.mode != "RGB" or img.size != (512, 512):
                raise ValueError("The original RGB native-size procedural PNG is required")
            rgb = np.asarray(img)
            achromatic = np.all(rgb == rgb[..., :1], axis=2)
            return np.asarray(img.convert("L")).copy(), float(achromatic.mean())

    def correlate(a, b):
        a = a.astype(float)
        b = b.astype(float)
        a -= a.mean()
        b -= b.mean()
        denominator = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
        return None if denominator == 0 else float(np.sum(a * b) / denominator)

    def shifted(a, b):
        crop = a[48:464, 48:464]
        return [dict(dx=dx, dy=dy, correlation=correlate(crop, b[48+dy:464+dy, 48+dx:464+dx]))
                for dx,dy in SHIFTS]

    def sift_pair(a, b, enabled):
        a, a_exposure = normalize_exposure(a, enabled)
        b, b_exposure = normalize_exposure(b, enabled)
        matcher = SiftLibraryMatcher(dict(nfeatures=2048, ratio_threshold=.75))
        b_features = matcher.query_features(b)
        p, q, _ = matcher.match(a, b_features)
        return dict(ratio_unique_matches=len(p),
                    within_one_pixel_of_fixed_sensor_coordinates=int((np.linalg.norm(p-q, axis=1) <= 1.).sum()),
                    reference_exposure=a_exposure, query_exposure=b_exposure)

    images, file_rows = {}, {}
    for row in run_meta["records"]:
        image, achromatic = load(row)
        key = (row["mode"], row["stimulus"])
        images[key] = image
        hist = np.bincount(image.ravel(), minlength=256)
        file_rows[row["filename"]] = dict(achromatic_fraction=achromatic,
            code_histogram={int(i): int(hist[i]) for i in np.flatnonzero(hist)},
            raw_standard_deviation=float(image.std()))

    results = {}
    for mode, _ in MODES:
        a, b, repeat, ramp = (images[(mode, stimulus)] for stimulus, _ in STIMULI)
        ramp_residual = ramp.astype(float) - ramp.mean(axis=0, keepdims=True)
        results[mode] = dict(
            measured_dither_intensity=run_meta["configuration"][mode]["dither_intensity"],
            repeat_fraction_identical=float(np.mean(a == repeat)), flat_cross_intensity_shift_controls=shifted(a, b),
            ramp_column_residual_vs_flat_shift_controls=shifted(ramp_residual, a),
            ramp_column_residual_std=float(ramp_residual.std()),
            flat_cross_intensity_sift_original=sift_pair(a, b, False),
            flat_cross_intensity_sift_frozen_exposure=sift_pair(a, b, True),
        )
    write(output, dict(
        protocol_sha256=expected, renderer_receipt_sha256=seal["rendered_sha256"], render_calendar_seconds=seal["outer_calendar_seconds"],
        software=dict(numpy=np.__version__, opencv=cv2.__version__), configuration=run_meta["configuration"],
        rows=file_rows, per_dither=results,
        default_equals_explicit_one={stimulus: bool(np.array_equal(images["default", stimulus], images["one", stimulus]))
                                     for stimulus, _ in STIMULI},
        scope=("Strictly procedural Cycles emission, no lunar data or pose estimation. The write_still, color transform, "
               "native resolution and renderer match the archived setting; the emission scene intentionally differs "
               "from diffuse terrain illumination. A positive test demonstrates causation in this fixture only."),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    modes.add_parser("freeze").add_argument("--output", type=Path, required=True)
    for command in ("run", "render-worker", "score"):
        args = modes.add_parser(command)
        args.add_argument("--protocol", type=Path, required=True)
        args.add_argument("--protocol-sha256", required=True)
        args.add_argument("--output", type=Path, required=True)
        if command == "score":
            args.add_argument("--render-folder", type=Path, required=True)
    a = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if a.command == "freeze":
        freeze(a.output)
    elif a.command == "run":
        run(a.protocol, a.protocol_sha256, a.output)
    elif a.command == "render-worker":
        render_worker(a.protocol, a.protocol_sha256, a.output)
    else:
        sys.path.insert(0, str(Path.cwd()))
        score(a.protocol, a.protocol_sha256, a.render_folder, a.output)
