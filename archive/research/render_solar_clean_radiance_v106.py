"""Re-render sealed selection images and export three encodings from each result."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import zlib


SOURCE = "render_solar_clean_radiance_v106.py"
MASTER = "reports/solar_development_admission_v67.json"
AUDIT = "reports/solar_full_geometry_audit_v70.json"
OUTPUT = Path("runs/solar-clean-radiance-v106/render")
BLENDER_SHA = "7492b7f776bbd1d27a33f1912247f325f3f6314f0305417736a6f36c15cbbed5"
BLENDER_BUILD = "8bf95cbd38d1"
SIZE = 512
PNG = b"\x89PNG\r\n\x1a\n"
DEPENDENCIES = (
    "render_solar_development_v67.py", "render_solar_completion_v68.py",
    "render_solar_data_v65.py", "solar_data_v65.py", "solar_physics_v63.py",
    "north_camera_scene_v59.py", "north_camera_correspondences_v59.py",
    "north_development_plan_v35.py", "north_camera_variation_v59.py",
    "north_render_plan_v31.py", "north_terrain_pilot_v32.py",
    "blender_camera_control_v31.py",
)
ENCODINGS = (("default_rgb8", 8), ("clean_rgb8", 8), ("clean_rgb16", 16))


def digest(path):
    hash_ = sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            hash_.update(data)
    return hash_.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) != value:
        raise ValueError("Require canonical repository-relative path")
    if path.resolve() != Path.cwd() / path:
        raise ValueError("Refuse symlinked inputs and outputs")
    return path


def publication(protocol, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Require a full published Git commit")
    for name in (SOURCE, *DEPENDENCIES, MASTER, AUDIT, protocol):
        public = subprocess.check_output(["git", "show", f"{commit}:{name}"], timeout=5)
        if public != Path(name).read_bytes():
            raise ValueError("Published source or protocol differs: " + name)


def metadata(protocol, output):
    """Check only metadata. Optical and geometry payloads stay closed here."""
    path = relative(protocol)
    root = relative(output)
    p = read(path)
    if (root != OUTPUT or p["output_root"] != output
            or p["split"] != "model_selection" or p["calibration_or_final_access"] is not False
            or p["query_truth_in_inference"] is not False or p["cpu_threads"] != 2
            or p["native_shape"] != [SIZE, SIZE] or p["cap_seconds"] != 900
            or p["original_master"] != MASTER or p["geometry_audit"] != AUDIT):
        raise ValueError("Require original model-selection provenance and the exclusive v106 output root")
    pins = p["bindings"]
    for name in (MASTER, AUDIT):
        if digest(name) != pins[name]:
            raise ValueError("Frozen historical metadata differs: " + name)
    if p["original_master_sha256"] != pins[MASTER] or p["geometry_audit_sha256"] != pins[AUDIT]:
        raise ValueError("Frozen historical identity bindings differ")
    master, audit = read(MASTER), read(AUDIT)
    if (not audit["passed"] or not audit["complete_dataset"]
            or audit["calibration_or_final_access"] or master["calibration_or_final_access"]
            or audit["original_master_sha256"] != pins[MASTER]):
        raise ValueError("Require the completed historical development geometry audit")
    audit_units = {entry["unit_index"]: entry for entry in audit["units"]}
    if not 0 < len(p["renders"]) == p["render_count"] <= 120:
        raise ValueError("Require a bounded list of at most 120 source-frozen native renders")
    groups = defaultdict(list)
    keys, outputs = set(), set()
    original_metadata = {}
    for row in p["renders"]:
        index, role, light = (row[field] for field in ("unit_index", "role", "light_index"))
        if (type(index) is not int or index not in audit_units or role not in ("reference", "query")
                or type(light) is not int or not 0 <= light < 24
                or row["key"] != f"u{index:03d}/{role}/light-{light}" or row["key"] in keys):
            raise ValueError("Invalid or repeated original native image identity")
        keys.add(row["key"])
        entry, unit = audit_units[index], master["units"][index]
        if (entry["split"] != "model_selection" or not entry["eligible"]
                or unit["site"]["split"] != "model_selection" or unit["calibration_or_final_access"]
                or unit["cpu_threads"] != 2 or len(unit["camera_pairs"]) != 1
                or entry["component"] != unit["site"]["component"]
                or row["mesh_path"] != unit["mesh"] or row["mesh_sha256"] != unit["mesh_sha256"]):
            raise ValueError("Require the same qualified native camera and finite terrain")
        archived = relative(entry["source_output"])
        if index not in original_metadata:
            archive_summary, archive_seal = archived / "summary.json", archived / "COMPLETE.json"
            if (digest(archive_summary) != entry["source_summary_sha256"]
                    or digest(archive_seal) != entry["source_seal_sha256"]):
                raise ValueError("Historical optical seal or summary differs")
            summary, seal = read(archive_summary), read(archive_seal)
            if (not summary["complete"] or summary["calibration_or_final_access"]
                    or summary["rendered_images"] != 48):
                raise ValueError("Historical unit is incomplete")
            original_metadata[index] = (summary, seal, archive_summary, archive_seal)
        summary, seal, archive_summary, archive_seal = original_metadata[index]
        images = [image for image in summary["images"] if image["role"] == role and image["light"] == light]
        view = summary["views"][role]
        if (len(images) != 1 or str(archive_summary) != row["source_summary"]
                or row["source_summary_sha256"] != entry["source_summary_sha256"]
                or row["source_seal"] != str(archive_seal)
                or row["source_seal_sha256"] != entry["source_seal_sha256"]
                or str(archived / images[0]["path"]) != row["archive_path"]
                or images[0]["sha256"] != row["archive_sha256"]
                or seal["sha256"][images[0]["path"]] != row["archive_sha256"]
                or str(archived / view["path"]) != row["archive_geometry_path"]
                or view["sha256"] != row["archive_geometry_sha256"]
                or seal["sha256"][view["path"]] != row["archive_geometry_sha256"]
                or not view["complete_surface"]):
            raise ValueError("Frozen optical path or geometry does not match the original sealed unit")
        for tag, _ in ENCODINGS:
            destination = relative(row["output_" + tag])
            if (not destination.is_relative_to(root) or destination.suffix != ".png"
                    or destination in outputs or destination.is_relative_to(root / "seals")):
                raise ValueError("Every new image must have a unique PNG path in the exclusive output root")
            outputs.add(destination)
        groups[index].append(row)
    return p, master, groups


def png_header(data, bit_depth):
    expected = (SIZE, SIZE, bit_depth, 2, 0, 0, 0)
    if (len(data) < 33 or data[:16] != PNG + b"\x00\x00\x00\x0dIHDR"
            or struct.unpack(">IIBBBBB", data[16:29]) != expected):
        raise ValueError("Require original native noninterlaced RGB PNG encoding")


def rgb8_pixels(data):
    """Decode native RGB8 losslessly when file-level bytes differ."""
    png_header(data, 8)
    offset, chunks = 8, []
    while offset + 12 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], "big")
        tag = data[offset + 4:offset + 8]
        end = offset + size + 12
        if end > len(data):
            raise ValueError("Truncated PNG chunk")
        if tag == b"IDAT":
            chunks.append(data[offset + 8:offset + 8 + size])
        if tag == b"IEND":
            break
        offset = end
    rows = zlib.decompress(b"".join(chunks))
    stride = 3 * SIZE
    if len(rows) != SIZE * (stride + 1):
        raise ValueError("Unexpected native PNG raster size")
    pixels, up = bytearray(), bytearray(stride)
    for pos in range(0, len(rows), stride + 1):
        mode = rows[pos]
        row = bytearray(rows[pos + 1:pos + 1 + stride])
        if mode not in range(5):
            raise ValueError("Unknown PNG row filter")
        if mode:
            for i, residual in enumerate(row):
                left = row[i - 3] if i >= 3 else 0
                top_left = up[i - 3] if i >= 3 else 0
                top = up[i]
                if mode == 1:
                    predictor = left
                elif mode == 2:
                    predictor = top
                elif mode == 3:
                    predictor = (left + top) // 2
                else:
                    trial = left + top - top_left
                    a, b, c = abs(trial - left), abs(trial - top), abs(trial - top_left)
                    predictor = left if a <= b and a <= c else top if b <= c else top_left
                row[i] = (residual + predictor) & 255
        pixels.extend(row)
        up = row
    return pixels


def audit_geometry(bvh, pose, intrinsics, origin, enu, row, clock):
    import numpy as np
    from render_solar_data_v65 import ray_geometry

    path = relative(row["archive_geometry_path"])
    if digest(path) != row["archive_geometry_sha256"]:
        raise ValueError("Sealed geometry payload differs")
    actual = ray_geometry(bvh, pose, intrinsics, clock)
    actual.update(body_origin=origin, enu_rows=enu)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(actual):
            raise ValueError("Native geometry array inventory differs")
        for name, observed in actual.items():
            expected = archive[name]
            equal_nan = observed.dtype.kind in "fc"
            if (observed.dtype != expected.dtype or observed.shape != expected.shape
                    or not np.array_equal(observed, expected, equal_nan=equal_nan)):
                raise ValueError("Recomputed native geometry differs: " + name)
    if actual["valid"].shape != (SIZE, SIZE) or not actual["valid"].all():
        raise ValueError("Require full native geometry support")
    return dict(exact_camera_depth_and_rays=True, native_rays=SIZE * SIZE, compared_arrays=sorted(actual))


def render_one(bpy, scene, sun, row, light, geometry, clock):
    from mathutils import Vector

    archived = relative(row["archive_path"]).read_bytes()
    if sha256(archived).hexdigest() != row["archive_sha256"]:
        raise ValueError("Sealed historical RGB8 input differs")
    png_header(archived, 8)
    paths = {tag: relative(row["output_" + tag]) for tag, _ in ENCODINGS}
    for path in paths.values():
        if path.exists():
            raise FileExistsError("Preserve existing render output: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
    settings = scene.render.image_settings
    original_dither = scene.render.dither_intensity
    if (original_dither != 1.0 or settings.file_format != "PNG" or settings.color_mode != "RGB"
            or settings.color_depth != "8" or scene.view_settings.view_transform != "Standard"
            or scene.view_settings.exposure != 0.0 or scene.view_settings.gamma != 1.0
            or scene.display_settings.display_device != "sRGB"):
        raise ValueError("Historical Standard RGB8 factory-default writer settings differ")
    sun.rotation_euler = (-Vector(light["sun_enu"])).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(paths["default_rgb8"].resolve())
    bpy.context.view_layer.update()
    clock()
    start = time.monotonic()
    result = bpy.ops.render.render(write_still=True)
    if result != {"FINISHED"}:
        raise ValueError("Blender did not complete its single historical render")
    render_result = bpy.data.images["Render Result"]
    parity_path = OUTPUT / "seals" / row["key"] / "default-save-render.png"
    parity_path.parent.mkdir(parents=True)
    clock()
    render_result.save_render(str(parity_path.resolve()), scene=scene)
    parity = parity_path.read_bytes()
    legacy = paths["default_rgb8"].read_bytes()
    png_header(parity, 8)
    writer_bytes_equal = parity == legacy
    if not writer_bytes_equal and rgb8_pixels(parity) != rgb8_pixels(legacy):
        raise ValueError("Render Result default export differs from historical write_still RGB8 pixels")
    try:
        scene.render.dither_intensity = 0.0
        for tag, depth in ENCODINGS[1:]:
            clock()
            settings.color_depth = str(depth)
            render_result.save_render(str(paths[tag].resolve()), scene=scene)
    finally:
        scene.render.dither_intensity = original_dither
        settings.color_depth = "8"
    legacy_equal = legacy == archived
    if not legacy_equal and rgb8_pixels(legacy) != rgb8_pixels(archived):
        raise ValueError("Re-rendered historical RGB8 pixels differ from the archive")
    output = {}
    for tag, depth in ENCODINGS:
        data = paths[tag].read_bytes()
        png_header(data, depth)
        output[tag] = dict(path=str(paths[tag]), sha256=sha256(data).hexdigest(), bit_depth=depth)
    return dict(
        **{key: row[key] for key in ("key", "unit_index", "role", "light_index", "archive_path",
                                    "archive_sha256", "archive_geometry_path", "archive_geometry_sha256")},
        **output, geometry=geometry, legacy_sha_equal=legacy_equal, legacy_rgb_pixels_equal=True,
        default_writer_control=dict(path=str(parity_path), sha256=sha256(parity).hexdigest(),
                                    byte_equal=writer_bytes_equal, exact_rgb_pixels=True),
        renderer_invocations=1, display_transform="Standard/sRGB", default_dither=original_dither,
        clean_dither=0.0, render_and_export_seconds=time.monotonic() - start,
    )


def run(protocol, output, commit, deadline):
    import bpy
    from mathutils import Matrix
    import numpy as np
    from north_camera_scene_v59 import setup_scene
    from solar_physics_v63 import lighting_schedule

    publication(protocol, commit)
    p, master, groups = metadata(protocol, output)
    if (tuple(bpy.app.version[:3]) != (4, 5, 9) or bpy.app.build_hash.decode() != BLENDER_BUILD
            or digest(bpy.app.binary_path) != BLENDER_SHA):
        raise ValueError("Require the exact original Blender executable and build")
    out = Path(output)
    out.mkdir()
    tick = time.monotonic()
    rows = []

    def clock():
        if (datetime.now(timezone.utc) >= deadline
                or time.monotonic() - tick >= p["cap_seconds"]):
            raise TimeoutError("Outer v106 render allocation elapsed")

    try:
        for index, requested in groups.items():
            clock()
            unit = master["units"][index]
            mesh = relative(unit["mesh"])
            if digest(mesh) != unit["mesh_sha256"]:
                raise ValueError("Original sealed terrain mesh changed")
            with np.load(mesh, allow_pickle=False) as data:
                vertices, faces = data["vertices"].copy(), data["faces"].copy()
                origin, enu = data["body_origin"].copy(), data["enu_rows"].copy()
            pair = unit["camera_pairs"][0]
            scene, camera, sun, nodes, bvh = setup_scene(
                out, vertices, faces, np.asarray(pair["reference"]["intrinsics"]))
            for node in nodes:
                scene.node_tree.nodes.remove(node)
            lights = {row["index"]: row for row in lighting_schedule(unit["site"]["centre_latitude_deg"])}
            for role in ("reference", "query"):
                subset = [row for row in requested if row["role"] == role]
                if not subset:
                    continue
                pose, intrinsics = (np.asarray(pair[role][name]) for name in ("cam2world", "intrinsics"))
                if len({(r["archive_geometry_path"], r["archive_geometry_sha256"]) for r in subset}) != 1:
                    raise ValueError("Frozen geometry changed between lighting conditions")
                geometry = audit_geometry(bvh, pose, intrinsics, origin, enu, subset[0], clock)
                camera.data.lens = 36 * intrinsics[0, 0] / SIZE
                world = pose.copy()
                world[:3, :3] = pose[:3, :3] @ np.diag([1., -1., -1.])
                camera.matrix_world = Matrix(world.tolist())
                for row in sorted(subset, key=lambda r: r["light_index"]):
                    light = lights[row["light_index"]]
                    if light["sun_enu"] != row["sun_enu"]:
                        raise ValueError("Frozen original solar direction differs")
                    saved = render_one(bpy, scene, sun, row, light, geometry, clock)
                    seal = out / "seals" / saved["key"] / "receipt.json"
                    write(seal, saved)
                    rows.append(saved)
                    print("v106 rendered", saved["key"], flush=True)
            del vertices, faces, bvh
        clock()
        if len(rows) != len(p["renders"]):
            raise ValueError("Incomplete fixed native rendering inventory")
        write(out / "COMPLETE.json", dict(
            complete=True, split="model_selection", calibration_or_final_access=False,
            cohort_sha256=digest(protocol), publication_commit=commit, renderer_sha256=BLENDER_SHA,
            renderer_build=BLENDER_BUILD, cpu_threads=2, rows=rows, elapsed_seconds=time.monotonic() - tick,
        ))
    except BaseException as exc:
        write(out / "FAILED.json", dict(complete=False, error_type=type(exc).__name__,
                                       error=str(exc), completed_render_count=len(rows)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--publication-commit")
    parser.add_argument("--deadline-utc")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if args.validate_only:
        p, _, _ = metadata(args.protocol, args.output)
        print("Frozen historical metadata checked for", len(p["renders"]), "native images")
        return
    if args.publication_commit is None or args.deadline_utc is None:
        parser.error("Rendering requires a published commit and the outer UTC deadline")
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.utcoffset() is None or deadline <= datetime.now(timezone.utc):
        parser.error("Use the active absolute outer UTC deadline")
    sys.path.insert(0, str(Path.cwd()))
    run(args.protocol, args.output, args.publication_commit, deadline)


if __name__ == "__main__":
    main()
