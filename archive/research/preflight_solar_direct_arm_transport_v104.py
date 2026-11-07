"""Read-only payload preflight for the original full24 Direct ARM comparator."""

import argparse
from collections import Counter
from pathlib import Path
from struct import unpack
import subprocess

from direct_render_adapter_v73 import render_profile_inputs, validate_render_profile
from run_solar_distributional_evaluation_v90 import fixed_v86_cases
from solar_capsule_v75 import (
    MAX_FILES, MAX_JSON_BYTES, MAX_MEMBER_BYTES, MAX_TOTAL_BYTES, allowed, portability_findings,
)
from solar_direct_arm_bridge_v103 import reference_identity
from solar_portable_evidence_v75 import file, new_json, pin, read, sha
from solar_public_transport_v98 import (
    PATH as PRIVATE_CONTRACT, TEXT_SUFFIXES, scan_public_text, validate_release_inputs,
)


ARM_PREFLIGHT = "runs/solar-modern-arm-preflight-only-v101.json"
ARM_PREFLIGHT_SHA = "513122ba9bd640ad797347b8953a02516231225345269ff5a10c63c7f29e90ee"
DIRECT_PROTOCOL = "reports/solar-full-direct-render-v92.json"
DIRECT_PROTOCOL_SHA = "9421f06bdaad9d3b8b9da8e424d4d21b6e5ceda90f6a1d228848842d24c0f909"
HISTORICAL_PUBLIC_HEAD = "21025b6787228cd05a051249fc7224525c0218bc"
ARM_PROFILES = ("frozen_step0_A1B0C1_quantization_aware", "SIFT2048", "loma_r_common", "romav2_common")
BRIDGE = "solar_direct_arm_bridge_v103.py"
ORIGINAL_ASSETS = ("dataset_summary", "geometry_audit", "original_master", "reference_assets_index")


def merged(*groups):
    result = {}
    for group in groups:
        for name, digest in group.items():
            if name in result and result[name] != digest:
                raise ValueError("Conflicting immutable transport pin: " + name)
            result[name] = digest
    return result


def historical_direct():
    path = pin(".", DIRECT_PROTOCOL, DIRECT_PROTOCOL_SHA)
    public = subprocess.check_output(["git", "show", f"{HISTORICAL_PUBLIC_HEAD}:{DIRECT_PROTOCOL}"], timeout=5)
    if path.read_bytes() != public:
        raise ValueError("Full24 profile differs from its archived public protocol")
    protocol = read(path)
    if (protocol["split"] != "model_selection" or protocol["calibration_or_final_access"] is not False
            or protocol["tracks"] != ["estimated_sun"] or protocol["accepted_results"] is not False
            or len(protocol["profiles"]) != 1):
        raise ValueError("Original development-only full24 Direct profile required")
    return protocol


def native_reference_png(path):
    # Hashing proves the complete source bytes. The fixed header check proves
    # native content without using a decoder or exposing pixels to this tool.
    with path.open("rb") as stream:
        header = stream.read(29)
    if len(header) != 29 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR":
        raise ValueError("Original reference render is not a PNG: " + str(path))
    width, height, depth, mode, compression, filtering, _ = unpack(">IIBBBBB", header[16:])
    if (width, height, depth) != (512, 512, 8) or mode not in (0, 2) or compression or filtering:
        raise ValueError("Original native512 uint8 reference render required: " + str(path))


def direct_closure(profile, cases, base_inputs):
    manifest, libraries, provenance = validate_render_profile(profile)
    library_protocol = read(profile["render_evidence"]["protocol"])
    inputs = merged(
        render_profile_inputs(profile),
        {library_protocol[key]: library_protocol[key + "_sha256"] for key in ORIGINAL_ASSETS},
    )
    sources = merged(profile["source_sha256"], library_protocol["source_sha256"],
                     {BRIDGE: sha(file(".", BRIDGE))})
    # Reuse exactly the 32 eligible v80 reference cameras, including each
    # materialized array's original raw camera geometry. The inherited
    # v103 bridge verifies the historical Git blob and full map identities.
    identities = {}
    reference_images = set()
    for case in cases:
        if not case["positive"] or case["unit_index"] in identities:
            continue
        identity = reference_identity(case, base_inputs)
        key = identity["reference_geometry_sha256"]
        library = libraries[key]
        if (library["unit_index"] != case["unit_index"] or library["component"] != case["component"]
                or identity["mesh_sha256"] != library["map_mesh_sha256"]
                or identity["scene_mesh_sha256"] != library["scene_mesh_sha256"]):
            raise ValueError("An ARM reference case differs from the original Direct library")
        raw_geometry = case["reference_asset_metadata"]["source_reference_geometry_path"]
        inputs = merged(inputs, {raw_geometry: key})
        identities[case["unit_index"]] = key
        for entry in library["entries"]:
            name = entry["image"]
            if name in reference_images or Path(name).parent.name != "reference":
                raise ValueError("Each original reference-role image must be unique")
            reference_images.add(name)
    if (len(libraries) != 32 or len(identities) != 32 or len(reference_images) != 768
            or set(identities.values()) != set(libraries) or manifest["query_role_images_used"]):
        raise ValueError("All original reference cameras and their 768 renders are required")
    return inputs, sources, reference_images, provenance


def inspect_members(inputs, sources, proved_direct, reference_images):
    joined = merged(inputs, sources)
    if len(joined) > MAX_FILES:
        raise ValueError("Logical payload exceeds the original file count")
    contract = read(file(".", PRIVATE_CONTRACT))
    sizes = {}
    for name, digest in joined.items():
        role = "source" if name in sources else "input"
        allowed(name, role)
        path = pin(".", name, digest) if name in proved_direct else file(".", name)
        if not path.is_file():
            raise ValueError("A historically staged ARM payload member is missing: " + name)
        size = path.stat().st_size
        if size > MAX_MEMBER_BYTES or (name.endswith(".json") and size > MAX_JSON_BYTES):
            raise ValueError("Logical payload exceeds an original member bound: " + name)
        sizes[name] = size
        if name in reference_images:
            native_reference_png(path)
        if name in proved_direct and name.endswith(TEXT_SUFFIXES):
            scan_public_text(name, path.read_bytes(), contract)
    if sum(sizes.values()) >= MAX_TOTAL_BYTES:
        raise ValueError("Logical payload exceeds the original 4 GiB ceiling")
    portability = portability_findings(".", sorted(proved_direct))
    if portability["failures"]:
        raise ValueError("New Direct closure contains nonportable metadata")
    return sizes


def preflight():
    base = read(pin(".", ARM_PREFLIGHT, ARM_PREFLIGHT_SHA))
    expected = (base["split"], base["tracks"], base["cpu_threads"], base["blender_executable"],
                base["calibration_or_final_access"], base["accepted_results"])
    if (expected != ("model_selection", ["estimated_sun"], 2, None, False, False)
            or tuple(p["name"] for p in base["profiles"]) != ARM_PROFILES):
        raise ValueError("Use the preserved four-arm local preflight only")
    validate_release_inputs(base, Path.cwd())
    basis, _, cases = fixed_v86_cases(full=True)
    direct = historical_direct()
    ids = [case["case_id"] for case in cases]
    if (len(ids) != 384 or ids != base["case_ids"] or ids != direct["case_ids"]
            or base["pilot_case_ids"] != basis["pilot_case_ids"]):
        raise ValueError("Historical case identity, order or qualification selection changed")
    direct_inputs, direct_sources, images, provenance = direct_closure(
        direct["profiles"][0], cases, base["input_sha256"],
    )
    sources = merged(base["source_sha256"], direct_sources)
    inputs = merged(base["input_sha256"], direct_inputs)
    proved_direct = set(direct_inputs) | set(direct_sources)
    sizes = inspect_members(inputs, sources, proved_direct, images)
    base_members = set(base["source_sha256"]) | set(base["input_sha256"])
    extras = set(sizes) - base_members
    base_bytes = sum(sizes[n] for n in base_members)
    extra_bytes = sum(sizes[n] for n in extras)
    return dict(
        schema="solar_direct_arm_payload_preflight_v104",
        complete=True, scientific_execution=False, publication_or_reservation=False,
        source_sha256=sha(Path(__file__)),
        base_preflight_sha256=ARM_PREFLIGHT_SHA, historical_direct_protocol_sha256=DIRECT_PROTOCOL_SHA,
        original_direct_publication_commit=direct["profiles"][0]["publication_commit"],
        historical_direct_protocol_scope=("The original published v92 execution protocol is only a Git proof "
                                          "of the profile and stays outside the capsule. Its historical local "
                                          "machine binding is not an input to the Direct adapter."),
        original_case_count=len(ids), qualification_case_count=len(base["pilot_case_ids"]),
        original_reference_cameras=32, exact_native_reference_pngs=len(images),
        direct_members_rehashed=len(proved_direct),
        extra_members_by_role=dict(sorted(Counter("source" if n in sources else "input" for n in extras).items())),
        direct_inputs=dict(sorted(direct_inputs.items())), direct_sources=dict(sorted(direct_sources.items())),
        base_logical_bytes=base_bytes, additional_direct_bytes=extra_bytes, combined_logical_bytes=sum(sizes.values()),
        remaining_logical_headroom_bytes=MAX_TOTAL_BYTES-sum(sizes.values()), logical_limit_bytes=MAX_TOTAL_BYTES,
        offline_original_reference_preparation_seconds=provenance["offline_original_reference_preparation_seconds"],
        transport_scope=("Existing private preflight remains local. All new Direct and overlapping Direct pins "
                         "were rehashed and release-scanned; inherited ARM-only assets were checked for existence "
                         "and size. No archive, receipt for dispatch, render, inference or reservation was created."),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve any existing preflight evidence")
    receipt = preflight()
    new_json(args.output, receipt)
