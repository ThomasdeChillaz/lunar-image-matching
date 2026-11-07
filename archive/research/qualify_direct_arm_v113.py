"""Exact Direct-only v113 freeze and logical overlay. No allocation or dispatch."""

import base64
from pathlib import Path
import hashlib
import json
import os
import subprocess

import solar_capsule_v75 as capsule
from solar_capsule_overlay_v110 import BASE
from solar_portable_evidence_v75 import file, pin, read, time_utc
import solar_remote_relay_v98 as original
from plan_solar_matched_arm_v113 import (
    ARM_PROTOCOL, ARM_PUBLISHED, ARM_WATCHDOG, COHORT, DIRECT_HISTORICAL_PROTOCOL,
    DIRECT_TRANSPORT_PREFLIGHT, DIRECT_QUALIFICATION_CAP, V112_PROTECTED, V112_STOP, RUN_CAP, pinned, outstanding_v112,
)
from solar_direct_transport_v113 import validate_binding, inventory, hash_path

PROTOCOL = "reports/solar-direct-arm-v113/pilot-protocol.json"
AUTHORIZATION = "reports/solar-direct-arm-v113/authorization.json"
OUTPUT = "runs/solar-direct-arm-v113/pilot"
MODE = "matched_direct_arm_cost_pilot_v113"
HISTORICAL_DIRECT_COMMIT = "21025b6787228cd05a051249fc7224525c0218bc"
WHEELS = "requirements_arm_cp312_v113.txt"
SOURCES = {
    "solar_direct_transport_v113.py", "qualify_direct_arm_v113.py", "solar_direct_controls_v113.py",
    "plan_solar_matched_arm_v113.py", "verify_solar_matched_arm_v113.py", "run_solar_direct_arm_v113.py",
    "solar_remote_relay_direct_v113.py", WHEELS, ".github/workflows/solar-remote-direct-cpu-v113.yml",
    ARM_PROTOCOL[0], ARM_PUBLISHED[0], ARM_WATCHDOG[0], DIRECT_TRANSPORT_PREFLIGHT[0], V112_STOP[0],
}
MUTABLE = {
    "profiles", "stage_mode", "output", "created_utc", "latest_start_utc", "absolute_deadline_utc",
    "baseline_admission", "source_sha256", "input_sha256", "direct_supplement",
}
CONTROLS = {
    "source": "control_direct_render_v90.py", "native_shape": [512, 512], "image_seed": 20261101,
    "generic_lights": 24, "translation_px": [8, -4], "target_light_index": 7,
    "minimum_matches_and_inliers": 6, "median_native_error_px_max": 1.0,
    "translation_error_m_max": 0.05, "rotation_error_deg_max": 0.5,
    "invalid_support": "zero_reference_depth", "blank": "native512_uint8_zero",
    "extra_boundary_control": "injected_query_sun_rejected",
}


def _merged(first, second):
    if any(first[name] != second[name] for name in first.keys() & second.keys()):
        raise ValueError("Frozen source/input closure has conflicting pins")
    return first | second


def direct_profile(root):
    archived = pinned(root, DIRECT_HISTORICAL_PROTOCOL)
    public = subprocess.check_output([
        "git", "-C", str(root), "show", HISTORICAL_DIRECT_COMMIT + ":" + DIRECT_HISTORICAL_PROTOCOL[0],
    ])
    if hashlib.sha256(public).hexdigest() != DIRECT_HISTORICAL_PROTOCOL[1]:
        raise ValueError("Original v92 public Direct profile proof changed")
    profile, = archived["profiles"]
    return profile


def original_profile_in_public_history(root):
    """Only profile metadata travels. Never stage the old machine-bound protocol."""
    public = subprocess.check_output([
        "git", "-C", str(root), "show", HISTORICAL_DIRECT_COMMIT + ":" + DIRECT_HISTORICAL_PROTOCOL[0],
    ])
    if hashlib.sha256(public).hexdigest() != DIRECT_HISTORICAL_PROTOCOL[1]:
        raise ValueError("Original v92 public Direct profile proof changed")
    profile, = json.loads(public)["profiles"]
    return profile


def supplement_binding(receipt):
    if (
        receipt.get("schema") != "solar_direct_arm_transport_v113"
        or receipt.get("input_members") != 677
        or receipt.get("scientific_execution") is not False
    ):
        raise ValueError("Original complete Direct-only supplement inventory required")
    value = {"part": receipt["asset"], "manifest_sha256": receipt["manifest_sha256"]}
    validate_binding(value["part"], value["manifest_sha256"])
    return value


def source_and_input_pins(root):
    previous = pinned(root, ARM_PROTOCOL)
    exact_direct = pinned(root, DIRECT_TRANSPORT_PREFLIGHT)
    sources = _merged(previous["source_sha256"], exact_direct["direct_sources"])
    new = {name: hash_path(file(root, name)) for name in sorted(SOURCES)}
    for pair in (ARM_PROTOCOL, ARM_PUBLISHED, ARM_WATCHDOG, DIRECT_TRANSPORT_PREFLIGHT, V112_STOP):
        if new[pair[0]] != pair[1]:
            raise ValueError("Frozen qualification predecessor evidence changed")
    inputs = _merged(previous["input_sha256"], exact_direct["direct_inputs"])
    inputs = _merged(inputs, {COHORT[0]: COHORT[1]})
    return _merged(sources, new), inputs


def _scope_metadata(protocol, root="."):
    old = pinned(root, ARM_PROTOCOL)
    cohort = pinned(root, COHORT)
    if (
        protocol.get("stage_mode") != MODE or protocol.get("output") != OUTPUT
        or protocol.get("purpose") != "cost_pilot" or protocol.get("target_purpose") != "modern_comparison"
        or protocol.get("max_calendar_seconds") != DIRECT_QUALIFICATION_CAP
        or protocol.get("pilot_case_ids") != cohort["pilot_case_ids"] or protocol.get("case_ids") != old["case_ids"]
        or protocol.get("case_cache_seals") or protocol.get("calibration_or_final_access") is not False
        or protocol["baseline_admission"].get("v112_stop_replay_sha256") != V112_STOP[1]
        or protocol.get("qualification_controls_v113") != CONTROLS
        or protocol.get("accepted_results") is not False
        or protocol.get("cpu_threads") != 2 or protocol.get("tracks") != ["estimated_sun"]
        or protocol.get("profiles") != [original_profile_in_public_history(root)]
    ):
        raise ValueError("Exact Direct-only profile, eight cases and procedural controls required")
    if (
        {key: value for key, value in protocol.items() if key not in MUTABLE | {"qualification_controls_v113"}}
        != {key: value for key, value in old.items() if key not in MUTABLE}
    ):
        raise ValueError("No inherited inference, geometry or resource policy may change")
    expected = pinned(root, DIRECT_TRANSPORT_PREFLIGHT)
    expected_sources = _merged(old["source_sha256"], expected["direct_sources"])
    expected_inputs = _merged(_merged(old["input_sha256"], expected["direct_inputs"]), {COHORT[0]: COHORT[1]})
    if (
        protocol["input_sha256"] != expected_inputs
        or set(protocol["source_sha256"]) != set(expected_sources) | SOURCES
        or any(protocol["source_sha256"].get(name) != digest for name, digest in expected_sources.items())
    ):
        raise ValueError("Inherited ARM and original Direct source/input pins must remain exact")
    for pair in (ARM_PROTOCOL, ARM_PUBLISHED, ARM_WATCHDOG, DIRECT_TRANSPORT_PREFLIGHT, V112_STOP):
        if protocol["source_sha256"].get(pair[0]) != pair[1]:
            raise ValueError("Exact public predecessor proof required")
    binding = protocol["direct_supplement"]
    if set(binding) != {"part", "manifest_sha256"}:
        raise ValueError("Exactly one Direct supplement is permitted")
    validate_binding(binding["part"], binding["manifest_sha256"])
    return old


def verify_v110_receipt(root="."):
    expected = pinned(root, ARM_PUBLISHED)
    watchdog = pinned(root, ARM_WATCHDOG)
    reservation = watchdog["original_reservation"]["reservation"]
    if (
        expected["fresh_replayed_rows"] != 32 or expected["projected_full_cost_seconds"] != 65729
        or expected["protocol_sha256"] != ARM_PROTOCOL[1] or expected["remote_watchdog_sha256"] != ARM_WATCHDOG[1]
        or watchdog["workflow_run_id"] != 41930272861 or watchdog["run_attempt"] != 1
        or watchdog["complete"] is not True or watchdog["child_returncode"] != 0
        or reservation["job_id"] != ARM_PROTOCOL[1] + "/evaluation"
        or reservation["seconds"] != 1800 or reservation["category"] != "baseline"
        or reservation["refund_on_failure"] is not False
    ):
        raise ValueError("Exact previously successful ARM run and its canonical debit required")
    return reservation


def prospective_protocol(receipt, *, now=None, v112_unreserved=None):
    from solar_evaluation_protocol_v74 import lease_fields
    from solar_learning_pipeline_v73 import journal_state, load_contract, remaining

    root = Path.cwd()
    previous = pinned(root, ARM_PROTOCOL)
    prior_entry = verify_v110_receipt(root)
    contract = load_contract(previous["learning_contract"])
    _, journal, _ = journal_state(previous["compute_ledger"], previous["compute_ledger_sha256"], previous["reservation_journal"])
    headroom = remaining(contract)["baseline"]
    verified_protected = outstanding_v112(root, journal)
    if v112_unreserved is not None and v112_unreserved != verified_protected:
        raise ValueError("Protected v112 remainder must match actual charges and preregistered stop")
    v112_unreserved = verified_protected
    if (
        not 0 <= v112_unreserved <= V112_PROTECTED
        or journal["reservations"].count(prior_entry) != 1
        or headroom < DIRECT_QUALIFICATION_CAP + RUN_CAP + v112_unreserved
    ):
        raise ValueError("Preserve the exact original charge and the remaining v112/comparison envelopes")
    sources, inputs = source_and_input_pins(root)
    admission = dict(
        previous["baseline_admission"], initial_reservations=journal["reservations"],
        remaining_at_preparation_seconds=headroom, unreserved_v112_protected_seconds=v112_unreserved,
        v112_stop_replay_sha256=V112_STOP[1],
    )
    protocol = dict(
        previous, stage_mode=MODE, output=OUTPUT, profiles=[direct_profile(root)], baseline_admission=admission,
        input_sha256=inputs, source_sha256=sources, direct_supplement=supplement_binding(receipt),
        qualification_controls_v113=CONTROLS,
    )
    protocol.update(lease_fields(DIRECT_QUALIFICATION_CAP, 3600, created=now))
    _scope_metadata(protocol, root)
    return protocol


def primary_scope(protocol_path, protocol, at=None, check_assets=True):
    from solar_evaluation_protocol_v74 import qualified_cases, validate_runtime
    from solar_public_transport_v98 import validate_release_inputs
    from solar_weight_transport_v96 import LOCK, validate_declaration

    if str(protocol_path) != PROTOCOL:
        raise ValueError("Exact separate v113 protocol path required")
    _scope_metadata(protocol)
    validate_runtime(protocol, protocol_path, at=at)
    if not check_assets:
        return protocol["max_calendar_seconds"]
    validate_release_inputs(protocol, Path.cwd())
    validate_declaration(protocol, original.read(LOCK))
    manifest, _, cases = qualified_cases(protocol["manifest"], protocol["geometry_audit"])
    if protocol["case_ids"] != [case["case_id"] for case in cases]:
        raise ValueError("The full ordered 384-case development authority changed")
    for name, digest in _merged(protocol["input_sha256"], protocol["source_sha256"]).items():
        pin(".", name, digest)
    original.validate_primary_archives(cases, manifest["reference_precompute_receipts"])
    return protocol["max_calendar_seconds"]


def _snapshot(auth):
    raw = base64.b64decode(auth["canonical_journal_bytes_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != auth["canonical_journal_sha256"]:
        raise ValueError("Authorization journal hash changed")
    return json.loads(raw)


def _authorized(old_auth, auth, protocol, binding, checkout):
    from solar_evaluation_protocol_v74 import validate_lease

    old, new = _snapshot(old_auth), _snapshot(auth)
    history = new["reservations"]
    prefix = protocol["baseline_admission"]["initial_reservations"]
    prior = verify_v110_receipt(checkout)
    entry = auth["reservation"]["reservation"]
    _, latest, _ = validate_lease(protocol)
    created = time_utc(protocol["created_utc"])
    reserved = time_utc(entry["reserved_utc"])
    issued = time_utc(auth["created_utc"])
    job = binding["protocol_sha256"] + "/evaluation"
    ids = [row["job_id"] for row in history]
    if (
        auth["protocol_path"] != PROTOCOL or auth["protocol_sha256"] != binding["protocol_sha256"]
        or auth["source_publication_commit"] != binding["source_publication_commit"]
        or auth["nonce"] != binding["nonce"] or auth["nonce"] == old_auth["nonce"]
        or auth["nonce"] != original.charge_nonce(binding["protocol_sha256"], entry)
        or auth["case_ids"] != protocol["case_ids"] or auth["tracks"] != ["estimated_sun"]
        or auth["original_latest_start_utc"] != protocol["latest_start_utc"]
        or auth["original_absolute_deadline_utc"] != protocol["absolute_deadline_utc"]
        or auth["max_calendar_seconds"] != DIRECT_QUALIFICATION_CAP
        or auth["authorized_attempt"] != 1 or auth["automatic_retry"] is not False or auth["refunds"] is not False
        or new["ledger_sha256"] != old["ledger_sha256"] or new["ledger_sha256"] != protocol["compute_ledger_sha256"]
        or history[:len(old["reservations"])] != old["reservations"] or history != [*prefix, entry]
        or prefix.count(prior) != 1 or len(ids) != len(set(ids)) or entry["job_id"] != job
        or entry["category"] != "baseline" or entry["seconds"] != DIRECT_QUALIFICATION_CAP
        or entry["refund_on_failure"] is not False
        or not time_utc(prior["reserved_utc"]) < created <= reserved <= issued <= latest
    ):
        raise ValueError("Separate fresh one-use charge must extend both historical ARM attempts")


def _git_bytes(root, commit, name):
    return subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + name], timeout=10)


def plan_overlay(hydrated, checkout, metadata, binding, supplement_root=None):
    root, checkout, metadata = map(Path, (hydrated, checkout, metadata))
    additions, _ = inventory(checkout)
    base_files = root / "payload"
    old_manifest = read(pin(root, "manifest.json", BASE["manifest_sha256"]))
    old_protocol = read(pin(base_files, BASE["protocol_path"], BASE["protocol_sha256"]))
    old_auth = read(pin(base_files, BASE["authorization_path"], BASE["authorization_sha256"]))
    protocol_path = pin(checkout, PROTOCOL, binding["protocol_sha256"])
    auth_path = pin(metadata, AUTHORIZATION, binding["authorization_sha256"])
    protocol, auth = read(protocol_path), read(auth_path)
    _scope_metadata(protocol, checkout)
    if binding["base"] != BASE or binding["direct_supplement"] != protocol["direct_supplement"]:
        raise ValueError("Signed base and Direct supplement bindings changed")
    _authorized(old_auth, auth, protocol, binding, checkout)
    old_members = _merged(old_protocol["source_sha256"], old_protocol["input_sha256"])
    if old_manifest["files"] != old_members | {
        BASE["protocol_path"]: BASE["protocol_sha256"], BASE["authorization_path"]: BASE["authorization_sha256"],
    }:
        raise ValueError("Original hydrated base closure changed")
    commit = binding["source_publication_commit"]
    if protocol_path.read_bytes() != _git_bytes(checkout, commit, PROTOCOL):
        raise ValueError("Exact published Direct protocol bytes required")
    source_paths = {}
    for name, digest in protocol["source_sha256"].items():
        path = pin(checkout, name, digest)
        if path.read_bytes() != _git_bytes(checkout, commit, name):
            raise ValueError("Published Direct source bytes differ: " + name)
        source_paths[name] = path
    input_paths = {}
    for name, digest in protocol["input_sha256"].items():
        if name in old_members:
            if old_members[name] != digest:
                raise ValueError("Inherited original member changed")
            input_paths[name] = file(base_files, name)
        elif name in additions:
            if additions[name] != digest:
                raise ValueError("Direct supplement may not substitute input")
            input_paths[name] = pin(supplement_root or checkout, name, digest)
        elif name == COHORT[0]:
            path = pin(checkout, name, digest)
            if path.read_bytes() != _git_bytes(checkout, commit, name):
                raise ValueError("Original published v103 cohort bytes changed")
            input_paths[name] = path
        else:
            raise ValueError("No additional unreviewed Direct inputs permitted")
    all_paths = input_paths | source_paths | {PROTOCOL: protocol_path, AUTHORIZATION: auth_path}
    files = _merged(protocol["input_sha256"], protocol["source_sha256"]) | {
        PROTOCOL: binding["protocol_sha256"], AUTHORIZATION: binding["authorization_sha256"],
    }
    if set(files) != set(all_paths) or len(files) > capsule.MAX_FILES:
        raise ValueError("Exact complete Direct closure required")
    total = 0
    for name, path in all_paths.items():
        capsule.allowed(name, "source" if name in source_paths else "input")
        size = path.stat().st_size
        if size > min(capsule.MAX_MEMBER_BYTES, old_manifest["max_member_bytes"]):
            raise ValueError("Original capsule member cap exceeded")
        total += size
    if total > min(capsule.MAX_TOTAL_BYTES, old_manifest["max_total_bytes"]):
        raise ValueError("Combined logical capsule exceeds the original 4 GiB cap")
    manifest = dict(
        old_manifest, files=dict(sorted(files.items())), protocol_path=PROTOCOL, publication_commit=commit,
        protocol_sha256=binding["protocol_sha256"], total_bytes=total,
        pre_reservation_authorization={"path": AUTHORIZATION, "sha256": binding["authorization_sha256"]},
        scope="Immutable historical primary assets plus separately signed Direct-only qualification metadata.",
    )
    return (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(), all_paths


def assemble_overlay(hydrated, checkout, metadata, destination, binding, supplement_root):
    manifest, files = plan_overlay(hydrated, checkout, metadata, binding, supplement_root)
    digest = hashlib.sha256(manifest).hexdigest()
    if digest != binding["assembled_manifest_sha256"]:
        raise ValueError("Deterministic Direct overlay differs from the signed manifest")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name, source in files.items():
        target = file(root / "payload", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target, follow_symlinks=False)
    with (root / "manifest.json").open("xb") as stream:
        stream.write(manifest)
    return {"manifest_sha256": digest, "member_count": len(files)}
