"""Prototype for assembling fresh metadata over a verified v98 science capsule.

Caller responsibilities: verify the signed exact v110 ticket/checkout and
public release metadata, download the single pinned v98 archive by SHA256,
and run unchanged v98 hydration against the pinned old full manifest. This
module never publishes, reserves, downloads, restores a journal, or runs science.
"""

import base64
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

import solar_capsule_v75 as capsule
from solar_portable_evidence_v75 import file, pin, read, time_utc
from solar_remote_relay_v98 import charge_nonce

BASE = {
    "archive_url": "https://github.com/ThomasdeChillaz/lunar-image-matching/releases/download/solar-arm-transport-v98/solar-arm-v98-public.zip",
    "archive_bytes": 563675388,
    "archive_sha256": "2d25dca8b56f86f10444289b7458aeed4ba9a6755bf1ff19fca7014afaa1dee0",
    "manifest_sha256": "825ca518c562e269c6128fd485d133aef9091f7eb43c07c8a0261868c757522a",
    "protocol_path": "reports/solar-modern-arm-v98/pilot-protocol.json",
    "protocol_sha256": "9ac4fe0828069b32d26741477428ce41ccd3a06a8e16b6df64426b3c8c192a25",
    "authorization_path": "reports/solar-modern-arm-v98/authorization.json",
    "authorization_sha256": "903cea12904896143902ff32f836ce67e14293d7e5a8f6f2f62bc3ab4cee27e7",
}
PROTOCOL = "reports/solar-modern-arm-v110/pilot-protocol.json"
AUTHORIZATION = "reports/solar-modern-arm-v110/authorization.json"
OUTPUT = "runs/solar-modern-arm-v110/pilot"
REPOSITORY = "ThomasdeChillaz/lunar-image-matching"
ADDED_SOURCES = {
    "solar_portable_catalogue_v110.py", "solar_capsule_overlay_v110.py",
    "solar_remote_relay_v110.py", "run_solar_evaluation_v110.py",
    ".github/workflows/solar-remote-cpu-v110.yml",
    "reports/solar_generic_sun_arm_probe_v110.json",
    "reports/solar_arm_v98_remote_watchdog.json",
}
MUTABLE_METADATA = {
    "output", "created_utc", "latest_start_utc", "absolute_deadline_utc",
    "baseline_admission", "source_sha256",
}


def _json_bytes(data):
    return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _snapshot(auth):
    raw = base64.b64decode(auth["canonical_journal_bytes_base64"], validate=True)
    if _sha(raw) != auth["canonical_journal_sha256"]:
        raise ValueError("Authorization journal snapshot digest differs")
    return json.loads(raw)


def validate_new_authorization(old_auth, auth, protocol, binding):
    """Bind the new one-use charge to an append-only journal, never import it."""
    if (
        auth.get("schema_version") != 75 or auth.get("repository") != REPOSITORY
        or auth.get("protocol_path") != PROTOCOL
        or auth.get("protocol_sha256") != binding["protocol_sha256"]
        or auth.get("source_publication_commit") != binding["source_publication_commit"]
        or auth.get("canonical_journal") != protocol["reservation_journal"]
        or auth.get("authorized_attempt") != 1
        or auth.get("refunds") is not False or auth.get("automatic_retry") is not False
        or auth.get("nonce") != binding["nonce"] or auth["nonce"] == old_auth["nonce"]
        or auth.get("original_latest_start_utc") != protocol["latest_start_utc"]
        or auth.get("original_absolute_deadline_utc") != protocol["absolute_deadline_utc"]
        or auth.get("max_calendar_seconds") != protocol["max_calendar_seconds"]
        or auth.get("case_ids") != protocol["case_ids"] or auth.get("tracks") != ["estimated_sun"]
    ):
        raise ValueError("Fresh authorization and reviewed v110 protocol differ")

    from solar_evaluation_protocol_v74 import validate_lease

    _, latest, _ = validate_lease(protocol)
    old = _snapshot(old_auth)
    new = _snapshot(auth)
    entry = auth["reservation"]["reservation"]
    created, reserved, issued = map(time_utc, (
        protocol["created_utc"], entry["reserved_utc"], auth["created_utc"],
    ))
    if not time_utc(old_auth["created_utc"]) < created <= reserved <= issued <= latest:
        raise ValueError("Fresh authorization must follow the original charge and its own freeze")
    history = new["reservations"]
    old_history = old["reservations"]
    job_ids = [row["job_id"] for row in history]
    job_id = binding["protocol_sha256"] + "/evaluation"
    if (
        old["ledger_sha256"] != new["ledger_sha256"]
        or new["ledger_sha256"] != protocol["compute_ledger_sha256"]
        or len(history) <= len(old_history) or history[:len(old_history)] != old_history
        or history[:-1] != protocol["baseline_admission"]["initial_reservations"]
        or len(job_ids) != len(set(job_ids)) or job_ids.count(job_id) != 1
        or entry != history[-1] or entry["job_id"] != job_id
        or entry["category"] != "baseline"
        or entry["seconds"] != protocol["max_calendar_seconds"]
        or entry.get("refund_on_failure") is not False
        or auth["nonce"] != charge_nonce(binding["protocol_sha256"], entry)
    ):
        raise ValueError("Fresh reservation must extend, never replace or consume, v98 history")


def _verify_public_source(checkout, commit, name, path):
    public = subprocess.check_output(["git", "-C", str(checkout), "show", commit + ":" + name])
    if public != path.read_bytes():
        raise ValueError("New overlay source bytes differ from the published source commit: " + name)


def plan_overlay(hydrated, checkout, metadata, binding):
    """Return deterministic logical manifest bytes and exact source paths. Read only."""
    if (
        binding.get("base") != BASE
        or binding.get("protocol_path") != PROTOCOL
        or binding.get("authorization_path") != AUTHORIZATION
        or not re.fullmatch(r"[0-9a-f]{40}", binding.get("source_publication_commit", ""))
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", binding.get(field, ""))
            for field in ("protocol_sha256", "authorization_sha256", "nonce")
        )
        or binding["protocol_sha256"] == BASE["protocol_sha256"]
        or binding["authorization_sha256"] == BASE["authorization_sha256"]
    ):
        raise ValueError("Exact frozen v98 input and distinct v110 metadata identities required")
    hydrated, checkout, metadata = map(Path, (hydrated, checkout, metadata))
    payload = hydrated / "payload"
    old_manifest = read(pin(hydrated, "manifest.json", BASE["manifest_sha256"]))
    if (
        old_manifest["protocol_path"] != BASE["protocol_path"]
        or old_manifest["protocol_sha256"] != BASE["protocol_sha256"]
        or old_manifest["pre_reservation_authorization"] != {
            "path": BASE["authorization_path"], "sha256": BASE["authorization_sha256"],
        }
        or old_manifest["schema_version"] != 75
        or old_manifest["capsule_kind"] != "model_selection_primary_full_sun"
        or old_manifest["complete"] is not True
    ):
        raise ValueError("Base capsule provenance changed")
    old_protocol_path = pin(payload, BASE["protocol_path"], BASE["protocol_sha256"])
    old_auth_path = pin(payload, BASE["authorization_path"], BASE["authorization_sha256"])
    old_protocol, old_auth = read(old_protocol_path), read(old_auth_path)
    expected_old = {
        **old_protocol["input_sha256"], **old_protocol["source_sha256"],
        BASE["protocol_path"]: BASE["protocol_sha256"],
        BASE["authorization_path"]: BASE["authorization_sha256"],
    }
    if old_manifest["files"] != expected_old:
        raise ValueError("Original data/source allowlist differs from the base manifest")

    new_protocol_path = pin(checkout, PROTOCOL, binding["protocol_sha256"])
    new_auth_path = pin(metadata, AUTHORIZATION, binding["authorization_sha256"])
    protocol, auth = read(new_protocol_path), read(new_auth_path)
    if protocol.get("overlay_transport_v110") != BASE:
        raise ValueError("Published v110 protocol must freeze the exact original release asset")
    science_new = {
        name: value for name, value in protocol.items()
        if name not in MUTABLE_METADATA | {"overlay_transport_v110"}
    }
    science_old = {name: value for name, value in old_protocol.items() if name not in MUTABLE_METADATA}
    if science_new != science_old or protocol.get("output") != OUTPUT:
        raise ValueError("v110 overlay may not change any scientific case, profile, data or budget policy")
    old_admission, new_admission = old_protocol["baseline_admission"], protocol["baseline_admission"]
    if any(
        new_admission.get(key) != value for key, value in old_admission.items()
        if key not in {"initial_reservations", "remaining_at_preparation_seconds"}
    ) or set(old_admission) != set(new_admission):
        raise ValueError("Original baseline admission policy changed")
    old_sources, new_sources = old_protocol["source_sha256"], protocol["source_sha256"]
    if (
        any(new_sources.get(name) != digest for name, digest in old_sources.items())
        or set(new_sources) - set(old_sources) != ADDED_SOURCES
        or set(ADDED_SOURCES) & set(old_protocol["input_sha256"])
    ):
        raise ValueError("All inherited source bytes and exact v110 additions are required")
    validate_new_authorization(old_auth, auth, protocol, binding)
    old_initial = old_admission["initial_reservations"]
    new_initial = new_admission["initial_reservations"]
    if new_initial[:len(old_initial)] != old_initial:
        raise ValueError("The original freeze's reservation history must remain a prefix")
    additional_baseline = sum(
        entry["seconds"] for entry in new_initial[len(old_initial):]
        if entry["category"] == "baseline"
    )
    remaining = old_admission["remaining_at_preparation_seconds"] - additional_baseline
    if (
        type(new_admission["remaining_at_preparation_seconds"]) not in (int, float)
        or new_admission["remaining_at_preparation_seconds"] != remaining
        or remaining < protocol["max_calendar_seconds"] + new_admission["maximum_full_calendar_seconds"]
    ):
        raise ValueError("Fresh baseline headroom must match the exact appended charges")
    commit = binding["source_publication_commit"]
    _verify_public_source(checkout, commit, PROTOCOL, new_protocol_path)
    for name, digest in old_sources.items():
        _verify_public_source(checkout, commit, name, pin(checkout, name, digest))

    inherited = {
        name: file(payload, name) for name in expected_old
        if name not in {BASE["protocol_path"], BASE["authorization_path"]}
    }
    if any(not member.is_file() for member in inherited.values()):
        raise ValueError("Hydrated science capsule is incomplete")
    added = {PROTOCOL: new_protocol_path, AUTHORIZATION: new_auth_path}
    for name in sorted(ADDED_SOURCES):
        added[name] = pin(checkout, name, new_sources[name])
        _verify_public_source(checkout, commit, name, added[name])
    if set(inherited) & set(added):
        raise ValueError("Overlay cannot overwrite a historical payload member")

    files = {
        **protocol["input_sha256"], **new_sources,
        PROTOCOL: binding["protocol_sha256"], AUTHORIZATION: binding["authorization_sha256"],
    }
    all_sources = inherited | added
    if set(all_sources) != set(files) or len(files) > capsule.MAX_FILES:
        raise ValueError("Reassembled capsule does not match the exact frozen v110 member closure")
    for name, source in all_sources.items():
        capsule.allowed(name, "source" if name in new_sources else "input")
        size = source.stat().st_size
        if size > min(capsule.MAX_MEMBER_BYTES, old_manifest["max_member_bytes"]) or (
            name.endswith(".json") and size > capsule.MAX_JSON_BYTES
        ):
            raise ValueError("Reassembled capsule member exceeds the original bound")
    size = sum(path.stat().st_size for path in all_sources.values())
    if size > min(capsule.MAX_TOTAL_BYTES, old_manifest["max_total_bytes"]):
        raise ValueError("Reassembled capsule exceeds the original transport bound")
    manifest = dict(
        old_manifest, files=dict(sorted(files.items())), protocol_path=PROTOCOL,
        protocol_sha256=binding["protocol_sha256"], publication_commit=binding["source_publication_commit"],
        pre_reservation_authorization={"path": AUTHORIZATION, "sha256": binding["authorization_sha256"]},
        total_bytes=size,
        scope="Original v98 science and measured pack time with distinct frozen v110 transport metadata.",
    )
    return _json_bytes(manifest), all_sources


def assemble_overlay(hydrated, checkout, metadata, destination, binding):
    """Materialize one same-filesystem scratch overlay, with no mutation of inputs."""
    manifest, sources = plan_overlay(hydrated, checkout, metadata, binding)
    digest = _sha(manifest)
    if digest != binding.get("assembled_manifest_sha256"):
        raise ValueError("Deterministic logical overlay manifest differs from the signed ticket")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / "payload"
    for name, source in sources.items():
        target = file(payload, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Old bytes were verified by v98 hydration and are checked again by
        # the unchanged v75 capsule gate before installation into the checkout.
        os.link(source, target, follow_symlinks=False)
    with (destination / "manifest.json").open("xb") as output:
        output.write(manifest)
    return {"manifest_sha256": digest, "member_count": len(sources)}
