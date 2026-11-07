"""Freeze and verify the 96-case five-arm ARM comparison and its transport."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import solar_capsule_v75 as capsule
from plan_solar_matched_arm_v113 import (
    ARM_PROTOCOL, ARM_PUBLISHED, ARM_REPLAY, ARM_SUMMARY, ARM_WATCHDOG, COHORT,
    DIRECT_HISTORICAL_PROTOCOL, RUN_CAP, cost_plan, outstanding_v112, pinned,
    read_direct, read_v110,
)
from qualify_direct_arm_v113 import BASE, _merged, _snapshot, verify_v110_receipt
from solar_direct_transport_v113 import inventory, validate_binding
from solar_portable_evidence_v75 import file, pin, read, sha, time_utc
from verify_solar_matched_arm_v113 import exact_case_ids


PROTOCOL = "reports/solar-five-arm-v115/comparison-protocol.json"
AUTHORIZATION = "reports/solar-five-arm-v115/authorization.json"
OUTPUT = "runs/solar-five-arm-v115/comparison"
MODE = "matched_modern_direct_arm_v113"
COUNT = 96
DIRECT_PROTOCOL = ("reports/solar-direct-arm-v113/pilot-protocol.json",
                   "e8c139efea3e1007a7ec18c6681d5bb66b075c1785a062af15efadd3fa5f0cf3")
ARM_ARCHIVE = ("reports/solar_arm_v110_pilot_artifacts.zip",
               "afc669527e1ef43926fdb989c78c778c413e119424c1a3436ac820574f2222fb")
DIRECT_ARCHIVE = ("reports/solar-direct-arm-v113-replay-artifacts.zip",
                  "1a89758991addca5cfa151ec8a5576947d450678f2a6e5f1a3ef7740dd5dde06")
DIRECT_RECEIPT = ("direct_qualification_receipt_v113.json",
                  "824e1a7948f80ca45011d5d59027da4429d9e81b007a88b2f2079c4cf4d41993")
# The archived v92 protocol has machine-local historical fields. Its pinned
# Git blob is read for witness validation and stays outside the capsule.
GIT_INPUTS = dict((COHORT, ARM_ARCHIVE, DIRECT_ARCHIVE, DIRECT_PROTOCOL, ARM_REPLAY))
NEW_SOURCES = {
    "qualify_five_arm_v115.py",
    "run_solar_five_arm_v115.py",
    "solar_remote_relay_five_arm_v115.py",
    "summarize_solar_five_arm_v115.py",
    ".github/workflows/solar-remote-five-arm-v115.yml",
}
MUTABLE = {"purpose", "stage_mode", "output", "profiles", "created_utc", "latest_start_utc",
           "absolute_deadline_utc", "max_calendar_seconds", "baseline_admission",
           "source_sha256", "input_sha256"}
ADDED = {"comparison_case_count", "comparison_case_ids", "qualified_witnesses_v115"}


def _archive_member(archive, relative, expected):
    matches = [item for item in archive.infolist() if item.filename == relative]
    if len(matches) != 1 or matches[0].file_size > 2 * 1024**2 or matches[0].is_dir():
        raise ValueError("Missing or ambiguous bounded timing evidence member: " + relative)
    data = archive.read(matches[0])
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Prior timing evidence changed: " + relative)
    return data


def _store(root, relative, data):
    path = file(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return path


def _journal(root, protocol, journal):
    actual = read(file(root, protocol["reservation_journal"])) if journal is None else journal
    if (actual["ledger_sha256"] != protocol["compute_ledger_sha256"]
            or len({row["job_id"] for row in actual["reservations"]}) != len(actual["reservations"])):
        raise ValueError("Original canonical reservation history required")
    return actual


def _verified_timing(root, journal):
    root = Path(root).resolve()
    original = pinned(root, ARM_PROTOCOL)
    original_direct = pinned(root, DIRECT_PROTOCOL)
    journal = _journal(root, original, journal)
    prior_arm = verify_v110_receipt(root)
    ledger = pinned(root, (original["compute_ledger"], original["compute_ledger_sha256"]))
    headroom = ledger["remaining_baseline_wall_seconds"] - sum(
        row["seconds"] for row in journal["reservations"] if row["category"] == "baseline"
    )
    if ledger["accounting_complete"] is not True or journal["reservations"].count(prior_arm) != 1:
        raise ValueError("The original completed modern ARM debit must remain")
    with TemporaryDirectory(prefix="solar-five-arm-timing-") as tmp:
        evidence = Path(tmp)
        for path, digest in (ARM_PROTOCOL, ARM_PUBLISHED, ARM_WATCHDOG, ARM_REPLAY, COHORT,
                             DIRECT_HISTORICAL_PROTOCOL):
            _store(evidence, path, pin(root, path, digest).read_bytes())
        _store(evidence, original["reservation_journal"],
               json.dumps(journal, allow_nan=False).encode())
        with ZipFile(pin(root, *ARM_ARCHIVE)) as archive:
            _store(evidence, ARM_SUMMARY[0], _archive_member(archive, *ARM_SUMMARY))
        direct_dir = evidence / "direct_witness"
        with ZipFile(pin(root, *DIRECT_ARCHIVE)) as archive:
            raw_receipt = _archive_member(archive, *DIRECT_RECEIPT)
            receipt = json.loads(raw_receipt)
            receipt_path = _store(direct_dir, DIRECT_RECEIPT[0], raw_receipt)
            for role, item in receipt["files"].items():
                data = (pin(root, *DIRECT_PROTOCOL).read_bytes() if role == "protocol"
                        else _archive_member(archive, item["path"], item["sha256"]))
                _store(direct_dir, item["path"], data)
        arm_protocol, software, _, arm = read_v110(evidence)
        direct = read_direct(evidence, receipt_path, arm_protocol, software)
        watchdog = pinned(direct_dir, (receipt["files"]["watchdog"]["path"],
                                        receipt["files"]["watchdog"]["sha256"]))
    prior_direct = watchdog["original_reservation"]["reservation"]
    if (watchdog["workflow_run_id"] != 41938284734
            or watchdog["protocol_sha256"] != DIRECT_PROTOCOL[1]
            or original_direct["direct_supplement"]["part"]["sha256"]
               != "6eccb4a5ebccaca5db2daa78dc8c08f5e76b616c4d3bfbcb1a80138b95288c7c"
            or journal["reservations"].count(prior_direct) != 1):
        raise ValueError("Exact successful Direct ARM qualification required")
    forecast = cost_plan(arm, headroom, outstanding_v112(root, journal), direct=direct)
    return (prior_arm, prior_direct), forecast


def verify_predecessors(root=".", journal=None):
    return _verified_timing(root, journal)[0]


def measured_forecast(root=".", journal=None):
    return _verified_timing(root, journal)[1]


def _witness_metadata(forecast):
    return dict(modern_run=41930272861, direct_run=41938284734,
                modern_archive_sha256=ARM_ARCHIVE[1], direct_archive_sha256=DIRECT_ARCHIVE[1],
                direct_receipt_sha256=DIRECT_RECEIPT[1], cost_forecast=forecast)


def _scope_metadata(protocol, root="."):
    previous = pinned(root, DIRECT_PROTOCOL)
    original = pinned(root, ARM_PROTOCOL)
    cohort = pinned(root, COHORT)
    expected_inputs = _merged(previous["input_sha256"], GIT_INPUTS)
    if (
        protocol.get("purpose") != "matched_development_comparison" or protocol.get("stage_mode") != MODE
        or protocol.get("output") != OUTPUT or protocol.get("comparison_case_count") != COUNT
        or protocol.get("max_calendar_seconds") != RUN_CAP or protocol.get("prepared_only") is not False
        or protocol.get("profiles") != [*original["profiles"], *previous["profiles"]]
        or protocol.get("input_sha256") != expected_inputs
        or set(protocol["source_sha256"]) != previous["source_sha256"].keys() | NEW_SOURCES
        or any(protocol["source_sha256"].get(path) != h for path, h in previous["source_sha256"].items())
        or {k: v for k, v in protocol.items() if k not in MUTABLE | ADDED | {"prepared_only"}}
           != {k: v for k, v in previous.items() if k not in MUTABLE | {"prepared_only"}}
    ):
        raise ValueError("Exact native five-arm profiles, methods, and unchanged primary scope required")
    exact_case_ids(protocol, cohort)
    admission = protocol["baseline_admission"]
    frozen_journal = dict(ledger_sha256=protocol["compute_ledger_sha256"],
                          reservations=admission["initial_reservations"])
    _, forecast = _verified_timing(root, frozen_journal)
    expected_admission = dict(previous["baseline_admission"],
                              initial_reservations=frozen_journal["reservations"],
                              remaining_at_preparation_seconds=forecast["canonical_remaining_seconds"],
                              unreserved_v112_protected_seconds=0)
    if (admission != expected_admission or protocol["qualified_witnesses_v115"] != _witness_metadata(forecast)
            or forecast["largest_measured_budget_fit"] != COUNT
            or forecast["effective_larger_run_cap_seconds"] != RUN_CAP
            or forecast["unreserved_v112_protected_seconds"] != 0):
        raise ValueError("Both replayed timings must admit the exact 96-case roster under the original cap")
    return previous


def prospective_protocol(evidence_dir=None, *, now=None):
    from solar_evaluation_protocol_v74 import lease_fields

    if evidence_dir is not None:
        raise ValueError("Use the two exact already-published evidence archives")
    previous = pinned(".", DIRECT_PROTOCOL)
    original = pinned(".", ARM_PROTOCOL)
    cohort = pinned(".", COHORT)
    journal = _journal(".", previous, None)
    forecast = measured_forecast(".", journal)
    sources = _merged(previous["source_sha256"], {name: sha(file(".", name)) for name in sorted(NEW_SOURCES)})
    admission = dict(previous["baseline_admission"], initial_reservations=journal["reservations"],
                     remaining_at_preparation_seconds=forecast["canonical_remaining_seconds"],
                     unreserved_v112_protected_seconds=0)
    protocol = dict(previous, purpose="matched_development_comparison", stage_mode=MODE, output=OUTPUT,
                    comparison_case_count=COUNT, comparison_case_ids=cohort["subsets"][str(COUNT)]["case_ids"],
                    profiles=[*original["profiles"], *previous["profiles"]], max_calendar_seconds=RUN_CAP,
                    prepared_only=False, source_sha256=sources,
                    input_sha256=_merged(previous["input_sha256"], GIT_INPUTS), baseline_admission=admission,
                    qualified_witnesses_v115=_witness_metadata(forecast))
    protocol.update(lease_fields(RUN_CAP, 3600, created=now))
    _scope_metadata(protocol)
    return protocol


def primary_scope(protocol_path, protocol, at=None, check_assets=True):
    from solar_evaluation_protocol_v74 import qualified_cases, validate_runtime
    from solar_public_transport_v98 import validate_release_inputs
    from solar_remote_relay_v98 import validate_primary_archives
    from solar_weight_transport_v96 import LOCK, validate_declaration

    if str(protocol_path) != PROTOCOL:
        raise ValueError("Exact distinct comparison protocol path required")
    _scope_metadata(protocol)
    validate_runtime(protocol, protocol_path, at=at)
    if check_assets:
        validate_release_inputs(protocol, Path.cwd())
        validate_declaration(protocol, read(LOCK))
        manifest, _, cases = qualified_cases(protocol["manifest"], protocol["geometry_audit"])
        if protocol["case_ids"] != [case["case_id"] for case in cases]:
            raise ValueError("Original complete 384-case eligibility changed")
        for name, digest in _merged(protocol["input_sha256"], protocol["source_sha256"]).items():
            pin(".", name, digest)
        validate_primary_archives(cases, manifest["reference_precompute_receipts"])
    return RUN_CAP


def _authorized(old, auth, protocol, binding, checkout):
    from solar_evaluation_protocol_v74 import validate_lease
    from solar_remote_relay_v98 import charge_nonce

    previous, current = _snapshot(old), _snapshot(auth)
    prefix = protocol["baseline_admission"]["initial_reservations"]
    prior_arm, prior_direct = verify_predecessors(checkout, dict(
        ledger_sha256=protocol["compute_ledger_sha256"], reservations=prefix))
    entry = auth["reservation"]["reservation"]
    _, latest, _ = validate_lease(protocol)
    created, reserved, issued = map(time_utc, (protocol["created_utc"], entry["reserved_utc"], auth["created_utc"]))
    if (
        current["ledger_sha256"] != previous["ledger_sha256"] or current["ledger_sha256"] != protocol["compute_ledger_sha256"]
        or current["reservations"] != [*prefix, entry]
        or prefix[:len(previous["reservations"])] != previous["reservations"]
        or prefix.count(prior_arm) != 1 or prefix.count(prior_direct) != 1
        or entry["job_id"] in {row["job_id"] for row in prefix}
        or entry["job_id"] != binding["protocol_sha256"] + "/evaluation"
        or (entry["category"], entry["seconds"], entry["refund_on_failure"]) != ("baseline", RUN_CAP, False)
        or auth["nonce"] != binding["nonce"] or auth["nonce"] == old["nonce"]
        or auth["nonce"] != charge_nonce(binding["protocol_sha256"], entry)
        or auth["protocol_path"] != PROTOCOL or auth["protocol_sha256"] != binding["protocol_sha256"]
        or auth["source_publication_commit"] != binding["source_publication_commit"]
        or auth["case_ids"] != protocol["case_ids"] or auth["tracks"] != protocol["tracks"]
        or auth["original_latest_start_utc"] != protocol["latest_start_utc"]
        or auth["original_absolute_deadline_utc"] != protocol["absolute_deadline_utc"]
        or auth["max_calendar_seconds"] != RUN_CAP or auth["authorized_attempt"] != 1
        or auth["refunds"] is not False or auth["automatic_retry"] is not False
        or auth["reservation"]["remaining_after_reservation"]
           != protocol["baseline_admission"]["remaining_at_preparation_seconds"] - RUN_CAP
        or not time_utc(prior_direct["reserved_utc"]) < created <= reserved <= issued <= latest
    ):
        raise ValueError("Fresh one-use charge must extend the exact two completed ARM qualifications")


def _public_file(checkout, commit, name, digest):
    path = pin(checkout, name, digest)
    blob = subprocess.check_output(["git", "-C", str(checkout), "show", f"{commit}:{name}"], timeout=10)
    if hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("Published comparison file differs: " + name)
    return path


def plan_overlay(hydrated, checkout, metadata, binding, supplement_root=None):
    root, checkout, metadata = map(Path, (hydrated, checkout, metadata))
    extra, _ = inventory(checkout)
    base_files = root / "payload"
    old_manifest = read(pin(root, "manifest.json", BASE["manifest_sha256"]))
    old_protocol = read(pin(base_files, BASE["protocol_path"], BASE["protocol_sha256"]))
    old_auth = read(pin(base_files, BASE["authorization_path"], BASE["authorization_sha256"]))
    commit = binding["source_publication_commit"]
    protocol_path = _public_file(checkout, commit, PROTOCOL, binding["protocol_sha256"])
    _public_file(checkout, commit, *DIRECT_HISTORICAL_PROTOCOL)
    auth_path = pin(metadata, AUTHORIZATION, binding["authorization_sha256"])
    protocol, auth = read(protocol_path), read(auth_path)
    _scope_metadata(protocol, checkout)
    if binding["base"] != BASE or binding["direct_supplement"] != protocol["direct_supplement"]:
        raise ValueError("Signed base and Direct supplement identities changed")
    validate_binding(protocol["direct_supplement"]["part"], protocol["direct_supplement"]["manifest_sha256"])
    _authorized(old_auth, auth, protocol, binding, checkout)
    old_members = _merged(old_protocol["source_sha256"], old_protocol["input_sha256"])
    if old_manifest["files"] != old_members | {
        BASE["protocol_path"]: BASE["protocol_sha256"], BASE["authorization_path"]: BASE["authorization_sha256"],
    }:
        raise ValueError("Original hydrated base closure changed")
    sources = {name: _public_file(checkout, commit, name, digest)
               for name, digest in protocol["source_sha256"].items()}
    inputs = {}
    for name, digest in protocol["input_sha256"].items():
        if name in old_members and old_members[name] == digest:
            inputs[name] = file(base_files, name)
        elif name in extra and extra[name] == digest:
            inputs[name] = pin(supplement_root or checkout, name, digest)
        elif GIT_INPUTS.get(name) == digest:
            inputs[name] = _public_file(checkout, commit, name, digest)
        else:
            raise ValueError("Unreviewed comparison input or changed inherited hash: " + name)
    files = _merged(protocol["input_sha256"], protocol["source_sha256"]) | {
        PROTOCOL: binding["protocol_sha256"], AUTHORIZATION: binding["authorization_sha256"],
    }
    paths = inputs | sources | {PROTOCOL: protocol_path, AUTHORIZATION: auth_path}
    if set(files) != set(paths) or len(files) > capsule.MAX_FILES:
        raise ValueError("Exact complete five-arm closure required")
    total = 0
    for name, path in paths.items():
        capsule.allowed(name, "source" if name in sources else "input")
        size = path.stat().st_size
        if size > min(capsule.MAX_MEMBER_BYTES, old_manifest["max_member_bytes"]):
            raise ValueError("Original capsule member cap exceeded")
        total += size
    if total > min(capsule.MAX_TOTAL_BYTES, old_manifest["max_total_bytes"]):
        raise ValueError("Combined logical capsule exceeds original 4 GiB cap")
    manifest = dict(old_manifest, files=dict(sorted(files.items())), protocol_path=PROTOCOL,
                    publication_commit=commit, protocol_sha256=binding["protocol_sha256"], total_bytes=total,
                    pre_reservation_authorization={"path": AUTHORIZATION, "sha256": binding["authorization_sha256"]},
                    scope="Immutable historical primary assets and separate signed 96-case ARM comparison.")
    return (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(), paths


def assemble_overlay(hydrated, checkout, metadata, destination, binding, supplement_root):
    manifest, files = plan_overlay(hydrated, checkout, metadata, binding, supplement_root)
    digest = hashlib.sha256(manifest).hexdigest()
    if digest != binding["assembled_manifest_sha256"]:
        raise ValueError("Five-arm overlay differs from signed manifest")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name, source in files.items():
        target = file(root / "payload", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target, follow_symlinks=False)
    _store(root, "manifest.json", manifest)
    return {"manifest_sha256": digest, "member_count": len(files)}
