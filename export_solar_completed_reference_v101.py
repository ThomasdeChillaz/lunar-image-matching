"""Private v75 export of the exact independently signed completed v80 artifacts.

This verifier is deliberately separate from every live reference launch path.
It cannot authorize a job and does not execute the historical programs.
"""

import argparse
import json
from pathlib import Path
import shutil
from urllib.request import ProxyHandler, Request, build_opener

import solar_reference_pipeline_v74 as original
from solar_learning_pipeline_v73 import CANONICAL_JOURNAL, full_geometry_gate, journal_state, pinned
from solar_portable_evidence_v75 import (
    PHASES, file, full_physics, new_json, portable_completed_reference_preparation,
    publication_proof, read, safe, sha, verify_publication_proof,
)

CONTRACT = "reports/solar_reference_contract_v80.json"
CONTRACT_SHA = "918e4a77630c995c4448d9440b3d09fccaf74fc6cea0d52e245696705321be98"
BASIS = "reports/solar-preselection-compact-controls-cost-pilot-v86.json"
BASIS_SHA = "9e14f9c1b95eb12a796ec0533e8c9e50af292a86895e763912a91f786f6b0912"
BASIS_COMMIT = "61bf3ed690a6d51913ccb7f30b29593fe4df090f"
OLD_EXPORTER_COMMIT = "21025b6787228cd05a051249fc7224525c0218bc"
OLD_EXPORTER_SOURCE = "solar_portable_evidence_v75.py"
COMMITS = {
    "reference_one": "ecf8bd471ba0268093af037d8900bb0f12675da6",
    "reference_four": "f6aabc034f2b6d8c2a055645d985078f1925a06c",
    "reference_full": "b90e2b14308f45c55c8eda6e03dad86febc2d355",
}
REPO_API = "https://api.github.com/repos/ThomasdeChillaz/lunar-image-matching"
EXPECTED_EVIDENCE = "reports/solar-modern-arm-v98/portable-reference/evidence.json"


def require_public_signature(commit):
    if commit not in {BASIS_COMMIT, *COMMITS.values()}:
        raise ValueError("Only the declared historical signed publications are supported")
    url = REPO_API + "/git/commits/" + commit
    opener = build_opener(ProxyHandler({}))
    request = Request(url, headers={"Accept": "application/vnd.github+json",
                                   "User-Agent": "completed-reference-proof-v101"})
    with opener.open(request, timeout=15) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
        if response.status != 200 or response.geturl() != url or len(raw) > 2 * 1024 * 1024:
            raise ValueError("The exact historical public commit response was unavailable")
    commit_row = json.loads(raw)
    signature = commit_row["verification"]
    if commit_row["sha"] != commit or signature["verified"] is not True or signature["reason"] != "valid":
        raise ValueError("A declared historical source commit lacks valid GitHub signature evidence")
    return dict(commit=commit, verified=True, reason="valid")


def completed_contract():
    pinned(CONTRACT, CONTRACT_SHA)
    c = read(CONTRACT)
    if (c.get("prepared_only", False) or c["schema_version"] != 74 or c["contract_path"] != CONTRACT
            or c["calibration_or_final_access"] is not False or c["query_truth_in_primary_inference"] is not False
            or c["reference_precompute"] != "full"):
        raise ValueError("The exact original completed v80 full reference contract is required")
    for name in ("original_master", "dataset_summary", "geometry_audit", "compute_ledger"):
        pinned(c[name], c[name + "_sha256"])
    audit = read(c["geometry_audit"])
    ids, groups, pilots = original.schedule(audit)
    inventory = original.selection_metadata(audit)
    if ((c["selection_units"], c["selection_groups"], c["pilot_units"]) != (ids, groups, pilots)
            or any(c[name] != value for name, value in inventory.items())):
        raise ValueError("Original 96-camera geometry qualification or eligible reference inventory changed")
    for name, digest in c["source_sha256"].items():
        if sha(name) != digest:
            raise ValueError("Frozen original physical source changed: " + name)
    if (c["reservation_journal"] != CANONICAL_JOURNAL
            or c["canonical_journal_absolute"] != str(Path(CANONICAL_JOURNAL).absolute())):
        raise ValueError("The original local append-only journal must provide the snapshot")
    ledger, journal, charges = journal_state(c["compute_ledger"], c["compute_ledger_sha256"],
                                            c["reservation_journal"])
    initial = c["initial_reservations"]
    if (journal["reservations"][:len(initial)] != initial
            or charges["baseline"] > ledger["remaining_baseline_wall_seconds"]):
        raise ValueError("Original canonical journal history or allowance changed")
    for i in ids:
        paths = original.unit_paths(c, i)
        if (read(paths["precompute"]), read(paths["lookup"])) != original.physical_protocols(c, i):
            raise ValueError("The original per-reference physical recipe changed")
    # The physical children launched Blender directly. Preserve its exact
    # original executable check. The old controller Python remains an opaque
    # historical pin in the verified contract and is never executed here.
    if sha(c["blender_executable"]) != c["blender_executable_sha256"]:
        raise ValueError("Frozen original Blender executable changed")
    full_geometry_gate(c["original_master"], c["dataset_summary"], c["geometry_audit"])
    return c, inventory


def verified_inputs(check_signature=require_public_signature):
    root = Path.cwd()
    c, inventory = completed_contract()
    pinned(BASIS, BASIS_SHA)
    anchor_proof = publication_proof(root, [{"commit": BASIS_COMMIT, "paths": {BASIS: BASIS_SHA}}])
    verify_publication_proof(root, anchor_proof)
    signature = [check_signature(commit) for commit in (*COMMITS.values(), BASIS_COMMIT)]
    basis = read(BASIS)
    if basis["split"] != "model_selection" or basis["calibration_or_final_access"] is not False:
        raise ValueError("A signed development-only evaluation anchor is required")
    anchor = basis["input_sha256"]

    pins = {CONTRACT: CONTRACT_SHA}
    claims = []
    phase_seconds = {}
    reserved = 0
    artifact_paths = set()
    for stage in PHASES:
        phase_path, folder = original.stage_paths(c, stage)
        phase = read(phase_path)
        summary = original.completion(c, stage, required=True)
        if summary["publication_commit"] != COMMITS[stage]:
            raise ValueError("Completed phase differs from the known signed source publication")
        phase_files = {str(folder / "summary.json"), str(folder / "COMPLETE.json")}
        for path in (phase_path, *sorted(phase_files)):
            pins[path] = sha(path)
            if anchor.get(path) != pins[path]:
                raise ValueError("Original phase record differs from the independently signed v86 freeze")
        artifact_paths.update(phase_files)
        names = {CONTRACT, phase_path, *c["source_sha256"]}
        for i in c["selection_units"]:
            paths = original.unit_paths(c, i)
            names.update(paths[n] for n in ("precompute", "lookup"))
        names.update(job["protocol"] for job in phase["jobs"])
        claims.append(dict(commit=COMMITS[stage], paths={name: sha(name) for name in sorted(names)}))
        reserved += phase["max_calendar_seconds"]
        phase_seconds[stage] = summary["calendar_seconds"]

    protocol_path, full_folder = original.stage_paths(c, "reference_full")
    summary_path = str(full_folder / "summary.json")
    if basis["reference_preparation_evidence"] != {"protocol": protocol_path, "summary": summary_path}:
        raise ValueError("The independently signed v86 anchor references different physical evidence")
    final_summary = read(summary_path)
    outer, inner = {}, {}
    for row in final_summary["reference_rows"]:
        paths = original.unit_paths(c, row["unit_index"])
        for field in ("archive", "program_receipt", "watchdog_receipt"):
            path, digest = row[field], row[field + "_sha256"]
            if anchor.get(path) != digest:
                raise ValueError("Completed artifact differs from the independently signed v86 freeze")
            pins[path] = digest
            artifact_paths.add(path)
        for field in ("precompute", "lookup"):
            pins[paths[field]] = sha(paths[field])
        outer[row["archive"]] = row["outer_calendar_seconds"]
        inner[row["archive"]] = row["internal_calendar_seconds"]
    if (anchor.get(CONTRACT) != CONTRACT_SHA or len(artifact_paths) != 102
            or len(outer) != len(c["selection_units"])):
        raise ValueError("The signed v86 anchor must cover all 32 original reference archives and their seals")

    original_result = dict(
        pinned_inputs=dict(pins), outer_calendar_seconds_by_archive=outer,
        internal_calendar_seconds_by_archive=inner, reference_archives=sorted(outer),
        reference_unit_indices=sorted(row["unit_index"] for row in final_summary["reference_rows"]),
        baseline_reserved_seconds=reserved, phase_calendar_seconds=phase_seconds,
        scope=("Completed physical reference preparation only. Outer costs include Blender startup and child shutdown; "
               "full phase costs also include serial validation. Historical replay grants no job admission."),
        **inventory,
    )
    for key in ("original_master", "dataset_summary", "geometry_audit", "compute_ledger"):
        pins[c[key]] = c[key + "_sha256"]
    pins.update(c["source_sha256"])
    for row in read(c["dataset_summary"])["units"]:
        for filename, digest_field in (("summary.json", "summary_sha256"), ("COMPLETE.json", "seal_sha256")):
            pins[row["output"] + "/" + filename] = row[digest_field]
    for i in c["selection_units"]:
        pre = read(original.unit_paths(c, i)["precompute"])
        for key in ("reference_geometry", "mesh", "scene_mesh", "geometry_audit"):
            pins[pre[key]] = pre[key + "_sha256"]
    pins[OLD_EXPORTER_SOURCE] = sha(OLD_EXPORTER_SOURCE)
    claims.append(dict(commit=OLD_EXPORTER_COMMIT, paths={OLD_EXPORTER_SOURCE: pins[OLD_EXPORTER_SOURCE]}))
    proof = publication_proof(root, claims)
    verify_publication_proof(root, proof)
    audit = dict(schema_version=101, completed_only=True, live_launch_authority=False,
                 original_python_current_bytes_used=False, preserved_blender_binary_sha256=c["blender_executable_sha256"],
                 signed_anchor=dict(commit=BASIS_COMMIT, path=BASIS, sha256=BASIS_SHA, artifacts=len(artifact_paths)),
                 signed_phase_commits=signature, completed_references=len(outer), baseline_reserved_seconds=reserved)
    return c, protocol_path, summary_path, pins, proof, original_result, audit


def export_verified():
    out = Path(EXPECTED_EVIDENCE)
    ledger_snapshot = out.parent / "ledger-snapshot.json"
    journal_snapshot = out.parent / "journal-snapshot.json"
    snapshots = {ledger_snapshot: "compute_ledger", journal_snapshot: "reservation_journal"}
    if any(path.exists() for path in (out, *snapshots)):
        raise FileExistsError("Preserve previous evidence and snapshots")
    c, protocol, summary, pins, proof, result, audit = verified_inputs()
    root = Path.cwd()
    for index in c["selection_units"]:
        archive = original.unit_paths(c, index)["archive"]
        full_physics(root, archive, c["reference_units"][str(index)]["latitude_deg"])
    out.parent.mkdir(parents=True, exist_ok=True)
    for target, key in snapshots.items():
        with file(root, c[key]).open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
        pins[str(target)] = sha(target)
    evidence = dict(schema_version=75, kind="completed_reference_physics", complete=True, read_only=True,
                    query_truth_used=False, reference_precompute="full", protocol=protocol, summary=summary,
                    contract=CONTRACT, original_master=c["original_master"], dataset_summary=c["dataset_summary"],
                    geometry_audit=c["geometry_audit"], ledger_snapshot=str(ledger_snapshot),
                    journal_snapshot=str(journal_snapshot),
                    historical_executable_identities={
                        name: dict(path=c[name], sha256=c[name + "_sha256"])
                        for name in ("python_executable", "blender_executable")},
                    original_canonical_journal=c["reservation_journal"],
                    original_canonical_absolute_identity=c["canonical_journal_absolute"],
                    pinned_inputs=pins, publication_proof=proof, result=result,
                    scope="Completed signed v80 artifacts only. Historical executable identities are opaque.",
                    completed_only_export_v101=audit)
    new_json(out, evidence)
    verified = portable_completed_reference_preparation(root, safe(str(out)))
    if len(verified["reference_archives"]) != audit["completed_references"]:
        raise ValueError("Unchanged v75 verification disagrees with the signed completed evidence")
    return dict(audit, private_export_verified=True, evidence_sha256=sha(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify-existing", "private-export"))
    args = parser.parse_args()
    result = verified_inputs()[-1] if args.action == "verify-existing" else export_verified()
    print(json.dumps(result, indent=2))
