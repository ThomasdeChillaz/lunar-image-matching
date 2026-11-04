"""A fixed eight-case development comparison with native MatchAnything.

The existing v86 cohort, native adapter, common AP3P, scorer and raw trace
reporting stay byte-identical. A successful actual v74 two-thread qualification
is mandatory. This runner never selects a model, opens a reserve or calibrates
acceptance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


BASIS = "reports/solar-preselection-compact-controls-cost-pilot-v86.json"
BASIS_SHA256 = "9e14f9c1b95eb12a796ec0533e8c9e50af292a86895e763912a91f786f6b0912"
CANONICAL_JOURNAL = "runs/solar-compute-reservations-v67.json"
CONTROLLER = "run_solar_distributional_evaluation_v90.py"
METHOD_LOCK = "matchanything_model_lock.json"
METHOD_NAME = "MatchAnything-EfficientLoFTR-native512"
PURPOSE = "matched_matchanything_development_v90"
SETTINGS = {
    "control_protocol", "control_publication_commit", "output",
    "max_calendar_seconds", "publication_dispatch_lease_seconds",
}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit repository-relative path required")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in ("", "."):
        raise ValueError("Repository-relative path required")
    if path.as_posix() != value or not path.resolve().is_relative_to(Path.cwd()):
        raise ValueError("Canonical repository path required")
    return value


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != SETTINGS:
        raise ValueError("Exact settings and a separate original cap required")
    relative_path(settings["control_protocol"])
    if not relative_path(settings["output"]).startswith("runs/"):
        raise ValueError("A versioned output under runs is required")
    commit = settings["control_publication_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Exact successful native-control publication commit required")
    cap = settings["max_calendar_seconds"]
    lease = settings["publication_dispatch_lease_seconds"]
    if type(cap) is not int or not 5 < cap <= 18000:
        raise ValueError("An explicit original calendar cap within the baseline study ceiling is required")
    if type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("An explicit original finite publication lease is required")


def profile_and_evidence(settings):
    from native_pretrained_adapters_v73 import native_profile, validate_native_profile

    control_path = settings["control_protocol"]
    control = read(control_path)
    if (control.get("control_kind"), control.get("method"), control.get("author_profile")) != (
        "legacy_matchanything", "matchanything", "native512"
    ):
        raise ValueError("Only the actual v74 MatchAnything native512 control is admissible")
    if control.get("cpu_threads") != 2 or control.get("reservation_journal") != CANONICAL_JOURNAL:
        raise ValueError("A two-thread control charged to the canonical journal is required")
    control_output = Path(relative_path(control["output"]))
    child_output = Path(relative_path(control["child_output"]))
    if child_output != control_output / "native":
        raise ValueError("Outer control must contain its original native child artifacts")
    dependency_dir = control["dependency_dir"]
    if dependency_dir is not None and not Path(dependency_dir).is_dir():
        raise ValueError("Pinned native-control dependency directory is missing")
    receipt_path = str(control_output / "receipt.json")
    outer = dict(
        protocol=control_path, protocol_sha256=sha(control_path),
        receipt=receipt_path, receipt_sha256=sha(receipt_path),
        publication_commit=settings["control_publication_commit"],
    )
    profile = native_profile(
        METHOD_NAME, "matchanything", "native512", child_output, ".",
        dependency_dir=dependency_dir, external_control=outer,
    )
    # This verifies the successful, source-published, bounded outer receipt and
    # the four unchanged procedural artifacts before any model constructor.
    gate = validate_native_profile(profile)
    evidence = gate["external_control"]
    if not evidence or evidence["passed"] is not True:
        raise ValueError("Actual bounded native control has not qualified")
    metadata = dict(evidence["input_sha256"])
    for name, digest in gate["evidence_sha256"].items():
        metadata[profile["pretrained_evidence"][name]] = digest
    sources = dict(gate["source_sha256"])
    sources.update(evidence["source_sha256"])
    return profile, metadata, sources


def artifact_pins(basis, manifest, cases, metadata):
    lock = read(METHOD_LOCK)
    assets = {lock["safe_weight_path"]: lock["safe_weight_sha256"]}
    for name, record in lock["source_files"].items():
        assets[str(Path(lock["source_directory"]) / name)] = record["sha256"]
    for case in cases:
        for name, digest in case["input_sha256"].items():
            if name in assets and assets[name] != digest:
                raise ValueError("The same asset has conflicting frozen bytes")
            assets[name] = digest
    for name in ("manifest", "geometry_audit", "learning_contract", "compute_ledger"):
        path = basis[name]
        assets[path] = basis["input_sha256"].get(path) or sha(path)
    for path in sorted({case["reference_geometry"] for case in cases}):
        receipt = manifest["reference_precompute_receipts"][path]
        assets[receipt["receipt_path"]] = receipt["sha256"]
    for name, digest in metadata.items():
        if name in assets and assets[name] != digest:
            raise ValueError("Native-control evidence has conflicting frozen bytes")
        assets[name] = digest
    return assets


def identities(settings):
    from run_solar_distributional_evaluation_v90 import fixed_v86_cases
    from solar_evaluation_protocol_v74 import source_closure

    validate_settings(settings)
    if sha(BASIS) != BASIS_SHA256:
        raise ValueError("The exact original v86 pilot cohort must be retained")
    basis, manifest, cases = fixed_v86_cases(BASIS)
    if (len(cases), sum(case["positive"] for case in cases)) != (8, 4):
        raise ValueError("Exact v86 four positive and four nonoverlap cases required")
    profile, metadata, control_sources = profile_and_evidence(settings)
    metadata[BASIS] = BASIS_SHA256
    sources = source_closure(".")
    for name, digest in control_sources.items():
        if name in sources and sources[name] != digest:
            raise ValueError("The actual control differs from the common evaluator sources")
        sources[name] = digest
    for name in (CONTROLLER, Path(__file__).name):
        sources[name] = sha(name)
    return basis, manifest, cases, profile, sources, metadata, artifact_pins(basis, manifest, cases, metadata)


def prepare(settings_path, protocol_path):
    """Derive a blocked freeze from real controls; root owns dates/publication."""
    relative_path(str(protocol_path))
    settings = read(settings_path)
    basis, manifest, cases, profile, sources, metadata, assets = identities(settings)
    if Path(settings["output"]).exists():
        raise FileExistsError("Preserve every original completed or partial evaluation")
    from solar_learning_pipeline_v73 import journal_state

    _, journal, _ = journal_state(basis["compute_ledger"], basis["compute_ledger_sha256"], CANONICAL_JOURNAL)
    protocol = dict(
        prepared_only=True, purpose=PURPOSE, split="model_selection",
        calibration_or_final_access=False, accepted_results=False, automatic_model_promotion=False,
        cpu_threads=2, tracks=["estimated_sun"], settings=settings,
        profiles=[profile], output=settings["output"], max_calendar_seconds=settings["max_calendar_seconds"],
        publication_dispatch_lease_seconds=settings["publication_dispatch_lease_seconds"],
        created_utc=None, latest_start_utc=None, absolute_deadline_utc=None,
        basis=BASIS, manifest=basis["manifest"], geometry_audit=basis["geometry_audit"],
        full_case_count=len(manifest["cases"]), case_ids=[case["case_id"] for case in cases],
        compute_ledger=basis["compute_ledger"], compute_ledger_sha256=basis["compute_ledger_sha256"],
        reservation_journal=CANONICAL_JOURNAL, initial_reservations=journal["reservations"],
        blender_executable=basis.get("blender_executable"), input_sha256=assets,
        source_sha256=sources, metadata_sha256=metadata,
        scope="Exact v86 eight-case raw development pilot. Same native images, geometry and estimator. The full original case bank is not measured by this pilot.",
    )
    with Path(protocol_path).open("x") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return protocol


def admission(protocol_path, commit, started):
    from run_solar_distributional_evaluation_v90 import JOB_SUFFIX, check_publication, validate_dispatch
    from solar_learning_pipeline_v73 import journal_state, load_contract, remaining

    relative_path(str(protocol_path))
    protocol = read(protocol_path)
    settings = protocol["settings"]
    validate_settings(settings)
    if (protocol.get("prepared_only") is not False or protocol.get("purpose") != PURPOSE
            or protocol.get("split") != "model_selection" or protocol.get("calibration_or_final_access") is not False
            or protocol.get("accepted_results") is not False or protocol.get("automatic_model_promotion") is not False
            or protocol.get("cpu_threads") != 2 or protocol.get("tracks") != ["estimated_sun"]
            or protocol.get("reservation_journal") != CANONICAL_JOURNAL):
        raise ValueError("Only the source-frozen two-thread raw development pilot is admissible")
    cap = settings["max_calendar_seconds"]
    if any(protocol[name] != settings[name] for name in ("output", "max_calendar_seconds", "publication_dispatch_lease_seconds")):
        raise ValueError("Original output, deadline or cap differs from the frozen settings")
    hard_deadline = validate_dispatch(protocol, started, 18000)

    def check():
        if datetime.now(timezone.utc) >= hard_deadline:
            raise TimeoutError("Original calendar allowance exhausted during admission")

    check()
    basis, manifest, cases, profile, sources, metadata, assets = identities(settings)
    expected = dict(
        basis=BASIS, manifest=basis["manifest"], geometry_audit=basis["geometry_audit"],
        full_case_count=len(manifest["cases"]), case_ids=[case["case_id"] for case in cases],
        profiles=[profile], compute_ledger=basis["compute_ledger"], compute_ledger_sha256=basis["compute_ledger_sha256"],
        blender_executable=basis.get("blender_executable"), source_sha256=sources, metadata_sha256=metadata, input_sha256=assets,
    )
    for field, value in expected.items():
        if protocol.get(field) != value:
            raise ValueError("Frozen comparison identity differs: " + field)
    check_publication(protocol_path, sources, metadata, commit, check)
    for path, digest in assets.items():
        check()
        if sha(path) != digest:
            raise ValueError("Frozen matcher/case/geometry asset changed: " + path)
    contract = load_contract(basis["learning_contract"])
    _, journal, _ = journal_state(protocol["compute_ledger"], protocol["compute_ledger_sha256"], CANONICAL_JOURNAL)
    initial = protocol["initial_reservations"]
    if journal["reservations"][:len(initial)] != initial:
        raise ValueError("Frozen canonical baseline accounting history changed")
    # A charged worker is checked again by the shared parent/worker guard.
    # Its existing exact charge must never be counted twice at this step.
    job_id = sha(protocol_path) + JOB_SUFFIX
    owned = [entry for entry in journal["reservations"] if entry["job_id"] == job_id]
    if len(owned) > 1 or any(row["category"] != "baseline" or row["seconds"] != cap for row in owned):
        raise ValueError("Existing comparison charge differs from the exact original cap")
    if not owned and cap > remaining(contract)["baseline"]:
        raise ValueError("Original cap exceeds the remaining canonical baseline budget")
    check()
    return protocol, cases, hard_deadline


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--settings", required=True)
    prep.add_argument("--protocol", required=True)
    for action in ("run", "worker"):
        sub = actions.add_parser(action)
        sub.add_argument("--protocol", required=True)
        sub.add_argument("--publication-commit", required=True)
        if action == "run":
            sub.add_argument("--python", default=sys.executable)
        else:
            sub.add_argument("--controller-pid", required=True, type=int)
    args = cli.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    else:
        from run_solar_distributional_evaluation_v90 import guarded_run, guarded_worker

        if args.action == "run":
            guarded_run(args.protocol, args.publication_commit, args.python, admission, Path(__file__).name)
        else:
            from solar_evaluation_v70 import Backend

            guarded_worker(args.protocol, args.publication_commit, args.controller_pid, admission, Backend)


if __name__ == "__main__":
    main()
