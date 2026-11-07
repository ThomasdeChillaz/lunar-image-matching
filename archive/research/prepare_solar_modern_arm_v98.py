"""Artifact-driven preparation of the four-profile, eight-case ARM v98 pilot."""
import argparse
from pathlib import Path
from time import perf_counter

from run_solar_distributional_evaluation_v90 import fixed_v86_cases, read, sha, write
from solar_evaluation_protocol_v74 import lease_fields, source_closure
from solar_learning_pipeline_v73 import journal_state, load_contract, remaining

BASIS = "reports/solar-preselection-compact-controls-cost-pilot-v86.json"
BASIS_SHA256 = "9e14f9c1b95eb12a796ec0533e8c9e50af292a86895e763912a91f786f6b0912"
CACHE = "data/models/modern-v2-torch-cache"
DINO = CACHE + "/hub/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
LOCK = "reports/compact_baseline_lock_v2.json"
EVIDENCE = "reports/solar-modern-arm-v98/archived-native-controls"
SOURCES = {
    "prepare_solar_modern_arm_v98.py", "solar_remote_relay_v98.py", "run_solar_evaluation_v98.py",
    ".github/workflows/solar-remote-cpu-v98.yml", "modern_baselines_v2.py", "cpu_budget_v2.py",
    "solar_capsule_v75.py", "solar_portable_evidence_v75.py", "solar_aligned_scoring_v91.py",
    "run_solar_distributional_evaluation_v90.py", "verify_solar_evaluation_v98.py",
    "solar_weight_transport_v96.py", "solar_public_transport_v98.py",
}
METHODS = {
    "loma_r": dict(source="data/models/loma-upstream", internal=504, source_artifact=13065656627,
                   run_id=41259046837, zip_sha256="7b0e346e24ad68ddd198bdf47ad8dd4f745becebd9377ee97e6116f096b70b1d"),
    "romav2": dict(source="data/models/romav2-upstream", internal=512, source_artifact=13065053315,
                   run_id=41259085058, zip_sha256="24ec9adcea89a2f145cf40439fc0565f19123350ef37ff7585cddca6220ffeaf"),
}
ARCHIVED = {
    "loma_r": {'pretrained_control': '7e4ba6f205fc1f1ddd4cd2e355c3fc3bf9db573cdff579538a5c145018350c40', 'pretrained_provenance': 'a7b8bad4cabe2aaab7bf72303fc45d229c861435e5aa7818f1ea3ae6fee05aaf', 'hardware_control': '9daa0c11f0d019813775c7efc9c1b2961fd86887741b95066be7d3067cf4c97d'},
    "romav2": {'pretrained_control': '2b0cc282823b945ed2eb51d4e77ad16a3c4be7b6d21a9f1e527b954e592b3a40', 'pretrained_provenance': '126045ff88ec7ce1e76c4e9a4c9a7bbb7fdcccde002efd4d0802d1b4c7407b54', 'hardware_control': '9daa0c11f0d019813775c7efc9c1b2961fd86887741b95066be7d3067cf4c97d'},
}


def merge(destination, incoming):
    for path, checksum in incoming.items():
        if path in destination and destination[path] != checksum:
            raise ValueError("Conflicting input evidence for " + path)
        destination[path] = checksum


def check_worker_runtime():
    """Verify the actual unchanged supported CPU stratum before either model."""
    from importlib.metadata import version
    import os
    import platform
    import torch
    from cpu_budget_v2 import inspect_cpu

    if (platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64")
            or torch.__version__ != "2.8.0+cpu" or version("torchvision") != "0.23.0"
            or torch.get_num_threads() != 2 or not inspect_cpu()["native_bf16"]
            or Path(os.environ.get("TORCH_HOME", "")).resolve() != Path(CACHE).resolve()):
        raise ValueError("Preserved two-thread ARM Torch2.8 native-BF16 runtime required")


def fresh_pretrained_control(profile, model):
    """Run the original known-transform and blank/mask gates on this worker."""
    from native_baseline_pose_v7 import controls

    archived = read(profile["pretrained_provenance"])
    actual = model.provenance
    keys = ("method", "profile", "revision", "device", "torch", "opencv", "common_input_size",
            "coordinate_convention", "settings", "tree_sha256")
    if any(archived[key] != actual[key] for key in keys):
        raise ValueError("Actual ARM matcher semantics differ from qualified original")
    expected_weights = read(LOCK)["modern"][profile["method"]]["weights"]
    if any(actual["weights"].get(name) != checksum for name, checksum in expected_weights.items()):
        raise ValueError("Actual ARM official weights differ")
    tick = perf_counter()
    outcome = controls(model)
    if not outcome["passed"]:
        raise ValueError("Fresh actual ARM known-transform/invalid-input controls failed")
    return dict(outcome, actual_control_calendar_seconds=perf_counter() - tick, provenance=actual)


def modern_profile(method, inputs, selected_weights):
    spec = METHODS[method]
    profile = dict(
        name=method + "_common", backend="modern", method=method, author_profile="common",
        source=spec["source"],
        pretrained_control=f"{EVIDENCE}/{method}/controls.json",
        pretrained_provenance=f"{EVIDENCE}/{method}/provenance.json",
        hardware_control=f"{EVIDENCE}/{method}/native-hardware.json",
        archived_native_control=dict(artifact_id=spec["source_artifact"], run_id=spec["run_id"],
                                     zip_sha256=spec["zip_sha256"]),
    )
    for field, checksum in ARCHIVED[method].items():
        path = profile[field]
        if sha(path) != checksum:
            raise ValueError("Original downloaded archive control changed: " + path)
        inputs[path] = checksum
    controls, provenance, hardware = (read(profile[field]) for field in (
        "pretrained_control", "pretrained_provenance", "hardware_control",
    ))
    lock = read(LOCK)["modern"][method]
    if (not controls["passed"] or not hardware["native_bf16"] or hardware["machine"] != "aarch64"
            or provenance["method"] != method or provenance["profile"] != "common"
            or provenance["device"] != "cpu" or provenance["torch"] != "2.8.0+cpu"
            or provenance["revision"] != lock["revision"] or provenance["settings"] != lock["settings"]
            or provenance["common_input_size"] != spec["internal"]):
        raise ValueError("Exact previously measured ARM common control required")
    for name, checksum in provenance["tree_sha256"].items():
        path = str(Path(spec["source"]) / name)
        if sha(path) != checksum:
            raise ValueError("Stage the exact preserved official source tree: " + path)
        inputs[path] = checksum
    for name, checksum in lock["weights"].items():
        path = CACHE + "/hub/checkpoints/" + name
        if sha(path) != checksum or provenance["weights"].get(name) != checksum:
            raise ValueError("Stage the exact locked official pretrained weight: " + path)
        inputs[path] = selected_weights[path] = checksum
    return profile


def prepare(settings_path, protocol_path):
    from solar_capsule_v75 import MAX_MEMBER_BYTES, MAX_TOTAL_BYTES, allowed, portability_findings
    from solar_evaluation_v70 import validate_profiles, validate_profile_training
    from solar_portable_evidence_v75 import safe
    from solar_remote_relay_v98 import RUNTIME
    from solar_weight_transport_v96 import declaration, validate_declaration
    from solar_public_transport_v98 import PATH, PUBLIC_BLOB, notice_pins, validate_release_inputs

    settings = read(settings_path)
    if set(settings) != {"output", "portable_reference_evidence", "max_calendar_seconds",
                         "maximum_full_calendar_seconds", "publication_dispatch_lease_seconds"}:
        raise ValueError("Declare only portable evidence, unused output, and original limits")
    for path in (settings_path, protocol_path, settings["output"], settings["portable_reference_evidence"]):
        safe(path)
    if not settings["output"].startswith("runs/") or Path(settings["output"]).exists():
        raise ValueError("Use an unused relative run output")
    if sha(BASIS) != BASIS_SHA256:
        raise ValueError("Preserve the exact original v86 eight-case cohort")
    basis, manifest, cases = fixed_v86_cases(BASIS)
    if len(cases) != 8 or sum(case["positive"] for case in cases) != 4:
        raise ValueError("Preserve every original positive and matched nonoverlap")
    p = dict(basis)
    p.update(
        purpose="cost_pilot", target_purpose="modern_comparison", stage_mode="matched_modern_arm_v98",
        settings_path=settings_path, settings_sha256=sha(settings_path), output=settings["output"],
        blender_executable=None, remote_cpu_stratum=RUNTIME, case_cache_seals={},
        scope="Exact eight matched v86 cases on one Linux ARM worker. Two modern common profiles, original frozen compact and SIFT. No selection, acceptance, or pooled timings across strata.",
    )
    compact = next(profile for profile in basis["profiles"] if profile["name"] == "frozen_step0_A1B0C1_quantization_aware")
    sift = next(profile for profile in basis["profiles"] if profile["backend"] == "sift")
    inputs = dict(basis["input_sha256"])
    inputs.pop(basis["settings_path"], None)
    inputs[settings_path] = p["settings_sha256"]
    inputs[LOCK] = sha(LOCK)
    weights = {}
    p["profiles"] = [compact, sift] + [modern_profile(method, inputs, weights) for method in METHODS]
    p["selected_weight_inputs"] = weights
    p["external_weight_transport_v96"] = declaration()
    dino_python = sorted(Path(DINO).rglob("*.py"))
    if len(dino_python) < 100 or not (Path(DINO) / "hubconf.py").is_file():
        raise ValueError("Stage the preserved exact offline DINOv3 source cache")
    for path in dino_python + [Path(DINO) / "LICENSE.md"]:
        inputs[str(path)] = sha(path)
    evidence_path = settings["portable_reference_evidence"]
    evidence = read(evidence_path)
    if (not evidence["complete"] or not evidence["read_only"] or evidence["query_truth_used"]
            or evidence["reference_precompute"] != "full"
            or any(evidence[key] != p["reference_preparation_evidence"][key] for key in ("protocol", "summary"))):
        raise ValueError("Actual portable full-generic physics evidence required")
    merge(inputs, evidence["pinned_inputs"])
    inputs[evidence_path] = sha(evidence_path)
    merge(inputs, notice_pins(Path.cwd()))
    p["external_public_git_blob_v98"] = {PATH: PUBLIC_BLOB}
    history = Path(compact["training_evidence"]["summary"]).parent
    study = read(compact["training_evidence"]["summary"])
    merge(inputs, {str(history / "history_2000.json"): study["all_history_sha256"]})
    # The capsule requires the complete sealed history and all original
    # checkpoint artifacts. Explicitly allowlist these byte-exact weights.
    for name, checksum in study["checkpoint_sha256"].items():
        path = str(history / name)
        inputs[path] = checksum
    seal = history / "COMPLETE.json"
    if seal.is_file():
        inputs[str(seal)] = sha(seal)
    train_protocol = read(compact["training_evidence"]["protocol"])
    merge(inputs, train_protocol["source_sha256"])
    contract = load_contract(p["learning_contract"])
    for key in ("original_master", "dataset_summary", "geometry_audit", "lighting_schedule", "compute_ledger"):
        merge(inputs, {contract[key]: contract[key + "_sha256"]})
    p["input_sha256"] = inputs
    p["source_sha256"] = source_closure(".")
    merge(p["source_sha256"], {path: sha(path) for path in sorted(SOURCES)})
    # These exact historical source roles are required by v75 recursive input
    # validation and by the original learning contract on the ARM worker.
    # The owner pins are also checked against the actual published bytes.
    for owner in (contract, read(evidence["contract"]), train_protocol):
        merge(p["source_sha256"], owner["source_sha256"])
    validate_declaration(p, read(LOCK))
    validate_release_inputs(p, Path.cwd())
    cap, full, lease = (settings[key] for key in (
        "max_calendar_seconds", "maximum_full_calendar_seconds", "publication_dispatch_lease_seconds",
    ))
    if (type(cap) is not int or type(full) is not int or not 0 < cap <= 18000 or not 0 < full <= 18000
            or cap + full > remaining(contract)["baseline"]):
        raise ValueError("Original finite modern limits must fit the canonical remaining allowance")
    _, journal, _ = journal_state(p["compute_ledger"], p["compute_ledger_sha256"], p["reservation_journal"])
    p["baseline_admission"] = dict(
        category="baseline", initial_reservations=journal["reservations"], refund_on_failure=False,
        remaining_at_preparation_seconds=remaining(contract)["baseline"], maximum_full_calendar_seconds=full,
    )
    p["max_calendar_seconds"] = cap
    p.update(lease_fields(cap, lease))
    validate_profiles(p["profiles"], p["purpose"])
    for profile in p["profiles"]:
        validate_profile_training(profile, p["purpose"])
    assets = {**p["input_sha256"], **p["source_sha256"]}
    # Fail before publication and any reservation if a known transfer bound
    # or frozen path rule is already impossible.
    for name, checksum in assets.items():
        allowed(name, "source" if name in p["source_sha256"] else "input")
        if sha(name) != checksum or Path(name).stat().st_size > MAX_MEMBER_BYTES:
            raise ValueError("Portable frozen asset differs or exceeds member limit: " + name)
    total = sum(Path(name).stat().st_size for name in assets)
    if total >= MAX_TOTAL_BYTES:
        raise ValueError("4GiB transport shortfall before any charge: " + str(total))
    findings = portability_findings(Path.cwd(), assets, historical_contracts={evidence["contract"], evidence_path})
    if findings["failures"]:
        raise ValueError("Nonportable frozen paths before any charge: " + str(findings["failures"][:12]))
    write(protocol_path, p)
    return dict(protocol_path=protocol_path, pilot_rows=32, all_bank_cases=len(manifest["cases"]), known_payload_bytes=total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    print(prepare(args.settings, args.protocol))
