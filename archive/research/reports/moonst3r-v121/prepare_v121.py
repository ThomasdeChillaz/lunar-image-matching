"""Build the prospective MOONSt3R roster from immutable development metadata."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIR = "reports/moonst3r-v121"
PROTOCOL = DIR + "/comparison-protocol.json"
RUNNER = DIR + "/run_v121.py"
OUTPUT = "runs/moonst3r-v121/comparison"
V115 = "reports/solar-five-arm-v115/comparison-protocol.json"
V115_SHA = "c43b7a0a5236b27055a7179caf1cfa1a274b18743a9fd868d0054fa96d23878c"
V115_SOURCE_COMMIT = "24046f9c34488fe2efd83464386b0ca8e46906e2"
V115_TICKET_COMMIT = "a95d290d9ecd001b2316e90285ae7bb2bff28c87"
V115_TICKET_SHA = "5394f9ed554357c3c5e20c0a4d9d98132c6e3b5a71ac330e15a43970212b7c36"
V115_NONCE = "f4185e74104c0a62d3b14d64e61cc6b3ef622feec4b28c3191ceba871855dd11"
V115_VERIFIER_SHA = "d77bfa4a298eb0497f3f72a9b15acf62a78d97917c23ff3fcf51a3747acaeb5b"
COHORT = "reports/solar_matched_arm_cohorts_v103.json"
COHORT_SHA = "3f5754829bcef72d90574975b509bf9354a108531848cb2961c5590d914aa9aa"
OVERLAP = "reports/moonst3r_overlap_disclosure_v73.json"
OVERLAP_SHA = "56709368370e576e4ce2673ac6efd8c87799f05408333bad4f2b1b57b1b7eaf4"
LOCK = "moonst3r_model_lock_v19.json"
CONTROL_DIR = "runs/localization-v19-moonst3r-native-control"
CONTROL_HASHES = {
    "protocol": "0c41ee7de980a0d14f3e863655fd5ed620495644b51c523edc37e9ad6af97c23",
    "summary": "6c983e8305f5e75e68d7212bdd7d806b3878210a9c892d13d6f09b9e95ae8f16",
    "controls": "9170ad5e8b054c72c019a051736cfa15195eda98c641c47e9b1b07c09653912c",
    "provenance": "455df7115e9c6e435a815b6dfcd2992448ff1d1a5e8e67abba1b27a3c9f30290",
}
LIGHTS = ((0, 0), (16, 16), (0, 17), (16, 1))
EXPECTED = (
    "u013-l00-17-positive", "u013-to-u021-l00-17-nonoverlap",
    "u022-l00-00-positive", "u022-to-u026-l00-00-nonoverlap",
    "u024-l16-16-positive", "u024-to-u056-l16-16-nonoverlap",
    "u072-l16-01-positive", "u072-to-u092-l16-01-nonoverlap",
)
CAP = 1800
JOB_SUFFIX = "/moonst3r-v121"
SHA_SOURCE = "ed6ea257475abb8a5235dff3a4d6167214b6f697"
SHA_WEIGHTS = "f339a2c46f4e37ba7ed0ad3b7df51a1cb6aa8664"
SHA_SAFE = "f1f356da8f400d503f2c675623775e08f736bb24066e75f48301a1aff8cb3d5f"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def repo_file(root, name):
    part = Path(name)
    if part.is_absolute() or ".." in part.parts or root / part != (root / part).resolve():
        raise ValueError("Use exact repository-relative nonsymlink paths")
    return root / part


def checked(root, name, wanted):
    raw = repo_file(root, name).read_bytes()
    if sha(raw) != wanted:
        raise ValueError(f"Frozen digest differs for {name}")
    return raw


def select_cases(cohort, parent, manifest):
    by_id = {row["case_id"]: row for row in manifest["cases"]}
    selected_96 = parent["comparison_case_ids"]
    subset_48 = cohort["subsets"]["48"]["case_ids"]
    if (len(by_id) != 384 or len(selected_96) != 96 or len(set(selected_96)) != 96
            or len(subset_48) != 48 or len(set(subset_48)) != 48 or not set(subset_48) <= set(selected_96)
            or set(cohort["pilot_case_ids"]) & set(selected_96)):
        raise ValueError("Original v115 and nested 48-case roster identity required")
    used, keep = set(), set()
    for lighting in LIGHTS:
        positives = [by_id[k] for k in subset_48 if by_id[k]["positive"]
                     and (by_id[k]["light_a"], by_id[k]["light_b"]) == lighting and by_id[k]["component"] not in used]
        positive = min(positives, key=lambda case: sha(f"moonst3r-v121/{case['component']}".encode()))
        negative, = [by_id[k] for k in subset_48 if not by_id[k]["positive"]
                     and by_id[k]["matched_positive_case"] == positive["case_id"]]
        used.add(positive["component"])
        keep.update((positive["case_id"], negative["case_id"]))
    ordered = [by_id[k] for k in selected_96 if k in keep]
    if tuple(case["case_id"] for case in ordered) != EXPECTED or any(c["split"] != "model_selection" for c in ordered):
        raise ValueError("Prospective eight-case roster changed")
    return ordered


def build(root=ROOT):
    root = Path(root).resolve()
    parent = json.loads(checked(root, V115, V115_SHA))
    cohort = json.loads(checked(root, COHORT, COHORT_SHA))
    manifest_path = parent["manifest"]
    manifest_hash = parent["input_sha256"][manifest_path]
    manifest = json.loads(checked(root, manifest_path, manifest_hash))
    cases = select_cases(cohort, parent, manifest)
    control_paths = {key: f"{CONTROL_DIR}/{key}.json" for key in CONTROL_HASHES}
    old_control_protocol = json.loads(checked(root, control_paths["protocol"], CONTROL_HASHES["protocol"]))
    sources = dict(parent["source_sha256"])
    for name, checksum in old_control_protocol["source_sha256"].items():
        if name in sources and sources[name] != checksum:
            raise ValueError(f"Historical control source differs from v115 for {name}")
        sources[name] = checksum
    for name in (RUNNER, DIR + "/prepare_v121.py", DIR + "/test_v121_preparation.py"):
        sources[name] = sha(repo_file(root, name).read_bytes())
    for name, checksum in sorted(sources.items()):
        checked(root, name, checksum)
    lock = json.loads(checked(root, LOCK, sources[LOCK]))
    if (lock["source_revision"], lock["weight_revision"], lock["safe_weight_sha256"]) != (SHA_SOURCE, SHA_WEIGHTS, SHA_SAFE):
        raise ValueError("Pinned MOONSt3R safe checkpoint or author source changed")
    proof = {name: value for key, path in control_paths.items() for name, value in
             ((key, path), (key + "_sha256", CONTROL_HASHES[key]))}
    inputs = {
        V115: V115_SHA, COHORT: COHORT_SHA, OVERLAP: OVERLAP_SHA, manifest_path: manifest_hash,
        **{parent[name]: parent["input_sha256"][parent[name]] for name in ("geometry_audit", "learning_contract", "compute_ledger")},
        **{path: CONTROL_HASHES[key] for key, path in control_paths.items()},
    }
    for case in cases:
        for name, checksum in case["input_sha256"].items():
            if parent["input_sha256"].get(name) != checksum or inputs.get(name, checksum) != checksum:
                raise ValueError("Selected case input pin differs from v115")
            inputs[name] = checksum
    # Only metadata and source bytes are opened above. Case assets, model
    # files and all current workflow outcome artifacts remain unopened.
    return dict(
        schema="moonst3r_same_v115_development_pilot_v121", prepared_only=True, execution_authorized=False,
        purpose="cost_pilot", split="model_selection", tracks=["estimated_sun"], output=OUTPUT,
        cpu_threads=2, max_calendar_seconds=CAP, accepted_results=False, calibration_or_final_access=False,
        automatic_retry=False, accuracy_only=True, rank_or_superiority_claim=False, same_cpu_latency_comparison=False,
        prerequisite_v115=dict(protocol=V115, sha256=V115_SHA, source_commit=V115_SOURCE_COMMIT,
                               ticket_commit=V115_TICKET_COMMIT, ticket_sha256=V115_TICKET_SHA, nonce=V115_NONCE,
                               artifact_verifier_source_sha256=V115_VERIFIER_SHA,
                               run_id=41941904269, run_attempt=1, requires_terminal_success_and_independent_replay=True),
        cohort=dict(path=COHORT, sha256=COHORT_SHA, candidate_subset=48, lighting_order=[list(pair) for pair in LIGHTS],
                    method="Lowest SHA256 of moonst3r-v121/<component> among unused terrains for each listed light pair, with its matching negative, retaining v115 order."),
        manifest=manifest_path, geometry_audit=parent["geometry_audit"], case_ids=parent["case_ids"],
        pilot_case_ids=list(EXPECTED), pilot_case_metadata=[
            {key: c[key] for key in ("case_id", "component", "query_component", "positive", "light_a", "light_b",
                                    "matched_positive_case") if key in c} for c in cases
        ],
        profiles=[dict(name="MOONSt3R-common-native512", method="moonst3r", backend="native_pretrained",
                       author_profile="common_native512_full_field_of_view", repository_root=".", cpu_threads=2,
                       pretrained_evidence=proof)],
        weight_and_source=dict(safe_path="data/models/moonst3r/model.safetensors", safe_sha256=SHA_SAFE,
                               author_checkout="data/models/moonst3r-upstream", source_revision=SHA_SOURCE,
                               official_weight_revision=SHA_WEIGHTS, lock=LOCK, redistribution_authorized=False,
                               training_overlap_disclosure=OVERLAP),
        runtime=dict(machine="x86_64", python="3.12.14", dependencies=lock["dependencies"], interop_threads=1),
        budget=dict(ledger=parent["compute_ledger"], ledger_sha256=parent["compute_ledger_sha256"],
                    learning_contract=parent["learning_contract"], learning_contract_sha256=parent["learning_contract_sha256"],
                    canonical_journal=parent["reservation_journal"], job_suffix=JOB_SUFFIX,
                    accounting="A separate pre-existing no-refund 1,800-second baseline reservation is required. Preparation reserves zero."),
        source_sha256=dict(sorted(sources.items())), input_sha256=dict(sorted(inputs.items())),
    )


def protocol_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = protocol_bytes(build())
    if args.output is None:
        if repo_file(ROOT, PROTOCOL).read_bytes() != payload:
            raise SystemExit("Prepared protocol bytes differ")
    else:
        with args.output.open("xb") as file:
            file.write(payload)
