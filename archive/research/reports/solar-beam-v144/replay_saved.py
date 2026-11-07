"""Public read-only v143 replay. Never imports the matcher or execution authority."""
import json
from analyze_saved import ROOT, RUN, OUT, METHODS, digest, load_module, read, require, summarize


def replay():
    protocol = read(ROOT / "reports/solar-beam-v143/protocol.json")
    for name, expected in {**protocol["sources"], **protocol["inputs"]}.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and digest(path) == expected,
                "Missing or changed frozen dependency: " + name)
    report = read(OUT / "results.json")
    for name, expected in report["saved_evidence_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and digest(path) == expected,
                "Changed saved evidence: " + name)
    for name, expected in report["report_sources_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(OUT) and path.is_file() and digest(path) == expected,
                "Changed report or replay source: " + name)
    require(report["source_protocol_sha256"] == digest(ROOT / "reports/solar-beam-v143/protocol.json"),
            "Frozen protocol changed")
    require(read(RUN / "PREDICTIONS_SEALED.json") == dict(rows=8, sha256=digest(RUN / "predictions.json")),
            "Complete prediction seal required")

    def checked(name):
        require(name in protocol["inputs"] and digest(ROOT / name) == protocol["inputs"][name],
                "Unpinned replay input: " + name)
        return ROOT / name

    verifier = load_module("frozen_independent_replay_v144", "reports/solar-beam-v143/replay_beam_v143.py")
    verification = verifier.verify(protocol, checked, lambda name: read(RUN / name))
    require(verification == read(RUN / "replay.json"), "Independent geometry or branch replay differs")
    require(verification["passed"] and verification["method_case_cells"] == 56
            and verification["historical_control_identities"] == 24,
            "All seven arms and historical identities required")
    recomputed = summarize(read(RUN / "predictions.json"), read(RUN / "scores.json"),
                           read(ROOT / protocol["scoring_manifest"]))
    for key, value in recomputed.items():
        require(value == report[key], "Summary or terrain arithmetic differs: " + key)
    diagnosis = load_module("public_branch_diagnosis_v144", "reports/solar-beam-v144/diagnose_saved.py")
    require(diagnosis.diagnose() == read(OUT / "branch-diagnosis.json"),
            "Post hoc saved-branch diagnosis differs")
    receipt = read(OUT / "completion-receipt.json")
    require(receipt["complete"] is True and receipt["fixed_charge_seconds"] == 1800
            and receipt["source_commit"] == "dbeb77776b330518c7d5a1c02ccf98e02ccd2bdb"
            and receipt["protocol_sha256"] == report["source_protocol_sha256"]
            and receipt["elapsed_seconds"] == report["actual_controller_elapsed_seconds"],
            "Public completion scope differs")
    return dict(passed=True, method_case_cells=56, historical_control_identities=24,
        sources=len(protocol["sources"]), inputs=len(protocol["inputs"]),
        independently_recomputed_geometry=True, branch_retention_and_field_accounting=True,
        summary_and_component_analysis_exact=True, no_matching_or_fitting=True,
        saved_branch_diagnosis_exact=True,
        private_authority_used=False, accepted_pose_study=False,
        methods={method: {key: report["methods"][method][key]
            for key in ("correct", "strict", "wrong_positive", "raw_nonoverlap")} for method in METHODS})


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
