"""Describe the frozen v143 study using saved outputs only; no model selection."""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/solar-beam-v143"
METHODS = ("direct", "legacy_no_extra", "legacy_active32", "single_uniform128",
           "beam_uniform128", "beam_geometry128", "beam_active128")
NON_BEAM = METHODS[:4]
EVIDENCE = ("predictions.json", "PREDICTIONS_SEALED.json", "scores.json", "replay.json",
            "timing.json", "timing-runtime.json", "predict-runtime.json", "score-runtime.json",
            "replay-runtime.json")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, relative):
    spec = spec_from_file_location(name, ROOT / relative)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise ValueError(message)


def summarize(raw, scores, labels):
    require(len(raw["rows"]) == len(scores) == len(labels) == 8
            and sum(label["positive"] for label in labels) == 4
            and raw["query_truth_used"] is False and raw["accepted_results"] is False,
            "Complete eight-case truth-free uncalibrated study required")
    intervals = load_module("terrain_intervals_v144", "reports/solar-blind-v142/analyze_saved.py")
    positive = [i for i, label in enumerate(labels) if label["positive"]]
    negative = [i for i, label in enumerate(labels) if not label["positive"]]
    for record, scored, label in zip(raw["rows"], scores, labels, strict=True):
        require(record["id"] == scored["id"] == label["id"], "Case order changed")
        require(tuple(record["result"]["methods"]) == tuple(scored["methods"]) == METHODS,
                "All seven methods in frozen order required")
    output = {}
    for method in METHODS:
        records = [row["methods"][method] for row in scores]
        times = [raw["rows"][i]["result"]["methods"][method]["online_seconds"] for i in positive]
        all_times = [row["result"]["methods"][method]["online_seconds"] for row in raw["rows"]]
        angles = [records[i]["sun_proposal_error_deg"] for i in positive
                  if records[i]["sun_proposal_error_deg"] is not None]
        output[method] = dict(
            **{field: sum(int(row[field]) for row in records)
               for field in ("correct", "strict", "wrong_positive", "raw_nonoverlap")},
            positive_pairs=len(positive), nonoverlap_pairs=len(negative),
            positive_pose_outputs=sum(int(records[i]["pose_returned"]) for i in positive),
            raw_recovery_percent=100 * sum(int(records[i]["correct"]) for i in positive) / len(positive),
            median_positive_online_seconds=median(times), positive_online_seconds=times,
            median_all_case_online_seconds=median(all_times), all_case_online_seconds=all_times,
            native_correspondences={threshold: {
                key: sum(records[i]["correspondence"][threshold][key] for i in positive)
                for key in ("correct", "visible", "returned")}
                for threshold in ("1", "2", "4", "8", "12")},
            sun_proposals=len(angles), median_sun_proposal_error_deg=median(angles) if angles else None)
    comparisons = {}
    for control in METHODS[:-1]:
        comparisons[control] = dict(
            paired_gain_pp=output["beam_active128"]["raw_recovery_percent"]-output[control]["raw_recovery_percent"],
            paired_component_interval=intervals.positive_interval(labels, scores, "beam_active128", control),
            median_positive_runtime_ratio=output["beam_active128"]["median_positive_online_seconds"] /
                                          output[control]["median_positive_online_seconds"])
    best = max(output[method]["correct"] for method in NON_BEAM)
    strongest = [method for method in NON_BEAM if output[method]["correct"] == best]
    return dict(schema="sealed_persistent_beam_summary_v144", methods=output,
        strongest_nonbeam_controls_by_raw_count=strongest,
        strongest_nonbeam_correct=best,
        active_gain_over_strongest_nonbeam_pp=100 * (output["beam_active128"]["correct"]-best) / len(positive),
        active_minus_control=comparisons,
        negative_endpoint_intervals={method: intervals.negative_endpoint_interval(labels, scores, method)
                                     for method in METHODS},
        positive_terrain_components=sorted({labels[i]["component"] for i in positive}),
        case_results=[dict(id=label["id"], source_case=label["source_case"], positive=label["positive"],
            reference_component=label["component"], query_component=label["query_component"],
            methods=scored["methods"],
            retained_branches=len(record["result"]["initialization"]["photometric_hypotheses"]),
            common_effective_native_fields=record["result"]["common_effective_native_fields"],
            shared_reference_candidate_capacity=record["result"]["shared_reference_candidate_capacity"],
            acquisitions={method: {key: record["result"]["methods"][method]["acquisition"].get(key)
                for key in ("attempted_native_fields", "rounds", "retained_branch_ids", "pruning")}
                for method in METHODS[3:]})
            for record, label, scored in zip(raw["rows"], labels, scores, strict=True)],
        accepted_pose_study=False, new_trainable_parameters=0,
        query_sun_input=False, continuous_sun_estimation=False,
        source="Frozen v143 seven-method eight-case persistent camera/light study",
        claim_limits="Exposed development terrain; no calibrated acceptance, fresh test, learned gain or modern-model ranking")


def main():
    protocol = read(ROOT / "reports/solar-beam-v143/protocol.json")
    replay = read(RUN / "replay.json")
    complete = read(RUN / "COMPLETE.json")
    require(read(RUN / "PREDICTIONS_SEALED.json") == dict(rows=8, sha256=digest(RUN / "predictions.json"))
            and complete["complete"] is True
            and complete["commit"] == "dbeb77776b330518c7d5a1c02ccf98e02ccd2bdb"
            and complete["protocol_sha256"] == digest(ROOT / "reports/solar-beam-v143/protocol.json")
            and not (RUN / "FAILED.json").exists()
            and replay["passed"] is True and replay["method_case_cells"] == 56
            and replay["historical_control_identities"] == 24,
            "Successful sealed and independently replayed full study required")
    result = summarize(read(RUN / "predictions.json"), read(RUN / "scores.json"),
                       read(ROOT / protocol["scoring_manifest"]))
    for method in METHODS:
        for key in ("correct", "strict", "wrong_positive", "raw_nonoverlap"):
            require(result["methods"][method][key] == replay["methods"][method][key], "Summary and replay differ")
    result["saved_evidence_sha256"] = {str((RUN / name).relative_to(ROOT)): digest(RUN / name) for name in EVIDENCE}
    result["source_protocol_sha256"] = digest(ROOT / "reports/solar-beam-v143/protocol.json")
    result["runtime"] = {operation: read(RUN / (operation + "-runtime.json"))
                         for operation in ("timing", "predict", "score", "replay")}
    result["timing"] = read(RUN / "timing.json")
    result["actual_controller_elapsed_seconds"] = complete["elapsed_seconds"]
    result["fixed_comparison_charge_seconds"] = 1800
    result["report_sources_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.glob("*.py"))}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    receipt = dict(schema="sanitized_persistent_beam_completion_v144", complete=True,
        source_commit=complete["commit"], protocol_sha256=result["source_protocol_sha256"],
        elapsed_seconds=complete["elapsed_seconds"], fixed_charge_seconds=1800,
        original_accounting_unchanged=True, prior_extra_fixed_charges_seconds=3300, refunds=False,
        stages=["timing", "predict", "score", "replay"],
        source_files=len(protocol["sources"]), input_files=len(protocol["inputs"]),
        method_case_cells=56, historical_control_identities=24,
        query_truth_used_in_inference=False, known_query_sun=False,
        accepted_pose_study=False, calibration_or_final_access=False,
        new_trainable_parameters=0, paid_compute=False,
        private_approval_and_execution_records_excluded=True)
    (OUT / "completion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({name: {key: row[key] for key in
        ("correct", "strict", "wrong_positive", "raw_nonoverlap", "median_positive_online_seconds")}
        for name, row in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
