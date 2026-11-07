from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from summarize_solar_five_arm_v115 import (
    CASE_COUNT, CASE_FIELDS, COHORT, NAMES, OUTCOMES, PURPOSE, REPLAY_SOURCE, RUN_CAP, STAGE, TRACK,
    _digest_json, analyze, cohort_cases, recover_rows, validate_evidence,
)


@pytest.fixture
def experiment():
    groups = list("abcdefgh")
    lighting = [(0, 0), (0, 17), (8, 1), (8, 17), (16, 1), (16, 16)]
    cases = []
    for i, g in enumerate(groups):
        successor = groups[(i + 1) % len(groups)]
        for j, (a, b) in enumerate(lighting):
            case = dict(case_id=f"{g}-{j}-positive", split="model_selection", component=g, query_component=g,
                        positive=True, unit_index=2*i+j//3, query_unit_index=2*i+j//3, light_a=a, light_b=b)
            cases.extend([case, dict(case, case_id=f"{g}-{j}-negative", positive=False, query_component=successor,
                                     query_unit_index=2*((i + 1) % len(groups))+j//3,
                                     matched_positive_case=case["case_id"])])
    ids = [case["case_id"] for case in cases]
    manifest = dict(split="model_selection", calibration_or_final_access=False, terrain_groups=groups,
                    cases=cases + [dict(case_id=f"excluded-{i}") for i in range(384 - len(cases))])
    cohort = dict(terrain_groups=groups, manifest="manifest.json", geometry_audit="geometry.json",
                  manifest_sha256="manifest-digest", geometry_audit_sha256="geometry-digest",
                  subsets={str(CASE_COUNT): dict(case_ids=ids, positives=48, nonoverlaps=48,
                      positive_lighting=[dict(reference=a, query=b, cases=8) for a, b in lighting],
                      negative_endpoints=[dict(case_id=case["case_id"], reference=case["component"],
                                               query=case["query_component"]) for case in cases if not case["positive"]])})
    profiles = [dict(name=name) for name in NAMES]
    pid = 33
    snapshots = [dict(profile=name, profile_index=i, worker_pid=pid, completed=True, fresh_inference_cases=96,
                      cached_cases=0, before=dict(worker_pid=pid), after=dict(worker_pid=pid))
                 for i, name in enumerate(NAMES)]
    summary = dict(purpose=PURPOSE, complete=True, cached_cases=0, fresh_inference_cases=480,
                   accepted_results=False, calibration_or_final_access=False, automatic_model_promotion=False,
                   scientific_model_evidence=True, case_ids=ids, profiles=profiles, rows=[],
                   protocol_sha256="protocol-digest", manifest_sha256="manifest-digest",
                   source_publication_commit="source-commit", calendar_seconds=900,
                   software=dict(platform="Linux-test", machine="aarch64", cpu_threads=2, torch="2.8.0+cpu"),
                   worker_peak_rss=dict(worker_pid=pid, peak_rss_bytes=123),
                   memory_measurement=dict(worker_pid=pid, profile_order=list(NAMES), profile_snapshots=snapshots,
                                           job_scope="shared_multi_profile_worker_process",
                                           per_model_memory_ranking_supported=False))
    replay = dict(passed=True, replayed_rows=480, rows=[], summary_sha256="summary-digest",
                  replay_source_sha256="replay-digest")
    for name in NAMES:
        for case in cases:
            positive = case["positive"]
            correct = name == NAMES[0] and positive and case["component"] == "a"
            wrong = name == NAMES[0] and case["case_id"] == "b-0-positive"
            bad_negative = name == NAMES[0] and case["case_id"] == "a-0-negative"
            failed = name == NAMES[0] and case["case_id"] == "b-1-positive"
            metrics = dict(positive=positive, correct=correct, strict=correct, pose_returned=correct or wrong or bad_negative,
                           wrong_positive=wrong, raw_nonoverlap=bad_negative)
            key = dict(profile=name, case_id=case["case_id"], track=TRACK)
            summary["rows"].append(dict(key, **{k: case[k] for k in CASE_FIELDS}, metrics=metrics,
                                        reused_case_cache=False, initialization_only_cost_control=False,
                                        inference_failure=dict(status="timeout") if failed else None,
                                        inference_seconds=1, complete_localization_cold_seconds=2))
            replay["rows"].append(dict(key, **{k: metrics[k] for k in OUTCOMES}))
    return cohort, manifest, summary, replay


def test_complete_paired_arithmetic_and_both_negative_endpoints(experiment):
    cohort, manifest, summary, replay = experiment
    cases, groups = cohort_cases(cohort, manifest)
    report = analyze(summary, replay, cases, groups)
    assert report["method_case_cells"] == 480
    assert len(report["paired"]) == 10
    primary = report["methods"][NAMES[0]]["strata"]["overall"]
    assert (primary["correct"], primary["wrong_positive"], primary["no_positive_pose"]) == (6, 1, 41)
    assert primary["operational_failures"] == {"timeout": 1}
    pair = report["paired"][0]["strata"]["overall"]
    assert pair["complete_without_operational_failures"] is False
    assert pair["positive_correct"]["delta_percentage_points"] == 12.5
    assert Counter(pair["positive_correct"]["net_correct_by_terrain"].values()) == {0: 7, 6: 1}
    low, center, high = pair["positive_correct"]["bootstrap_2_5_50_97_5_percentage_points"]
    assert low == 0 and center == 12.5 and high > center
    negative = pair["negative_raw_pose"]
    assert negative["cases"] == 48 and negative["first_only"] == 1
    for omitted, count in (("a", 0), ("b", 0), ("c", 1)):
        reduced = negative["leave_one_terrain_out"][omitted]
        assert (reduced["cases"], reduced["first_only"]) == (36, count)
    changed = report["paired"][0]["strata"]["changed_light_index"]
    assert changed["positive_correct"]["cases"] == 32
    assert changed["negative_raw_pose"]["first_only"] == 0
    assert report == analyze(summary, replay, cases, groups)


@pytest.mark.parametrize("mutation, expected", [
    (lambda s, r: s["rows"].__setitem__(-1, deepcopy(s["rows"][0])), "Complete fresh blind"),
    (lambda s, r: r["rows"][0].__setitem__("correct", False), "independent replay"),
    (lambda s, r: s["rows"][1].__setitem__("query_component", "c"), "frozen endpoint"),
    (lambda s, r: s["rows"][0].__setitem__("reused_case_cache", True), "Fresh outcome"),
    (lambda s, r: s["memory_measurement"]["profile_snapshots"][2].__setitem__("worker_pid", 44), "Shared worker"),
])
def test_rejects_mixed_or_incomplete_evidence(experiment, mutation, expected):
    cohort, manifest, summary, replay = experiment
    mutation(summary, replay)
    cases, groups = cohort_cases(cohort, manifest)
    with pytest.raises(ValueError, match=expected):
        analyze(summary, replay, cases, groups)


def test_rejects_watchdog_overrun_and_cross_run_replay(experiment):
    cohort, manifest, summary, replay = experiment
    protocol = dict(stage_mode=STAGE, purpose=PURPOSE, comparison_case_count=96,
                    comparison_case_ids=summary["case_ids"], case_ids=[c["case_id"] for c in manifest["cases"]],
                    manifest=cohort["manifest"], geometry_audit=cohort["geometry_audit"], tracks=[TRACK], cpu_threads=2,
                    max_calendar_seconds=RUN_CAP, accepted_results=False, calibration_or_final_access=False,
                    case_cache_seals=[], profiles=summary["profiles"], output="runs/comparison",
                    input_sha256={COHORT[0]: COHORT[1], cohort["manifest"]: cohort["manifest_sha256"],
                                  cohort["geometry_audit"]: cohort["geometry_audit_sha256"]},
                    source_sha256={REPLAY_SOURCE: "replay-digest"})
    seal = dict(complete=True, external_watchdog_passed=True, summary_sha256="summary-digest",
                protocol_sha256="protocol-digest", original_allowance_seconds=RUN_CAP, observed_calendar_seconds=950)
    watchdog = dict(complete=True, protocol_sha256="protocol-digest", source_publication_commit="source-commit",
                    run_attempt=1, workflow_run_id=456, cap_seconds=RUN_CAP, child_returncode=0, failure=None,
                    archive_failure=None, primary_only=True, calendar_seconds=1000, automatic_retry=False,
                    reservation_refunded=False, original_reservation={"reservation": dict(
                        job_id="protocol-digest/evaluation", category="baseline", seconds=RUN_CAP, refund_on_failure=False)},
                    result_artifact=dict(scientific_output=protocol["output"]))
    args = (protocol, "protocol-digest", summary, "summary-digest", seal, replay, watchdog, cohort, manifest)
    validate_evidence(*args)
    watchdog["calendar_seconds"] = RUN_CAP + 1
    with pytest.raises(ValueError, match="calendar limits"):
        validate_evidence(*args)
    watchdog["calendar_seconds"] = 1000
    replay["summary_sha256"] = "other-summary"
    with pytest.raises(ValueError, match="same frozen execution"):
        validate_evidence(*args)


def test_partial_recovery_only_counts_matching_seals(experiment, tmp_path):
    cohort, manifest, summary, _ = experiment
    cases, _ = cohort_cases(cohort, manifest)
    protocol = dict(profiles=summary["profiles"], output="runs/comparison", source_sha256={"source": "digest"})
    rows = {r["case_id"]: r for r in summary["rows"] if r["profile"] == NAMES[0]}
    for case_id, arrays in (("b-1-positive", b"sealed timeout"), ("b-2-positive", b"tampered")):
        case = next(c for c in cases if c["case_id"] == case_id)
        identity = _digest_json(dict(case=case, profile=summary["profiles"][0], track=TRACK,
                                     sources=protocol["source_sha256"]))
        arrays_sha = sha256(b"sealed timeout").hexdigest()
        remote = f"{protocol['output']}/{identity}"
        row = dict(rows[case_id], saved_arrays=remote + ".npz", saved_arrays_sha256=arrays_sha)
        row_bytes = json.dumps(row).encode()
        (tmp_path / (identity + ".json")).write_bytes(row_bytes)
        (tmp_path / (identity + ".npz")).write_bytes(arrays)
        (tmp_path / (identity + ".COMPLETE.json")).write_text(json.dumps(dict(
            identity=identity, complete=True, source_protocol_sha256="protocol-digest",
            arrays_sha256=arrays_sha, row_sha256=sha256(row_bytes).hexdigest(),
            row_path=remote + ".json", arrays_path=remote + ".npz")))
    report = recover_rows(tmp_path, protocol, "protocol-digest", cases)
    assert report["complete"] is False and report["expected_cells"] == 480 and report["sealed_cells"] == 1
    assert report["methods"][NAMES[0]]["statuses"] == {"timeout": 1}
    assert len(report["invalid_markers"]) == 1
    assert "b-2-positive" in report["methods"][NAMES[0]]["missing_case_ids"]
    assert "paired" not in report
