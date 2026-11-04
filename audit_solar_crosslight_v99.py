"""Read the exact v99 dispatch, sealed progress and terminal mechanism counts."""

import argparse
from collections import Counter, defaultdict
from datetime import timedelta
import json
from pathlib import Path
from statistics import median

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, fixed_v86_cases, read, sha, utc, utc_now, validate_dispatch,
)
from run_solar_full_crosslight_rescue_v99 import DIRECT, FULL_CEILING, PROFILES, completion, verify_replayed
from run_solar_matched_direct_render_v91 import budget, relative
from solar_evaluation_v70 import digest


DISPATCH = "reports/solar-full-crosslight-rescue-dispatch-v99.json"
DISPATCH_SHA256 = "5ca6a4a5de7c7dcd3987db92007d7485074485bb746ebcecf5efef8b08612817"
AUDIT_FIELDS = ("component", "query_component", "unit_index", "query_unit_index", "light_a", "light_b", "positive")


def guard():
    if sha(DISPATCH) != DISPATCH_SHA256:
        raise ValueError("Only the exact published v99 development dispatch is admitted")
    p = read(DISPATCH)
    _, manifest, cases = fixed_v86_cases(p["basis"], full=True)
    if (p["split"] != "model_selection" or p["calibration_or_final_access"] is not False or p["accepted_results"]
            or p["cpu_threads"] != 2 or p["tracks"] != ["estimated_sun"] or p["full_case_count"] != 384
            or len(manifest["terrain_groups"]) != 8 or sum(case["positive"] for case in cases) != 192
            or p["case_ids"] != [case["case_id"] for case in cases]
            or tuple(profile["name"] for profile in p["profiles"]) != PROFILES):
        raise ValueError("Exact model-selection-only case bank and fresh method grid required")
    root = relative(p["output"])
    if not root.is_relative_to("runs/solar-full-crosslight-rescue-v99"):
        raise ValueError("Only the frozen development run directory is allowed")
    execution = read(root / "execution.json")
    charge = read(root / "reservation.json")["reservation"]
    start = utc(execution["started_utc"])
    deadline = validate_dispatch(p, start, FULL_CEILING)
    entries, remaining = budget(p)
    if (execution["protocol_sha256"] != DISPATCH_SHA256 or utc(execution["hard_deadline_utc"]) != deadline
            or utc(execution["work_deadline_utc"]) != deadline - timedelta(seconds=5)
            or entries.count(charge) != 1 or charge["job_id"] != DISPATCH_SHA256 + JOB_SUFFIX
            or charge["category"] != "baseline" or charge["seconds"] != p["max_calendar_seconds"]
            or charge["refund_on_failure"] or not start <= utc(charge["reserved_utc"]) <= deadline):
        raise ValueError("One matching original deadline and canonical charge required")
    check_publication(DISPATCH, p["source_sha256"], p["metadata_sha256"], execution["publication_commit"], lambda: None)
    return p, cases, dict(protocol_sha256=DISPATCH_SHA256, publication_commit=execution["publication_commit"],
                          controller_pid=execution["controller_pid"], hard_deadline_utc=deadline.isoformat(),
                          within_hard_deadline=utc_now() < deadline, charged_baseline_seconds=charge["seconds"],
                          canonical_remaining_baseline_seconds=remaining, calibration_or_final_cases=0,
                          split=p["split"], tracks=p["tracks"], terrain_groups=len(manifest["terrain_groups"]),
                          observed_pairs_per_arm=len(cases), physical_images_or_checkpoints_opened=False)


def progress(p, cases):
    """Audit the immutable completed prefix only, without reading a partial cell."""
    root = Path(p["output"])
    paths = set(root.glob("*.COMPLETE.json"))
    keys = [(profile["name"], case, digest(dict(case=case, profile=profile, track="estimated_sun", sources=p["source_sha256"])))
            for profile in p["profiles"] for case in cases]
    prefix = Counter()
    seen_gap = False
    for profile, case, key in keys:
        marker = root / (key + ".COMPLETE.json")
        if marker not in paths:
            seen_gap = True
            continue
        if seen_gap:
            raise ValueError("Frozen arm-major progression has a gap")
        receipt = read(marker)
        row_path, array_path = root / (key + ".json"), root / (key + ".npz")
        if (not receipt["complete"] or receipt["identity"] != key
                or receipt["row_path"] != str(row_path) or receipt["arrays_path"] != str(array_path)
                or receipt["row_sha256"] != sha(row_path) or receipt["arrays_sha256"] != sha(array_path)):
            raise ValueError("Completed cell no longer agrees with its marker")
        row = read(row_path)
        if (row["case_id"] != case["case_id"] or row["profile"] != profile or row["track"] != "estimated_sun"
                or any(row[field] != case[field] for field in AUDIT_FIELDS)
                or row["saved_arrays_sha256"] != receipt["arrays_sha256"]):
            raise ValueError("Sealed cell changed its frozen case endpoints")
        if row["inference_failure"] is None:
            search = row["render_search"]
            if (search["library_entries_considered"] != 24 or search["exact_known_sun"]
                    or len(search["observed_pose_calls"]) != 9 or not search["reference_viewpoint_only"]):
                raise ValueError("Successful sealed cell differs from the fixed library and nine-call budget")
        prefix[profile] += 1
    if len(paths) != sum(prefix.values()):
        raise ValueError("Unexpected completed-cell marker outside frozen grid")
    watchdog_path = root / "watchdog.json"
    watchdog = read(watchdog_path) if watchdog_path.is_file() else None
    return dict(sealed_snapshot_cells={profile: prefix[profile] for profile in PROFILES},
                sealed_snapshot_total=sum(prefix.values()), expected_total=len(keys),
                worker_pid=int((root / "WORKER_STARTED").read_text()) if (root / "WORKER_STARTED").is_file() else None,
                watchdog_terminal=None if watchdog is None else dict(complete=watchdog["complete"],
                    failure=watchdog["failure"], child_returncode=watchdog["child_returncode"]),
                final_seal_present=(root / "COMPLETE.json").is_file(),
                accuracy_from_partial_prefix_reported=False)


def stage(direct, rescue):
    if direct["inference_failure"] is not None or rescue["inference_failure"] is not None:
        return "backend_failure"
    if direct["metrics"]["pose_returned"]:
        return "direct_pose_already_present"
    trace = rescue["render_search"]["crosslight_pooling"]
    if not trace["eligible"]:
        original = direct["render_search"]["observed_pose_calls"]
        if original[7]["pairs"] >= 6:
            return "miss_original_eighth_has_six_raw_pairs"
        if any(call["pose_returned"] for call in original[:7]):
            return "miss_original_candidate_pose_not_reproduced"
        raise ValueError("Skipped pool has no original eligibility exclusion")
    evidence = trace["evidence"]
    if evidence["distinct_pairs"] < 6:
        return "miss_pool_fewer_than_six_unique_pairs"
    if len(evidence["contributing_reference_light_indices"]) < 2:
        return "miss_pool_one_reference_appearance"
    if not trace["attempted"]:
        raise ValueError("Sufficient independent pool evidence was never tested")
    if not trace["selected_pooled"]:
        return "miss_pooled_solver_no_pose"
    if rescue["metrics"]["correct"]:
        return "new_correct_pose"
    return "new_wrong_pose" if direct["positive"] else "new_nonoverlap_pose"


def mechanism(path, p, cases):
    relative(path)
    summary, _, seal, rows, _ = completion(DISPATCH, p)
    independent = verify_replayed(path, DISPATCH, summary, seal, rows, two_arms=True)
    cells = list(zip(rows[:len(cases)], rows[len(cases):], independent["paired_mechanism_trace"], cases, strict=True))
    details = []
    for direct, rescue, verified, case in cells:
        if any(row["case_id"] != case["case_id"] for row in (direct, rescue, verified)):
            raise ValueError("Terminal mechanism case IDs differ")
        label = stage(direct, rescue)
        if label == "backend_failure":
            evidence = None
        else:
            audit = rescue["render_search"]["crosslight_pooling"]
            if verified["verified"] is None or any(verified["verified"][k] != audit[k] for k in ("eligible", "attempted", "selected_pooled")):
                raise ValueError("Original paired trace and independent replay disagree")
            evidence = audit.get("evidence")
        if evidence is None:
            pooling = None
        else:
            count = max(audit["original_match_counts"].values())
            pooling = dict(raw_pairs=evidence["raw_pairs"], distinct_pairs=evidence["distinct_pairs"],
                           duplicate_pairs_removed=evidence["duplicate_pairs_removed"],
                           conflicting_endpoint_pairs_removed=evidence["conflicting_endpoint_pairs_removed"],
                           added_distinct_over_largest_template=evidence["distinct_pairs"] - count,
                           reference_lights=len(evidence["contributing_reference_light_indices"]))
        details.append(dict(case_id=case["case_id"], stage=label, positive=case["positive"], component=case["component"],
                            query_component=case["query_component"], changed_light=case["light_a"] != case["light_b"],
                            pooling=pooling))
    groups = defaultdict(list)
    for item in details:
        group = "positive" if item["positive"] else "nonoverlap"
        lighting = "changed" if item["changed_light"] else "same"
        groups[group].append(item)
        groups[group + "_" + lighting].append(item)
        if item["positive"]:
            groups["positive_terrain_" + item["component"]].append(item)
    strata = {}
    for group, items in sorted(groups.items()):
        pools = [item["pooling"] for item in items if item["pooling"] is not None]
        fields = ("distinct_pairs", "added_distinct_over_largest_template", "duplicate_pairs_removed",
                  "conflicting_endpoint_pairs_removed", "reference_lights")
        strata[group] = dict(cases=len(items), pool_evidence_cases=len(pools),
                            stages=dict(sorted(Counter(item["stage"] for item in items).items())),
                            median_pool=None if not pools else {field: median(pool[field] for pool in pools) for field in fields})
    return dict(exact_terminal_replay_sha256=sha(path), strata=strata, cases=details,
                claim_scope="Descriptive mechanism accounting on previously observed development; excludes no failed cases.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", help="After completion, use the original independent v99 replay path")
    args = parser.parse_args()
    p, cases, audit = guard()
    audit["progress"] = progress(p, cases)
    if args.replay:
        audit["mechanism"] = mechanism(args.replay, p, cases)
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
