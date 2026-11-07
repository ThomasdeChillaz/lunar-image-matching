"""Published cost pilot for additive native windows on the eight v91 development cases."""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_common_protocol, check_publication, guarded_run, read, sha,
    utc, utc_now, validate_dispatch, write,
)
from run_solar_full_direct_render_v92 import (
    FULL_CEILING, PILOT, PILOT_REPLAY, PILOT_REPLAY_SHA256, PILOT_SHA256, completion,
)
from run_solar_matched_direct_render_v91 import budget, identities, merge_pins, relative

SOURCE = "run_solar_direct_additive_windows_v97.py"
BACKEND = "solar_direct_additive_windows_v96.py"
SELECTOR = "solar_translation_acquisition_v93.py"
PREDECESSOR = "reports/solar-full-native-baselines-dispatch-v94.json"
PREDECESSOR_SHA256 = "82e51b9a3e325e8eafc3022c7ac613dbce1a2400b71405a32eef4aca3dd8f1bc"
PURPOSE = "direct_additive_windows_development_cost_pilot_v97"
OUTPUT = "runs/solar-direct-additive-windows-v97/pilot"
TRACK = "estimated_sun"
ARMS = ("frozen_direct", "uniform", "translation_information")
WINDOWS = 128
PILOT_CEILING = 360
TIMING = (
    "Separate fresh direct SIFT, pose and optional native windows for each arm. "
    "Per-case calendar includes input verification/decode, matcher, trace serialization, scoring, "
    "common pose replay and NPZ write; the watchdog includes admission and all output writes. "
    "OS file caches and process lifetime RSS are shared. Arm order is frozen."
)


def historical():
    """Bind the independently replayed v91 control without opening an inference backend."""
    if sha(PILOT) != PILOT_SHA256 or sha(PILOT_REPLAY) != PILOT_REPLAY_SHA256:
        raise ValueError("Exact v91 control and its independent replay required")
    prior = read(PILOT)
    expected, cases = identities(prior["settings"])
    if any(prior.get(k) != value for k, value in expected.items()):
        raise ValueError("Original v91 identities differ")
    summary, _, seal, rows, sealed, files = completion(PILOT, prior)
    report = read(PILOT_REPLAY)
    if (report["passed"] is not True or report["replayed_rows"] != 8
            or report["protocol_sha256"] != PILOT_SHA256 or report["accepted_results"] is not False
            or report["summary_sha256"] != seal["summary_sha256"]
            or report["watchdog_sha256"] != seal["watchdog_sha256"]
            or [r["case_id"] for r in report["rows"]] != prior["case_ids"]):
        raise ValueError("Original paired saved replay is incomplete")
    evidence = {path: sha(path) for path in files.values()}
    for row, marker in sealed:
        receipt = read(marker)
        merge_pins(evidence, {
            str(marker): sha(marker), receipt["row_path"]: receipt["row_sha256"],
            row["saved_arrays"]: row["saved_arrays_sha256"],
        })
    # v90's loader remembers one last image pair and one reference geometry.
    # Require neither cache key to survive across the frozen profile boundary.
    for fields in (("reference", "query"), ("reference_geometry", "solar_lookup_protocol")):
        if tuple(cases[0].get(k) for k in fields) == tuple(cases[-1].get(k) for k in fields):
            raise ValueError("Pilot arm boundary could reuse the previous arm's loader state")
    return prior, cases, summary["software"], rows, evidence


def terminal_predecessor():
    """Refuse pilot launch while the currently prescribed v94 stage can still be running."""
    if sha(PREDECESSOR) != PREDECESSOR_SHA256:
        raise ValueError("Original full native baseline stage identity changed")
    protocol = read(PREDECESSOR)
    root = Path(protocol["output"])
    watchdog_path = root / "watchdog.json"
    if not watchdog_path.exists():
        raise ValueError("Wait for the original full native baseline watchdog to finish")
    watchdog = read(watchdog_path)
    charge = read(root / "reservation.json")
    entries, _ = budget(protocol)
    start, end = utc(watchdog["started_utc"]), utc(watchdog["ended_utc"])
    hard = validate_dispatch(protocol, start, FULL_CEILING)
    cap = protocol["max_calendar_seconds"]
    if (watchdog["protocol_sha256"] != PREDECESSOR_SHA256
            or charge != watchdog["compute_reservation"] or charge["reservation"] not in entries
            or charge["reservation"]["job_id"] != PREDECESSOR_SHA256 + JOB_SUFFIX
            or charge["reservation"]["category"] != "baseline" or charge["reservation"]["seconds"] != cap
            or charge["reservation"]["refund_on_failure"] or watchdog["cap_seconds"] != cap
            or utc(watchdog["hard_deadline_utc"]) != hard or not start < end <= utc_now()
            or type(watchdog["child_returncode"]) is not int):
        raise ValueError("Original full native baseline has no verified terminal worker receipt")
    return str(watchdog_path), sha(watchdog_path)


def frozen(settings):
    from control_historical_native_v74 import source_closure

    if not isinstance(settings, dict) or set(settings) != {
        "output", "max_calendar_seconds", "publication_dispatch_lease_seconds",
    }:
        raise ValueError("Explicit pilot output, cap and dispatch lease required")
    if str(relative(settings["output"])) != OUTPUT:
        raise ValueError("Use the fixed fresh v97 cost-pilot output")
    cap, lease = settings["max_calendar_seconds"], settings["publication_dispatch_lease_seconds"]
    if type(cap) is not int or not 5 < cap <= PILOT_CEILING or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Pilot cap must fit within three original 120-second control caps and the original lease ceiling")
    if sha(PREDECESSOR) != PREDECESSOR_SHA256:
        raise ValueError("Original full native baseline stage identity changed")
    prior, cases, software, _, evidence = historical()
    sources, public, assets = (dict(prior[k]) for k in ("source_sha256", "metadata_sha256", "input_sha256"))
    merge_pins(sources, source_closure({SOURCE, BACKEND, SELECTOR, "run_solar_full_direct_render_v92.py"}))
    merge_pins(public, {PILOT: PILOT_SHA256, PILOT_REPLAY: PILOT_REPLAY_SHA256, PREDECESSOR: PREDECESSOR_SHA256})
    merge_pins(assets, public)
    merge_pins(assets, evidence)
    parent, = prior["profiles"]
    profiles = [
        dict(parent, name=policy, backend="direct_additive_windows", method=policy,
             configuration=dict(policy=policy, budget=WINDOWS), direct_profile=parent)
        for policy in ARMS
    ]
    protocol = {k: prior[k] for k in (
        "basis", "manifest", "geometry_audit", "full_case_count", "case_ids", "compute_ledger",
        "compute_ledger_sha256", "reservation_journal", "blender_executable", "scoring_revision",
    )}
    protocol.update(
        **settings, settings=settings, purpose=PURPOSE, profiles=profiles, tracks=[TRACK],
        split="model_selection", cpu_threads=2, calibration_or_final_access=False,
        accepted_results=False, automatic_model_promotion=False, expected_software=software,
        source_sha256=sources, metadata_sha256=public, input_sha256=assets,
        incremental_window_cap=WINDOWS, cache_scope="No cross-arm inference or reference/query decode cache hit.",
        scope="Cost/mechanism pilot only. The v91 cases were previously observed. No final reserve.",
        timing_scope=TIMING,
    )
    return protocol, cases


def prepare(settings_path, path, *, dispatch=False):
    relative(path)
    protocol, _ = frozen(read(settings_path))
    if Path(protocol["output"]).exists():
        raise FileExistsError("Preserve every previous attempt")
    entries, remaining = budget(protocol)
    cap = protocol["max_calendar_seconds"]
    if cap > remaining:
        raise ValueError("Original baseline budget cannot admit the pilot")
    created = latest = absolute = None
    if dispatch:
        terminal_predecessor()
        created = utc_now()
        latest = created + timedelta(seconds=protocol["publication_dispatch_lease_seconds"])
        absolute = latest + timedelta(seconds=cap)
    protocol.update(prepared_only=not dispatch, initial_reservations=entries,
                    created_utc=None if created is None else created.isoformat(),
                    latest_start_utc=None if latest is None else latest.isoformat(),
                    absolute_deadline_utc=None if absolute is None else absolute.isoformat())
    write(path, protocol)


def validate_frozen(protocol):
    expected, cases = frozen(protocol["settings"])
    fields = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (protocol.get("prepared_only") is not False or set(protocol) != set(expected) | fields
            or any(protocol.get(k) != value for k, value in expected.items())):
        raise ValueError("Published pilot differs from the exact frozen identity")
    check_common_protocol(protocol)
    return cases


def admission(path, commit, started):
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    protocol = read(path)
    deadline = validate_dispatch(protocol, started, PILOT_CEILING)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original pilot allowance exhausted during admission")

    check()
    cases = validate_frozen(protocol)
    terminal_predecessor()
    check_publication(path, protocol["source_sha256"], protocol["metadata_sha256"], commit, check)
    for name, checksum in protocol["input_sha256"].items():
        check()
        if sha(name) != checksum:
            raise ValueError("Frozen control, case or render input changed: " + name)
    if software_identity() != protocol["expected_software"]:
        raise ValueError("Use the original control CPU/software stratum")
    entries, remaining = budget(protocol)
    initial = protocol["initial_reservations"]
    cap = protocol["max_calendar_seconds"]
    if entries[:len(initial)] != initial:
        raise ValueError("Original canonical reservation history changed")
    owned = [row for row in entries if row["job_id"] == sha(path) + JOB_SUFFIX]
    if len(owned) > 1 or any(row["category"] != "baseline" or row["seconds"] != cap
                             or row["refund_on_failure"] for row in owned):
        raise ValueError("Existing pilot charge differs from its original cap")
    if not owned and cap > remaining:
        raise ValueError("Pilot cap exceeds the remaining original baseline allowance")
    check()
    return protocol, cases, deadline


def array_fingerprint(encoded, arrays):
    from run_solar_evaluation_v70 import RAW_ARRAY_PREFIX

    digest = hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for key in sorted(k for k in arrays if k.startswith(RAW_ARRAY_PREFIX)):
        array = arrays[key]
        digest.update(repr((key, array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fingerprint(result):
    from run_solar_evaluation_v70 import pack_raw_inference

    arrays = {}
    encoded, metadata = pack_raw_inference(result, arrays)
    if not metadata["encoding_complete"]:
        raise ValueError("Direct seed must be serializable without loss")
    return array_fingerprint(encoded, arrays)


class SeedObserver:
    """Read the same seed call the v96 backend already makes."""
    def __init__(self, backend):
        self.backend = backend
        self.seed = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def infer(self, inp):
        self.seed = self.backend.infer(inp)
        return self.seed


def case_key(protocol, profile, case):
    from solar_evaluation_v70 import digest

    return digest(dict(case=case, profile=profile, track=TRACK, sources=protocol["source_sha256"]))


class TraceBackend:
    """Sidecars are observed before scoring; the inference result is returned unchanged."""
    def __init__(self, output, profile, case_identities):
        from solar_direct_additive_windows_v96 import DirectAdditiveWindowsBackend

        self.profile = profile
        self.backend = DirectAdditiveWindowsBackend(profile)
        self.observer = SeedObserver(self.backend.direct)
        self.backend.direct = self.observer
        self.provenance = self.backend.provenance
        self.remaining = iter(case_identities)
        self.output = Path(output)

    def infer(self, inp):
        case_id, key = next(self.remaining)
        if inp.known_query_sun is not None:
            raise ValueError("Primary inference may not use query Sun truth")
        tick = time.perf_counter()
        self.observer.seed = None
        result, trace = self.backend.infer_with_trace(inp)
        inference_seconds = time.perf_counter() - tick
        seed = self.observer.seed
        if seed is None or (result is seed) != trace["used_original_seed"]:
            raise ValueError("Original frozen direct seed/fallback identity changed")
        tick = time.perf_counter()
        seed_digest = fingerprint(seed)
        observer_seconds = time.perf_counter() - tick
        write(self.output / (key + ".windows.json"), dict(
            key=key, case_id=case_id, profile=self.profile["name"],
            seed_fingerprint=seed_digest, seed_pose_returned=seed["pose"] is not None,
            seed_match_count=len(seed["reference_xy"]), inference_without_sidecar_seconds=inference_seconds,
            seed_observer_seconds=observer_seconds, trace=trace,
        ))
        return result


def traces(protocol, cases, rows):
    root = Path(protocol["output"])
    result = {}
    index = {}
    for row, (profile, case) in zip(rows, ((p, c) for p in protocol["profiles"] for c in cases), strict=True):
        key = case_key(protocol, profile, case)
        if (row["case_id"], row["profile"], row["track"]) != (case["case_id"], profile["name"], TRACK):
            raise ValueError("Pilot case, profile or track order changed")
        path = root / (key + ".windows.json")
        if not path.exists() and row["inference_failure"] is not None:
            continue
        record = read(path)
        if (record["key"], record["case_id"], record["profile"]) != (key, case["case_id"], profile["name"]):
            raise ValueError("Trace has the wrong scientific-row identity")
        result[profile["name"], case["case_id"]] = record
        index[str(path)] = sha(path)
    if set(index) != {str(p) for p in root.glob("*.windows.json")}:
        raise ValueError("Unexpected acquisition sidecar inventory")
    return result, index


def worker(path, commit, pid):
    from run_solar_matched_matchanything_recovery_v91 import guarded_aligned_worker

    protocol = read(path)
    cases = None

    def admit(*args):
        nonlocal cases
        frozen_protocol, cases, deadline = admission(*args)
        return frozen_protocol, cases, deadline

    def factory(profile):
        # Build artifact identities outside inference. The adapter receives
        # only those opaque identifiers, never any case or scoring fields.
        keys = [(c["case_id"], case_key(protocol, profile, c)) for c in cases]
        return TraceBackend(protocol["output"], profile, keys)

    guarded_aligned_worker(path, commit, pid, admit, factory)
    root = Path(protocol["output"])
    summary_path = root / "summary.json"
    summary = read(summary_path)
    _, index = traces(protocol, cases, summary["rows"])
    predecessor_path, predecessor_sha = terminal_predecessor()
    summary.update(acquisition_trace_sha256=index, predecessor_watchdog={predecessor_path: predecessor_sha},
                   cost_scope=TIMING)
    staged = root / "summary.with-window-traces.json"
    write(staged, summary)
    # Only this new pilot's summary is augmented before the original
    # controller seals it; no prior bytes, row JSON or NPZ are changed.
    os.replace(staged, summary_path)


def audit_trace(record, profile):
    import math

    trace = record["trace"]
    uniform = profile == "uniform"
    control = profile == "frozen_direct"
    pool = trace["eligible_grid_ids"]
    attempts = trace["attempted_windows"]
    used_ids = [attempt["grid_id"] for attempt in attempts]
    acquired = [attempt for attempt in attempts if attempt["query_xy"] is not None]
    budget = 0 if control else WINDOWS
    if (record["profile"] != profile or trace["policy"] != profile or trace["requested_budget"] != budget
            or sorted(set(pool)) != pool or any(type(i) is not int or not 0 <= i < 256 for i in pool)
            or len(used_ids) != len(set(used_ids)) or not set(used_ids).issubset(pool)
            or len(used_ids) != min(budget, len(pool)) or not 0 <= trace["ncc_calls"] <= 2 * len(used_ids)
            or trace["accepted_additions"] != len(acquired)
            or trace["additional_common_pnp_calls"] != int(bool(acquired))):
        raise ValueError("Acquisition sidecar violates its frozen availability or measurement cap")
    if ((control or not record["seed_pose_returned"]) and (pool or attempts or not trace["used_original_seed"])
            or trace["used_original_seed"] != bool(trace["reason"])
            or (uniform and any(a["policy"] != "uniform" for a in attempts))
            or any(a["policy"] not in (profile, "uniform") for a in attempts)):
        raise ValueError("Acquisition sidecar violates arm isolation or fallback")
    for key in ("inference_without_sidecar_seconds", "seed_observer_seconds"):
        if not math.isfinite(record[key]) or record[key] < 0:
            raise ValueError("Inclusive pilot measurement requires finite actual times")


def sealed_output(protocol, cases, summary):
    from run_solar_matched_matchanything_recovery_v91 import _replay_metrics

    root = Path(protocol["output"])
    expected_markers = set()
    checks = []
    for row, (profile, case) in zip(summary["rows"], ((p, c) for p in protocol["profiles"] for c in cases), strict=True):
        key = case_key(protocol, profile, case)
        marker = root / (key + ".COMPLETE.json")
        row_path, arrays_path = root / (key + ".json"), root / (key + ".npz")
        expected_markers.add(marker)
        seal = read(marker)
        if (seal["complete"] is not True or seal["identity"] != key or row != read(row_path)
                or seal["row_path"] != str(row_path) or seal["row_sha256"] != sha(row_path)
                or seal["arrays_path"] != str(arrays_path) or row["saved_arrays"] != str(arrays_path)
                or seal["arrays_sha256"] != sha(arrays_path) or row["saved_arrays_sha256"] != seal["arrays_sha256"]):
            raise ValueError("Fresh measurement differs from its sealed case artifact")
        checks.append(dict(profile=profile["name"], **_replay_metrics(row, case)))
    if set(root.glob("*.COMPLETE.json")) != expected_markers:
        raise ValueError("Scientific result receipt inventory differs")
    return checks


def archived_fingerprint(row):
    import numpy as np

    if row["inference_failure"] is not None or not row["raw_inference_reporting"]["encoding_complete"]:
        raise ValueError("Complete exact parent raw inference is required")
    with np.load(row["saved_arrays"], allow_pickle=False) as arrays:
        return array_fingerprint(row["raw_inference_result"], arrays)


def paired_audit(cases, rows, records, historical_rows):
    import numpy as np

    grid = {(r["profile"], r["case_id"]): r for r in rows}
    by_id = {r["case_id"]: r for r in historical_rows}
    report = []
    for case in cases:
        cid = case["case_id"]
        reference = by_id[cid]
        seed = archived_fingerprint(reference)
        pools, details = [], {}
        for policy in ARMS:
            row = grid[policy, cid]
            record = records.get((policy, cid))
            if row["inference_failure"] is not None or record is None:
                details[policy] = dict(inference_failure=row["inference_failure"], missing_trace=record is None)
                continue
            audit_trace(record, policy)
            trace = record["trace"]
            if record["seed_fingerprint"] != seed or record["seed_pose_returned"] != reference["metrics"]["pose_returned"]:
                raise ValueError("Fresh arm changed the exact historical direct result")
            if trace["used_original_seed"]:
                if archived_fingerprint(row) != seed or row["metrics"] != reference["metrics"]:
                    raise ValueError("Control or fallback does not reproduce exact direct output")
            else:
                with np.load(reference["saved_arrays"], allow_pickle=False) as original, \
                        np.load(row["saved_arrays"], allow_pickle=False) as current:
                    for key in ("reference_xy", "query_xy"):
                        n = record["seed_match_count"]
                        if n != len(original[key]) or not np.array_equal(current[key][:n], original[key]):
                            raise ValueError("Additive result changed its original correspondences")
                        if len(current[key]) != n + trace["accepted_additions"]:
                            raise ValueError("Additive result differs from its acquired window count")
            if policy != "frozen_direct":
                pools.append(trace["eligible_grid_ids"])
            details[policy] = dict(windows=len(trace["attempted_windows"]), ncc_calls=trace["ncc_calls"],
                                   additions=trace["accepted_additions"], used_original_seed=trace["used_original_seed"],
                                   reason=trace["reason"], correct=row["metrics"]["correct"])
        if len(pools) == 2 and pools[0] != pools[1]:
            raise ValueError("The two active arms had different nominal candidate availability")
        report.append(dict(case_id=cid, positive=case["positive"], seed_pose_returned=reference["metrics"]["pose_returned"],
                           light_a=case["light_a"], light_b=case["light_b"], arms=details))
    return report


def full_cost(protocol, rows, watchdog, report, remaining):
    import math

    from run_solar_evaluation_v70 import measured_cost

    if any(row["inference_failure"] is not None for row in rows):
        return dict(qualified=False, reason="inference_failure_or_unavailable")
    for policy in ARMS[1:]:
        if not any(case["arms"][policy].get("windows") == WINDOWS for case in report):
            return dict(qualified=False, reason="no_full_incremental_window_measurement", profile=policy)
    for row in rows:
        for key in ("case_calendar_seconds", "profile_load_seconds"):
            if type(row[key]) not in (float, int) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Measured pilot time must be finite and nonnegative")
    n = protocol["full_case_count"]
    projection = measured_cost(rows, n, protocol["tracks"], protocol["profiles"])
    billed_in_rows = sum(row["case_calendar_seconds"] + row["profile_load_seconds"] for row in rows)
    one_time_seconds = max(0., watchdog["calendar_seconds"] - billed_in_rows)
    proposed = projection["conservative_total_seconds"] + math.ceil(1.5 * one_time_seconds) + 5
    return dict(qualified=5 < proposed <= min(FULL_CEILING, remaining),
                projected_full_cap_seconds=proposed, all_case_estimates=projection,
                measured_one_time_calendar_seconds=one_time_seconds,
                overhead_rule="1.5 times measured startup and unallocated watchdog time, plus five seconds to seal.",
                original_full_ceiling_seconds=FULL_CEILING, remaining_baseline_seconds=remaining,
                full_execution_authorized=False,
                scope="Projections only. A separately published full-development protocol must include every arm and case.")


def replay(path, output):
    from solar_evaluation_v70 import aggregate_rows

    relative(path)
    relative(output)
    protocol = read(path)
    cases = validate_frozen(protocol)
    root = Path(protocol["output"])
    summary, watchdog, seal, execution, reservation = (
        read(root / name) for name in ("summary.json", "watchdog.json", "COMPLETE.json", "execution.json", "reservation.json")
    )
    digest = sha(path)
    cap = protocol["max_calendar_seconds"]
    start, end = utc(watchdog["started_utc"]), utc(watchdog["ended_utc"])
    hard = validate_dispatch(protocol, start, PILOT_CEILING)
    charge = reservation["reservation"]
    entries, remaining = budget(protocol)
    commit = watchdog["publication_commit"]
    predecessor_path, predecessor_sha = terminal_predecessor()
    if (seal["external_watchdog_passed"] is not True or watchdog["complete"] is not True or summary["complete"] is not True
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or any(x["protocol_sha256"] != digest for x in (summary, watchdog, seal, execution))
            or seal["summary_sha256"] != sha(root / "summary.json")
            or seal["watchdog_sha256"] != sha(root / "watchdog.json")
            or watchdog["cap_seconds"] != cap or not 0 < watchdog["calendar_seconds"] < cap
            or hard != utc(watchdog["hard_deadline_utc"]) or not start <= end < hard
            or any(execution[k] != watchdog[k] for k in ("started_utc", "hard_deadline_utc", "work_deadline_utc"))
            or summary["publication_commit"] != commit or execution["publication_commit"] != commit
            or summary["purpose"] != protocol["purpose"] or summary["profiles"] != protocol["profiles"]
            or summary["calibration_or_final_access"] or summary["selection_performed"]
            or summary["software"] != protocol["expected_software"]
            or charge not in entries or charge["job_id"] != digest + JOB_SUFFIX or charge["category"] != "baseline"
            or charge["seconds"] != cap or charge["refund_on_failure"]
            or reservation != watchdog["compute_reservation"]
            or summary["predecessor_watchdog"] != {predecessor_path: predecessor_sha}
            or utc(read(predecessor_path)["ended_utc"]) > start):
        raise ValueError("Actual serial completion, original reservation or measured CPU proof differs")
    check_publication(path, protocol["source_sha256"], protocol["metadata_sha256"], commit, lambda: None)
    for name, checksum in protocol["input_sha256"].items():
        if sha(name) != checksum:
            raise ValueError("Frozen pilot input changed: " + name)
    rows = summary["rows"]
    checks = sealed_output(protocol, cases, summary)
    records, index = traces(protocol, cases, rows)
    if index != summary["acquisition_trace_sha256"] or summary["table"] != aggregate_rows(rows):
        raise ValueError("Sealed trace or raw aggregate differs")
    _, _, _, historical_rows, _ = historical()
    paired = paired_audit(cases, rows, records, historical_rows)
    cost = full_cost(protocol, rows, watchdog, paired, remaining)
    write(output, dict(passed=True, replayed_rows=len(checks), rows=checks, paired=paired,
                       inclusive_calendar_seconds=watchdog["calendar_seconds"], measured_full_projection=cost,
                       protocol_sha256=digest, summary_sha256=seal["summary_sha256"],
                       watchdog_sha256=seal["watchdog_sha256"], accepted_results=False,
                       calibration_or_final_access=False, scope=protocol["scope"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "freeze", "run", "worker", "replay"):
        command = commands.add_parser(action)
        command.add_argument("--protocol", required=True)
        if action in ("prepare", "freeze"):
            command.add_argument("--settings", required=True)
        elif action in ("run", "worker"):
            command.add_argument("--publication-commit", required=True)
        else:
            command.add_argument("--output", required=True)
        if action == "run":
            command.add_argument("--python", default=sys.executable)
        elif action == "worker":
            command.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    if args.action in ("prepare", "freeze"):
        prepare(args.settings, args.protocol, dispatch=args.action == "freeze")
    elif args.action == "run":
        with Path("runs/.solar-direct-additive-v97.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guarded_run(args.protocol, args.publication_commit, args.python, admission, SOURCE)
    elif args.action == "worker":
        worker(args.protocol, args.publication_commit, args.controller_pid)
    else:
        replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
