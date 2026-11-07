"""Guarded development mechanism comparison of flat and terrain-warped native NCC."""
import argparse
from collections import Counter
from datetime import timedelta
import fcntl
import hashlib
import math
import os
from pathlib import Path
import sys

from run_solar_crosslight_mechanism_v98 import terminal_predecessors
from run_solar_direct_additive_windows_v97 import (
    TraceBackend, SeedObserver, archived_fingerprint, audit_trace, case_key, traces,
)
from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_common_protocol, check_publication, fixed_v86_cases, guarded_run,
    read, sha, utc, utc_now, validate_dispatch, write,
)
from run_solar_full_crosslight_rescue_v99 import (
    HISTORICAL, HISTORICAL_SHA256, HISTORICAL_REPLAY, HISTORICAL_REPLAY_SHA256,
    completion, historical, timing_projection,
)
from run_solar_full_direct_render_v92 import FULL_CEILING
from run_solar_matched_direct_render_v91 import BASIS, budget, merge_pins, relative
from solar_perspective_windows_v99 import METHOD, TerrainPerspectiveWindowsBackend


SOURCE = Path(__file__).name
PURPOSE = "matched_terrain_perspective_windows_v100"
ROOT = "runs/solar-perspective-windows-v100/pilot"
PRIOR = "reports/solar-direct-additive-windows-pilot-v97.json"
PRIOR_SHA = "89ce084f6ec3d76eff93de1be2f7f27d760924664c437e9ce8462853721eb6ff"
PRIOR_REPLAY = "reports/solar_direct_additive_windows_pilot_replay_v97.json"
PRIOR_REPLAY_SHA = "5ec50f5ad57683c6a990a25e7a072fff698f09b519dd563fc66cdb43c09721d3"
CROSS_LIGHT = "reports/solar-full-crosslight-rescue-dispatch-v99.json"
CROSS_LIGHT_SHA = "5ca6a4a5de7c7dcd3987db92007d7485074485bb746ebcecf5efef8b08612817"
DIRECT = "frozen_direct"
FLAT = "uniform"
ARMS = (DIRECT, FLAT, METHOD)
SAME = ((0, 0), (16, 16))
CHANGED = ((0, 17), (16, 1), (8, 17), (8, 1))
FAMILIES = ("training_motion", "unseen_roll")
BUDGET = 128
SCOPE = ("Development-only mechanism pilot. Cases and camera balance come from metadata without querying outcomes. "
         "These cases already appeared in the v92 direct-only development comparison. "
         "No calibration, accepted accuracy, or final inference.")
TIMING = ("Three fresh arms, direct first then flat and warped. Identical paid window IDs in the two additive arms. "
          "Case wall includes input verification/decode, inference, scoring, trace and NPZ; "
          "controller wall also includes admission and sealing. Process lifetime RSS and OS cache order are shared.")


def cohort(all_cases, audit, previous_ids):
    """Metadata-only 24 positives and their 24 original matched nonoverlaps."""
    positives = [c for c in all_cases if c["positive"]]
    old_units = {c["unit_index"] for c in positives if c["case_id"] in previous_ids}
    negatives = {c["matched_positive_case"]: c for c in all_cases if not c["positive"]}
    units = {u["unit_index"]: u for u in audit["units"] if u["split"] == "model_selection"}
    groups = sorted({c["component"] for c in positives})
    if len(groups) != 8 or len(old_units) != 4:
        raise ValueError("Original eight terrain groups and four exposed v97 positive cameras required")
    chosen = []
    free_group_index = 0
    for group_index, group in enumerate(groups):
        excluded = [units[i]["family"] for i in old_units if units[i]["component"] == group]
        if len(excluded) > 1:
            raise ValueError("At most one earlier positive camera per terrain group")
        if excluded:
            majority = FAMILIES[1 - FAMILIES.index(excluded[0])]
        else:
            majority = FAMILIES[free_group_index % 2]
            free_group_index += 1
        minority = FAMILIES[1 - FAMILIES.index(majority)]
        light_pairs = (SAME[group_index % 2], CHANGED[group_index % 4], CHANGED[(group_index + 1) % 4])
        used_units = set(old_units)
        for slot, lighting in enumerate(light_pairs):
            family = minority if slot == group_index % 3 else majority
            options = [c for c in positives if c["component"] == group and c["unit_index"] not in used_units
                       and units[c["unit_index"]]["family"] == family and (c["light_a"], c["light_b"]) == lighting]
            if not options:
                raise ValueError("Missing an exact metadata-only group/camera/lighting slot")
            selected = min(options, key=lambda c: hashlib.sha256(("20261104/" + c["case_id"]).encode()).digest())
            used_units.add(selected["unit_index"])
            chosen.append(selected)
    if (len({c["unit_index"] for c in chosen}) != 24
            or Counter(units[c["unit_index"]]["family"] for c in chosen) != Counter(dict.fromkeys(FAMILIES, 12))
            or Counter((c["light_a"], c["light_b"]) for c in chosen) != Counter(dict.fromkeys(SAME + CHANGED, 4))):
        raise ValueError("Frozen cohort must balance six light pairs and both camera families")
    ids = {c["case_id"] for c in chosen}
    ids.update(negatives[c["case_id"]]["case_id"] for c in chosen)
    result = [c for c in all_cases if c["case_id"] in ids]
    if len(result) != 48 or ids.intersection(previous_ids):
        raise ValueError("Exact new 48-case paired development cohort required")
    for fields in (("reference", "query"), ("reference_geometry", "solar_lookup_protocol")):
        if tuple(result[0].get(k) for k in fields) == tuple(result[-1].get(k) for k in fields):
            raise ValueError("The v90 single-entry loader must be cold across arm boundaries")
    return result


def cost_basis(prior, summary, watchdog, n_cases):
    """Use measured v97 costs and an explicit pre-lunar margin for the new warp."""
    observed = {}
    for name in (DIRECT, FLAT):
        rows = [r for r in summary["rows"] if r["profile"] == name]
        if len(rows) != 8 or any(r["inference_failure"] is not None for r in rows):
            raise ValueError("Both original arms must have eight complete measured v97 rows")
        for row in rows:
            for field in ("case_calendar_seconds", "profile_load_seconds"):
                value = row[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Finite measured v97 elapsed seconds required")
        observed[name] = dict(case_seconds=max(r["case_calendar_seconds"] for r in rows),
                              load_seconds=max(r["profile_load_seconds"] for r in rows))
    billed = sum(r["case_calendar_seconds"] + r["profile_load_seconds"] for r in summary["rows"])
    outside = max(0., watchdog["calendar_seconds"] - billed)
    # New projection/remap cost is still unmeasured on lunar scenes. Bound
    # this exploratory stage at twice the measured flat case cost before the
    # existing 1.5x rule. Full study must use new measured wall times.
    warp = dict(case_seconds=2 * observed[FLAT]["case_seconds"], load_seconds=observed[FLAT]["load_seconds"])
    elapsed = [observed[DIRECT], observed[FLAT], warp]
    cap = math.ceil(1.5 * (2 * outside + sum(m["load_seconds"] + n_cases * m["case_seconds"] for m in elapsed))) + 5
    if cap > FULL_CEILING or cap <= prior["max_calendar_seconds"]:
        raise ValueError("Broader pilot allowance outside the original baseline limits")
    return dict(measured_v97=observed, v97_unallocated_seconds=outside, warp_planning_bound=warp,
                conservative_controller_calendar_seconds=cap,
                rule="1.5× (three arm loads + 48 worst-case rows per arm + twice v97 overhead) + five seconds. "
                     "Warp estimates twice flat; actual lunar warp latency will be measured here.")


def previous_proof():
    if sha(PRIOR) != PRIOR_SHA or sha(PRIOR_REPLAY) != PRIOR_REPLAY_SHA:
        raise ValueError("Exact v97 measured controls and independent replay required")
    prior = read(PRIOR)
    summary, watchdog, seal, rows, files = completion(PRIOR, prior)
    replay = read(PRIOR_REPLAY)
    if (replay["passed"] is not True or replay["replayed_rows"] != 24
            or replay["accepted_results"] or replay["calibration_or_final_access"]
            or replay["protocol_sha256"] != PRIOR_SHA or replay["summary_sha256"] != seal["summary_sha256"]
            or replay["watchdog_sha256"] != seal["watchdog_sha256"]
            or replay["inclusive_calendar_seconds"] != watchdog["calendar_seconds"]):
        raise ValueError("Exact complete v97 replay required")
    for row, verified in zip(rows, replay["rows"], strict=True):
        if any(row[k] != verified[k] for k in ("profile", "case_id")) or any(
            row["metrics"][k] != verified[k] for k in ("correct", "strict", "pose_returned", "correspondence")
        ):
            raise ValueError("Measured v97 row differs from independent replay")
    return prior, summary, watchdog, {str(path): sha(path) for path in files}


def frozen(settings):
    from control_historical_native_v74 import source_closure
    from direct_render_adapter_v73 import validate_render_profile

    if not isinstance(settings, dict) or set(settings) != {"output", "publication_dispatch_lease_seconds"}:
        raise ValueError("Specify only fresh v100 output and the dispatch lease")
    lease = settings["publication_dispatch_lease_seconds"]
    if str(relative(settings["output"])) != ROOT or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Fixed fresh v100 output and original finite dispatch lease required")
    predecessors = {CROSS_LIGHT: CROSS_LIGHT_SHA}
    if sha(CROSS_LIGHT) != CROSS_LIGHT_SHA:
        raise ValueError("Exact active full cross-light predecessor required")
    # Select IDs before reading historical outcome payloads.
    _, _, full = fixed_v86_cases(BASIS, full=True)
    prior_meta = read(PRIOR)
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError("Exact v97 cohort metadata required")
    audit = read(prior_meta["geometry_audit"])
    cases = cohort(full, audit, prior_meta["case_ids"])
    prior, measured, watchdog, prior_files = previous_proof()
    old, history, _, historical_files = historical(full)
    direct, flat = prior["profiles"][:2]
    parent, = old["profiles"]
    if (direct["name"], flat["name"]) != (DIRECT, FLAT) or any(
        p["direct_profile"] != parent or p["configuration"] != dict(policy=p["name"], budget=BUDGET)
        for p in (direct, flat)
    ) or history["software"] != prior["expected_software"]:
        raise ValueError("Original v97 direct and flat configuration or v92 seed changed")
    warped = dict(flat, name=METHOD, method=METHOD, backend="terrain_perspective_windows",
                  configuration=dict(policy=METHOD, budget=BUDGET))
    profiles = [direct, flat, warped]
    cost = cost_basis(prior, measured, watchdog, len(cases))
    sources = dict(prior["source_sha256"])
    merge_pins(sources, source_closure({
        "solar_perspective_windows_v99.py", "run_solar_full_crosslight_rescue_v99.py",
    }))
    merge_pins(sources, {SOURCE: sha(Path(__file__))})
    metadata = dict(prior["metadata_sha256"])
    merge_pins(metadata, {PRIOR: PRIOR_SHA, PRIOR_REPLAY: PRIOR_REPLAY_SHA,
                          HISTORICAL: HISTORICAL_SHA256, HISTORICAL_REPLAY: HISTORICAL_REPLAY_SHA256, **predecessors})
    assets = dict(prior["input_sha256"])
    for pins in (metadata, prior_files, historical_files):
        merge_pins(assets, pins)
    _, libraries, _ = validate_render_profile(parent)
    historical_rows = {r["case_id"]: r for r in history["rows"]}
    for case in cases:
        merge_pins(assets, case["input_sha256"])
        library = libraries[case["reference_asset_metadata"]["source_reference_geometry_sha256"]]
        merge_pins(assets, {library["source_summary"]: library["source_summary_sha256"]})
        merge_pins(assets, {entry["image"]: entry["image_sha256"] for entry in library["entries"]})
        row = historical_rows[case["case_id"]]
        merge_pins(assets, {row["saved_arrays"]: row["saved_arrays_sha256"]})
    protocol = {k: prior[k] for k in (
        "basis", "manifest", "geometry_audit", "full_case_count", "compute_ledger", "compute_ledger_sha256",
        "reservation_journal", "blender_executable", "scoring_revision", "expected_software",
    )}
    protocol.update(**settings, settings=settings, purpose=PURPOSE, profiles=profiles, tracks=["estimated_sun"],
                    case_ids=[case["case_id"] for case in cases], split="model_selection", cpu_threads=2,
                    calibration_or_final_access=False, accepted_results=False, automatic_model_promotion=False,
                    serial_predecessors=predecessors, incremental_window_cap=BUDGET,
                    source_sha256=sources, metadata_sha256=metadata, input_sha256=assets, cost_basis=cost,
                    max_calendar_seconds=cost["conservative_controller_calendar_seconds"], scope=SCOPE, timing_scope=TIMING)
    return protocol, cases, historical_rows


def prepare(settings_path, path, *, dispatch=False):
    relative(path)
    protocol, _, _ = frozen(read(settings_path))
    if Path(protocol["output"]).exists():
        raise FileExistsError("Preserve all previous run directories")
    entries, remaining = budget(protocol)
    cap = protocol["max_calendar_seconds"]
    if cap > remaining:
        raise ValueError("The original baseline allowance cannot admit this measured-scope pilot")
    created = latest = absolute = None
    if dispatch:
        terminal_predecessors(protocol)
        created = utc_now()
        latest = created + timedelta(seconds=protocol["publication_dispatch_lease_seconds"])
        absolute = latest + timedelta(seconds=cap)
    protocol.update(prepared_only=not dispatch, initial_reservations=entries,
                    created_utc=None if created is None else created.isoformat(),
                    latest_start_utc=None if latest is None else latest.isoformat(),
                    absolute_deadline_utc=None if absolute is None else absolute.isoformat())
    write(path, protocol)


def exact(protocol):
    expected, cases, historical_rows = frozen(protocol["settings"])
    fields = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (protocol.get("prepared_only") is not False or set(protocol) != set(expected) | fields
            or any(protocol.get(k) != v for k, v in expected.items())):
        raise ValueError("Exact published v100 experiment identity required")
    check_common_protocol(protocol)
    return cases, historical_rows


def admission(path, commit, started):
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    p = read(path)
    deadline = validate_dispatch(p, started, FULL_CEILING)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original v100 allowance exhausted")

    check()
    cases, _ = exact(p)
    terminal_predecessors(p)
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, checksum in p["input_sha256"].items():
        check()
        if sha(asset) != checksum:
            raise ValueError("Frozen input changed: " + asset)
    if software_identity() != p["expected_software"]:
        raise ValueError("Original measured CPU/software stratum required")
    entries, remaining = budget(p)
    cap, initial = p["max_calendar_seconds"], p["initial_reservations"]
    own = [entry for entry in entries if entry["job_id"] == sha(path) + JOB_SUFFIX]
    if (entries[:len(initial)] != initial or len(own) > 1
            or any(e["category"] != "baseline" or e["seconds"] != cap or e["refund_on_failure"] for e in own)
            or (not own and cap > remaining)):
        raise ValueError("Original canonical budget history, cap or charge changed")
    check()
    return p, cases, deadline


class WarpTraceBackend(TraceBackend):
    """Use v97 trace behavior with the unchanged v99 scientific backend."""

    def __init__(self, output, profile, keys):
        if profile["name"] != METHOD:
            super().__init__(output, profile, keys)
            return
        self.profile = profile
        self.backend = TerrainPerspectiveWindowsBackend(profile)
        self.observer = SeedObserver(self.backend.direct)
        self.backend.direct = self.observer
        self.provenance = self.backend.provenance
        self.remaining = iter(keys)
        self.output = Path(output)


def worker(path, commit, pid):
    from run_solar_matched_matchanything_recovery_v91 import guarded_aligned_worker

    protocol = read(path)
    cases = None

    def admit(*args):
        nonlocal cases
        verified, cases, deadline = admission(*args)
        return verified, cases, deadline

    def factory(profile):
        keys = [(c["case_id"], case_key(protocol, profile, c)) for c in cases]
        return WarpTraceBackend(protocol["output"], profile, keys)

    guarded_aligned_worker(path, commit, pid, admit, factory)
    root = Path(protocol["output"])
    summary_path = root / "summary.json"
    summary = read(summary_path)
    _, index = traces(protocol, cases, summary["rows"])
    summary.update(acquisition_trace_sha256=index, predecessor_watchdogs=terminal_predecessors(protocol),
                   cost_scope=TIMING)
    staged = root / "summary.with-window-traces.json"
    write(staged, summary)
    os.replace(staged, summary_path)


def scored_additions(case, trace):
    import numpy as np

    from solar_aligned_scoring_v91 import load_aligned_scoring

    attempts = trace["attempted_windows"]
    returned = [a for a in attempts if a["query_xy"] is not None]
    reference = np.array([a["reference_xy"] for a in returned]).reshape(-1, 2)
    query = np.array([a["query_xy"] for a in returned]).reshape(-1, 2)
    scored = dict(paid_windows=len(attempts), returned=len(returned), ncc_calls=trace["ncc_calls"],
                  co_visible=None, within_px=None, median_error_px=None, fallback=trace["used_original_seed"],
                  reasons=dict(Counter(a["reason"] or "reciprocal_match" for a in attempts)))
    if case["positive"] and len(reference):
        truth, _ = load_aligned_scoring(case, reference)
        visible = truth["visible"]
        residual = np.linalg.norm(query[visible] - truth["query_xy"][visible], axis=1)
        scored.update(co_visible=int(visible.sum()), within_px={
            str(t): int((residual <= t).sum()) for t in (1, 2, 4, 8, 12)
        }, median_error_px=float(np.median(residual)) if len(residual) else None)
    elif case["positive"]:
        scored.update(co_visible=0, within_px={str(t): 0 for t in (1, 2, 4, 8, 12)})
    return scored, reference, query


def audit_pair(case, history, rows, records):
    import numpy as np

    cid = case["case_id"]
    seed = archived_fingerprint(history)
    schedules = {}
    addons = {}
    for name in ARMS:
        row, record = rows[name, cid], records.get((name, cid))
        if row["inference_failure"] is not None:
            addons[name] = dict(inference_failure=row["inference_failure"])
            continue
        if record is None:
            raise ValueError("Successful arm did not retain its trace")
        audit_trace(record, name)
        if record["seed_fingerprint"] != seed or record["seed_pose_returned"] != history["metrics"]["pose_returned"]:
            raise ValueError("Fresh inference differs from the exact original seed")
        trace = record["trace"]
        additions, reference, query = scored_additions(case, trace)
        addons[name] = additions
        attempts = trace["attempted_windows"]
        if name != DIRECT:
            schedule = [(a["grid_id"], a["reference_xy"], a["projected_query_xy"]) for a in attempts]
            schedules[name] = (trace["eligible_grid_ids"], schedule)
        for a in attempts:
            p, b = np.asarray(a["reference_xy"]), np.asarray(a["projected_query_xy"])
            if p.shape != (2,) or b.shape != (2,) or not np.isfinite([p, b]).all():
                raise ValueError("Native paired scheduled coordinates required")
            if a["query_xy"] is not None:
                q = np.asarray(a["query_xy"])
                if q.shape != (2,) or not np.isfinite(q).all() or (np.abs(q - b) > 16 + 1e-12).any():
                    raise ValueError("An added match exceeded its paid native search window")
        if trace["used_original_seed"]:
            if archived_fingerprint(row) != seed or row["metrics"] != history["metrics"]:
                raise ValueError("A control or fallback changed frozen direct output")
        else:
            n = record["seed_match_count"]
            with np.load(history["saved_arrays"], allow_pickle=False) as original, \
                    np.load(row["saved_arrays"], allow_pickle=False) as candidate:
                for field, extra in (("reference_xy", reference), ("query_xy", query)):
                    if (len(original[field]) != n or not np.array_equal(candidate[field][:n], original[field])
                            or not np.array_equal(candidate[field][n:], extra)):
                        raise ValueError("Saved refit arrays differ from the seed plus acquired matches")
    if len(schedules) == 2 and schedules[FLAT] != schedules[METHOD]:
        raise ValueError("Warped and flat arms did not pay for the exact same windows in the same order")
    return dict(case_id=cid, positive=case["positive"], component=case["component"],
                query_component=case["query_component"], light_a=case["light_a"], light_b=case["light_b"],
                paid_schedule_verified=len(schedules) == 2, acquisitions=addons)


def mechanisms(pairs):
    def subtotal(items, policy):
        present = [p["acquisitions"][policy] for p in items if "inference_failure" not in p["acquisitions"][policy]]
        paid = sum(r["paid_windows"] for r in present)
        returned = sum(r["returned"] for r in present)
        good = sum((r["within_px"] or {}).get("2", 0) for r in present)
        positive = bool(items) and items[0]["positive"]
        reasons = Counter()
        for r in present:
            reasons.update(r["reasons"])
        return dict(cases=len(items), completed_cases=len(present), paid_windows=paid, reciprocal_matches=returned,
                    correct_within_2px=good if positive else None,
                    correct_per_paid_window=good / paid if paid and positive else None,
                    correct_per_returned_match=good / returned if returned and positive else None,
                    ncc_calls=sum(r["ncc_calls"] for r in present), reasons=dict(reasons))

    return {label: {arm: subtotal(items, arm) for arm in (FLAT, METHOD)} for label, items in (
        ("same_light_positive", [p for p in pairs if p["positive"] and p["light_a"] == p["light_b"]]),
        ("changed_light_positive", [p for p in pairs if p["positive"] and p["light_a"] != p["light_b"]]),
        ("all_positive", [p for p in pairs if p["positive"]]),
        ("nonoverlap", [p for p in pairs if not p["positive"]]),
    )}


def replay(path, output):
    from run_solar_matched_matchanything_recovery_v91 import _replay_metrics
    from solar_evaluation_v70 import paired_component_bootstrap

    relative(path)
    relative(output)
    p = read(path)
    cases, history = exact(p)
    predecessor = terminal_predecessors(p)
    summary, watchdog, seal, rows, _ = completion(path, p)
    start = utc(watchdog["started_utc"])
    if (summary["predecessor_watchdogs"] != predecessor
            or any(utc(read(name)["ended_utc"]) > start for name in predecessor)):
        raise ValueError("Serial predecessor termination did not precede v100 inference")
    for name, digest in p["input_sha256"].items():
        if sha(name) != digest:
            raise ValueError("Pinned inference or scoring input changed: " + name)
    index = {case["case_id"]: case for case in cases}
    checked = [dict(profile=r["profile"], **_replay_metrics(r, index[r["case_id"]])) for r in rows]
    records, sidecars = traces(p, cases, rows)
    if sidecars != summary["acquisition_trace_sha256"]:
        raise ValueError("The sealed acquisition sidecar hashes changed")
    grid = {(r["profile"], r["case_id"]): r for r in rows}
    paired = [audit_pair(c, history[c["case_id"]], grid, records) for c in cases]
    measurements = {}
    if all(r["inference_failure"] is None for r in rows) and all(
        any(c["acquisitions"][arm]["paid_windows"] == BUDGET for c in paired) for arm in (FLAT, METHOD)
    ):
        projection = timing_projection(rows, p["full_case_count"], p["profiles"], watchdog["calendar_seconds"])
        _, remaining = budget(p)
        measurements.update(estimated_full_calendar=projection,
                            fits_original_ceiling=projection["conservative_controller_calendar_seconds"] <= FULL_CEILING,
                            fits_current_baseline_allowance=projection["conservative_controller_calendar_seconds"] <= remaining)
    else:
        measurements["reason"] = "Fresh measurements do not support full-grid projection"
    measurements["full_execution_authorized"] = False
    comparisons = {control: paired_component_bootstrap(rows, METHOD, control) for control in (DIRECT, FLAT)}
    write(output, dict(passed=True, replayed_rows=len(checked), rows=checked, table=summary["table"],
                       paired_acquisitions=paired, correspondence_mechanism=mechanisms(paired),
                       paired_pose_comparisons=comparisons, measured_cost=measurements,
                       inclusive_calendar_seconds=watchdog["calendar_seconds"],
                       protocol_sha256=sha(path), summary_sha256=seal["summary_sha256"],
                       watchdog_sha256=seal["watchdog_sha256"], accepted_results=False,
                       calibration_or_final_access=False, scope=SCOPE))


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
        with Path("runs/.solar-perspective-v100.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guarded_run(args.protocol, args.publication_commit, args.python, admission, SOURCE)
    elif args.action == "worker":
        worker(args.protocol, args.publication_commit, args.controller_pid)
    else:
        replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
