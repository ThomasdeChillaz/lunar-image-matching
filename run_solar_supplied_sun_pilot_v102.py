"""Secondary supplied-Sun-only matched pilot using the frozen v90 watchdog."""

import argparse
from collections import Counter
from datetime import timedelta
import math
from pathlib import Path
import sys
from types import FunctionType

import lunar_known_sun_composite_v102 as composite
import run_solar_distributional_evaluation_v90 as guard
import run_solar_full_crosslight_rescue_v99 as evidence
from run_solar_matched_direct_render_v91 import budget, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import _replay_metrics


TRACK = "known_sun"
METHODS = ("nearest_even", "two_reference_ic")
SOURCE = Path(__file__).name
PURPOSE = "secondary_supplied_sun_cost_pilot_v102"
V99 = "reports/solar-full-crosslight-rescue-dispatch-v99.json"
V99_SHA = "5ca6a4a5de7c7dcd3987db92007d7485074485bb746ebcecf5efef8b08612817"
V99_REPLAY = "reports/solar_full_crosslight_rescue_replay_v99.json"
V99_REPLAY_SHA = "8347b60efa4d5f3721f2e118087475ec791339c595f43e2cd204b2d678312182"
SUPPORT = "reports/solar_supplied_sun_metadata_support_v102.json"
SUPPORT_SHA = "8bc140055dc5d14480864f05876d946f2e75811ab09243044206c696e96f331d"
PREPROCESSING = "Reference sRGB linear luminance and linear min/max uint8 contrast. Same query grayscale preprocessing in both arms."
SCOPE = "Secondary known-Sun track on the same 48 previously observed v100 metadata-selected development cases."
# Exact v100 metadata-only positive camera/light selections, in original order.
POSITIVES = (
    "u013-l00-00-positive", "u014-l16-01-positive", "u015-l00-17-positive", "u020-l08-17-positive",
    "u021-l16-16-positive", "u022-l16-01-positive", "u024-l08-17-positive", "u025-l00-00-positive",
    "u027-l08-01-positive", "u057-l16-16-positive", "u058-l00-17-positive", "u059-l08-01-positive",
    "u061-l00-17-positive", "u062-l00-00-positive", "u063-l16-01-positive", "u068-l16-16-positive",
    "u069-l16-01-positive", "u070-l08-17-positive", "u072-l08-01-positive", "u073-l08-17-positive",
    "u075-l00-00-positive", "u093-l08-01-positive", "u094-l16-16-positive", "u095-l00-17-positive",
)


def isolated(function, **bindings):
    """Bind the audited guard to one secondary job without mutating any module."""
    symbols = dict(function.__globals__, **bindings)
    return FunctionType(function.__code__, symbols, function.__name__, function.__defaults__, function.__closure__)


def cohort(full, audit):
    selected = set(POSITIVES)
    cases = [case for case in full if case["case_id"] in selected or case.get("matched_positive_case") in selected]
    positives = [case for case in cases if case["positive"]]
    negatives = [case for case in cases if not case["positive"]]
    families = {unit["unit_index"]: unit["family"] for unit in audit["units"] if unit["split"] == "model_selection"}
    counts = Counter(families[case["unit_index"]] for case in positives)
    terrains = Counter(case["component"] for case in positives)
    lights = Counter((case["light_a"], case["light_b"]) for case in positives)
    if (tuple(case["case_id"] for case in positives) != POSITIVES or len(cases) != 48
            or {case["matched_positive_case"] for case in negatives} != selected
            or len(terrains) != 8 or set(terrains.values()) != {3} or len(lights) != 6 or set(lights.values()) != {4}
            or counts != {"training_motion": 12, "unseen_roll": 12}):
        raise ValueError("Exact balanced v100 case order and matched nonoverlaps required")
    return cases


def support_rows(cases, profile):
    if guard.sha(SUPPORT) != SUPPORT_SHA:
        raise ValueError("Exact full metadata-only Sun support audit required")
    report = guard.read(SUPPORT)
    source = profile["render_evidence"]
    if report["library_sha256"] != source["manifest_sha256"] or guard.sha(source["manifest"]) != source["manifest_sha256"]:
        raise ValueError("Reference catalogue identity differs from the support audit")
    audit = {row["case_id"]: row for row in report["rows"]}
    if len(audit) != 384:
        raise ValueError("Exact full Sun metadata support grid required")
    result = {case["case_id"]: audit[case["case_id"]] for case in cases}
    counts = Counter((row["positive"], len(row["component_indices"])) for row in result.values())
    if (counts != {(True, 1): 8, (True, 2): 16, (False, 2): 14, (False, 0): 10}
            or any((c["positive"], c["component"], c["query_component"], bool(c["light_b"] % 2))
                   != (r["positive"], r["terrain"], r["query_terrain"], r["odd_query_light"])
                   for c, r in zip(cases, result.values(), strict=True))):
        raise ValueError("Declared full or common support differs from the frozen cohort")
    return result, guard.read(source["manifest"])


def frozen(settings):
    if not isinstance(settings, dict) or set(settings) != {"output", "publication_dispatch_lease_seconds"}:
        raise ValueError("Specify only fresh output and publication lease")
    output, lease = relative(settings["output"]), settings["publication_dispatch_lease_seconds"]
    if not output.is_relative_to("runs/solar-supplied-sun-pilot-v102") or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Versioned secondary output and original finite dispatch lease required")
    basis, _, full = guard.fixed_v86_cases(full=True)
    cases = cohort(full, guard.read(basis["geometry_audit"]))
    if guard.sha(V99) != V99_SHA or guard.sha(V99_REPLAY) != V99_REPLAY_SHA:
        raise ValueError("Exact completed and replayed primary v99 evidence required")
    prior = guard.read(V99)
    summary, watchdog, seal, rows, receipts = evidence.completion(V99, prior)
    evidence.verify_replayed(V99_REPLAY, V99, summary, seal, rows, two_arms=True)
    original = prior["profiles"][0]
    support, manifest = support_rows(cases, original)
    profiles = [dict(name=name, backend="supplied_sun_v102", direct_profile=original, configuration=dict(
        method=name, preprocessing=PREPROCESSING, reference_library_indices=list(composite.EVEN_LIBRARY), max_pose_calls=1,
    )) for name in METHODS]
    direct_rows = [row for row in rows if row["profile"] == original["name"]]
    if len(direct_rows) != 384 or any(row["inference_failure"] is not None for row in direct_rows):
        raise ValueError("Complete measured direct precursor required")
    estimate = evidence.timing_projection(rows, len(cases), prior["profiles"], watchdog["calendar_seconds"])
    # Both methods read at most two templates; the completed 24-light v99
    # case envelope is a conservative pre-measurement limit, not our latency.
    cap = estimate["conservative_controller_calendar_seconds"]
    if not 5 < cap <= evidence.FULL_CEILING:
        raise ValueError("Pilot does not fit the original full ceiling")
    sources, metadata = (dict(prior[k]) for k in ("source_sha256", "metadata_sha256"))
    source = Path(composite.__file__)
    merge_pins(sources, {SOURCE: guard.sha(Path(__file__)), source.name: guard.sha(source)})
    merge_pins(sources, {path: guard.sha(path) for path in ("audit_supplied_sun_support_v102.py", "test_lunar_known_sun_composite_v102.py")})
    merge_pins(metadata, {V99: V99_SHA, V99_REPLAY: V99_REPLAY_SHA, SUPPORT: SUPPORT_SHA})
    assets = dict(metadata)
    merge_pins(assets, {str(path): guard.sha(path) for path in receipts})
    libraries = {library["unit_index"]: library for library in manifest["libraries"]}
    for case in cases:
        merge_pins(assets, case["input_sha256"])
        library = libraries[case["unit_index"]]
        row = support[case["case_id"]]
        used = {row["nearest_index"], *row["component_indices"]}
        merge_pins(assets, {entry["image"]: entry["image_sha256"] for entry in library["entries"] if entry["light_index"] in used})
    p = {k: prior[k] for k in ("basis", "manifest", "geometry_audit", "full_case_count", "compute_ledger",
                              "compute_ledger_sha256", "reservation_journal", "blender_executable", "scoring_revision")}
    p.update(**settings, settings=settings, purpose=PURPOSE, tracks=[TRACK], split="model_selection", cpu_threads=2,
             calibration_or_final_access=False, accepted_results=False, automatic_model_promotion=False, profiles=profiles,
             case_ids=[case["case_id"] for case in cases], expected_software=summary["software"], source_sha256=sources,
             metadata_sha256=metadata, input_sha256=assets, max_calendar_seconds=cap, cost_basis=estimate,
             metadata_support=dict(positive=24, nonoverlap=14, unsupported_nonoverlap=10), scope=SCOPE)
    return p, cases, support


def prepare(settings, path):
    relative(path)
    p, *_ = frozen(guard.read(settings))
    entries, remaining = budget(p)
    if Path(p["output"]).exists() or p["max_calendar_seconds"] > remaining:
        raise ValueError("Preserve previous outputs and original baseline allowance")
    p.update(prepared_only=True, initial_reservations=entries, created_utc=None, latest_start_utc=None, absolute_deadline_utc=None)
    guard.write(path, p)


def exact(p):
    expected, cases, support = frozen(p["settings"])
    staged = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (p.get("prepared_only") is not False or set(p) != set(expected) | staged
            or any(p[k] != value for k, value in expected.items())):
        raise ValueError("Exact prospective supplied-Sun protocol required")
    return cases, support


def admission(path, commit, started):
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    p = guard.read(path)
    deadline = guard.validate_dispatch(p, started, evidence.FULL_CEILING)

    def check():
        if guard.utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original supplied-Sun pilot allowance exhausted")

    cases, _ = exact(p)
    guard.check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, checksum in p["input_sha256"].items():
        check()
        if guard.sha(asset) != checksum:
            raise ValueError("Frozen source data changed: " + asset)
    if software_identity() != p["expected_software"]:
        raise ValueError("Original measured CPU/software stratum required")
    entries, remaining = budget(p)
    initial, cap = p["initial_reservations"], p["max_calendar_seconds"]
    own = [row for row in entries if row["job_id"] == guard.sha(path) + guard.JOB_SUFFIX]
    if entries[:len(initial)] != initial or len(own) > 1 or any(
        row["category"] != "baseline" or row["seconds"] != cap or row["refund_on_failure"] for row in own
    ) or not own and cap > remaining:
        raise ValueError("Original canonical reservation history or available allowance differs")
    check()
    return p, cases, deadline


class Backend:
    def __init__(self, profile):
        from solar_evaluation_v70 import common_pose

        self.profile = profile
        self.calls = 0

        def observed(p, q, inp):
            if self.calls:
                raise ValueError("One unchanged common pose call is the complete method budget")
            self.calls += 1
            return common_pose(p, q, inp)

        self.model = composite.SuppliedSunCompositeAdapter(profile["direct_profile"], profile["name"], estimator=observed)
        self.provenance = dict(self.model.direct.provenance, method=profile["name"], track=TRACK,
                               preprocessing=PREPROCESSING, maximum_inference_pose_calls=1)

    def infer(self, inp):
        self.calls = 0
        result = self.model.infer(inp)
        search = result["supplied_sun_control"]
        if self.calls != search["final_pose_calls"] or self.calls != bool(search["template_indices"]):
            raise ValueError("Exact actual pose budget required")
        search["observed_pose_calls"] = self.calls
        return result


def evaluate_known_profiles(p, cases, commit, check, backend_factory):
    """v90 immutable-cell loop for the explicit known-Sun inference boundary."""
    import time
    import numpy as np
    from run_solar_evaluation_v70 import InferenceResultCapture, InputLoader, capture_case_report, worker_peak_rss_snapshot
    from solar_aligned_scoring_v91 import load_aligned_scoring
    from solar_evaluation_v70 import digest, evaluate_case

    if p["tracks"] != [TRACK] or p["calibration_or_final_access"]:
        raise ValueError("Only the isolated secondary development track is allowed")
    root, rows, memory = Path(p["output"]), [], []
    for profile in p["profiles"]:
        loader = InputLoader(commit, p["blender_executable"])
        try:
            check()
            before = worker_peak_rss_snapshot()
            started = time.perf_counter()
            backend = InferenceResultCapture(backend_factory(profile))
            load = time.perf_counter() - started
            guard.write(root / (profile["name"] + ".provenance.json"), backend.provenance)
            for i, case in enumerate(cases):
                check()
                started = time.perf_counter()
                for path, checksum in case["input_sha256"].items():
                    if guard.sha(path) != checksum:
                        raise ValueError("Frozen case bytes changed: " + path)
                inp = loader.get(case, TRACK)
                if not np.array_equal(inp.known_query_sun, case["secondary_known_query_sun_world"]):
                    raise ValueError("Exact supplied Sun metadata is required at the common boundary")
                row, arrays = evaluate_case(backend, inp, case, scoring_loader=load_aligned_scoring)
                row["raw_inference_result"], row["raw_inference_reporting"] = capture_case_report(backend, arrays)
                key = digest(dict(case=case, profile=profile, track=TRACK, sources=p["source_sha256"]))
                row_path, array_path, marker = (root / (key + suffix) for suffix in (".json", ".npz", ".COMPLETE.json"))
                with array_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                row.update({field: case[field] for field in ("case_id", *evidence.CASE_FIELDS)})
                row.update(profile=profile["name"], track=TRACK, reference_setup=loader.last_setup.copy(),
                           complete_localization_cold_seconds=row["inference_seconds"] + sum(loader.last_setup.values()),
                           case_calendar_seconds=time.perf_counter() - started, profile_load_seconds=load if i == 0 else 0.,
                           saved_arrays=str(array_path), saved_arrays_sha256=guard.sha(array_path))
                guard.write(row_path, row)
                guard.write(marker, dict(complete=True, identity=key, row_path=str(row_path), row_sha256=guard.sha(row_path),
                                         arrays_path=str(array_path), arrays_sha256=row["saved_arrays_sha256"]))
                rows.append(row)
                print(profile["name"], case["case_id"], row["reason"], flush=True)
            memory.append(dict(profile=profile["name"], before=before, after=worker_peak_rss_snapshot()))
            del backend
        finally:
            loader.close()
    return rows, memory


def seal_known_rows(root, protocol, ids):
    from solar_evaluation_v70 import digest

    root = Path(root)
    manifest = guard.read(protocol["manifest"])
    cases = {case["case_id"]: case for case in manifest["cases"]}
    rows, pins, markers = [], {}, set()
    for profile in protocol["profiles"]:
        for case_id in ids:
            key = digest(dict(case=cases[case_id], profile=profile, track=TRACK, sources=protocol["source_sha256"]))
            row_path, arrays, marker = (root / (key + suffix) for suffix in (".json", ".npz", ".COMPLETE.json"))
            seal, row = guard.read(marker), guard.read(row_path)
            if (seal != dict(complete=True, identity=key, row_path=str(row_path), row_sha256=guard.sha(row_path),
                             arrays_path=str(arrays), arrays_sha256=guard.sha(arrays))
                    or row["track"] != TRACK or row["profile"] != profile["name"] or row["case_id"] != case_id
                    or row["saved_arrays"] != str(arrays) or row["saved_arrays_sha256"] != seal["arrays_sha256"]):
                raise ValueError("Sealed secondary native cell differs from its frozen identity")
            markers.add(marker)
            rows.append(row)
            merge_pins(pins, {str(marker): guard.sha(marker), str(row_path): seal["row_sha256"], str(arrays): seal["arrays_sha256"]})
    if set(root.glob("*.COMPLETE.json")) != markers:
        raise ValueError("Exact complete secondary grid required")
    return rows, pins


def audit_sun_trace(row, case, expected):
    import numpy as np

    if row["inference_failure"] is not None:
        return None
    raw = row["raw_inference_result"]
    record = raw["supplied_sun_control"]
    supported = bool(expected["component_indices"])
    nearest = row["profile"] == METHODS[0]
    indices = [expected["nearest_index"]] if nearest else expected["component_indices"]
    expected_calls = int(bool(indices))
    expected_exact = dict(method=row["profile"], known_sun_metadata_used=True, template_indices=indices,
                          supplied_sun_enu=case["secondary_known_query_sun_world"], nearest_index=expected["nearest_index"],
                          horizontal_bracket_available=supported, final_pose_calls=expected_calls, observed_pose_calls=expected_calls)
    if (raw["query_truth_used"] is not False or any(record[key] != value for key, value in expected_exact.items())
            or len(record["synthesis_weights"]) != len(expected["weights"])
            or not np.allclose(record["synthesis_weights"], expected["weights"], rtol=0., atol=1e-12)):
        raise ValueError("Supplied Sun metadata, method, support or actual pose budget differs")
    for key in ("nearest_angle_deg", "vertical_residual", "horizontal_condition"):
        a, b = record[key], expected[key]
        if (a is None) != (b is None) or a is not None and not np.isclose(a, b, rtol=0., atol=1e-12):
            raise ValueError("Supplied Sun geometric trace differs from the metadata audit")
    if not indices and (row["reason"] != "no_supported_horizontal_bracket" or row["metrics"]["pose_returned"]):
        raise ValueError("Unsupported composite must be retained as an explicit no-pose cell")
    return record


def strata(cases, rows, support):
    from solar_evaluation_v70 import paired_component_bootstrap

    groups = {"all": {c["case_id"] for c in cases},
              "common_supported": {c["case_id"] for c in cases if support[c["case_id"]]["component_indices"]},
              "odd_query_index": {c["case_id"] for c in cases if c["light_b"] % 2 == 1},
              "odd_query_common_supported": {c["case_id"] for c in cases if c["light_b"] % 2 == 1
                                               and support[c["case_id"]]["component_indices"]}}
    output = {}
    for name, ids in groups.items():
        part = [r for r in rows if r["case_id"] in ids]
        output[name] = dict(
            **{method: evidence.outcomes([r for r in part if r["profile"] == method]) for method in METHODS},
            paired_terrain_interval=paired_component_bootstrap(part, METHODS[1], METHODS[0], track=TRACK),
        )
    return output


def replay(path, output):
    import numpy as np

    relative(path)
    relative(output)
    p = guard.read(path)
    cases, support = exact(p)
    finish = isolated(evidence.completion, seal_rows=seal_known_rows)
    summary, _, seal, rows, _ = finish(path, p)
    by_case = {case["case_id"]: case for case in cases}
    traces, verified = [], []
    for row in rows:
        case = by_case[row["case_id"]]
        if any(row[key] != case[key] for key in evidence.CASE_FIELDS):
            raise ValueError("Original matched endpoints and lighting are required")
        for key in ("reference_geometry", "scoring_labels", "scoring_geometry"):
            path_asset = case.get(key)
            if path_asset is not None and guard.sha(path_asset) != case["input_sha256"][path_asset]:
                raise ValueError("Frozen scoring asset changed: " + path_asset)
        traces.append(dict(profile=row["profile"], case_id=row["case_id"],
                           observed=audit_sun_trace(row, case, support[row["case_id"]])))
        verified.append(dict(profile=row["profile"], track=TRACK, **_replay_metrics(row, case)))
    exact_cases = []
    for a, b in zip(rows[:len(cases)], rows[len(cases):], strict=True):
        plan = support[a["case_id"]]
        if plan["component_indices"] == [plan["nearest_index"]]:
            exact_match = False
            if a["inference_failure"] is None and b["inference_failure"] is None:
                with np.load(a["saved_arrays"], allow_pickle=False) as first, np.load(b["saved_arrays"], allow_pickle=False) as second:
                    exact_match = all(np.array_equal(first[k], second[k]) for k in ("reference_xy", "query_xy", "pose"))
            exact_cases.append(dict(case_id=a["case_id"], exact_native_outputs=exact_match))
    guard.write(output, dict(passed=True, replayed_rows=len(verified), rows=verified, table=summary["table"],
        track=TRACK, strata=strata(cases, rows, support), metadata_support=p["metadata_support"], planned_and_observed_trace=traces,
        exact_single_reference_control=exact_cases, protocol_sha256=guard.sha(path), summary_sha256=seal["summary_sha256"],
        watchdog_sha256=seal["watchdog_sha256"], accepted_results=False, full_run_authorized=False, scope=SCOPE,
        support_scope="All unsupported and failed cases are retained. Common support is the pre-inference horizontal bracket, not a pose-output filter."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "replay"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--protocol", required=True)
        if action in ("run", "worker"):
            cmd.add_argument("--publication-commit", required=True)
        if action == "prepare":
            cmd.add_argument("--settings", required=True)
        elif action == "run":
            cmd.add_argument("--python", default=sys.executable)
        elif action == "worker":
            cmd.add_argument("--controller-pid", type=int, required=True)
        else:
            cmd.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "run":
        check = isolated(guard.check_common_protocol, TRACKS=[TRACK])
        controller = isolated(guard.guarded_run, TRACKS=[TRACK], check_common_protocol=check)
        controller(args.protocol, args.publication_commit, args.python, admission, SOURCE)
    elif args.action == "worker":
        worker = isolated(guard.guarded_worker, evaluate_profiles=evaluate_known_profiles)
        worker(args.protocol, args.publication_commit, args.controller_pid, admission, Backend)
    else:
        replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
