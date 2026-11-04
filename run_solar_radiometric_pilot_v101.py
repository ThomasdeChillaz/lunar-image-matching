"""Bounded fresh matched development pilot for blind exposure normalization."""

import argparse
from collections import Counter
from datetime import timedelta
from functools import lru_cache
import math
from pathlib import Path
from statistics import median
import sys

from run_solar_crosslight_mechanism_v98 import verify_saved_final_call
from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, fixed_v86_cases, guarded_run, read, sha, utc_now, validate_dispatch, write,
)
from run_solar_full_crosslight_rescue_v99 import (
    CASE_FIELDS, FULL_CEILING, completion, outcomes, repeatability, timing_projection, verify_replayed,
)
from run_solar_matched_direct_render_v91 import BASIS, budget, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import _replay_metrics, guarded_aligned_worker
import solar_radiometric_control_v101 as radiometry


SOURCE = Path(__file__).name
PURPOSE = "matched_radiometric_control_cost_pilot_v101"
V99 = "reports/solar-full-crosslight-rescue-dispatch-v99.json"
V99_SHA256 = "5ca6a4a5de7c7dcd3987db92007d7485074485bb746ebcecf5efef8b08612817"
V99_REPLAY = "reports/solar_full_crosslight_rescue_replay_v99.json"
V99_REPLAY_SHA256 = "8347b60efa4d5f3721f2e118087475ec791339c595f43e2cd204b2d678312182"
PROFILES = (radiometry.DIRECT, radiometry.NORMALIZED)
SCOPE = "Exact v100 metadata-only cohort, already observed as development; 24 positives and 24 matched nonoverlaps."
TIMING = "Both fresh arms have identical observation logging, 24 renders and nine instrumented pose calls per successful inference."
# These are the exact 24 positive IDs produced by the independent v100
# metadata selector. The existing manifest supplies their paired negatives.
POSITIVE_IDS = (
    "u013-l00-00-positive", "u014-l16-01-positive", "u015-l00-17-positive", "u020-l08-17-positive",
    "u021-l16-16-positive", "u022-l16-01-positive", "u024-l08-17-positive", "u025-l00-00-positive",
    "u027-l08-01-positive", "u057-l16-16-positive", "u058-l00-17-positive", "u059-l08-01-positive",
    "u061-l00-17-positive", "u062-l00-00-positive", "u063-l16-01-positive", "u068-l16-16-positive",
    "u069-l16-01-positive", "u070-l08-17-positive", "u072-l08-01-positive", "u073-l08-17-positive",
    "u075-l00-00-positive", "u093-l08-01-positive", "u094-l16-16-positive", "u095-l00-17-positive",
)


def cohort(full, audit):
    positive_ids = set(POSITIVE_IDS)
    positives = [case for case in full if case["case_id"] in positive_ids and case["positive"]]
    units = {item["unit_index"]: item for item in audit["units"] if item["split"] == "model_selection"}
    terrain = Counter(case["component"] for case in positives)
    lighting = Counter((case["light_a"], case["light_b"]) for case in positives)
    families = Counter(units[case["unit_index"]]["family"] for case in positives)
    if (tuple(case["case_id"] for case in positives) != POSITIVE_IDS or len(terrain) != 8 or set(terrain.values()) != {3}
            or len(lighting) != 6 or set(lighting.values()) != {4}
            or families != {"training_motion": 12, "unseen_roll": 12} or len({c["unit_index"] for c in positives}) != 24):
        raise ValueError("Exact balanced v100 metadata-only positive cohort required")
    cases = [case for case in full if case["case_id"] in positive_ids or case.get("matched_positive_case") in positive_ids]
    negatives = [case for case in cases if not case["positive"]]
    if len(cases) != 48 or {case["matched_positive_case"] for case in negatives} != positive_ids:
        raise ValueError("Every original matched nonoverlap must stay in the fixed pilot")
    for fields in (("reference", "query"), ("reference_geometry", "solar_lookup_protocol")):
        if tuple(cases[0][k] for k in fields) == tuple(cases[-1][k] for k in fields):
            raise ValueError("Original loader cache must be cold across arms")
    return cases


def pilot_cost(rows, watchdog, n_cases):
    if len(rows) != 768:
        raise ValueError("Complete measured 768-cell predecessor grid required")
    for row in rows:
        for field in ("profile_load_seconds", "case_calendar_seconds"):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Finite measured predecessor wall times required")
    original = [row for row in rows if row["profile"] == radiometry.DIRECT]
    if len(original) != 384 or any(row["inference_failure"] is not None for row in original):
        raise ValueError("Complete measured original direct grid required")
    load = max(row["profile_load_seconds"] for row in original)
    peak = max(row["case_calendar_seconds"] for row in original)
    covered = sum(row["profile_load_seconds"] + row["case_calendar_seconds"] for row in rows)
    overhead = max(0., watchdog["calendar_seconds"] - covered)
    cap = math.ceil(1.5 * (2 * overhead + 2 * load + n_cases * (peak + 2 * peak))) + 5
    if not 5 < cap <= FULL_CEILING:
        raise ValueError("Exploratory allowance exceeds original full ceiling")
    return dict(v99_direct_load_seconds=load, v99_direct_worst_case_seconds=peak,
                v99_unallocated_seconds=overhead, treatment_pre_lunar_case_bound_seconds=2 * peak,
                conservative_controller_calendar_seconds=cap,
                rule="1.5× (twice v99 outside time and load, plus 48 worst direct and 48 double-worst treatment rows) + 5 seconds.")


def library_inventory(profile):
    evidence = profile["render_evidence"]
    if sha(evidence["manifest"]) != evidence["manifest_sha256"]:
        raise ValueError("Original reference-only library inventory changed")
    manifest = read(evidence["manifest"])
    if manifest["query_truth_used"] or manifest["query_role_images_used"] or manifest["calibration_or_final_access"]:
        raise ValueError("Original reference-only development library required")
    return {lib["reference_geometry_sha256"]: lib for lib in manifest["libraries"]}


def frozen(settings):
    if not isinstance(settings, dict) or set(settings) != {"output", "publication_dispatch_lease_seconds"}:
        raise ValueError("Set only a fresh output and lease; prior measured cost fixes the pilot cap")
    output = relative(settings["output"])
    lease = settings["publication_dispatch_lease_seconds"]
    if not output.is_relative_to("runs/solar-radiometric-pilot-v101") or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Versioned v101 output and original finite lease required")
    basis, _, full = fixed_v86_cases(BASIS, full=True)
    cases = cohort(full, read(basis["geometry_audit"]))
    if sha(V99) != V99_SHA256 or sha(V99_REPLAY) != V99_REPLAY_SHA256:
        raise ValueError("Exact terminal v99 and independent replay required")
    prior = read(V99)
    summary, watchdog, seal, prior_rows, receipts = completion(V99, prior)
    independent = verify_replayed(V99_REPLAY, V99, summary, seal, prior_rows, two_arms=True)
    if prior["case_ids"] != [case["case_id"] for case in full] or prior["calibration_or_final_access"]:
        raise ValueError("Exact original primary-development v99 precursor required")
    profiles = list(radiometry.matched_profiles(prior["profiles"][0]))
    timing = pilot_cost(prior_rows, watchdog, len(cases))
    sources, metadata = (dict(prior[field]) for field in ("source_sha256", "metadata_sha256"))
    backend = Path(radiometry.__file__)
    merge_pins(sources, {SOURCE: sha(Path(__file__)), backend.name: sha(backend)})
    merge_pins(metadata, {V99: V99_SHA256, V99_REPLAY: V99_REPLAY_SHA256})
    assets = dict(metadata)
    merge_pins(assets, {str(path): sha(path) for path in receipts})
    old = {row["case_id"]: row for row in prior_rows if row["profile"] == radiometry.DIRECT}
    libraries = library_inventory(profiles[0])
    for case in cases:
        merge_pins(assets, case["input_sha256"])
        lib = libraries[case["reference_asset_metadata"]["source_reference_geometry_sha256"]]
        merge_pins(assets, {lib["source_summary"]: lib["source_summary_sha256"]})
        merge_pins(assets, {entry["image"]: entry["image_sha256"] for entry in lib["entries"]})
        previous = old[case["case_id"]]
        merge_pins(assets, {previous["saved_arrays"]: previous["saved_arrays_sha256"]})
    p = {k: prior[k] for k in ("basis", "manifest", "geometry_audit", "full_case_count", "compute_ledger",
                              "compute_ledger_sha256", "reservation_journal", "blender_executable", "scoring_revision")}
    p.update(**settings, settings=settings, purpose=PURPOSE, profiles=profiles, tracks=["estimated_sun"],
             split="model_selection", cpu_threads=2, calibration_or_final_access=False, accepted_results=False,
             automatic_model_promotion=False, expected_software=summary["software"], source_sha256=sources,
             metadata_sha256=metadata, input_sha256=assets, max_calendar_seconds=timing["conservative_controller_calendar_seconds"],
             case_ids=[case["case_id"] for case in cases], cost_basis=timing, case_source="fixed exact v100 metadata cohort",
             full_run_authorized=False, scope=SCOPE, timing_scope=TIMING)
    checked = {row["case_id"]: row for row in independent["rows"] if row["profile"] == radiometry.DIRECT}
    return p, cases, old, checked


def prepare(settings, path):
    relative(path)
    p, *_ = frozen(read(settings))
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve every earlier attempt")
    entries, remaining = budget(p)
    if p["max_calendar_seconds"] > remaining:
        raise ValueError("Original canonical allowance cannot admit this pilot")
    p.update(prepared_only=True, initial_reservations=entries, created_utc=None, latest_start_utc=None, absolute_deadline_utc=None)
    write(path, p)


def exact(p):
    expected, cases, old, checked = frozen(p["settings"])
    staged = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (p.get("prepared_only") is not False or set(p) != set(expected) | staged
            or any(p[k] != v for k, v in expected.items())):
        raise ValueError("Exact published v101 pilot identity required")
    return cases, old, checked


def admission(path, commit, started):
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    p = read(path)
    deadline = validate_dispatch(p, started, FULL_CEILING)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original pilot allowance exhausted")

    cases, _, _ = exact(p)
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, expected in p["input_sha256"].items():
        check()
        if sha(asset) != expected:
            raise ValueError("Frozen case or precursor evidence changed: " + asset)
    if software_identity() != p["expected_software"]:
        raise ValueError("Original measured CPU/software stratum required")
    entries, remaining = budget(p)
    initial, cap = p["initial_reservations"], p["max_calendar_seconds"]
    own = [row for row in entries if row["job_id"] == sha(path) + JOB_SUFFIX]
    if entries[:len(initial)] != initial or len(own) > 1 or any(
        e["category"] != "baseline" or e["seconds"] != cap or e["refund_on_failure"] for e in own
    ):
        raise ValueError("Original canonical baseline accounting changed")
    if not own and cap > remaining:
        raise ValueError("Measured pilot cap exceeds canonical allowance")
    check()
    return p, cases, deadline


@lru_cache(maxsize=1024)
def image_statistics(path, checksum):
    """Independently compute the discrete rank and histogram accounting without SIFT."""
    import numpy as np
    from PIL import Image

    if sha(path) != checksum:
        raise ValueError("Original grayscale input changed: " + path)
    with Image.open(path) as source:
        if source.size != (512, 512) or source.mode not in ("L", "RGB"):
            raise ValueError("Exact original native grayscale-convertible image required")
        pixels = np.asarray(source.convert("L"), np.uint8).ravel().copy()
    rank = math.ceil(199 * len(pixels) / 200)
    white = int(np.partition(pixels, rank - 1)[rank - 1])
    tied = int((pixels == white).sum())
    return dict(p995_input_code=white, p995_input_rank=rank, percentile_plateau_pixels=tied,
                percentile_has_duplicate_codes=tied > 1, percentile_plateau_fraction=tied / len(pixels),
                input_distinct_codes=len(np.unique(pixels)), input_black_pixels=int((pixels == 0).sum()),
                input_saturated_pixels=int((pixels == 255).sum()), input_above_white_pixels=int((pixels > white).sum()),
                pixel_count=len(pixels))


def image_trace(path, checksum, observed, enabled):
    expected = dict(image_statistics(path, checksum))
    code = expected["p995_input_code"]
    if not enabled:
        mode = "identity_control"
    elif expected["input_distinct_codes"] == 1:
        mode = "uniform_unchanged"
    elif code in (0, 255):
        mode = "zero_white_unchanged" if code == 0 else "full_white_unchanged"
    else:
        mode = "scaled"
    changed = mode == "scaled"
    clipped = expected["input_above_white_pixels"] if changed else 0
    saturated = expected["percentile_plateau_pixels"] + clipped if changed else expected["input_saturated_pixels"]
    expected.update(action=mode, gain=255. / code if changed else 1., clipped_pixels=clipped, output_saturated_pixels=saturated)
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved exposure telemetry differs from the exact image: " + path)


def audit_trace(row, case, library):
    if row["inference_failure"] is not None:
        return None
    search = row["render_search"]
    trace = search["radiometry"]
    enabled = row["profile"] == radiometry.NORMALIZED
    if (trace["policy"] != (radiometry.POLICY if enabled else radiometry.CONTROL_POLICY) or not trace["same_native_fov"]
            or not search["reference_viewpoint_only"] or search["exact_known_sun"] or search["library_entries_considered"] != 24
            or search["common_pnp_candidates"] != 8 or search["final_common_pnp_calls"] != 1
            or row["raw_inference_result"]["query_truth_used"] is not False):
        raise ValueError("Original 24-light native primary pose budget and fixed exposure policy required")
    query = case["query"]
    image_trace(query, case["input_sha256"][query], trace["query"], enabled)
    query_keypoints = trace["query"]["detected_keypoints"]
    if type(query_keypoints) is not int or query_keypoints < 0:
        raise ValueError("Finite nonnegative SIFT query count required")
    refs = trace["reference_renders"]
    if [observed["light_index"] for observed in refs] != [entry["light_index"] for entry in library["entries"]]:
        raise ValueError("Every exact original reference light must appear in order")
    counts = {}
    for observed, entry in zip(refs, library["entries"], strict=True):
        image_trace(entry["image"], entry["image_sha256"], observed, enabled)
        count = observed["sift_ratio_unique_matches"]
        if type(count) is not int or not 0 <= count <= query_keypoints:
            raise ValueError("Reference ratio match count exceeds unique query support")
        counts[entry["light_index"]] = count
    calls = search["observed_pose_calls"]
    if ([call["pairs"] for call in calls[:8]] != sorted(counts.values(), reverse=True)[:8]
            or counts[search["selected_light_index"]] != calls[-1]["pairs"]):
        raise ValueError("Original SIFT-ranked pose input counts differ from saved telemetry")
    verify_saved_final_call(row)
    return trace


def exact_arrays(a, b):
    import numpy as np

    with np.load(a["saved_arrays"], allow_pickle=False) as first, np.load(b["saved_arrays"], allow_pickle=False) as second:
        for name in ("reference_xy", "query_xy", "pose"):
            old, new = first[name], second[name]
            if old.shape != new.shape or old.dtype != new.dtype or old.tobytes() != new.tobytes():
                return False
    return True


def mechanism(cases, current, traces):
    from solar_evaluation_v70 import paired_component_bootstrap

    half = len(cases)
    first, second = current[:half], current[half:]
    pairs = []
    for case, a, b, x, y in zip(cases, first, second, traces[:half], traces[half:], strict=True):
        def summarize(trace):
            if trace is None:
                return None
            refs = trace["reference_renders"]
            return dict(query=trace["query"], reference_renders=refs,
                        best_light_match_count=max(ref["sift_ratio_unique_matches"] for ref in refs))
        pairs.append(dict(case_id=case["case_id"], positive=case["positive"], light_a=case["light_a"], light_b=case["light_b"],
                          component=case["component"], query_component=case["query_component"], control=summarize(x),
                          exposure=summarize(y), control_inference_failure=a["inference_failure"],
                          exposure_inference_failure=b["inference_failure"], control_correct=a["metrics"]["correct"],
                          exposure_correct=b["metrics"]["correct"], exposure_wrong_positive=b["metrics"]["wrong_positive"],
                          exposure_raw_nonoverlap=b["metrics"]["raw_nonoverlap"]))
    strata = {}
    for name in ("all", "same_light", "changed_light"):
        def includes(record):
            return name == "all" or (record["light_a"] == record["light_b"]) == (name == "same_light")
        control, exposure = ([r for r in arm if includes(r)] for arm in (first, second))
        paired = [r for r in pairs if includes(r)]
        seen = [r for r in paired if r["control"] is not None and r["exposure"] is not None]
        strata[name] = dict(control=outcomes(control), exposure=outcomes(exposure), pairs_with_both_traces=len(seen),
            new_correct=sum(r["exposure_correct"] and not r["control_correct"] for r in paired),
            lost_correct=sum(r["control_correct"] and not r["exposure_correct"] for r in paired),
            query_keypoint_medians=None if not seen else [median(r[arm]["query"]["detected_keypoints"] for r in seen)
                                                         for arm in ("control", "exposure")],
            newly_six_raw_matches=sum(r["control"]["best_light_match_count"] < 6 <= r["exposure"]["best_light_match_count"]
                                      for r in seen),
            exposure_query_clip_fraction_median=None if not seen else median(
                r["exposure"]["query"]["clipped_pixels"] / r["exposure"]["query"]["pixel_count"] for r in seen),
            interval=paired_component_bootstrap(control + exposure, radiometry.NORMALIZED, radiometry.DIRECT))
    return strata, pairs


def replay(path, output):
    relative(path)
    relative(output)
    p = read(path)
    cases, historical, historical_metrics = exact(p)
    summary, watchdog, seal, rows, _ = completion(path, p)
    index = {case["case_id"]: case for case in cases}
    libraries = library_inventory(p["profiles"][0])
    checked, traces = [], []
    image_statistics.cache_clear()
    for row in rows:
        case = index[row["case_id"]]
        if any(row[field] != case[field] for field in CASE_FIELDS):
            raise ValueError("Case endpoints and labels must match exact frozen identities")
        for field in ("reference_geometry", "scoring_labels", "scoring_geometry"):
            asset = case.get(field)
            if asset is not None and sha(asset) != case["input_sha256"][asset]:
                raise ValueError("Frozen geometry or scoring input changed: " + asset)
        checked.append(dict(profile=row["profile"], **_replay_metrics(row, case)))
        library = libraries[case["reference_asset_metadata"]["source_reference_geometry_sha256"]]
        traces.append(audit_trace(row, case, library))
    strata, pairs = mechanism(cases, rows, traces)
    baseline = rows[:len(cases)]
    identities = repeatability(checked, [historical_metrics[row["case_id"]] for row in baseline])
    different = [row["case_id"] for row in baseline if not exact_arrays(row, historical[row["case_id"]])]
    identities.update(exact_native_arrays=len(baseline) - len(different), different_native_arrays=different)
    if different or identities["differences"] or any(row["inference_failure"] is not None for row in rows):
        projection = dict(available=False, reason="Exact original direct outputs or successful measured arm grid missing.")
    else:
        measured = timing_projection(rows, p["full_case_count"], p["profiles"], watchdog["calendar_seconds"])
        cap = measured["conservative_controller_calendar_seconds"]
        _, remaining = budget(p)
        projection = dict(available=True, **measured, fits_original_ceiling=cap <= FULL_CEILING,
                          fits_current_baseline_allowance=cap <= remaining)
    write(output, dict(passed=True, replayed_rows=len(checked), rows=checked, table=summary["table"], strata=strata,
                       paired_radiometry=pairs, original_v99_direct_repeatability=identities, protocol_sha256=sha(path),
                       summary_sha256=seal["summary_sha256"], watchdog_sha256=seal["watchdog_sha256"],
                       maximum_full_calendar_seconds=FULL_CEILING, original_full_cost_projection=projection,
                       full_run_authorized=False, accepted_results=False, timing_scope=TIMING, scope=SCOPE,
                       trace_scope="Pixel percentiles, clipping, solver counts and scores replay independently. SIFT was not rerun."))


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
            cmd.add_argument("--controller-pid", required=True, type=int)
        else:
            cmd.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, SOURCE)
    elif args.action == "worker":
        guarded_aligned_worker(args.protocol, args.publication_commit, args.controller_pid, admission, radiometry.RadiometricDirectBackend)
    else:
        replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
