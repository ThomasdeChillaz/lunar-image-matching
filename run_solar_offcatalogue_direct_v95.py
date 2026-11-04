"""Paired direct relighting with a globally fixed even-index render library."""

import argparse
from pathlib import Path
import sys

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, guarded_run, read, sha, utc_now, validate_dispatch, write,
)
from run_solar_full_direct_render_v92 import FULL_CEILING, completion, frozen as original_full
from run_solar_matched_direct_render_v91 import budget, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import _replay_metrics, guarded_aligned_worker


BASELINE = "reports/solar-full-direct-render-v92.json"
BASELINE_SHA = "9421f06bdaad9d3b8b9da8e424d4d21b6e5ceda90f6a1d228848842d24c0f909"
BASELINE_REPLAY = "reports/solar_full_direct_render_replay_v92.json"
BASELINE_REPLAY_SHA = "b11cc65d41367d82c85fc770b109a83fcd4fb65dca2a7da7c56ae039725af558"
EVEN = tuple(range(0, 24, 2))
NAME = "reference_camera_render_match_even12"
PURPOSE = "offcatalogue_direct_render_development_v95"
OUTCOMES = ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap", "correspondence")


def restrict_libraries(libraries):
    """Keep the same 12 reference-role indices for every camera."""
    restricted = {}
    for key, library in libraries.items():
        entries = library["entries"]
        if (len(entries) != 24 or {entry["light_index"] for entry in entries} != set(range(24))
                or any(entry["source_role"] != "reference" or entry["primary_generic"] is not True for entry in entries)):
            raise ValueError("Exact qualified 24-reference-render library required before restriction")
        selected = [entry for entry in entries if entry["light_index"] in EVEN]
        if len(selected) != len(EVEN):
            raise ValueError("Exact fixed global even-index catalogue required")
        restricted[key] = dict(library, entries=selected)
    if len(restricted) != 32:
        raise ValueError("Preserve all 32 original reference cameras")
    return restricted


def even_backend(profile):
    from solar_evaluation_v70 import Backend

    if (profile["name"] != NAME or profile["backend"] != "direct_render"
            or tuple(profile["reference_library_indices"]) != EVEN):
        raise ValueError("Exact even-catalogue direct-render profile required")
    backend = Backend(profile)
    # Backend constructs and validates the original complete profile first.
    # This per-instance replacement is finished before InputLoader sees a case.
    backend.model.libraries = restrict_libraries(backend.model.libraries)
    backend.provenance = dict(backend.provenance, active_reference_library_indices=list(EVEN),
                              active_reference_renders_per_camera=12, restricted_before_any_query=True)
    return backend


def baseline_evidence():
    if sha(BASELINE) != BASELINE_SHA or sha(BASELINE_REPLAY) != BASELINE_REPLAY_SHA:
        raise ValueError("Exact original v92 full evaluation and independent replay required")
    old = read(BASELINE)
    expected, cases = original_full(old["settings"])
    if (old["prepared_only"] is not False or any(old.get(k) != v for k, v in expected.items())
            or old["max_calendar_seconds"] != 1256):
        raise ValueError("Original v92 full run and measured cap changed")
    summary, _, seal, rows, _, files = completion(BASELINE, old)
    replay = read(BASELINE_REPLAY)
    if (replay["passed"] is not True or replay["replayed_rows"] != 384 or len(replay["rows"]) != 384
            or replay["protocol_sha256"] != BASELINE_SHA or replay["summary_sha256"] != seal["summary_sha256"]
            or replay["watchdog_sha256"] != seal["watchdog_sha256"] or replay["table"] != summary["table"]
            or replay["accepted_results"] is not False or summary["software"] != old["expected_software"]):
        raise ValueError("Complete independently replayed v92 evidence required")
    for row, check in zip(rows, replay["rows"], strict=True):
        if check["case_id"] != row["case_id"] or any(check[key] != row["metrics"][key] for key in OUTCOMES):
            raise ValueError("Original v92 replayed row differs")
    evidence = {path: sha(path) for path in files.values()}
    merge_pins(evidence, {BASELINE: BASELINE_SHA, BASELINE_REPLAY: BASELINE_REPLAY_SHA})
    return old, cases, rows, evidence


def frozen(settings):
    if not isinstance(settings, dict) or set(settings) != {"output", "publication_dispatch_lease_seconds"}:
        raise ValueError("Set only a new output and lease; keep the original v92 cap")
    root = relative(settings["output"])
    lease = settings["publication_dispatch_lease_seconds"]
    if root == Path("runs") or not root.is_relative_to("runs") or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Use an unused runs directory and finite original lease")
    old, cases, reference_rows, evidence = baseline_evidence()
    if str(root) == old["output"]:
        raise ValueError("Preserve the original v92 full outputs")
    sources, public = (dict(old[field]) for field in ("source_sha256", "metadata_sha256"))
    library = read(old["profiles"][0]["render_evidence"]["manifest"])
    excluded = {entry["image"] for view in library["libraries"] for entry in view["entries"]
                if entry["light_index"] not in EVEN}
    if len(excluded) != 384 or any(excluded.intersection(case["input_sha256"]) for case in cases):
        raise ValueError("No odd reference render may be an input to the new comparison")
    # Preserve the exact original library manifest and receipts but do not open
    # omitted odd-reference PNGs even in the new stage's input preflight.
    inputs = {path: digest for path, digest in old["input_sha256"].items() if path not in excluded}
    name = Path(__file__).name
    merge_pins(sources, {name: sha(name)})
    merge_pins(public, {BASELINE: BASELINE_SHA})
    merge_pins(inputs, evidence)
    profile = dict(old["profiles"][0], name=NAME, reference_library_indices=list(EVEN))
    protocol = dict(old)
    for field in ("prepared_only", "created_utc", "latest_start_utc", "absolute_deadline_utc", "initial_reservations"):
        protocol.pop(field)
    protocol.update(
        settings=settings, output=settings["output"], purpose=PURPOSE, publication_dispatch_lease_seconds=lease,
        profiles=[profile], source_sha256=sources, metadata_sha256=public, input_sha256=inputs,
        original_full24_artifacts=evidence, original_full24_profile=old["profiles"][0]["name"],
        cost_scope="Original v92 full24 cost estimate and 1256-second cap retained. The fixed even12 library uses fewer reference comparisons and the same final pose-candidate cap.",
        scope="Fresh primary model-selection results on all 384 original cases with a globally fixed even reference-render subset.",
    )
    return protocol, cases, reference_rows


def prepare(settings_path, path):
    relative(path)
    p, _, _ = frozen(read(settings_path))
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve previous outputs")
    entries, remaining = budget(p)
    if p["max_calendar_seconds"] > remaining:
        raise ValueError("Original unchanged cap exceeds the remaining allowance")
    p.update(prepared_only=True, initial_reservations=entries, created_utc=None,
             latest_start_utc=None, absolute_deadline_utc=None)
    write(path, p)


def admission(path, commit, started):
    from direct_render_adapter_v73 import validate_render_profile
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    p = read(path)
    deadline = validate_dispatch(p, started, FULL_CEILING)

    def check():
        if utc_now() >= deadline:
            raise TimeoutError("Original off-catalogue allowance exhausted")

    check()
    expected, cases, _ = frozen(p["settings"])
    if p.get("prepared_only") is not False or any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Published fixed-catalogue comparison changed")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, digest in p["input_sha256"].items():
        check()
        if sha(asset) != digest:
            raise ValueError("Original frozen input changed: " + asset)
    _, libraries, _ = validate_render_profile(p["profiles"][0])
    restrict_libraries(libraries)
    if software_identity() != p["expected_software"]:
        raise ValueError("Original measured CPU/software stratum required")
    entries, remaining = budget(p)
    initial = p["initial_reservations"]
    if entries[:len(initial)] != initial:
        raise ValueError("Original canonical reservation history changed")
    cap = p["max_calendar_seconds"]
    owned = [entry for entry in entries if entry["job_id"] == sha(path) + JOB_SUFFIX]
    if len(owned) > 1 or any(entry["category"] != "baseline" or entry["seconds"] != cap for entry in owned):
        raise ValueError("Existing charge differs from unchanged original cap")
    if not owned and cap > remaining:
        raise ValueError("Unchanged cap exceeds the remaining baseline allowance")
    check()
    return p, cases, deadline


def raw_counts(rows):
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    return dict(
        positives=len(positives), negatives=len(negatives),
        correct=sum(row["metrics"]["correct"] for row in positives),
        strict=sum(row["metrics"]["strict"] for row in positives),
        wrong_positive=sum(row["metrics"]["wrong_positive"] for row in positives),
        raw_nonoverlap=sum(row["metrics"]["raw_nonoverlap"] for row in negatives),
        inference_failures=sum(row["inference_failure"] is not None for row in rows),
    )


def replay(path, output):
    from solar_evaluation_v70 import aggregate_rows, paired_component_bootstrap

    relative(path)
    relative(output)
    p = read(path)
    expected, cases, reference_rows = frozen(p["settings"])
    if p.get("prepared_only") is not False or any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Original fixed-catalogue comparison identities differ")
    summary, _, seal, rows, _, _ = completion(path, p)
    if summary["software"] != p["expected_software"] or summary["table"] != aggregate_rows(rows):
        raise ValueError("Full software identity or sealed aggregate differs")
    checks = []
    for row, case in zip(rows, cases, strict=True):
        for asset, digest in case["input_sha256"].items():
            if sha(asset) != digest:
                raise ValueError("Original full case input changed")
        if row["inference_failure"] is None:
            search = row["render_search"]
            if (search["library_entries_considered"] != 12 or search["selected_light_index"] not in EVEN
                    or search["common_pnp_candidates"] != 8 or search["final_common_pnp_calls"] != 1
                    or search["exact_known_sun"] is not False):
                raise ValueError("Successful row did not execute the fixed even12 primary search")
        checks.append(_replay_metrics(row, case))
    strata = {}
    for label, parity in (("overall", None), ("even_query_index", 0), ("odd_query_index", 1)):
        subset = [row for row in rows if parity is None or row["light_b"] % 2 == parity]
        prior = [row for row in reference_rows if parity is None or row["light_b"] % 2 == parity]
        if [r["case_id"] for r in subset] != [r["case_id"] for r in prior]:
            raise ValueError("Exact paired case order required")
        candidate, baseline = raw_counts(subset), raw_counts(prior)
        interval = paired_component_bootstrap(subset + prior, NAME, p["original_full24_profile"])
        if interval["terrain_components"] != 8:
            raise ValueError("Preserve original eight terrain groups")
        strata[label] = dict(even12=candidate, full24=baseline,
                            paired_correct_delta=(candidate["correct"] - baseline["correct"]) / candidate["positives"],
                            paired_terrain_percentiles=interval)
    write(output, dict(
        passed=True, replayed_rows=len(checks), rows=checks, table=summary["table"], paired_by_query_index=strata,
        protocol_sha256=sha(path), summary_sha256=seal["summary_sha256"],
        watchdog_sha256=seal["watchdog_sha256"], accepted_results=False,
        scope="Independent saved metrics and paired raw outcomes on all original development cases.",
        parity_scope="Odd query indices are excluded from the even12 reference catalogue. In this manifest they also coincide with the changed-light conditions.",
    ))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "replay"):
        cmd = sub.add_parser(action)
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
    args = cli.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, Path(__file__).name)
    elif args.action == "worker":
        guarded_aligned_worker(args.protocol, args.publication_commit, args.controller_pid, admission, even_backend)
    else:
        replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
