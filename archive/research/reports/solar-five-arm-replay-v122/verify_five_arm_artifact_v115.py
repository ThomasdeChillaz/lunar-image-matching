"""Verify exact official v115 artifacts and replay saved rows without a model."""
import argparse
import base64
from datetime import datetime, timedelta, timezone
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
from zipfile import ZipFile


REPOSITORY = "ThomasdeChillaz/lunar-image-matching"
WORKFLOW = ".github/workflows/solar-remote-five-arm-v115.yml"
RELAY = "solar_remote_relay_five_arm_v115.py"
TICKET = "reports/solar-remote-arm-five-v115-ticket.json"
PROTOCOL = "reports/solar-five-arm-v115/comparison-protocol.json"
AUTH = "reports/solar-five-arm-v115/authorization.json"
OUTPUT = "runs/solar-five-arm-v115/comparison"
CONTEXT = TICKET + ".WORKER_CONTEXT.json"
HARDWARE = "runs/solar-arm-five-hardware-v115.json"
COHORT = "reports/solar_matched_arm_cohorts_v103.json"
REPLAY = "verify_solar_matched_arm_v113.py"
REPORTER = "summarize_solar_five_arm_v115.py"
CAP = 18000
FILE_COUNT = 30000
EXPANDED = 2**31
META_LIMIT = 8 * 1024**2
BASE_MANIFEST = "825ca518c562e269c6128fd485d133aef9091f7eb43c07c8a0261868c757522a"
BASE_ARCHIVE = "2d25dca8b56f86f10444289b7458aeed4ba9a6755bf1ff19fca7014afaa1dee0"
BASE_BYTES = 563675388
BASENAMES = {
    "execution.json", "reservation.json", "watchdog_admission.json", "worker.log",
    "summary.json", "COMPLETE.json", "WORKER_FAILED.json", "FAILED.json",
    "actual_ARM_Direct_control.json",
}
CELL_NAME = re.compile(r"[0-9a-f]{64}\.(?:npz|json|COMPLETE\.json|provenance\.json|actual_ARM_control\.json)")
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


def need(condition, error):
    if not condition:
        raise ValueError(error)


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def digest(raw):
    return sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def utc(value):
    result = datetime.fromisoformat(value)
    need(result.tzinfo is not None, "Expected original timezone-aware timestamp")
    return result.astimezone(timezone.utc)


def relative(name):
    path = PurePosixPath(name)
    need(path.parts and not path.is_absolute() and ".." not in path.parts
         and str(path) == name and "\\" not in name and all(ord(c) >= 32 for c in name),
         "Unsafe path: " + name)
    return path


def scoped_file(root, name):
    path = root / relative(name)
    need(path.is_file() and path.resolve().is_relative_to(root), "Missing or unsafe path: " + name)
    return path


def fetch(url, limit=META_LIMIT):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "lunar-five-arm-v115-artifact-verifier",
    })
    with urllib.request.urlopen(request, timeout=20) as response:
        need(response.geturl() == url, "Public GitHub metadata unexpectedly redirected")
        raw = response.read(limit + 1)
    need(len(raw) <= limit, "Public GitHub metadata exceeded its bound")
    return raw


def api(suffix):
    return json.loads(fetch("https://api.github.com/repos/" + REPOSITORY + suffix))


def public_blob(commit, name, expected):
    need(HEX40.fullmatch(commit) and HEX64.fullmatch(expected), "Missing exact Git binding")
    raw = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{relative(name)}")
    need(digest(raw) == expected, "Exact public Git blob changed: " + name)
    return raw


def bind_ticket(ticket_commit, expected_ticket_sha, get=api, get_blob=public_blob):
    signed = get("/git/commits/" + ticket_commit)
    verification = signed["verification"]
    need(signed["sha"] == ticket_commit and verification["verified"] is True
         and verification["reason"] == "valid", "GitHub did not verify the exact ticket commit")
    ticket_raw = get_blob(ticket_commit, TICKET, expected_ticket_sha)
    ticket = json.loads(ticket_raw)
    nonce = ticket["nonce"]
    source_commit = ticket["source_publication_commit"]
    need(HEX64.fullmatch(nonce) and HEX40.fullmatch(source_commit), "Invalid immutable ticket identity")
    branch = "solar-five-arm-v115-ticket-" + nonce
    title = "solar-five-arm-v115-" + nonce
    fixed = {
        "schema_version": 115, "repository": REPOSITORY, "workflow": WORKFLOW,
        "workflow_name": "Solar five-arm native CPU comparison v115",
        "transport": "v115_signed_five_arm_v98_and_v113_immutable_data",
        "authorized_attempt": 1, "protocol_path": PROTOCOL, "max_calendar_seconds": CAP,
        "required_runner": "ubuntu-24.04-arm", "cpu_threads": 2,
        "expected_branch": branch, "expected_run_title": title,
        "refunds": False, "automatic_retry": False, "primary_only": True,
    }
    need(all(ticket.get(key) == val for key, val in fixed.items())
         and ticket["base"]["archive_sha256"] == BASE_ARCHIVE
         and ticket["base"]["manifest_sha256"] == BASE_MANIFEST
         and ticket["base"]["archive_bytes"] == BASE_BYTES, "Unexpected v115 ticket contract")
    if source_commit not in [parent["sha"] for parent in signed["parents"]]:
        comparison = get(f"/compare/{source_commit}...{ticket_commit}")
        need(comparison["status"] == "ahead" and comparison["merge_base_commit"]["sha"] == source_commit,
             "The reviewed scientific publication is outside signed ticket history")
    protocol_raw = get_blob(source_commit, PROTOCOL, ticket["protocol_sha256"])
    need(get_blob(ticket_commit, PROTOCOL, ticket["protocol_sha256"]) == protocol_raw,
         "Ticket checkout changed the frozen protocol")
    protocol = json.loads(protocol_raw)
    need(protocol["output"] == OUTPUT and protocol["max_calendar_seconds"] == CAP
         and protocol["purpose"] == "matched_development_comparison"
         and protocol["stage_mode"] == "matched_modern_direct_arm_v113"
         and protocol["split"] == "model_selection" and protocol["tracks"] == ["estimated_sun"]
         and protocol["comparison_case_count"] == 96 and len(set(protocol["comparison_case_ids"])) == 96
         and protocol["case_cache_seals"] == {} and protocol["accepted_results"] is False
         and protocol["calibration_or_final_access"] is False
         and ticket["direct_supplement"] == protocol["direct_supplement"],
         "Signed source does not specify the 96-case primary comparison")
    inline = ticket["authorization"]
    auth_raw = base64.b64decode(inline["inline_base64"], validate=True)
    need(inline["path"] == AUTH and digest(auth_raw) == inline["sha256"], "Inline charge evidence changed")
    auth = json.loads(auth_raw)
    charge = auth["reservation"]["reservation"]
    journal_raw = base64.b64decode(auth["canonical_journal_bytes_base64"], validate=True)
    journal = json.loads(journal_raw)
    nonce_data = {"protocol_sha256": ticket["protocol_sha256"], "canonical_original_entry": charge}
    canonical = json.dumps(nonce_data, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode()
    entries = journal["reservations"]
    need(auth["schema_version"] == 75 and auth["repository"] == REPOSITORY
         and auth["protocol_path"] == PROTOCOL and auth["protocol_sha256"] == ticket["protocol_sha256"]
         and auth["source_publication_commit"] == source_commit and auth["authorized_attempt"] == 1
         and auth["canonical_journal"] == protocol["reservation_journal"]
         and auth["refunds"] is False and auth["automatic_retry"] is False
         and auth["case_ids"] == protocol["case_ids"] and auth["tracks"] == protocol["tracks"]
         and auth["max_calendar_seconds"] == CAP
         and auth["original_latest_start_utc"] == ticket["original_latest_start_utc"] == protocol["latest_start_utc"]
         and auth["original_absolute_deadline_utc"] == ticket["original_absolute_deadline_utc"] == protocol["absolute_deadline_utc"]
         and journal["ledger_sha256"] == protocol["compute_ledger_sha256"]
         and digest(journal_raw) == auth["canonical_journal_sha256"]
         and entries == [*protocol["baseline_admission"]["initial_reservations"], charge]
         and len({entry["job_id"] for entry in entries}) == len(entries)
         and charge["job_id"] == ticket["protocol_sha256"] + "/evaluation"
         and charge["category"] == "baseline" and charge["seconds"] == CAP
         and charge["refund_on_failure"] is False and auth["nonce"] == nonce
         and digest(b"solar-remote-charge-v75:" + canonical) == nonce,
         "Signed v115 one-use charge and original baseline journal differ")
    return {"ticket": ticket, "auth": auth, "ticket_commit": ticket_commit,
            "ticket_sha256": expected_ticket_sha, "ticket_raw": ticket_raw,
            "auth_raw": auth_raw, "protocol": protocol, "protocol_raw": protocol_raw}


def metadata(run_id, signed, get=api):
    ticket = signed["ticket"]
    commit = signed["ticket_commit"]
    workflow = get("/actions/workflows/" + PurePosixPath(WORKFLOW).name)
    run = get(f"/actions/runs/{run_id}")
    repository = run["repository"]
    need(run["id"] == run_id and run["workflow_id"] == workflow["id"] and workflow["path"] == WORKFLOW
         and run["path"].split("@")[0] == WORKFLOW and run["run_attempt"] == 1
         and run["event"] == "workflow_dispatch" and run["head_sha"] == commit
         and run["head_branch"] == ticket["expected_branch"] and run["display_title"] == ticket["expected_run_title"]
         and repository["full_name"] == REPOSITORY and repository["private"] is False
         and run["head_repository"]["id"] == repository["id"],
         "Public run does not match the exact first v115 dispatch")
    result = dict(run_id=run_id, workflow_id=workflow["id"], status=run["status"], conclusion=run["conclusion"],
                  run_started_at=run["run_started_at"], html_url=run["html_url"], ticket_commit=commit,
                  ticket_sha256=signed["ticket_sha256"], protocol_sha256=ticket["protocol_sha256"],
                  source_commit=ticket["source_publication_commit"], nonce=ticket["nonce"], artifacts={})
    if run["status"] != "completed":
        return result
    jobs = get(f"/actions/runs/{run_id}/attempts/1/jobs")
    matches = [entry for entry in jobs["jobs"] if entry["name"] == "native-cpu"]
    need(len(matches) == 1 or (not matches and run["conclusion"] != "success"), "Expected original native CPU job")
    if matches:
        job, = matches
        need(job["run_id"] == run_id and (job["conclusion"] == "success") == (run["conclusion"] == "success"),
             "Public job and workflow completion status disagree")
        result["job"] = {"id": job["id"], "conclusion": job["conclusion"],
                         "steps": [{k: item[k] for k in ("name", "conclusion")} for item in job["steps"]]}
    rows = get(f"/actions/runs/{run_id}/artifacts?per_page=100")
    need(len(rows["artifacts"]) == rows["total_count"], "Incomplete public artifact list")
    artifact_stem = f"{ticket['expected_run_title']}-{run_id}-attempt-1"
    for role, name in (("outer", artifact_stem), ("partial", artifact_stem + "-partial")):
        matches = [item for item in rows["artifacts"] if item["name"] == name]
        if not matches:
            continue
        need(len(matches) == 1, "More than one public artifact claims the exact attempt")
        item, = matches
        verified = get(f"/actions/artifacts/{item['id']}")
        keys = ("id", "name", "size_in_bytes", "digest", "expired", "archive_download_url")
        origin = verified["workflow_run"]
        need(all(item.get(key) == verified.get(key) for key in keys)
             and verified["expired"] is False and type(verified["size_in_bytes"]) is int
             and 0 < verified["size_in_bytes"] <= EXPANDED + 32 * 1024**2
             and re.fullmatch(r"sha256:[0-9a-f]{64}", verified["digest"] or "")
             and origin["id"] == run_id and origin["head_sha"] == commit
             and origin["head_branch"] == ticket["expected_branch"] and origin["repository_id"] == repository["id"]
             and origin["head_repository_id"] == repository["id"],
             "Public artifact digest, origin or identity differs")
        result["artifacts"][role] = {key: verified[key] for key in keys}
    if run["conclusion"] == "success":
        need(set(result["artifacts"]) == {"outer"}, "Successful v115 requires exactly the official outer artifact")
    return result


def scientific_name(name):
    basename = name.removeprefix(OUTPUT + "/")
    return name != basename and (basename in BASENAMES or CELL_NAME.fullmatch(basename) is not None)


def safe_entries(archive, translate, maximum=EXPANDED):
    entries = archive.infolist()
    need(len(entries) <= FILE_COUNT and sum(item.file_size for item in entries) <= maximum,
         "ZIP count or expanded bytes exceeded the scientific bound")
    mapped = {}
    for item in entries:
        relative(item.filename)
        name = translate(item.filename)
        need(name is not None and name not in mapped and not item.is_dir()
             and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG),
             "Unexpected, nonregular, or duplicate ZIP member: " + item.filename)
        mapped[name] = item
    return mapped


def extract(archive, item, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(item) as source, target.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    need(target.stat().st_size == item.file_size and (expected is None or sha(target) == expected),
         "Extracted file size or digest differs: " + item.filename)


def archive_download_matches(path, row):
    need(path.stat().st_size == row["size_in_bytes"] and "sha256:" + sha(path) == row["digest"],
         "Exact original ZIP does not match GitHub's size and digest")


def validate_watchdog(watchdog, signed, official):
    ticket = signed["ticket"]
    commit = signed["ticket_commit"]
    started = utc(official["run_started_at"])
    effective = min(started + timedelta(seconds=CAP), utc(ticket["original_absolute_deadline_utc"]))
    expected = {
        "workflow_run_id": official["run_id"], "run_attempt": 1, "workflow_source_commit": commit,
        "reviewed_ticket_commit": commit, "reviewed_branch": ticket["expected_branch"],
        "source_publication_commit": ticket["source_publication_commit"], "nonce": ticket["nonce"],
        "ticket_nonce": ticket["nonce"], "ticket_path": TICKET, "ticket_sha256": signed["ticket_sha256"],
        "protocol_sha256": ticket["protocol_sha256"], "authorization_sha256": ticket["authorization"]["sha256"],
        "original_reservation": signed["auth"]["reservation"], "automatic_retry": False,
        "reservation_refunded": False, "primary_only": True, "cap_seconds": CAP,
        "relay_source_sha256": signed["protocol"]["source_sha256"][RELAY],
        "workflow_source_sha256": signed["protocol"]["source_sha256"][WORKFLOW],
    }
    need(all(watchdog.get(key) == val for key, val in expected.items())
         and utc(watchdog["workflow_started_utc"]) == utc(watchdog["started_utc"])
         == utc(watchdog["outer_started_utc"]) == started
         and utc(watchdog["absolute_deadline_utc"]) == utc(watchdog["outer_absolute_deadline_utc"]) == effective,
         "Outer watchdog identity or actual GitHub wall clock differs")
    if official["conclusion"] == "success":
        need(watchdog["complete"] is True and watchdog["child_returncode"] == 0
             and watchdog["failure"] is None and watchdog["archive_failure"] is None
             and 0 < watchdog["calendar_seconds"] < min(CAP, (effective - started).total_seconds()),
             "Successful public run has an unsuccessful or over-budget outer watchdog")


def preserve_outer(path, destination, signed, official):
    nonce = signed["ticket"]["nonce"]
    remote = f"runs/solar-remote-five-arm-v115/{nonce}"
    watchdog_name = remote + "/remote_watchdog.json"
    nested_name = remote + "/scientific_results.zip"
    outer_allow = {HARDWARE, CONTEXT, watchdog_name, nested_name, remote + "/controller.log"}
    success = official["conclusion"] == "success"
    with ZipFile(path) as outer:
        # The upload action removes its common search root if it found only
        # one file on a setup failure. Canonicalize the sole hardware basename.
        def outer_name(name):
            if name in outer_allow:
                return name
            if not success and name == PurePosixPath(HARDWARE).name:
                return HARDWARE
            return None

        for name, entry in safe_entries(outer, outer_name, EXPANDED + 32 * 1024**2).items():
            extract(outer, entry, destination / name)
    watchdog_path = destination / watchdog_name
    nested = destination / nested_name
    if not watchdog_path.is_file():
        need(not success and not nested.exists(), "Missing original outer watchdog")
        return None
    watchdog = read(watchdog_path)
    validate_watchdog(watchdog, signed, official)
    result = watchdog["result_artifact"]
    if result is None:
        need(not success, "Completed v115 watchdog must bind the exact nested archive")
        return None
    need(result["path"] == nested_name and result["scientific_output"] == OUTPUT
         and result["includes_failed_partial_outputs"] is True
         and result["input_capsule_included"] is False and result["model_weights_included"] is False
         and sha(nested) == result["sha256"], "Nested archive identity or scientific output policy differs")
    signed_files = {PROTOCOL: signed["protocol_raw"], TICKET: signed["ticket_raw"], AUTH: signed["auth_raw"]}
    with ZipFile(nested) as scientific:
        entries = safe_entries(scientific, lambda name: name if name in signed_files or scientific_name(name) else None)
        need(len(entries) == result["file_count"]
             and sum(item.file_size for item in entries.values()) == result["expanded_bytes"],
             "Nested archive coverage differs from the outer watchdog")
        for name, raw in signed_files.items():
            need(name in entries and entries[name].file_size == len(raw) and scientific.read(entries[name]) == raw,
                 "Nested archive signed member differs: " + name)
        for name, entry in entries.items():
            if scientific_name(name):
                extract(scientific, entry, destination / name)
    if success:
        validate_complete(destination, signed, official, watchdog)
    return nested, watchdog_path


def preserve_partial(path, destination):
    with ZipFile(path) as partial:
        def translate(name):
            # GitHub's single-directory upload can strip the full directory.
            candidates = (name, OUTPUT + "/" + name, "runs/" + name, "runs/solar-five-arm-v115/" + name)
            return next((item for item in candidates if scientific_name(item)), None)

        for name, entry in safe_entries(partial, translate).items():
            extract(partial, entry, destination / name)


def validate_complete(destination, signed, official, watchdog):
    summary_path = destination / OUTPUT / "summary.json"
    summary = read(summary_path)
    seal = read(destination / OUTPUT / "COMPLETE.json")
    hardware = read(destination / HARDWARE)
    context = read(destination / CONTEXT)
    ticket = signed["ticket"]
    need(seal["complete"] is True and seal["external_watchdog_passed"] is True
         and seal["protocol_sha256"] == summary["protocol_sha256"] == ticket["protocol_sha256"]
         and seal["summary_sha256"] == sha(summary_path)
         and not (destination / OUTPUT / "FAILED.json").exists()
         and summary["fresh_inference_cases"] == 480 and summary["cached_cases"] == 0
         and len(summary["rows"]) == 480 and summary["accepted_results"] is False
         and summary["calibration_or_final_access"] is False
         and summary["case_ids"] == signed["protocol"]["comparison_case_ids"]
         and summary["profiles"] == signed["protocol"]["profiles"]
         and summary["source_publication_commit"] == ticket["source_publication_commit"]
         and hardware["machine"] == "aarch64" and hardware["native_bf16"] is True
         and context["ticket_sha256"] == signed["ticket_sha256"] and context["workflow_run_id"] == official["run_id"]
         and context["overlay"]["manifest_sha256"] == ticket["assembled_manifest_sha256"]
         and len(context["portable_catalogue_archives"]) == 32
         and read(destination / OUTPUT / "reservation.json") == signed["auth"]["reservation"],
         "Successful completion, native hardware, setup or case-count evidence differs")
    for key in ("outer_started_utc", "outer_absolute_deadline_utc", "controller_started_utc"):
        need(context[key] == watchdog[key], "Worker and parent timing identity differs: " + key)
    expected = [(profile["name"], case, "estimated_sun") for profile in summary["profiles"] for case in summary["case_ids"]]
    need([(row["profile"], row["case_id"], row["track"]) for row in summary["rows"]] == expected,
         "Successful result does not have the original ordered matched row grid")
    for row in summary["rows"]:
        name = row["saved_arrays"]
        need(scientific_name(name) and name.endswith(".npz") and row["reused_case_cache"] is False,
             "Unsafe, stale or missing case prediction binding")
        need(sha(scoped_file(destination, name)) == row["saved_arrays_sha256"], "Saved array differs: " + name)


def bytes_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def stage_sources(repository, destination, signed):
    sources = signed["protocol"]["source_sha256"]
    scripts = (REPLAY, REPORTER, "solar_evaluation_v70.py", "solar_aligned_scoring_v91.py",
               "plan_solar_matched_arm_v113.py")
    need(all(name in sources for name in scripts), "Independent reporter and replay are missing from the freeze")
    stage = destination / "frozen_source"
    for name, expected in sources.items():
        need(HEX64.fullmatch(expected), "Invalid frozen source SHA256")
        path = repository / relative(name)
        if path.is_file() and path.resolve().is_relative_to(repository) and sha(path) == expected:
            raw = path.read_bytes()
        else:
            raw = subprocess.check_output([
                "git", "-C", str(repository), "show", f"{signed['ticket']['source_publication_commit']}:{name}",
            ], stderr=subprocess.DEVNULL, timeout=15)
        need(digest(raw) == expected, "Source differs from the pre-inference freeze: " + name)
        bytes_new(stage / name, raw)
    return stage


def selected_inputs(protocol, manifest, cohort):
    need(manifest["split"] == "model_selection"
         and [case["case_id"] for case in manifest["cases"]] == protocol["case_ids"]
         and protocol["comparison_case_ids"] == cohort["subsets"]["96"]["case_ids"]
         and protocol["manifest"] == cohort["manifest"] and protocol["geometry_audit"] == cohort["geometry_audit"],
         "Exact frozen eligibility and cohort manifest differ")
    expected = protocol["input_sha256"]
    paths = {protocol["manifest"], protocol["geometry_audit"]}
    cases = {case["case_id"]: case for case in manifest["cases"]}
    for identifier in protocol["comparison_case_ids"]:
        for name, check in cases[identifier]["input_sha256"].items():
            need(expected.get(name) == check, "Frozen case and protocol input digests disagree: " + name)
            paths.add(name)
    return {name: expected[name] for name in sorted(paths)}


def stage_inputs(repository, base, destination, signed, include_case_inputs):
    protocol = signed["protocol"]
    expected = protocol["input_sha256"]
    cohort_path = scoped_file(repository, COHORT)
    cohort_raw = cohort_path.read_bytes()
    need(digest(cohort_raw) == expected[COHORT], "Frozen cohort digest differs")
    bytes_new(destination / COHORT, cohort_raw)
    manifest_name = protocol["manifest"]
    if not include_case_inputs:
        manifest = scoped_file(repository, manifest_name).read_bytes()
        need(digest(manifest) == expected[manifest_name], "Frozen case manifest differs")
        bytes_new(destination / manifest_name, manifest)
        selected_inputs(protocol, json.loads(manifest), json.loads(cohort_raw))
        return {COHORT: expected[COHORT], manifest_name: expected[manifest_name]}

    if base.is_dir():
        original_manifest = scoped_file(base, "manifest.json")
        need(sha(original_manifest) == BASE_MANIFEST, "Original private v98 base manifest digest differs")
        old_files = read(original_manifest)["files"]
        payload = (base / "payload").resolve()
        raw = scoped_file(payload, manifest_name).read_bytes()
        need(digest(raw) == expected[manifest_name], "Frozen case manifest differs")
        paths = selected_inputs(protocol, json.loads(raw), json.loads(cohort_raw))
        for name, check in paths.items():
            need(old_files.get(name) == check, "Selected input is outside original v98 closure: " + name)
            source = scoped_file(payload, name)
            need(sha(source) == check, "Frozen v98 case data changed: " + name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
    else:
        need(base.is_file() and base.stat().st_size == BASE_BYTES and sha(base) == BASE_ARCHIVE,
             "Original public v98 ZIP size or digest differs")
        with ZipFile(base) as archive:
            entries = safe_entries(archive, lambda name: name if name == "manifest.json"
                                   or name.startswith("payload/") else None, 4 * 2**30)
            member = entries["payload/" + manifest_name]
            need(member.file_size <= META_LIMIT, "Selected manifest exceeds metadata bound")
            raw = archive.read(member)
            need(digest(raw) == expected[manifest_name], "Frozen case manifest differs")
            paths = selected_inputs(protocol, json.loads(raw), json.loads(cohort_raw))
            for name, check in paths.items():
                extract(archive, entries["payload/" + name], destination / name, check)
    return paths | {COHORT: expected[COHORT]}


def no_model_script(stage, script, destination, *arguments):
    need(script in (REPLAY, REPORTER), "Only saved-array replay and frozen aggregation are permitted")
    excluded = {"GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"}
    env = {key: value for key, value in os.environ.items() if key not in excluded}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    subprocess.run([sys.executable, "-B", "-E", "-s", str(stage / script), *map(str, arguments)],
                   cwd=destination, env=env, check=True, stdout=subprocess.DEVNULL, timeout=1800)


def replay(destination, stage, nested, watchdog):
    independently_replayed = destination / "independent_replay_v115.json"
    aggregated = destination / "matched_comparison_v115.json"
    no_model_script(stage, REPLAY, destination, OUTPUT, "--protocol", PROTOCOL, "--output", independently_replayed)
    replayed = read(independently_replayed)
    need(replayed["passed"] is True and replayed["replayed_rows"] == 480,
         "Saved-array replay did not validate 480 rows")
    no_model_script(stage, REPORTER, destination, OUTPUT, "--root", destination, "--protocol", PROTOCOL,
                    "--output", aggregated, "--replay", independently_replayed, "--watchdog", watchdog,
                    "--archive", nested)
    return independently_replayed, aggregated


def inventory_partial(root, stage, rows, label):
    receipt = root / (label + "_cell_inventory_v115.json")
    no_model_script(stage, REPORTER, root, rows, "--root", root, "--protocol", PROTOCOL,
                    "--output", receipt, "--partial")
    return receipt, read(receipt)


def preserve(args, signed, official):
    repository, destination = args.repository.resolve(), args.output.resolve()
    need(repository.is_dir() and not destination.is_relative_to(repository) and not destination.exists(),
         "Use a fresh evidence directory outside the scientific checkout")
    need(official["status"] == "completed", "Only terminal official GitHub runs can be preserved")
    provided = {role: getattr(args, role).resolve() for role in ("outer", "partial") if getattr(args, role)}
    need(provided and set(provided) <= set(official["artifacts"]), "Provide ZIPs for the exact official artifact roles")
    for role, path in provided.items():
        archive_download_matches(path, official["artifacts"][role])
    success = official["conclusion"] == "success"
    need(not success or set(provided) == {"outer"}, "Completed successful evidence requires the outer ZIP")
    need(not (args.replay and success) or args.base, "Full saved-array replay needs the original v98 base")
    destination.mkdir(exist_ok=False)
    write_new(destination / "github_v115_metadata.json", official)
    bytes_new(destination / PROTOCOL, signed["protocol_raw"])
    outer = preserve_outer(provided["outer"], destination, signed, official) if "outer" in provided else None
    partial_path = destination / "raw_partial"
    if "partial" in provided:
        preserve_partial(provided["partial"], partial_path)
    records = {}
    if args.replay:
        stage = stage_sources(repository, destination, signed)
        paths = stage_inputs(repository, args.base.resolve() if args.base else None, destination, signed, success)
        write_new(destination / "replay_input_hashes_v115.json", paths)
        if (destination / OUTPUT).is_dir():
            path, inventory = inventory_partial(destination, stage, OUTPUT, "nested")
            records["nested_cell_inventory"] = {"path": path.name, "sha256": sha(path)}
            if success:
                need(inventory["sealed_cells"] == 480 and not inventory["invalid_markers"],
                     "Successful scientific archive lacks 480 individually sealed, correctly bound rows")
        if (partial_path / OUTPUT).is_dir():
            path, _ = inventory_partial(destination, stage, partial_path / OUTPUT, "raw_partial")
            records["raw_partial_cell_inventory"] = {"path": path.name, "sha256": sha(path)}
        if success:
            independently_replayed, aggregated = replay(destination, stage, *outer)
            records.update(replay={"path": independently_replayed.name, "sha256": sha(independently_replayed)},
                           comparison={"path": aggregated.name, "sha256": sha(aggregated)})
    if outer is not None:
        for role, item in zip(("scientific_archive", "outer_watchdog"), outer, strict=True):
            records[role] = {"path": item.relative_to(destination).as_posix(), "sha256": sha(item)}
    receipt = dict(schema="solar_five_arm_official_artifact_audit_v115", run_id=official["run_id"],
                   conclusion=official["conclusion"], complete_archive_verified=success,
                   independently_replayed=success and args.replay, new_inference=False,
                   ticket_sha256=signed["ticket_sha256"], protocol_sha256=signed["ticket"]["protocol_sha256"],
                   archived_downloads={role: official["artifacts"][role] for role in provided}, records=records)
    write_new(destination / "artifact_audit_v115.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--ticket-commit", required=True)
    parser.add_argument("--ticket-sha256", required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--outer", type=Path, help="exact original ZIP from GitHub Actions")
    parser.add_argument("--partial", type=Path, help="exact separate original partial ZIP, if provided")
    parser.add_argument("--output", type=Path, help="fresh isolated evidence directory")
    parser.add_argument("--base", type=Path, help="original v98 capsule directory or original public release ZIP")
    parser.add_argument("--replay", action="store_true", help="replay complete results, or inventory failed partial cells")
    args = parser.parse_args()
    if args.run_id <= 0 or not HEX40.fullmatch(args.ticket_commit) or not HEX64.fullmatch(args.ticket_sha256):
        parser.error("Use the exact official run ID, reviewed ticket commit and predispatch ticket SHA256")
    signed = bind_ticket(args.ticket_commit, args.ticket_sha256)
    official = metadata(args.run_id, signed)
    if args.outer is None and args.partial is None and args.output is None and not args.replay:
        print(json.dumps(official, indent=2))
        return
    if args.output is None or args.repository is None:
        parser.error("Preservation requires --repository and fresh --output")
    print(json.dumps(preserve(args, signed, official), indent=2))


if __name__ == "__main__":
    main()
