"""Signed one-use v115 comparison on a standard public Linux ARM worker."""

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
import urllib.request

from plan_solar_matched_arm_v113 import ARM_NAMES, DIRECT_NAME, outstanding_v112
from solar_capsule_overlay_v110 import BASE
from solar_direct_transport_v113 import RELEASE_TAG, extract, hash_path, inventory
import solar_remote_relay_direct_v113 as previous
import solar_remote_relay_v98 as legacy

WORKFLOW = ".github/workflows/solar-remote-five-arm-v115.yml"
WORKFLOW_NAME = "Solar five-arm native CPU comparison v115"
RUNNER = "run_solar_five_arm_v115.py"
PROTOCOL = "reports/solar-five-arm-v115/comparison-protocol.json"
AUTHORIZATION = "reports/solar-five-arm-v115/authorization.json"
TICKET = "reports/solar-remote-arm-five-v115-ticket.json"
OUTPUT = "runs/solar-five-arm-v115/comparison"
REMOTE = "runs/solar-remote-five-arm-v115"
RUN_PREFIX = "solar-five-arm-v115-"
BRANCH_PREFIX = "solar-five-arm-v115-ticket-"
TRANSPORT = "v115_signed_five_arm_v98_and_v113_immutable_data"
REPOSITORY = legacy.REPOSITORY
CANONICAL_JOURNAL = legacy.CANONICAL_JOURNAL
CAP = 18000
PREDECESSOR_RUNS = (41926059385, 41928100316, 41930272861, 41938284734)
TOKEN_ENV = ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN")
TICKET_KEYS = previous.TICKET_KEYS


def qualifier():
    import qualify_five_arm_v115 as module

    if (module.PROTOCOL, module.AUTHORIZATION, module.OUTPUT, module.BASE) != (PROTOCOL, AUTHORIZATION, OUTPUT, BASE):
        raise ValueError("v115 relay and qualifier identities differ")
    return module


def primary_scope(protocol_path, protocol, at=None, check_assets=True):
    if str(protocol_path) != PROTOCOL:
        raise ValueError("Exact v115 protocol path required")
    return qualifier().primary_scope(protocol_path, protocol, at=at, check_assets=check_assets)


def _scope_wire(protocol):
    """Stdlib-compatible rejection before installing any worker dependencies."""
    from solar_direct_transport_v113 import validate_binding

    subset = protocol.get("comparison_case_ids", [])
    if (
        protocol.get("prepared_only") is not False or protocol.get("output") != OUTPUT
        or protocol.get("stage_mode") != "matched_modern_direct_arm_v113"
        or protocol.get("purpose") != "matched_development_comparison"
        or protocol.get("split") != "model_selection" or protocol.get("cpu_threads") != 2
        or protocol.get("tracks") != ["estimated_sun"]
        or protocol.get("accepted_results") is not False or protocol.get("calibration_or_final_access") is not False
        or type(protocol.get("max_calendar_seconds")) is not int or protocol["max_calendar_seconds"] != CAP
        or protocol.get("reservation_journal") != CANONICAL_JOURNAL
        or len(protocol.get("case_ids", [])) != 384 or len(set(protocol["case_ids"])) != 384
        or protocol.get("comparison_case_count") != 96 or len(subset) != 96 or len(set(subset)) != 96
        or not set(subset) <= set(protocol["case_ids"]) or protocol.get("case_cache_seals")
        or tuple(item["name"] for item in protocol.get("profiles", [])) != (*ARM_NAMES, DIRECT_NAME)
    ):
        raise ValueError("Only the source-frozen 96-case, five-arm primary v115 protocol may execute")
    supplement = protocol["direct_supplement"]
    if set(supplement) != {"part", "manifest_sha256"}:
        raise ValueError("Exactly one original Direct supplement required")
    validate_binding(supplement["part"], supplement["manifest_sha256"])


def _reserve_frozen_history(protocol, protocol_digest, source_commit):
    """Append one exclusive charge while holding the original canonical lock."""
    path = Path(CANONICAL_JOURNAL)
    expected = protocol["baseline_admission"]["initial_reservations"]
    job_id = protocol_digest + "/evaluation"
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = legacy.read(path)
        if (
            journal["reservations"] != expected
            or journal["ledger_sha256"] != protocol["compute_ledger_sha256"]
            or any(entry["job_id"] == job_id for entry in expected)
        ):
            raise ValueError("Canonical journal changed since freeze; no v115 charge made")
        if legacy.sha(protocol["compute_ledger"]) != journal["ledger_sha256"]:
            raise ValueError("Canonical budget ledger changed after source publication")
        ledger = legacy.read(protocol["compute_ledger"])
        charged = sum(entry["seconds"] for entry in expected if entry["category"] == "baseline")
        ledger_available = ledger["remaining_baseline_wall_seconds"] - charged
        qualified_available = protocol["baseline_admission"]["remaining_at_preparation_seconds"]
        remaining_after = qualified_available - CAP
        if ledger.get("accounting_complete") is not True or remaining_after < 0 or qualified_available > ledger_available:
            raise ValueError("The original canonical baseline budget cannot cover v115")
        legacy._alive_lease({
            "original_latest_start_utc": protocol["latest_start_utc"],
            "original_absolute_deadline_utc": protocol["absolute_deadline_utc"],
        })
        entry = {
            "job_id": job_id, "category": "baseline", "seconds": CAP,
            "reserved_utc": datetime.now(timezone.utc).isoformat(), "refund_on_failure": False,
        }
        snapshot = previous._bytes(dict(journal, reservations=[*expected, entry]))
        authorization = {
            "schema_version": 75, "repository": REPOSITORY, "protocol_path": PROTOCOL,
            "protocol_sha256": protocol_digest, "source_publication_commit": source_commit,
            "nonce": legacy.charge_nonce(protocol_digest, entry), "authorized_attempt": 1,
            "reservation": {"reservation": entry, "remaining_after_reservation": remaining_after},
            "canonical_journal": CANONICAL_JOURNAL,
            "canonical_journal_sha256": previous._digest(snapshot),
            "canonical_journal_bytes_base64": base64.b64encode(snapshot).decode(),
            "original_latest_start_utc": protocol["latest_start_utc"],
            "original_absolute_deadline_utc": protocol["absolute_deadline_utc"],
            "max_calendar_seconds": CAP, "case_ids": protocol["case_ids"], "tracks": ["estimated_sun"],
            "created_utc": datetime.now(timezone.utc).isoformat(), "refunds": False, "automatic_retry": False,
        }
        previous._review_inline_bytes(previous._bytes(authorization))
        legacy.write_new(AUTHORIZATION + ".ISSUING.json", {
            "protocol_sha256": protocol_digest, "no_retry": True,
            "predecessor_qualification_runs": list(PREDECESSOR_RUNS[-2:]),
        })
        temp = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
                temp = Path(stream.name)
                stream.write(snapshot)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        legacy.write_new(AUTHORIZATION, authorization)
    return {"authorization": AUTHORIZATION, "nonce": authorization["nonce"], "seconds": CAP}


def pre_reserve(source_commit, base_capsule, supplement_archive, api=None, verify_head=None):
    """One separately authorized transaction; no launch or worker mutation."""
    from solar_learning_pipeline_v73 import load_contract, remaining
    from solar_portable_evidence_v75 import pin

    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise ValueError("A worker cannot write the local canonical journal")
    _local_attempt_absent()
    api = public_repository_api if api is None else api
    verify_head = legacy.verify_release_download_head if verify_head is None else verify_head
    legacy._hex(source_commit, legacy.HEX40, "Exact public v115 source commit required")
    protocol = legacy.read(PROTOCOL)
    _scope_wire(protocol)
    part = protocol["direct_supplement"]["part"]
    if Path(supplement_archive).stat().st_size != part["bytes"] or hash_path(supplement_archive) != part["sha256"]:
        raise ValueError("Exact local Direct supplement must still match the protocol")
    journal = legacy.read(CANONICAL_JOURNAL)
    _require_predecessors(journal)
    _public_repo_and_workflow(api)
    ref = api("/git/ref/heads/main")
    if (
        ref.get("ref") != "refs/heads/main" or ref.get("object", {}).get("type") != "commit"
        or ref.get("object", {}).get("sha") != source_commit
    ):
        raise ValueError("Reviewed v115 source and protocol must be the exact public main commit")
    _check_public_assets(protocol["direct_supplement"], api)
    verify_head(previous._science_part())
    verify_head(part)
    base_capsule = Path(base_capsule)
    pin(base_capsule, "manifest.json", BASE["manifest_sha256"])
    pin(base_capsule / "payload", BASE["protocol_path"], BASE["protocol_sha256"])
    pin(base_capsule / "payload", BASE["authorization_path"], BASE["authorization_sha256"])
    legacy.published(PROTOCOL, protocol, source_commit)
    primary_scope(PROTOCOL, protocol)
    budget = remaining(load_contract(protocol["learning_contract"]))["baseline"]
    if budget < CAP or budget != protocol["baseline_admission"]["remaining_at_preparation_seconds"]:
        raise ValueError("Live canonical baseline no longer matches the published freeze")
    return _reserve_frozen_history(protocol, legacy.sha(PROTOCOL), source_commit)


def _local_attempt_absent():
    if any(Path(path).exists() for path in (
        OUTPUT, AUTHORIZATION, AUTHORIZATION + ".ISSUING.json", AUTHORIZATION + ".TICKET_ISSUED.json",
        TICKET, TICKET + ".DISPATCHED.json",
    )):
        raise FileExistsError("Preserve any previous v115 authorization, output or dispatch marker")


def _require_predecessors(journal):
    rows = qualifier().verify_predecessors(journal=journal)
    if len(rows) != 2 or rows[0] == rows[1] or any(journal["reservations"].count(row) != 1 for row in rows):
        raise ValueError("Both original completed ARM qualification debits must remain exactly once")
    if outstanding_v112(Path.cwd(), journal) != 0:
        raise ValueError("The preregistered stopped v112 commitment is not independently resolved")


def _inline(ticket):
    entry = ticket["authorization"]
    if set(entry) != {"path", "sha256", "inline_base64"} or entry["path"] != AUTHORIZATION:
        raise ValueError("Exact distinct v115 authorization path required")
    raw = base64.b64decode(entry["inline_base64"], validate=True)
    auth = previous._review_inline_bytes(raw)
    if previous._digest(raw) != legacy._hex(entry["sha256"], legacy.HEX64, "Authorization digest required"):
        raise ValueError("Inline authorization differs from the signed digest")
    return raw, auth


def _overlay_binding(ticket):
    return {
        "base": ticket["base"], "direct_supplement": ticket["direct_supplement"], "protocol_path": ticket["protocol_path"],
        "protocol_sha256": ticket["protocol_sha256"], "authorization_path": ticket["authorization"]["path"],
        "authorization_sha256": ticket["authorization"]["sha256"], "source_publication_commit": ticket["source_publication_commit"],
        "nonce": ticket["nonce"], "assembled_manifest_sha256": ticket["assembled_manifest_sha256"],
    }


def freeze_ticket(base_capsule):
    if Path(TICKET).exists() or Path(AUTHORIZATION + ".TICKET_ISSUED.json").exists():
        raise FileExistsError("Preserve any previous v115 signed or prepared ticket")
    raw = Path(AUTHORIZATION).read_bytes()
    auth = previous._review_inline_bytes(raw)
    legacy.validate_authorization(auth, PROTOCOL, require_local_journal=True)
    _require_predecessors(legacy.read(CANONICAL_JOURNAL))
    protocol = legacy.read(PROTOCOL)
    _scope_wire(protocol)
    qualifier()._scope_metadata(protocol)
    ticket = {
        "schema_version": 115, "repository": REPOSITORY, "workflow": WORKFLOW,
        "workflow_name": WORKFLOW_NAME, "transport": TRANSPORT, "nonce": auth["nonce"], "authorized_attempt": 1,
        "source_publication_commit": auth["source_publication_commit"], "protocol_path": PROTOCOL,
        "protocol_sha256": auth["protocol_sha256"], "authorization": {
            "path": AUTHORIZATION, "sha256": previous._digest(raw), "inline_base64": base64.b64encode(raw).decode(),
        },
        "base": BASE, "direct_supplement": protocol["direct_supplement"], "assembled_manifest_sha256": None,
        "original_latest_start_utc": auth["original_latest_start_utc"],
        "original_absolute_deadline_utc": auth["original_absolute_deadline_utc"],
        "max_calendar_seconds": CAP, "required_runner": "ubuntu-24.04-arm", "cpu_threads": 2,
        "expected_run_title": RUN_PREFIX + auth["nonce"], "expected_branch": BRANCH_PREFIX + auth["nonce"],
        "refunds": False, "automatic_retry": False, "primary_only": True,
    }
    legacy._alive_lease(ticket)
    manifest, _ = qualifier().plan_overlay(base_capsule, Path.cwd(), Path.cwd(), _overlay_binding(ticket))
    ticket["assembled_manifest_sha256"] = previous._digest(manifest)
    legacy._alive_lease(ticket)
    legacy.write_new(AUTHORIZATION + ".TICKET_ISSUED.json", {
        "ticket_path": TICKET, "nonce": auth["nonce"], "no_reissue": True,
    })
    legacy.write_new(TICKET, ticket)
    return {"ticket": TICKET, "bytes": Path(TICKET).stat().st_size, "nonce": auth["nonce"]}


def validate_ticket(ticket_path, expected_sha=None, expected_nonce=None):
    if str(ticket_path) != TICKET or not 0 < Path(ticket_path).stat().st_size <= previous.MAX_TICKET_BYTES:
        raise ValueError("Exact bounded v115 ticket path required")
    if expected_sha is not None and legacy.sha(ticket_path) != legacy._hex(
        expected_sha, legacy.HEX64, "Exact signed v115 ticket digest required",
    ):
        raise ValueError("Signed ticket bytes differ")
    ticket = legacy.read(ticket_path)
    nonce = legacy._hex(ticket.get("nonce"), legacy.HEX64, "Exact fresh v115 nonce required")
    if (
        set(ticket) != TICKET_KEYS or ticket["schema_version"] != 115
        or ticket["repository"] != REPOSITORY or ticket["workflow"] != WORKFLOW
        or ticket["workflow_name"] != WORKFLOW_NAME or ticket["transport"] != TRANSPORT
        or ticket["protocol_path"] != PROTOCOL or ticket["base"] != BASE
        or ticket["required_runner"] != "ubuntu-24.04-arm" or ticket["cpu_threads"] != 2
        or ticket["authorized_attempt"] != 1 or ticket["max_calendar_seconds"] != CAP
        or ticket["refunds"] is not False or ticket["automatic_retry"] is not False
        or ticket["primary_only"] is not True or ticket["expected_run_title"] != RUN_PREFIX + nonce
        or ticket["expected_branch"] != BRANCH_PREFIX + nonce
        or (expected_nonce is not None and nonce != expected_nonce)
    ):
        raise ValueError("Only the fresh one-use v115 ticket is supported")
    legacy._hex(ticket["source_publication_commit"], legacy.HEX40, "Exact v115 source publication required")
    legacy._hex(ticket["assembled_manifest_sha256"], legacy.HEX64, "Exact v115 manifest digest required")
    _, auth = _inline(ticket)
    legacy.validate_authorization(auth, PROTOCOL)
    protocol = legacy.read(PROTOCOL)
    _scope_wire(protocol)
    if (
        ticket["protocol_sha256"] != auth["protocol_sha256"] or ticket["nonce"] != auth["nonce"]
        or ticket["source_publication_commit"] != auth["source_publication_commit"]
        or ticket["original_latest_start_utc"] != auth["original_latest_start_utc"]
        or ticket["original_absolute_deadline_utc"] != auth["original_absolute_deadline_utc"]
        or auth["max_calendar_seconds"] != CAP or ticket["protocol_sha256"] == BASE["protocol_sha256"]
        or ticket["authorization"]["sha256"] == BASE["authorization_sha256"]
        or ticket["direct_supplement"] != protocol["direct_supplement"]
    ):
        raise ValueError("Signed ticket, fresh charge and reviewed v115 source do not bind")
    return ticket


def _history_path(page):
    return "/actions/workflows/" + Path(WORKFLOW).name + "/runs?event=workflow_dispatch&per_page=100&page=" + str(page)


def public_repository_api(suffix):
    allowed = (
        r"(?:/git/ref/heads/(?:main|solar-five-arm-v115-ticket-[0-9a-f]{64})"
        r"|/git/commits/[0-9a-f]{40}|/releases/tags/(?:solar-arm-transport-v98|solar-arm-direct-v113-transport)"
        r"|/actions/workflows/solar-remote-five-arm-v115\.yml"
        r"(?:/runs\?event=workflow_dispatch&per_page=100&page=(?:[1-9]|1[0-9]|20))?)?"
    )
    if re.fullmatch(allowed, suffix) is None:
        raise ValueError("Unexpected public v115 metadata path")
    url = "https://api.github.com/repos/" + REPOSITORY + suffix
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "solar-browser-gate-v115",
    })
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.geturl() != url:
            raise ValueError("Public GitHub metadata unexpectedly redirected")
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("Public GitHub metadata exceeded its bound")
    return json.loads(data)


def _public_repo_and_workflow(api):
    repo = api("")
    workflow = api("/actions/workflows/" + Path(WORKFLOW).name)
    if (
        repo.get("full_name") != REPOSITORY or repo.get("private") is not False
        or repo.get("visibility") != "public" or repo.get("default_branch") != "main"
        or workflow.get("state") != "active" or workflow.get("path") != WORKFLOW
    ):
        raise ValueError("Distinct v115 workflow must be active on the public repository")


def _check_public_assets(supplement, api):
    return previous._public_science_asset(api), previous._public_supplement(supplement, api)


def _one_use_runs(api, title):
    matches = []
    for page in range(1, 21):
        rows = api(_history_path(page))["workflow_runs"]
        matches.extend(row for row in rows if row.get("display_title") == title)
        if len(rows) < 100:
            return matches
    raise ValueError("Workflow history exceeds the bounded v115 uniqueness audit")


def public_dispatch_evidence(ticket, ticket_commit, api=None):
    api = public_repository_api if api is None else api
    legacy._hex(ticket_commit, legacy.HEX40, "Exact reviewed v115 commit required")
    _public_repo_and_workflow(api)
    branch = ticket["expected_branch"]
    ref = api("/git/ref/heads/" + branch)
    commit = api("/git/commits/" + ticket_commit)
    signature = commit.get("verification", {})
    if (
        ref.get("ref") != "refs/heads/" + branch or ref.get("object", {}).get("type") != "commit"
        or ref.get("object", {}).get("sha") != ticket_commit or commit.get("sha") != ticket_commit
        or signature.get("verified") is not True or signature.get("reason") != "valid"
    ):
        raise ValueError("GitHub must verify the exact signed v115 branch commit")
    base, direct = _check_public_assets(ticket["direct_supplement"], api)
    return {
        "ticket_commit": ticket_commit, "branch": branch, "verified_commit": True,
        "release_asset_id": base["id"], "archive_bytes": base["size"], "archive_sha256": BASE["archive_sha256"],
        "direct_supplement_asset_id": direct["id"], "direct_supplement_bytes": direct["size"],
        "direct_supplement_sha256": ticket["direct_supplement"]["part"]["sha256"],
        "prior_successful_arm_run_ids": list(PREDECESSOR_RUNS[-2:]),
    }


def prepare_browser_dispatch(ticket_commit, api=None, verify_head=None):
    """Verify public signed launch readiness and write one manual dispatch marker."""
    api = public_repository_api if api is None else api
    verify_head = legacy.verify_release_download_head if verify_head is None else verify_head
    ticket = validate_ticket(TICKET)
    legacy._alive_lease(ticket)
    raw, auth = _inline(ticket)
    if Path(AUTHORIZATION).read_bytes() != raw:
        raise ValueError("Signed v115 ticket and local precharge differ")
    legacy.validate_authorization(auth, PROTOCOL, require_local_journal=True)
    _require_predecessors(legacy.read(CANONICAL_JOURNAL))
    previous._require_source_ancestor(ticket, ticket_commit)
    legacy.published(PROTOCOL, legacy.read(PROTOCOL), ticket_commit)
    if subprocess.check_output(["git", "show", ticket_commit + ":" + TICKET]) != Path(TICKET).read_bytes():
        raise ValueError("Exact v115 ticket bytes must already be published")
    evidence = public_dispatch_evidence(ticket, ticket_commit, api)
    verify_head(previous._science_part())
    verify_head(ticket["direct_supplement"]["part"])
    if _one_use_runs(api, ticket["expected_run_title"]):
        raise ValueError("This v115 nonce already has a workflow invocation")
    legacy._alive_lease(ticket)
    result = {
        "mode": "prepared_browser_dispatch_once", "reviewed_branch": ticket["expected_branch"],
        "dispatch_url": "https://github.com/" + REPOSITORY + "/actions/workflows/" + Path(WORKFLOW).name,
        "evidence": evidence, "authorized_attempt": 1, "automatic_retry": False,
        "inputs": {
            "ticket": TICKET, "ticket_sha256": legacy.sha(TICKET), "ticket_nonce": ticket["nonce"],
            "ticket_commit": ticket_commit,
        },
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    legacy.write_new(TICKET + ".DISPATCHED.json", result)
    return result


def workflow_binding(ticket, ticket_commit, environment=None, api=legacy.github_api):
    env = os.environ if environment is None else environment
    if (
        env.get("GITHUB_ACTIONS") != "true" or env.get("GITHUB_REPOSITORY") != REPOSITORY
        or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or env.get("GITHUB_RUN_ATTEMPT") != "1"
        or env.get("RUNNER_OS") != "Linux" or env.get("RUNNER_ARCH") != "ARM64"
        or env.get("RUNNER_ENVIRONMENT") != "github-hosted"
        or env.get("GITHUB_REF") != "refs/heads/" + ticket["expected_branch"]
    ):
        raise ValueError("Unique attempt one on standard hosted Linux ARM required")
    run_id = env.get("GITHUB_RUN_ID", "")
    head = legacy._hex(env.get("GITHUB_SHA"), legacy.HEX40, "Exact v115 workflow SHA required")
    if (
        not re.fullmatch(r"[1-9][0-9]*", run_id) or int(run_id) in PREDECESSOR_RUNS
        or head != legacy._hex(ticket_commit, legacy.HEX40, "Exact reviewed v115 commit required")
        or env.get("GITHUB_WORKFLOW_SHA") != head
        or env.get("GITHUB_WORKFLOW_REF") != REPOSITORY + "/" + WORKFLOW + "@" + env["GITHUB_REF"]
    ):
        raise ValueError("Actual v115 workflow and signed checkout bindings differ")
    prefix = "repos/" + REPOSITORY
    row = api(prefix + "/actions/runs/" + run_id)
    if (
        str(row.get("id")) != run_id or row.get("event") != "workflow_dispatch"
        or row.get("run_attempt") != 1 or row.get("head_sha") != head
        or row.get("head_branch") != ticket["expected_branch"] or row.get("path", "").split("@")[0] != WORKFLOW
        or row.get("display_title") != ticket["expected_run_title"]
        or row.get("repository", {}).get("full_name") != REPOSITORY
        or row.get("head_repository", {}).get("full_name") != REPOSITORY
    ):
        raise ValueError("Actual v115 workflow run/source/ticket bindings differ")
    same_nonce = _one_use_runs(lambda suffix: api(prefix + suffix), ticket["expected_run_title"])
    if len(same_nonce) != 1 or str(same_nonce[0]["id"]) != run_id or same_nonce[0].get("run_attempt") != 1:
        raise ValueError("Duplicate invocation, re-dispatch or rerun of a one-use v115 ticket")
    return {
        "workflow_run_id": int(run_id), "run_attempt": 1, "workflow_source_commit": head,
        "source_publication_commit": ticket["source_publication_commit"], "ticket_nonce": ticket["nonce"],
        "reviewed_branch": ticket["expected_branch"], "reviewed_ticket_commit": ticket_commit,
        "html_url": row.get("html_url"), "workflow_started_utc": row["run_started_at"],
    }


def _outer_bounds(ticket, workflow_started, now=None):
    started = legacy.utc(workflow_started)
    latest = legacy.utc(ticket["original_latest_start_utc"])
    absolute = legacy.utc(ticket["original_absolute_deadline_utc"])
    current = datetime.now(timezone.utc) if now is None else now
    cap = ticket["max_calendar_seconds"]
    deadline = min(started + timedelta(seconds=cap), absolute)
    if (
        cap != CAP or started > current or started > latest or current >= deadline
    ):
        raise TimeoutError("v115 original pipeline deadline expired or workflow start is invalid")
    return started, deadline, min(deadline, latest)


def verify_remote(ticket_path, expected_sha, expected_nonce, ticket_commit):
    ticket = validate_ticket(ticket_path, expected_sha, expected_nonce)
    legacy._alive_lease(ticket)
    binding = workflow_binding(ticket, ticket_commit)
    _outer_bounds(ticket, binding["workflow_started_utc"])
    previous._require_source_ancestor(ticket, ticket_commit)
    legacy.verify_ticket_checkout(ticket_path, ticket, ticket_commit)
    public_dispatch_evidence(ticket, ticket_commit)
    legacy.verify_release_download_head(previous._science_part())
    legacy.verify_release_download_head(ticket["direct_supplement"]["part"])
    legacy._alive_lease(ticket)
    _outer_bounds(ticket, binding["workflow_started_utc"])


def _bound_context(ticket_path, protocol_path, *, worker):
    ticket = validate_ticket(ticket_path)
    raw, auth = _inline(ticket)
    if str(protocol_path) != PROTOCOL or Path(AUTHORIZATION).read_bytes() != raw:
        raise ValueError("Installed v115 authorization differs from the signed charge")
    legacy.validate_authorization(auth, PROTOCOL, require_local_journal=True)
    context = legacy.read(TICKET + ".WORKER_CONTEXT.json")
    started, deadline, _ = _outer_bounds(ticket, context["workflow_started_utc"])
    if (
        context.get("ticket_sha256") != legacy.sha(ticket_path) or context.get("nonce") != ticket["nonce"]
        or str(context.get("workflow_run_id")) != os.environ.get("GITHUB_RUN_ID")
        or context.get("run_attempt") != 1 or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
        or context.get("cap_seconds") != CAP
        or legacy.utc(context["outer_started_utc"]) != started
        or legacy.utc(context["outer_absolute_deadline_utc"]) != deadline
    ):
        raise ValueError("Worker context differs from its signed v115 identity or cap")
    parent = os.getppid()
    if worker:
        watchdog = legacy.read(Path(OUTPUT) / "watchdog_admission.json")
        controller_parent = int(subprocess.check_output(["ps", "-o", "ppid=", "-p", str(parent)], text=True).strip())
        if (
            watchdog.get("controller_pid") != parent or controller_parent != context.get("parent_pid")
            or watchdog.get("protocol_sha256") != legacy.sha(PROTOCOL)
            or legacy.read(Path(OUTPUT) / "reservation.json") != auth["reservation"]
        ):
            raise ValueError("Worker must descend from the signed v115 relay and controller")
    elif context.get("parent_pid") != parent:
        raise ValueError("Controller must be the actual v115 relay child")
    return context, auth["reservation"]


def controller_context(ticket_path, protocol_path):
    return _bound_context(ticket_path, protocol_path, worker=False)[0]


def worker_context(ticket_path, protocol_path):
    return _bound_context(ticket_path, protocol_path, worker=True)[0]


def consume_pre_reservation(ticket_path, protocol_path):
    return _bound_context(ticket_path, protocol_path, worker=False)[1]


def validate_worker_ticket(ticket_path, protocol_path):
    return _bound_context(ticket_path, protocol_path, worker=True)[1]


@contextmanager
def _without_action_token():
    previous_env = {name: os.environ.pop(name) for name in TOKEN_ENV if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(previous_env)


def _scientific_env():
    env = {name: value for name, value in os.environ.items() if name not in TOKEN_ENV}
    return dict(env, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                VECLIB_MAXIMUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")


def _validate_hydrated_weights(hydration, protocol):
    official = protocol["external_weight_transport_v96"]
    receipts = hydration["official_checkpoint_receipts"]
    observed = {row["filename"]: (row["source"], row["sha256"], row["bytes"]) for row in receipts}
    expected = {
        Path(path).name: (entry["url"], entry["sha256"], entry["bytes"])
        for path, entry in official.items()
    }
    if len(receipts) != 4 or observed != expected:
        raise ValueError("The exact four locked official author checkpoints must be hydrated")


def execute_remote(ticket_path, expected_sha, expected_nonce, ticket_commit, python=sys.executable):
    """One wall-time cap includes runner bootstrap, setup, science and packing."""
    from solar_portable_catalogue_v110 import install_verified_capsule
    from solar_public_transport_v98 import hydrate_thin_zip

    ticket = validate_ticket(ticket_path, expected_sha, expected_nonce)
    legacy._alive_lease(ticket)
    context = workflow_binding(ticket, ticket_commit)
    started, deadline, setup_deadline = _outer_bounds(ticket, context["workflow_started_utc"])
    entered = datetime.now(timezone.utc)
    tick = time.monotonic()
    parent_allowance = (deadline - entered).total_seconds()
    if parent_allowance <= 0:
        raise TimeoutError("Original v115 pipeline cap expired before parent setup")
    root = Path(REMOTE) / ticket["nonce"]
    root.mkdir(parents=True, exist_ok=False)
    context.update(
        ticket_sha256=legacy.sha(ticket_path), nonce=ticket["nonce"], parent_pid=os.getpid(),
        cap_seconds=CAP, outer_started_utc=started.isoformat(), outer_absolute_deadline_utc=deadline.isoformat(),
        parent_entered_utc=entered.isoformat(),
    )
    pending = []
    previous_handlers = {s: signal.signal(s, lambda signum, _: pending.append(signum))
                         for s in (signal.SIGTERM, signal.SIGINT)}

    def alarm(*_):
        raise TimeoutError("Original v115 pipeline cap interrupted the parent")

    previous_handlers[signal.SIGALRM] = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, parent_allowance)
    phase = "verify"
    child = protocol = failure = result = archive_failure = setup_seconds = None
    complete = False

    def check():
        if pending or datetime.now(timezone.utc) >= deadline or time.monotonic() - tick >= parent_allowance:
            raise TimeoutError("Original v115 pipeline cap exhausted or interrupted")

    def check_setup():
        check()
        if datetime.now(timezone.utc) >= setup_deadline:
            raise TimeoutError("v115 setup consumed its original launch lease")

    def remaining_setup_seconds():
        check_setup()
        return (setup_deadline - datetime.now(timezone.utc)).total_seconds()

    try:
        previous._require_source_ancestor(ticket, ticket_commit)
        legacy.verify_ticket_checkout(ticket_path, ticket, ticket_commit)
        context["public_dispatch_evidence"] = public_dispatch_evidence(ticket, ticket_commit)
        legacy.verify_release_download_head(previous._science_part())
        legacy.verify_release_download_head(ticket["direct_supplement"]["part"])
        check_setup()
        protocol = legacy.read(PROTOCOL)
        downloaded, hydrated = root / "science-v98.zip", root / "hydrated-v98"
        phase = "immutable_science_transport"
        previous._fetch_science_data(ticket, downloaded, setup_deadline)
        hydration = hydrate_thin_zip(
            downloaded, hydrated, BASE["manifest_sha256"], BASE["protocol_sha256"],
            check_setup, remaining_setup_seconds,
        )
        _validate_hydrated_weights(hydration, protocol)
        context.update(hydration)
        phase = "direct_supplement_transport"
        supplement = ticket["direct_supplement"]
        part = supplement["part"]
        direct_zip, direct_files = root / "direct-supplement.zip", root / "direct-files"
        legacy.fetch_capsule({
            "capsule": {"parts": [part], "archive_bytes": part["bytes"], "archive_sha256": part["sha256"]},
            "original_latest_start_utc": ticket["original_latest_start_utc"],
        }, direct_zip, active_deadline=setup_deadline)
        additions, _ = inventory(Path.cwd())
        context["direct_supplement"] = extract(
            direct_zip, direct_files, part, supplement["manifest_sha256"], additions, check_setup,
        )
        phase = "assemble_signed_overlay"
        overlay = root / "assembled-v115"
        with TemporaryDirectory(dir=root, prefix="metadata-") as metadata:
            auth_path = Path(metadata) / AUTHORIZATION
            auth_path.parent.mkdir(parents=True)
            with auth_path.open("xb") as stream:
                stream.write(_inline(ticket)[0])
            context["overlay"] = qualifier().assemble_overlay(
                hydrated, Path.cwd(), metadata, overlay, _overlay_binding(ticket), direct_files,
            )
        check_setup()
        phase = "portable_capsule_validation"
        install = install_verified_capsule(overlay, Path.cwd(), ticket["assembled_manifest_sha256"])
        context["portable_catalogue_archives"] = install["archives"]
        check_setup()
        legacy.restore_precharged_journal(ticket)
        _require_predecessors(legacy.read(CANONICAL_JOURNAL))
        phase = "official_source_setup"
        primary_scope(PROTOCOL, protocol)
        legacy.published(PROTOCOL, protocol, ticket["source_publication_commit"])
        with _without_action_token():
            legacy.prepare_modern_sources(protocol, active_deadline=setup_deadline)
        check_setup()
        context.update(workflow_binding(ticket, ticket_commit))
        if (
            legacy.utc(context["workflow_started_utc"]) != started
            or public_dispatch_evidence(ticket, ticket_commit) != context["public_dispatch_evidence"]
        ):
            raise ValueError("Public v115 run identity changed during setup")
        check_setup()
        setup_seconds = time.monotonic() - tick
        context.update(capsule_setup_seconds=setup_seconds, controller_started_utc=datetime.now(timezone.utc).isoformat())
        legacy.write_new(TICKET + ".WORKER_CONTEXT.json", context)
        phase = "evaluation"
        command = [python, RUNNER, "--protocol", PROTOCOL, "--publication-commit", ticket["source_publication_commit"],
                   "--python", python, "--pre-reserved-ticket", TICKET]
        with (root / "controller.log").open("x") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=_scientific_env(),
            )
            while child.poll() is None:
                check()
                time.sleep(0.1)
            check()
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
            seal = legacy.read(Path(OUTPUT) / "COMPLETE.json")
            if (
                seal.get("complete") is not True or seal.get("protocol_sha256") != ticket["protocol_sha256"]
                or seal.get("external_watchdog_passed") is not True
            ):
                raise ValueError("Original inner scientific watchdog did not seal the comparison")
            complete = True
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc), "phase": phase}
        if child is not None and child.poll() is None:
            legacy.terminate_descendants(child)
        raise
    finally:
        try:
            # If the cap already expired, the workflow's separate partial
            # artifact preserves uncompressed completed cells without replay.
            if protocol is not None and Path(AUTHORIZATION).is_file():
                check()
                result = legacy.preserve_results(root, protocol, TICKET, PROTOCOL)
                check()
        except BaseException as exc:
            archive_failure = {"error_type": type(exc).__name__, "error": str(exc)}
            complete = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        wall = (datetime.now(timezone.utc) - started).total_seconds()
        if complete and (result is None or wall >= CAP or datetime.now(timezone.utc) >= deadline):
            complete = False
            failure = {"error_type": "TimeoutError", "error": "Original pipeline bound exceeded", "phase": "result_preservation"}
        record = {
            **context, "complete": complete, "protocol_sha256": ticket["protocol_sha256"], "ticket_path": TICKET,
            "relay_source_sha256": legacy.sha(__file__), "workflow_source_sha256": legacy.sha(WORKFLOW),
            "authorization_sha256": ticket["authorization"]["sha256"], "original_reservation": _inline(ticket)[1]["reservation"],
            "started_utc": started.isoformat(), "absolute_deadline_utc": deadline.isoformat(),
            "calendar_seconds": wall, "parent_calendar_seconds": time.monotonic() - tick, "setup_seconds": setup_seconds,
            "child_returncode": None if child is None else child.returncode, "phase": phase,
            "result_artifact": result, "failure": failure, "archive_failure": archive_failure,
            "automatic_retry": False, "reservation_refunded": False, "primary_only": True,
        }
        try:
            legacy.write_new(root / "remote_watchdog.json", record)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    if not complete:
        raise RuntimeError("v115 evaluation or preservation failed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--write-live-protocol", action="store_true")
    reserve = sub.add_parser("pre-reserve")
    reserve.add_argument("--source-commit", required=True)
    reserve.add_argument("--base-capsule", required=True)
    reserve.add_argument("--supplement-archive", required=True)
    sub.add_parser("freeze-ticket").add_argument("--base-capsule", required=True)
    sub.add_parser("prepare-browser-dispatch").add_argument("--ticket-publication-commit", required=True)
    for name in ("verify-ticket", "remote"):
        command = sub.add_parser(name)
        command.add_argument("--ticket", required=True)
        command.add_argument("--ticket-sha256", required=True)
        command.add_argument("--ticket-nonce", required=True)
        command.add_argument("--ticket-commit", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.write_live_protocol:
            _local_attempt_absent()
        protocol = qualifier().prospective_protocol()
        _scope_wire(protocol)
        if args.write_live_protocol:
            legacy.write_new(PROTOCOL, protocol)
        value = {"protocol": PROTOCOL, "cases": 96, "profiles": 5, "charged": False,
                 "wrote_live_protocol": args.write_live_protocol}
    elif args.command == "pre-reserve":
        value = pre_reserve(args.source_commit, args.base_capsule, args.supplement_archive)
    elif args.command == "freeze-ticket":
        value = freeze_ticket(args.base_capsule)
    elif args.command == "prepare-browser-dispatch":
        value = prepare_browser_dispatch(args.ticket_publication_commit)
    elif args.command == "verify-ticket":
        verify_remote(args.ticket, args.ticket_sha256, args.ticket_nonce, args.ticket_commit)
        value = {"ticket_verified": True, "immutable_release_assets": 2}
    else:
        execute_remote(args.ticket, args.ticket_sha256, args.ticket_nonce, args.ticket_commit)
        return
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
