"""Separate one-use ARM qualification after a sealed v98 setup-only failure.

The old ZIP supplies immutable scientific assets only. A new canonical
charge, nonce, signed ticket and worker context authorize this attempt.
There is no API to dispatch, retry, refund, or reset a deadline.
"""

import argparse
import base64
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
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

from qualify_direct_arm_v113 import (
    AUTHORIZATION, BASE, OUTPUT, PROTOCOL, assemble_overlay, plan_overlay, primary_scope,
    prospective_protocol, verify_v110_receipt, _scope_metadata,
)
from plan_solar_matched_arm_v113 import DIRECT_QUALIFICATION_CAP, RUN_CAP, outstanding_v112
from solar_direct_transport_v113 import RELEASE_TAG, inventory, extract, hash_path
import solar_remote_relay_v98 as legacy
from solar_public_transport_v98 import MACHINE_PATH, SECRET_PATTERNS

WORKFLOW = ".github/workflows/solar-remote-direct-cpu-v113.yml"
WORKFLOW_NAME = "Solar Direct-only native CPU qualification v113"
TICKET = "reports/solar-remote-arm-direct-v113-ticket.json"
RUN_PREFIX = "solar-direct-v113-"
BRANCH_PREFIX = "solar-direct-v113-ticket-"
TRANSPORT = "v113_signed_direct_supplement_v98_immutable_data"
REPOSITORY = legacy.REPOSITORY
CANONICAL_JOURNAL = legacy.CANONICAL_JOURNAL
PROBE = "reports/solar_generic_sun_arm_probe_v110.json"
PROBE_SHA256 = "c27fa7920581a94c3ac1341cdd20e8e1a115bba6368c2079c6641207870a41ac"
FAILURE = "reports/solar_arm_v98_remote_watchdog.json"
FAILURE_SHA256 = "52a5eda77286b26d8e4753db8ce7de07f5f2c99aac25e571bc98834bf780016c"
FAILED_RUN = 41926059385
PROBE_RUN = 41928100316
MAX_INLINE_AUTH_BYTES = 128 * 1024
MAX_TICKET_BYTES = 256 * 1024
AUTH_KEYS = {
    "schema_version", "repository", "protocol_path", "protocol_sha256",
    "source_publication_commit", "nonce", "authorized_attempt", "reservation",
    "canonical_journal", "canonical_journal_sha256", "canonical_journal_bytes_base64",
    "original_latest_start_utc", "original_absolute_deadline_utc", "max_calendar_seconds",
    "case_ids", "tracks", "created_utc", "refunds", "automatic_retry",
}
ENTRY_KEYS = {"category", "job_id", "refund_on_failure", "reserved_utc", "seconds"}
TICKET_KEYS = {
    "schema_version", "repository", "workflow", "workflow_name", "transport", "nonce",
    "authorized_attempt", "source_publication_commit", "protocol_path", "protocol_sha256",
    "authorization", "base", "direct_supplement", "assembled_manifest_sha256", "original_latest_start_utc",
    "original_absolute_deadline_utc", "max_calendar_seconds", "required_runner", "cpu_threads",
    "expected_run_title", "expected_branch", "refunds", "automatic_retry", "primary_only",
}


def _bytes(data):
    return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _require_sha(value, pattern, message):
    return legacy._hex(value, pattern, message)


def _reserve_frozen_history(protocol, protocol_digest, source_commit):
    """Use the original journal and lock; a stale freeze cannot consume budget."""
    path = Path(CANONICAL_JOURNAL)
    expected = protocol["baseline_admission"]["initial_reservations"]
    cap = protocol["max_calendar_seconds"]
    job_id = protocol_digest + "/evaluation"
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = legacy.read(path)
        if (
            journal["reservations"] != expected
            or journal["ledger_sha256"] != protocol["compute_ledger_sha256"]
            or any(entry["job_id"] == job_id for entry in expected)
        ):
            raise ValueError("Canonical journal changed since freeze; no v110 charge made")
        legacy._alive_lease({
            "original_latest_start_utc": protocol["latest_start_utc"],
            "original_absolute_deadline_utc": protocol["absolute_deadline_utc"],
        })
        # The old canonical helper uses the ledger ceiling, while the launch
        # also verifies the stricter learning-contract ceiling before locking.
        if legacy.sha(protocol["compute_ledger"]) != journal["ledger_sha256"]:
            raise ValueError("Canonical budget ledger changed after the published freeze")
        ledger = legacy.read(protocol["compute_ledger"])
        charged = sum(entry["seconds"] for entry in expected if entry["category"] == "baseline")
        remaining_after = ledger["remaining_baseline_wall_seconds"] - charged - cap
        if ledger.get("accounting_complete") is not True or remaining_after < 0:
            raise ValueError("The original canonical baseline budget is exhausted")
        entry = {
            "job_id": job_id, "category": "baseline", "seconds": cap,
            "reserved_utc": datetime.now(timezone.utc).isoformat(), "refund_on_failure": False,
        }
        snapshot = _bytes(dict(journal, reservations=[*expected, entry]))
        authorization = {
            "schema_version": 75, "repository": REPOSITORY, "protocol_path": PROTOCOL,
            "protocol_sha256": protocol_digest, "source_publication_commit": source_commit,
            "nonce": legacy.charge_nonce(protocol_digest, entry), "authorized_attempt": 1,
            "reservation": {"reservation": entry, "remaining_after_reservation": remaining_after},
            "canonical_journal": CANONICAL_JOURNAL, "canonical_journal_sha256": _digest(snapshot),
            "canonical_journal_bytes_base64": base64.b64encode(snapshot).decode(),
            "original_latest_start_utc": protocol["latest_start_utc"],
            "original_absolute_deadline_utc": protocol["absolute_deadline_utc"],
            "max_calendar_seconds": cap, "case_ids": protocol["case_ids"], "tracks": ["estimated_sun"],
            "created_utc": datetime.now(timezone.utc).isoformat(), "refunds": False, "automatic_retry": False,
        }
        _review_inline_bytes(_bytes(authorization))
        # Once set, a crash or partial publication never allows another charge.
        legacy.write_new(AUTHORIZATION + ".ISSUING.json", {
            "protocol_sha256": protocol_digest, "no_retry": True, "previous_run_id": FAILED_RUN,
        })
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(snapshot)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        legacy.write_new(AUTHORIZATION, authorization)
    return {"authorization": AUTHORIZATION, "nonce": authorization["nonce"], "seconds": cap}


def pre_reserve(source_commit, base_capsule, supplement_archive, api=None, verify_head=None):
    """One authorized local transaction after exact public source checks."""
    from solar_learning_pipeline_v73 import load_contract, remaining
    from solar_portable_evidence_v75 import pin

    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise ValueError("A worker cannot write a canonical v110 reservation")
    api = public_repository_api if api is None else api
    verify_head = legacy.verify_release_download_head if verify_head is None else verify_head
    _require_sha(source_commit, legacy.HEX40, "Exact published source commit required")
    prior = verify_v110_receipt()
    if (
        Path(OUTPUT).exists() or Path(AUTHORIZATION).exists()
        or Path(AUTHORIZATION + ".ISSUING.json").exists() or Path(TICKET).exists()
    ):
        raise FileExistsError("Preserve previous v110 authorization or output")
    protocol = legacy.read(PROTOCOL)
    _scope_metadata(protocol)
    supplement = protocol["direct_supplement"]["part"]
    if Path(supplement_archive).stat().st_size != supplement["bytes"] or hash_path(supplement_archive) != supplement["sha256"]:
        raise ValueError("Exact local Direct supplement must still match the protocol")
    journal = legacy.read(CANONICAL_JOURNAL)
    if journal["reservations"].count(prior) != 1:
        raise ValueError("The successful v110 qualification charge must remain intact")
    _public_repo_and_workflow(api)
    ref = api("/git/ref/heads/main")
    ref_object = ref.get("object", {})
    if (
        ref.get("ref") != "refs/heads/main" or ref_object.get("type") != "commit"
        or ref_object.get("sha") != source_commit
    ):
        raise ValueError("The exact reviewed v110 protocol must already be on public main")
    _public_science_asset(api)
    verify_head(_science_part())
    _public_supplement(protocol["direct_supplement"], api)
    verify_head(supplement)
    base_capsule = Path(base_capsule)
    pin(base_capsule, "manifest.json", BASE["manifest_sha256"])
    pin(base_capsule / "payload", BASE["protocol_path"], BASE["protocol_sha256"])
    pin(base_capsule / "payload", BASE["authorization_path"], BASE["authorization_sha256"])
    legacy.published(PROTOCOL, protocol, source_commit)
    primary_scope(PROTOCOL, protocol)
    contract = load_contract(protocol["learning_contract"])
    budget = remaining(contract)["baseline"]
    protected = protocol["baseline_admission"]["unreserved_v112_protected_seconds"]
    if (budget != protocol["baseline_admission"]["remaining_at_preparation_seconds"]
            or protected != outstanding_v112(Path.cwd(), journal)
            or budget < protected + RUN_CAP + DIRECT_QUALIFICATION_CAP):
        raise ValueError("The live canonical baseline allowance or protected commitment changed")
    return _reserve_frozen_history(protocol, legacy.sha(PROTOCOL), source_commit)


def _review_inline_bytes(raw):
    """Only a typed, public-scanned journal snapshot may enter a signed ticket."""
    if not 0 < len(raw) <= MAX_INLINE_AUTH_BYTES:
        raise ValueError("Inline authorization exceeds its small reviewed bound")
    auth = json.loads(raw)
    if set(auth) != AUTH_KEYS or set(auth["reservation"]) != {"reservation", "remaining_after_reservation"}:
        raise ValueError("Unexpected authorization metadata field")
    snapshot = base64.b64decode(auth["canonical_journal_bytes_base64"], validate=True)
    for data in (raw, snapshot):
        if SECRET_PATTERNS.search(data) or MACHINE_PATH.search(data):
            raise ValueError("Authorization transport contains a private path or credential pattern")
    if _digest(snapshot) != auth["canonical_journal_sha256"]:
        raise ValueError("Exact authorization journal digest required")
    journal = json.loads(snapshot)
    if set(journal) != {"ledger_sha256", "reservations"} or not isinstance(journal["reservations"], list):
        raise ValueError("Unexpected canonical journal shape")
    _require_sha(journal["ledger_sha256"], legacy.HEX64, "Canonical ledger digest required")
    for entry in journal["reservations"]:
        if (
            set(entry) != ENTRY_KEYS or entry["category"] not in {"training", "baseline"}
            or type(entry["seconds"]) is not int or entry["seconds"] <= 0
            or entry["refund_on_failure"] is not False
        ):
            raise ValueError("Unexpected canonical journal entry")
        legacy.relative(entry["job_id"])
        legacy.utc(entry["reserved_utc"])
    return auth


def _inline(ticket):
    item = ticket["authorization"]
    if set(item) != {"path", "sha256", "inline_base64"} or item["path"] != AUTHORIZATION:
        raise ValueError("Exact distinct v110 authorization transport required")
    raw = base64.b64decode(item["inline_base64"], validate=True)
    auth = _review_inline_bytes(raw)
    if _digest(raw) != _require_sha(item["sha256"], legacy.HEX64, "Authorization digest required"):
        raise ValueError("Inline authorization bytes differ from the signed digest")
    return raw, auth


def _overlay_binding(ticket):
    return {
        "base": ticket["base"], "direct_supplement": ticket["direct_supplement"], "protocol_path": ticket["protocol_path"],
        "protocol_sha256": ticket["protocol_sha256"], "authorization_path": ticket["authorization"]["path"],
        "authorization_sha256": ticket["authorization"]["sha256"],
        "source_publication_commit": ticket["source_publication_commit"], "nonce": ticket["nonce"],
        "assembled_manifest_sha256": ticket["assembled_manifest_sha256"],
    }


def freeze_ticket(base_capsule):
    """Freeze a small reviewable ticket using the exact precharged metadata."""
    if Path(TICKET).exists() or Path(AUTHORIZATION + ".TICKET_ISSUED.json").exists():
        raise FileExistsError("Preserve any existing signed or prepared v110 ticket")
    raw = Path(AUTHORIZATION).read_bytes()
    auth = _review_inline_bytes(raw)
    legacy.validate_authorization(auth, PROTOCOL, require_local_journal=True)
    protocol = legacy.read(PROTOCOL)
    _scope_metadata(protocol)
    ticket = {
        "schema_version": 113, "repository": REPOSITORY, "workflow": WORKFLOW,
        "workflow_name": WORKFLOW_NAME, "transport": TRANSPORT, "nonce": auth["nonce"],
        "authorized_attempt": 1, "source_publication_commit": auth["source_publication_commit"],
        "protocol_path": PROTOCOL, "protocol_sha256": auth["protocol_sha256"],
        "authorization": {
            "path": AUTHORIZATION, "sha256": _digest(raw), "inline_base64": base64.b64encode(raw).decode(),
        },
        "base": BASE, "direct_supplement": protocol["direct_supplement"], "assembled_manifest_sha256": None,
        "original_latest_start_utc": auth["original_latest_start_utc"],
        "original_absolute_deadline_utc": auth["original_absolute_deadline_utc"],
        "max_calendar_seconds": auth["max_calendar_seconds"], "required_runner": "ubuntu-24.04-arm",
        "cpu_threads": 2, "expected_run_title": RUN_PREFIX + auth["nonce"],
        "expected_branch": BRANCH_PREFIX + auth["nonce"],
        "refunds": False, "automatic_retry": False, "primary_only": True,
    }
    legacy._alive_lease(ticket)
    manifest, _ = plan_overlay(base_capsule, Path.cwd(), Path.cwd(), _overlay_binding(ticket))
    ticket["assembled_manifest_sha256"] = _digest(manifest)
    legacy.write_new(AUTHORIZATION + ".TICKET_ISSUED.json", {
        "ticket_path": TICKET, "nonce": auth["nonce"], "no_reissue": True,
    })
    legacy.write_new(TICKET, ticket)
    return {"ticket": TICKET, "bytes": Path(TICKET).stat().st_size, "nonce": auth["nonce"]}


def validate_ticket(ticket_path, expected_sha=None, expected_nonce=None):
    if str(ticket_path) != TICKET or not 0 < Path(ticket_path).stat().st_size <= MAX_TICKET_BYTES:
        raise ValueError("The exact bounded v110 ticket path is required")
    if expected_sha is not None and legacy.sha(ticket_path) != _require_sha(
        expected_sha, legacy.HEX64, "Exact signed v110 ticket digest required"
    ):
        raise ValueError("Signed ticket bytes differ")
    ticket = legacy.read(ticket_path)
    nonce = _require_sha(ticket.get("nonce"), legacy.HEX64, "Exact new one-use nonce required")
    if (
        set(ticket) != TICKET_KEYS or ticket["schema_version"] != 113
        or ticket["repository"] != REPOSITORY or ticket["workflow"] != WORKFLOW
        or ticket["workflow_name"] != WORKFLOW_NAME or ticket["transport"] != TRANSPORT
        or ticket["protocol_path"] != PROTOCOL or ticket["base"] != BASE
        or ticket["required_runner"] != "ubuntu-24.04-arm" or ticket["cpu_threads"] != 2
        or ticket["authorized_attempt"] != 1 or ticket["max_calendar_seconds"] != 1800
        or ticket["refunds"] is not False or ticket["automatic_retry"] is not False
        or ticket["primary_only"] is not True or ticket["expected_run_title"] != RUN_PREFIX + nonce
        or ticket["expected_branch"] != BRANCH_PREFIX + nonce
        or (expected_nonce is not None and nonce != expected_nonce)
    ):
        raise ValueError("Only the new separately charged v110 one-use ticket is supported")
    _require_sha(ticket["source_publication_commit"], legacy.HEX40, "Exact v110 source publication required")
    _require_sha(ticket["assembled_manifest_sha256"], legacy.HEX64, "Exact logical v110 manifest required")
    _, auth = _inline(ticket)
    legacy.validate_authorization(auth, PROTOCOL)
    if (
        ticket["protocol_sha256"] != auth["protocol_sha256"] or ticket["nonce"] != auth["nonce"]
        or ticket["source_publication_commit"] != auth["source_publication_commit"]
        or ticket["original_latest_start_utc"] != auth["original_latest_start_utc"]
        or ticket["original_absolute_deadline_utc"] != auth["original_absolute_deadline_utc"]
        or ticket["max_calendar_seconds"] != auth["max_calendar_seconds"]
        or ticket["protocol_sha256"] == BASE["protocol_sha256"]
        or ticket["authorization"]["sha256"] == BASE["authorization_sha256"]
    ):
        raise ValueError("Fresh ticket and fresh authorization do not bind the same charge")
    protocol = legacy.read(PROTOCOL)
    _scope_metadata(protocol)
    if ticket["direct_supplement"] != protocol["direct_supplement"]:
        raise ValueError("The signed ticket must pin the exact Direct supplement")
    return ticket


def _science_part():
    return {"url": BASE["archive_url"], "sha256": BASE["archive_sha256"], "bytes": BASE["archive_bytes"]}


def _history_path(page):
    return "/actions/workflows/" + Path(WORKFLOW).name + "/runs?event=workflow_dispatch&per_page=100&page=" + str(page)


def public_repository_api(suffix):
    """Only public repository/ref/signature/workflow/old-asset metadata."""
    pattern = (
        r"(?:/git/ref/heads/(?:main|solar-direct-v113-ticket-[0-9a-f]{64})"
        r"|/git/commits/[0-9a-f]{40}|/releases/tags/(?:solar-arm-transport-v98|solar-arm-direct-v113-transport)"
        r"|/actions/workflows/solar-remote-direct-cpu-v113\.yml"
        r"(?:/runs\?event=workflow_dispatch&per_page=100&page=(?:[1-9]|1[0-9]|20))?)?"
    )
    if re.fullmatch(pattern, suffix) is None:
        raise ValueError("Unexpected public metadata path")
    url = "https://api.github.com/repos/" + REPOSITORY + suffix
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "solar-browser-gate-v110",
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
        raise ValueError("The separate v110 workflow must already be active on this public repository")


def _one_use_runs(api, title):
    hits = []
    for page in range(1, 21):
        runs = api(_history_path(page))["workflow_runs"]
        hits.extend(run for run in runs if run.get("display_title") == title)
        if len(runs) < 100:
            return hits
    raise ValueError("Workflow history exceeds the bounded uniqueness audit")


def public_dispatch_evidence(ticket, ticket_commit, api=None):
    api = public_repository_api if api is None else api
    _require_sha(ticket_commit, legacy.HEX40, "Exact v110 ticket commit required")
    _public_repo_and_workflow(api)
    branch = ticket["expected_branch"]
    ref = api("/git/ref/heads/" + branch)
    commit = api("/git/commits/" + ticket_commit)
    verification = commit.get("verification", {})
    if (
        ref.get("ref") != "refs/heads/" + branch
        or ref.get("object", {}).get("type") != "commit"
        or ref.get("object", {}).get("sha") != ticket_commit or commit.get("sha") != ticket_commit
        or verification.get("verified") is not True or verification.get("reason") != "valid"
    ):
        raise ValueError("GitHub must verify the exact signed v110 ticket branch and commit")
    asset = _public_science_asset(api)
    extra = _public_supplement(ticket["direct_supplement"], api)
    return {
        "ticket_commit": ticket_commit, "branch": branch, "verified_commit": True,
        "release_asset_id": asset["id"], "archive_bytes": asset["size"], "archive_sha256": BASE["archive_sha256"],
        "previous_failed_run_id": FAILED_RUN, "verified_arithmetic_run_id": PROBE_RUN,
        "direct_supplement_asset_id": extra["id"], "prior_successful_arm_run_id": 41930272861,
    }


def _public_science_asset(api):
    part = _science_part()
    release = api("/releases/tags/solar-arm-transport-v98")
    expected_name = part["url"].rsplit("/", 1)[-1]
    assets = [item for item in release.get("assets", []) if item.get("name") == expected_name]
    if (
        release.get("tag_name") != "solar-arm-transport-v98" or release.get("draft") is not False
        or not release.get("published_at") or len(assets) != 1
    ):
        raise ValueError("The original independent science release is unavailable")
    asset = assets[0]
    if (
        asset.get("state") != "uploaded" or asset.get("size") != part["bytes"]
        or asset.get("digest") != "sha256:" + part["sha256"] or asset.get("browser_download_url") != part["url"]
        or type(asset.get("id")) is not int or asset["id"] <= 0
    ):
        raise ValueError("Original immutable science asset size, URL or SHA differs")
    return asset


def _public_supplement(binding, api):
    part = binding["part"]
    release = api("/releases/tags/" + RELEASE_TAG)
    name = part["url"].rsplit("/", 1)[-1]
    candidates = [asset for asset in release.get("assets", []) if asset.get("name") == name]
    if (
        release.get("tag_name") != RELEASE_TAG or release.get("draft") is not False
        or not release.get("published_at") or len(candidates) != 1
    ):
        raise ValueError("The separate Direct supplemental release is unavailable")
    asset = candidates[0]
    if (
        asset.get("state") != "uploaded" or asset.get("size") != part["bytes"]
        or asset.get("digest") != "sha256:" + part["sha256"] or asset.get("browser_download_url") != part["url"]
        or type(asset.get("id")) is not int or asset["id"] <= 0
    ):
        raise ValueError("Exact published Direct supplement size, URL and digest required")
    return asset


def _require_source_ancestor(ticket, ticket_commit):
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ticket["source_publication_commit"], ticket_commit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode:
        raise ValueError("The signed v110 ticket must descend from the reviewed source publication")


def prepare_browser_dispatch(ticket_commit, api=None, verify_head=None):
    """Write the exclusive marker with exact inputs for one manual browser action."""
    api = public_repository_api if api is None else api
    verify_head = legacy.verify_release_download_head if verify_head is None else verify_head
    ticket = validate_ticket(TICKET)
    legacy._alive_lease(ticket)
    raw, auth = _inline(ticket)
    if Path(AUTHORIZATION).read_bytes() != raw:
        raise ValueError("Signed ticket and the local precharge differ")
    legacy.validate_authorization(auth, PROTOCOL, require_local_journal=True)
    _require_source_ancestor(ticket, ticket_commit)
    legacy.published(PROTOCOL, legacy.read(PROTOCOL), ticket_commit)
    if subprocess.check_output(["git", "show", ticket_commit + ":" + TICKET]) != Path(TICKET).read_bytes():
        raise ValueError("Signed ticket bytes have not been published")
    evidence = public_dispatch_evidence(ticket, ticket_commit, api=api)
    verify_head(_science_part())
    verify_head(ticket["direct_supplement"]["part"])
    if _one_use_runs(api, ticket["expected_run_title"]):
        raise ValueError("This v110 nonce already has a workflow invocation")
    legacy._alive_lease(ticket)
    marker = {
        "mode": "prepared_browser_dispatch_once", "reviewed_branch": ticket["expected_branch"],
        "dispatch_url": "https://github.com/" + REPOSITORY + "/actions/workflows/" + Path(WORKFLOW).name,
        "evidence": evidence, "authorized_attempt": 1, "automatic_retry": False,
        "inputs": {
            "ticket": TICKET, "ticket_sha256": legacy.sha(TICKET),
            "ticket_nonce": ticket["nonce"], "ticket_commit": ticket_commit,
        },
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    legacy.write_new(TICKET + ".DISPATCHED.json", marker)
    return marker


def workflow_binding(ticket, ticket_commit, environment=None, api=legacy.github_api):
    """Require the exact v110 job, signed branch and unique first attempt."""
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
    head = _require_sha(env.get("GITHUB_SHA"), legacy.HEX40, "Exact workflow source SHA required")
    if (
        not re.fullmatch(r"[1-9][0-9]*", run_id) or int(run_id) in {FAILED_RUN, PROBE_RUN, 41930272861}
        or head != _require_sha(ticket_commit, legacy.HEX40, "Exact v110 ticket commit required")
        or env.get("GITHUB_WORKFLOW_SHA") != head
        or env.get("GITHUB_WORKFLOW_REF") != REPOSITORY + "/" + WORKFLOW + "@" + env["GITHUB_REF"]
    ):
        raise ValueError("Actual v110 workflow and signed checkout bindings differ")
    prefix = "repos/" + REPOSITORY
    current = api(prefix + "/actions/runs/" + run_id)
    if (
        str(current.get("id")) != run_id or current.get("event") != "workflow_dispatch"
        or current.get("run_attempt") != 1 or current.get("head_sha") != head
        or current.get("head_branch") != ticket["expected_branch"]
        or current.get("path", "").split("@")[0] != WORKFLOW
        or current.get("display_title") != ticket["expected_run_title"]
        or current.get("repository", {}).get("full_name") != REPOSITORY
        or current.get("head_repository", {}).get("full_name") != REPOSITORY
    ):
        raise ValueError("Actual v110 workflow run/source/ticket bindings differ")
    hits = _one_use_runs(lambda suffix: api(prefix + suffix), ticket["expected_run_title"])
    if len(hits) != 1 or str(hits[0]["id"]) != run_id or hits[0].get("run_attempt") != 1:
        raise ValueError("Duplicate invocation, re-dispatch or rerun of a one-use v110 ticket")
    return {
        "workflow_run_id": int(run_id), "run_attempt": 1, "workflow_source_commit": head,
        "source_publication_commit": ticket["source_publication_commit"], "ticket_nonce": ticket["nonce"],
        "reviewed_branch": ticket["expected_branch"], "reviewed_ticket_commit": ticket_commit,
        "html_url": current.get("html_url"), "workflow_started_utc": current.get("run_started_at"),
    }


def verify_remote(ticket_path, expected_sha, expected_nonce, ticket_commit):
    ticket = validate_ticket(ticket_path, expected_sha, expected_nonce)
    legacy._alive_lease(ticket)
    workflow_binding(ticket, ticket_commit)
    _require_source_ancestor(ticket, ticket_commit)
    legacy.verify_ticket_checkout(ticket_path, ticket, ticket_commit)
    public_dispatch_evidence(ticket, ticket_commit)
    legacy.verify_release_download_head(_science_part())
    legacy.verify_release_download_head(ticket["direct_supplement"]["part"])
    legacy._alive_lease(ticket)


def _bound_auth(ticket_path, protocol_path):
    ticket = validate_ticket(ticket_path)
    raw, auth = _inline(ticket)
    if str(protocol_path) != PROTOCOL or Path(AUTHORIZATION).read_bytes() != raw:
        raise ValueError("Installed v110 authorization differs from the signed charge")
    legacy.validate_authorization(auth, PROTOCOL, require_local_journal=True)
    context = legacy.read(str(ticket_path) + ".WORKER_CONTEXT.json")
    if (
        context.get("ticket_sha256") != legacy.sha(ticket_path) or context.get("nonce") != ticket["nonce"]
        or str(context.get("workflow_run_id")) != os.environ.get("GITHUB_RUN_ID")
        or context.get("run_attempt") != 1 or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
    ):
        raise ValueError("Only the authenticated first v110 worker can consume this charge")
    return context, auth["reservation"]


def consume_pre_reservation(ticket_path, protocol_path):
    context, reservation = _bound_auth(ticket_path, protocol_path)
    if context.get("parent_pid") != os.getppid():
        raise ValueError("Only the authenticated v110 parent controller can consume this charge")
    return reservation


def validate_worker_ticket(ticket_path, protocol_path):
    context, reservation = _bound_auth(ticket_path, protocol_path)
    watchdog = legacy.read(Path(OUTPUT) / "watchdog_admission.json")
    controller_parent = int(subprocess.check_output(
        ["ps", "-o", "ppid=", "-p", str(os.getppid())], text=True,
    ).strip())
    if (
        watchdog.get("controller_pid") != os.getppid() or controller_parent != context.get("parent_pid")
        or watchdog.get("protocol_sha256") != legacy.sha(PROTOCOL)
        or legacy.read(Path(OUTPUT) / "reservation.json") != reservation
    ):
        raise ValueError("Worker must descend from the actual one-use v110 parent and controller")
    return reservation


def _fetch_science_data(ticket, destination, deadline):
    part = _science_part()
    # Pass a data-only identity to the inherited downloader. Do not load the
    # original v98 ticket or use its lease or authorization to access a worker.
    binding = {
        "capsule": {"parts": [part], "archive_bytes": part["bytes"], "archive_sha256": part["sha256"]},
        "original_latest_start_utc": ticket["original_latest_start_utc"],
    }
    legacy.fetch_capsule(binding, destination, active_deadline=deadline)


def execute_remote(ticket_path, expected_sha, expected_nonce, ticket_commit, python=sys.executable):
    """Keep setup, science and result preservation inside one original allowance."""
    from solar_portable_catalogue_v110 import install_verified_capsule
    from solar_public_transport_v98 import hydrate_thin_zip

    ticket = validate_ticket(ticket_path, expected_sha, expected_nonce)
    legacy._alive_lease(ticket)
    context = workflow_binding(ticket, ticket_commit)
    root = Path("runs/solar-remote-direct-v113") / ticket["nonce"]
    root.mkdir(parents=True, exist_ok=False)
    started, tick = datetime.now(timezone.utc), time.monotonic()
    cap = ticket["max_calendar_seconds"]
    deadline = min(started + timedelta(seconds=cap), legacy.utc(ticket["original_absolute_deadline_utc"]))
    setup_deadline = min(deadline, legacy.utc(ticket["original_latest_start_utc"]))
    context.update(ticket_sha256=legacy.sha(ticket_path), nonce=ticket["nonce"], parent_pid=os.getpid())
    pending = []
    previous = {s: signal.signal(s, lambda signum, _: pending.append(signum)) for s in (signal.SIGTERM, signal.SIGINT)}
    phase = "verify"
    child = protocol = failure = result = archive_failure = setup_seconds = None
    complete = False

    def check():
        if pending or datetime.now(timezone.utc) >= deadline or time.monotonic() - tick >= cap:
            raise TimeoutError("One-use v110 parent allowance exhausted or interrupted")

    def check_setup():
        check()
        if datetime.now(timezone.utc) >= setup_deadline:
            raise TimeoutError("v110 setup consumed its original fresh launch lease")

    def remaining_setup_seconds():
        check_setup()
        return (setup_deadline - datetime.now(timezone.utc)).total_seconds()

    try:
        _require_source_ancestor(ticket, ticket_commit)
        legacy.verify_ticket_checkout(ticket_path, ticket, ticket_commit)
        context["public_dispatch_evidence"] = public_dispatch_evidence(ticket, ticket_commit)
        legacy.verify_release_download_head(_science_part())
        legacy.verify_release_download_head(ticket["direct_supplement"]["part"])
        check_setup()
        downloaded, hydrated = root / "science-v98.zip", root / "hydrated-v98"
        phase = "immutable_science_transport"
        _fetch_science_data(ticket, downloaded, deadline)
        check_setup()
        context.update(hydrate_thin_zip(
            downloaded, hydrated, BASE["manifest_sha256"], BASE["protocol_sha256"],
            check_setup, remaining_setup_seconds,
        ))
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
        overlay = root / "assembled-direct-v113"
        with TemporaryDirectory(dir=root, prefix="metadata-") as metadata:
            auth_file = Path(metadata) / AUTHORIZATION
            auth_file.parent.mkdir(parents=True)
            with auth_file.open("xb") as stream:
                stream.write(_inline(ticket)[0])
            context["overlay"] = assemble_overlay(hydrated, Path.cwd(), metadata, overlay, _overlay_binding(ticket), direct_files)
        check_setup()
        phase = "v110_capsule_validation"
        installed = install_verified_capsule(overlay, Path.cwd(), ticket["assembled_manifest_sha256"])
        context["portable_catalogue_archives"] = installed["archives"]
        check_setup()
        legacy.restore_precharged_journal(ticket)
        protocol = legacy.read(PROTOCOL)
        phase = "official_source_setup"
        primary_scope(PROTOCOL, protocol)
        legacy.published(PROTOCOL, protocol, ticket["source_publication_commit"])
        # The original Direct source is in the exact signed checkout. No author model runs.
        check_setup()
        context.update(workflow_binding(ticket, ticket_commit))
        if public_dispatch_evidence(ticket, ticket_commit) != context["public_dispatch_evidence"]:
            raise ValueError("Public dispatch evidence changed during setup")
        check_setup()
        setup_seconds = time.monotonic() - tick
        context.update(capsule_setup_seconds=setup_seconds, controller_started_utc=datetime.now(timezone.utc).isoformat())
        legacy.write_new(TICKET + ".WORKER_CONTEXT.json", context)
        phase = "evaluation"
        command = [
            python, "run_solar_direct_arm_v113.py", "--protocol", PROTOCOL,
            "--publication-commit", ticket["source_publication_commit"],
            "--python", python, "--pre-reserved-ticket", TICKET,
        ]
        with (root / "controller.log").open("x") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=dict(
                    os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                    VECLIB_MAXIMUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                ),
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
                raise ValueError("The independent inner scientific watchdog did not complete")
            complete = True
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc), "phase": phase}
        if child is not None and child.poll() is None:
            legacy.terminate_descendants(child)
        raise
    finally:
        try:
            if protocol is not None:
                result = legacy.preserve_results(root, protocol, TICKET, PROTOCOL)
        except BaseException as exc:
            archive_failure = {"error_type": type(exc).__name__, "error": str(exc)}
            complete = False
        if complete and (pending or datetime.now(timezone.utc) >= deadline or time.monotonic() - tick >= cap):
            complete = False
            failure = {
                "error_type": "TimeoutError", "error": "Result preservation exceeded original allowance",
                "phase": "result_preservation",
            }
        record = {
            **context, "complete": complete, "protocol_sha256": ticket["protocol_sha256"], "ticket_path": TICKET,
            "relay_source_sha256": legacy.sha(__file__), "workflow_source_sha256": legacy.sha(WORKFLOW),
            "authorization_sha256": ticket["authorization"]["sha256"],
            "original_reservation": _inline(ticket)[1]["reservation"],
            "started_utc": started.isoformat(), "absolute_deadline_utc": deadline.isoformat(), "cap_seconds": cap,
            "calendar_seconds": time.monotonic() - tick, "setup_seconds": setup_seconds,
            "child_returncode": None if child is None else child.returncode, "phase": phase,
            "result_artifact": result, "failure": failure, "archive_failure": archive_failure,
            "automatic_retry": False, "reservation_refunded": False, "primary_only": True,
        }
        try:
            legacy.write_new(root / "remote_watchdog.json", record)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if not complete:
        raise RuntimeError("v110 evaluation or preservation failed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--supplement-receipt", required=True)
    prepare.add_argument("--unreserved-v112-seconds", type=int)
    prepare.add_argument("--write-live-protocol", action="store_true")
    reserve = commands.add_parser("pre-reserve")
    reserve.add_argument("--source-commit", required=True)
    reserve.add_argument("--base-capsule", required=True)
    reserve.add_argument("--supplement-archive", required=True)
    commands.add_parser("freeze-ticket").add_argument("--base-capsule", required=True)
    commands.add_parser("prepare-browser-dispatch").add_argument("--ticket-publication-commit", required=True)
    for name in ("verify-ticket", "remote"):
        command = commands.add_parser(name)
        command.add_argument("--ticket", required=True)
        command.add_argument("--ticket-sha256", required=True)
        command.add_argument("--ticket-nonce", required=True)
        command.add_argument("--ticket-commit", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepared = prospective_protocol(legacy.read(args.supplement_receipt), v112_unreserved=args.unreserved_v112_seconds)
        if args.write_live_protocol:
            legacy.write_new(PROTOCOL, prepared)
        result = {
            "protocol": PROTOCOL, "cases": 8, "profiles": 1, "charged": False,
            "source_count": len(prepared["source_sha256"]), "input_count": len(prepared["input_sha256"]),
            "baseline_remaining": prepared["baseline_admission"]["remaining_at_preparation_seconds"],
            "wrote_live_protocol": args.write_live_protocol,
        }
    elif args.command == "pre-reserve":
        result = pre_reserve(args.source_commit, args.base_capsule, args.supplement_archive)
    elif args.command == "freeze-ticket":
        result = freeze_ticket(args.base_capsule)
    elif args.command == "prepare-browser-dispatch":
        result = prepare_browser_dispatch(args.ticket_publication_commit)
    elif args.command == "verify-ticket":
        verify_remote(args.ticket, args.ticket_sha256, args.ticket_nonce, args.ticket_commit)
        result = {"ticket_verified": True, "released_science_asset_unchanged": True}
    else:
        result = execute_remote(args.ticket, args.ticket_sha256, args.ticket_nonce, args.ticket_commit)
    if args.command != "remote":
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
