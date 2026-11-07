"""One-use public free-CPU relay for an already frozen evaluation.

No models are imported here. The local machine charges the one canonical
journal before exporting a capsule. A remote controller consumes precisely
that entry; it never reserves again, refunds, resets a lease, or imports its
journal back into the local one. No automatic dispatch/retry is performed.
"""
import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPOSITORY = "ThomasdeChillaz/lunar-image-matching"
WORKFLOW = ".github/workflows/solar-remote-cpu-v98.yml"
WORKFLOW_NAME = "Solar paired native modern CPU pilot v98"
CANONICAL_JOURNAL = "runs/solar-compute-reservations-v67.json"
TORCH_HOME = "data/models/modern-v2-torch-cache"
RUNTIME = dict(runner="ubuntu-24.04-arm", cpu_threads=2, torch="2.8.0+cpu", torchvision="0.23.0", torch_home=TORCH_HOME)
UPSTREAM = {
    "loma_r": ("https://github.com/davnords/loma.git", "415caedde23a9ed479172ed174fe4ff7b4514d4b"),
    "romav2": ("https://github.com/Parskatt/RoMaV2.git", "95c9968145c8906b7b59383258e9f73b02853d89"),
}
MAX_EXECUTION_SECONDS = 18000  # Leave headroom below the hosted six-hour limit.
MAX_ARCHIVE_BYTES = 4 * 1024**3
MAX_PART_BYTES = 1_900_000_000
MAX_RESULT_BYTES = 2 * 1024**3
SELF = "solar_remote_relay_v98.py"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_PATH = re.compile(r"[A-Za-z0-9_.\-/]{1,350}\Z")
RELEASE_URL = re.compile(r"https://github\.com/ThomasdeChillaz/lunar-image-matching/releases/download/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def relative(path):
    if not isinstance(path, str) or not SAFE_PATH.fullmatch(path):
        raise ValueError("Safe repository-relative path required")
    p = PurePosixPath(path)
    if p.is_absolute() or str(p) != path or any(x in ("", ".", "..") for x in p.parts):
        raise ValueError("No absolute, aliased, traversal or shell-expression paths")
    return path


def utc(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timezone-aware original UTC timestamp required")
    return result.astimezone(timezone.utc)


def _hex(value, pattern, description):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(description)
    return value


def charge_nonce(protocol_sha256, original_entry):
    """One original canonical charge has exactly one cross-worker identity."""
    payload = dict(protocol_sha256=_hex(protocol_sha256, HEX64, "Exact original protocol SHA required"),
        canonical_original_entry=original_entry)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(b"solar-remote-charge-v75:" + raw).hexdigest()


def published(protocol_path, protocol, commit):
    _hex(commit, HEX40, "Exact published source commit required")
    for name in (relative(str(protocol_path)), *protocol["source_sha256"]):
        relative(name)
        public = subprocess.check_output(["git", "show", commit + ":" + name], stderr=subprocess.DEVNULL)
        if public != Path(name).read_bytes():
            raise ValueError("Byte-identical publication missing: " + name)


def primary_scope(protocol_path, protocol, at=None, check_assets=True):
    """New launch checks only; never use this to invalidate old completed work."""
    from solar_evaluation_protocol_v74 import validate_runtime
    from solar_weight_transport_v96 import LOCK, validate_declaration
    from solar_public_transport_v98 import validate_release_inputs
    if (protocol.get("builder_schema_version") != 74 or protocol.get("schema_version") != 70
            or protocol.get("prepared_only", False) or protocol.get("split") != "model_selection"
            or protocol.get("calibration_or_final_access") is not False or protocol.get("cpu_threads") != 2
            or protocol.get("tracks") != ["estimated_sun"]):
        raise ValueError("Frozen primary-only development protocol required")
    exact_pilot = [
        "u012-l00-00-positive", "u012-to-u020-l00-00-nonoverlap",
        "u026-l00-17-positive", "u026-to-u058-l00-17-nonoverlap",
        "u060-l16-16-positive", "u060-to-u068-l16-16-nonoverlap",
        "u074-l08-01-positive", "u074-to-u094-l08-01-nonoverlap",
    ]
    if (protocol.get("purpose") != "cost_pilot" or protocol.get("target_purpose") != "modern_comparison"
            or protocol.get("stage_mode") != "matched_modern_arm_v98" or protocol.get("pilot_case_ids") != exact_pilot
            or [row["name"] for row in protocol["profiles"]] !=
            ["frozen_step0_A1B0C1_quantization_aware", "SIFT2048", "loma_r_common", "romav2_common"]):
        raise ValueError("The v98 transport only admits the fixed four-profile, eight-case modern pilot")
    cap = protocol.get("max_calendar_seconds")
    if type(cap) is not int or not 0 < cap <= MAX_EXECUTION_SECONDS:
        raise ValueError("Explicit cap must fit the predeclared five-hour free-worker execution limit")
    if protocol.get("reservation_journal") != CANONICAL_JOURNAL:
        raise ValueError("Only the existing local canonical journal may authorize remote work")
    if protocol.get("remote_cpu_stratum") != RUNTIME:
        raise ValueError("Explicit unchanged Torch2.8 ARM stratum and fixed portable cache required")
    for profile in protocol["profiles"]:
        backend = profile["backend"]
        if backend not in ("solar", "sift", "modern"):
            raise ValueError("No model/runtime porting: ARM stratum permits compact/SIFT/LoMa/RoMa only")
        if backend == "modern" and (profile.get("method") not in ("loma_r", "romav2") or profile.get("author_profile") != "common"):
            raise ValueError("Untouched pinned LoMa-R/RoMa-v2 common profiles required")
        if backend == "modern":
            relative(profile["source"])
            if not profile["source"].startswith("data/models/"):
                raise ValueError("Dedicated portable official source directory required")
    for name in (str(protocol_path), protocol["output"], protocol["manifest"], protocol["geometry_audit"],
                 protocol["compute_ledger"], protocol["learning_contract"], *protocol["input_sha256"], *protocol["source_sha256"]):
        relative(name)
    if not {SELF, WORKFLOW, "solar_capsule_v75.py", "solar_portable_evidence_v75.py"} <= set(protocol["source_sha256"]):
        raise ValueError("Relay and exact workflow must be source-pinned before protocol freeze")
    validate_runtime(protocol, protocol_path, at=at)
    if check_assets:
        validate_declaration(protocol, read(LOCK))
        validate_release_inputs(protocol, Path.cwd())
        from solar_evaluation_protocol_v74 import qualified_cases
        manifest, audit, cases = qualified_cases(protocol["manifest"], protocol["geometry_audit"])
        if protocol["case_ids"] != [case["case_id"] for case in cases]:
            raise ValueError("Preserve every ordered eligible 12*N case and the raw32 inventory")
        for name, expected in {**protocol["source_sha256"], **protocol["input_sha256"]}.items():
            if sha(name) != expected:
                raise ValueError("Frozen source/input changed: " + name)
        validate_primary_archives(cases, manifest["reference_precompute_receipts"])
        validate_modern_assets(p=protocol)
    return cap


def validate_primary_archives(cases, receipts):
    """Fail before any neural construction if one generic physical column is absent."""
    import numpy as np
    for name in sorted({case["reference_geometry"] for case in cases}):
        pinned = receipts[name]
        if sha(pinned["receipt_path"]) != pinned["sha256"]:
            raise ValueError("Full reference preparation sidecar changed")
        receipt = read(pinned["receipt_path"])
        catalogue = receipt.get("sun_catalogue", [])
        if (receipt.get("complete") is not True or receipt.get("precompute") != "full"
                or receipt.get("uncomputed_solar_columns") != 0 or not catalogue
                or receipt.get("computed_solar_columns") != len(catalogue)
                or receipt.get("query_truth_used") is not False or receipt.get("output_sha256") != sha(name)):
            raise ValueError("Source-bound full generic reference preparation required")
        with np.load(name, allow_pickle=False) as archive:
            required = {"solar_materialized_mask", "sun_parameters", "coarse_sun_mask", "sun_vectors", "solar_states", "solar_variants"}
            if not required <= set(archive.files):
                raise ValueError("Remote primary physics requires an explicit fully materialized generic archive")
            mask = archive["solar_materialized_mask"]
            parameters = archive["sun_parameters"]
            coarse = archive["coarse_sun_mask"]
            suns = archive["sun_vectors"]
            if mask.dtype != np.bool_ or mask.ndim != 1 or not mask.size or not mask.all():
                raise ValueError("Uncomputed physical columns cannot become UNKNOWN or silently disable physics")
            if parameters.shape != (len(mask), 2) or suns.shape != (len(mask), 3) or coarse.shape != mask.shape:
                raise ValueError("Full generic solar catalogue shape differs from computed-column mask")
            expected_parameters = np.asarray([[row["hour_angle_deg"], row["subsolar_latitude_deg"]] for row in catalogue], float)
            expected_suns = np.asarray([row["sun_enu"] for row in catalogue], float)
            if not np.array_equal(parameters, expected_parameters) or not np.array_equal(suns, expected_suns):
                raise ValueError("Archive dropped or changed published generic Sun columns")
            if not np.isfinite(parameters).all() or not np.isfinite(suns).all() or not np.allclose(np.linalg.norm(suns, axis=1), 1, atol=1e-6):
                raise ValueError("Finite unit generic Sun catalogue required")
            h, d = parameters.T
            if (not np.allclose(h/1.25, np.round(h/1.25), atol=1e-8)
                    or not np.allclose(d/.25, np.round(d/.25), atol=1e-8)
                    or (np.abs(h) > 90).any() or (np.abs(d) > 1.5).any()
                    or len(np.unique(parameters, axis=0)) != len(mask)):
                raise ValueError("No query-specific or off-grid solar vectors in the primary generic archive")
            expected_coarse = (np.remainder(np.round(h/1.25).astype(int), 4) == 0) & (np.remainder(np.round(d/.25).astype(int), 2) == 0)
            if coarse.dtype != np.bool_ or not np.array_equal(coarse, expected_coarse) or not coarse.any():
                raise ValueError("Original coarse initialization and bounded fine catalogue required")
            states, variants = archive["solar_states"], archive["solar_variants"]
            if states.shape != (256, len(mask)) or variants.shape != (*states.shape, 7) or not np.isin(states, [0, 1, 2]).all() or not np.isin(variants, [0, 1, 2]).all():
                raise ValueError("All declared generic rays and seven terrain perturbations must be present")


def validate_modern_assets(p):
    """Only byte checks; no torch import, checkpoint decode or source fetch."""
    lock = read("reports/compact_baseline_lock_v2.json")["modern"]
    for profile in p["profiles"]:
        if profile["backend"] != "modern":
            continue
        provenance = read(profile["pretrained_provenance"])
        source = Path(profile["source"])
        for name, expected in provenance["tree_sha256"].items():
            path = str(source / name)
            if p["input_sha256"].get(path) != expected or sha(path) != expected:
                raise ValueError("Portable author source must match saved native-control tree: " + path)
        for name, expected in lock[profile["method"]]["weights"].items():
            path = TORCH_HOME + "/hub/checkpoints/" + name
            if p["input_sha256"].get(path) != expected or p.get("selected_weight_inputs", {}).get(path) != expected or sha(path) != expected:
                raise ValueError("Missing frozen official cache weight before neural construction: " + name)
        if profile["method"] == "romav2":
            directory = Path(TORCH_HOME) / "hub/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
            files = list(directory.rglob("*.py"))
            if not (directory / "hubconf.py").is_file() or len(files) < 100:
                raise ValueError("Pinned offline DINOv3 source cache required; no silent TorchHub fetch")
            if any(p["input_sha256"].get(str(path)) != sha(path) for path in files):
                raise ValueError("Every cached DINOv3 Python source must be frozen as a portable input")


def prepare_modern_sources(p, active_deadline=None):
    """Recreate Git provenance from only the exact two official source revisions."""
    validate_modern_assets(p)
    def command(args):
        deadline = min(utc(p["latest_start_utc"]), active_deadline or utc(p["latest_start_utc"]))
        remaining = (deadline-datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0: raise TimeoutError("Official source setup consumed original launch allowance")
        subprocess.run(args, check=True, timeout=min(120., remaining), env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    for profile in p["profiles"]:
        if profile["backend"] != "modern":
            continue
        source = Path(profile["source"])
        url, revision = UPSTREAM[profile["method"]]
        if not (source / ".git").exists():
            command(["git", "init", str(source)])
            command(["git", "-C", str(source), "fetch", "--depth=1", url, revision])
            command(["git", "-C", str(source), "checkout", "--detach", "--force", revision])
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
        if head != revision or status:
            raise ValueError("Cold source checkout differs from exact clean official author implementation")
    validate_modern_assets(p)


def pre_reserve(protocol_path, source_commit, authorization_path):
    """The ONLY reservation write. Run locally, before capsule export/dispatch."""
    protocol_path = relative(str(protocol_path)); authorization_path = relative(str(authorization_path))
    p = read(protocol_path)
    primary_scope(protocol_path, p)
    published(protocol_path, p, source_commit)
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve prior evaluation outputs before any new charge")
    if Path(authorization_path).exists():
        raise FileExistsError("Preserve prior authorization; no automatic reissue")
    write_new(authorization_path + ".ISSUING.json", dict(protocol_sha256=sha(protocol_path), no_retry=True))
    from solar_compute_budget_v67 import reserve
    reservation = reserve(p["compute_ledger"], p["compute_ledger_sha256"], CANONICAL_JOURNAL,
                          sha(protocol_path) + "/evaluation", "baseline", p["max_calendar_seconds"])
    raw = Path(CANONICAL_JOURNAL).read_bytes()
    value = dict(schema_version=75, repository=REPOSITORY, protocol_path=protocol_path,
        protocol_sha256=sha(protocol_path), source_publication_commit=source_commit,
        nonce=charge_nonce(sha(protocol_path), reservation["reservation"]), authorized_attempt=1, reservation=reservation,
        canonical_journal=CANONICAL_JOURNAL, canonical_journal_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_journal_bytes_base64=base64.b64encode(raw).decode("ascii"),
        original_latest_start_utc=p["latest_start_utc"], original_absolute_deadline_utc=p["absolute_deadline_utc"],
        max_calendar_seconds=p["max_calendar_seconds"], case_ids=p["case_ids"], tracks=p["tracks"],
        created_utc=datetime.now(timezone.utc).isoformat(), refunds=False, automatic_retry=False)
    write_new(authorization_path, value)
    return value


def validate_authorization(value, protocol_path, require_local_journal=False):
    p = read(protocol_path)
    if (value.get("schema_version") != 75 or value.get("repository") != REPOSITORY
            or value.get("protocol_path") != str(protocol_path) or value.get("protocol_sha256") != sha(protocol_path)
            or value.get("canonical_journal") != CANONICAL_JOURNAL or value.get("authorized_attempt") != 1
            or value.get("refunds") is not False or value.get("automatic_retry") is not False):
        raise ValueError("Original one-use authorization differs from this protocol")
    _hex(value["nonce"], HEX64, "One-use nonce required")
    raw = base64.b64decode(value["canonical_journal_bytes_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != value["canonical_journal_sha256"]:
        raise ValueError("Precharged canonical snapshot bytes changed")
    journal = json.loads(raw)
    entries = journal["reservations"]
    ids = [entry["job_id"] for entry in entries]
    entry = value["reservation"]["reservation"]
    if (len(ids) != len(set(ids)) or ids.count(sha(protocol_path) + "/evaluation") != 1
            or entries[ids.index(sha(protocol_path) + "/evaluation")] != entry
            or entry["category"] != "baseline" or entry["seconds"] != p["max_calendar_seconds"]
            or entry.get("refund_on_failure") is not False or journal["ledger_sha256"] != p["compute_ledger_sha256"]):
        raise ValueError("Exact original canonical job/cap/UTC entry required; no duplicate/refund")
    if value["nonce"] != charge_nonce(sha(protocol_path), entry):
        raise ValueError("One-use nonce must identify the complete original canonical charge; no alternate ticket identity")
    if (value["original_latest_start_utc"] != p["latest_start_utc"]
            or value["original_absolute_deadline_utc"] != p["absolute_deadline_utc"]
            or value["max_calendar_seconds"] != p["max_calendar_seconds"]
            or value["case_ids"] != p["case_ids"] or value["tracks"] != ["estimated_sun"]):
        raise ValueError("Ticket cannot refresh lease/deadline, shrink cases, or expose known query Sun")
    if require_local_journal:
        current = read(CANONICAL_JOURNAL)
        if current["ledger_sha256"] != journal["ledger_sha256"] or current["reservations"][:len(entries)] != entries:
            raise ValueError("Local append-only canonical history changed; never import a worker journal")
    return raw


def freeze_ticket(authorization_path, capsule_manifest_path, archive_path, parts, ticket_path):
    """Create reviewable ticket metadata; publishing/tagging/sending are separate."""
    authorization_path = relative(str(authorization_path)); ticket_path = relative(str(ticket_path))
    auth = read(authorization_path); validate_authorization(auth, auth["protocol_path"], True)
    manifest = read(capsule_manifest_path)
    if (manifest.get("schema_version") != 75 or manifest.get("capsule_kind") != "model_selection_primary_full_sun"
            or manifest.get("protocol_path") != auth["protocol_path"] or manifest.get("protocol_sha256") != auth["protocol_sha256"]
            or manifest.get("publication_commit") != auth["source_publication_commit"]
            or manifest.get("pre_reservation_authorization") != dict(path=authorization_path, sha256=sha(authorization_path))
            or manifest.get("files", {}).get(authorization_path) != sha(authorization_path)):
        raise ValueError("Capsule must carry the identical post-charge authorization/snapshot")
    if not isinstance(parts, list) or len(parts) != 1:
        raise ValueError("The thin capsule must be one bounded public release asset")
    total = 0
    for part in parts:
        if set(part) != {"url", "sha256", "bytes"} or not RELEASE_URL.fullmatch(part["url"]):
            raise ValueError("Only safe public assets in this repository; no arbitrary URL/token")
        _hex(part["sha256"], HEX64, "Hash every original public chunk")
        if type(part["bytes"]) is not int or not 0 < part["bytes"] <= MAX_PART_BYTES:
            raise ValueError("Explicit bounded public asset size required")
        total += part["bytes"]
    if len({part["url"] for part in parts}) != len(parts) or total > MAX_ARCHIVE_BYTES or Path(archive_path).stat().st_size != total:
        raise ValueError("Ordered archive chunks must cover one bounded immutable ZIP")
    with Path(archive_path).open("rb") as original:
        for part in parts:
            remaining = part["bytes"]; h = hashlib.sha256()
            while remaining:
                chunk = original.read(min(8 * 1024 * 1024, remaining))
                if not chunk: raise ValueError("Truncated original ZIP part")
                remaining -= len(chunk); h.update(chunk)
            if h.hexdigest() != part["sha256"]:
                raise ValueError("Published part hashes do not match the frozen local ZIP")
    p = read(auth["protocol_path"])
    target_software = None
    if p["purpose"] != "cost_pilot":
        pilot = read(p["cost_pilot_summary"])
        target_software = pilot["software"]
        if target_software.get("torch") != "2.8.0+cpu" or target_software.get("cpu_threads") != 2 or target_software.get("machine") not in ("aarch64", "arm64"):
            raise ValueError("Full remote crosscheck must use a measured ARM2.8 pilot, never the local Intel pilot")
    ticket = dict(schema_version=75, repository=REPOSITORY, workflow=WORKFLOW, workflow_name=WORKFLOW_NAME,
        nonce=auth["nonce"], authorized_attempt=1, source_publication_commit=auth["source_publication_commit"],
        protocol_path=auth["protocol_path"], protocol_sha256=auth["protocol_sha256"],
        authorization=dict(path=authorization_path, sha256=sha(authorization_path)),
        capsule=dict(parts=parts, archive_sha256=sha(archive_path), archive_bytes=total, manifest_sha256=sha(capsule_manifest_path),
                     transport="public_projected_full_closure_v98"),
        original_latest_start_utc=auth["original_latest_start_utc"], original_absolute_deadline_utc=auth["original_absolute_deadline_utc"],
        max_calendar_seconds=auth["max_calendar_seconds"], required_runner="ubuntu-24.04-arm", cpu_threads=2,
        expected_run_title="solar-v98-" + auth["nonce"], expected_branch="solar-v98-ticket-" + auth["nonce"],
        refunds=False, automatic_retry=False, primary_only=True, recorded_pilot_software=target_software)
    write_new(authorization_path + ".TICKET_ISSUED.json", dict(ticket_path=ticket_path, nonce=auth["nonce"], no_reissue=True))
    write_new(ticket_path, ticket)
    return ticket


def validate_ticket(ticket_path, expected_sha=None, expected_nonce=None):
    relative(str(ticket_path)); ticket = read(ticket_path)
    if expected_sha is not None and sha(ticket_path) != _hex(expected_sha, HEX64, "Exact ticket hash required"):
        raise ValueError("One-use ticket bytes changed")
    nonce = _hex(ticket.get("nonce"), HEX64, "Exact ticket nonce required")
    if expected_nonce is not None and nonce != expected_nonce:
        raise ValueError("Workflow input nonce differs from ticket")
    if (ticket.get("schema_version") != 75 or ticket.get("repository") != REPOSITORY or ticket.get("workflow") != WORKFLOW
            or ticket.get("authorized_attempt") != 1 or ticket.get("cpu_threads") != 2
            or ticket.get("required_runner") != "ubuntu-24.04-arm" or ticket.get("primary_only") is not True
            or ticket.get("refunds") is not False or ticket.get("automatic_retry") is not False
            or ticket.get("expected_run_title") != "solar-v98-" + nonce or ticket.get("expected_branch") != "solar-v98-ticket-" + nonce):
        raise ValueError("Only the original standard free ARM one-attempt ticket is supported")
    _hex(ticket["source_publication_commit"], HEX40, "Source commit required")
    relative(ticket["protocol_path"]); relative(ticket["authorization"]["path"])
    _hex(ticket["authorization"]["sha256"], HEX64, "Authorization hash required")
    _hex(ticket["capsule"]["archive_sha256"], HEX64, "Archive hash required")
    _hex(ticket["capsule"]["manifest_sha256"], HEX64, "Manifest hash required")
    if ticket["capsule"].get("transport") != "public_projected_full_closure_v98" or len(ticket["capsule"]["parts"]) != 1:
        raise ValueError("Exact thin capsule transport required")
    cap = ticket["max_calendar_seconds"]
    if type(cap) is not int or not 0 < cap <= MAX_EXECUTION_SECONDS:
        raise ValueError("Published five-hour remote limit exceeded")
    return ticket


def public_repository_api(suffix):
    """Anonymous, read-only GitHub access for exact refs, commits and assets."""
    allowed = r"(?:/git/ref/heads/solar-v98-ticket-[0-9a-f]{64}|/git/commits/[0-9a-f]{40}|/releases/tags/[A-Za-z0-9_.-]+|/actions/workflows/solar-remote-cpu-v98\.yml)?"
    if not re.fullmatch(allowed, suffix) or any(part in (".", "..") for part in suffix.split("/")):
        raise ValueError("Only this public repository and exact reviewed metadata are permitted")
    url = "https://api.github.com/repos/" + REPOSITORY + suffix
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "solar-browser-gate-v98",
    })
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.geturl() != url:
            raise ValueError("Public GitHub metadata unexpectedly redirected")
        data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Public GitHub metadata exceeded its bound")
    return json.loads(data)


def public_dispatch_evidence(ticket, ticket_commit, api=public_repository_api):
    """The signed exact branch and already-uploaded binary must exist before run."""
    _hex(ticket_commit, HEX40, "Exact reviewed ticket commit required")
    repository = api("")
    if repository.get("full_name") != REPOSITORY or repository.get("private") is not False or repository.get("visibility") != "public":
        raise ValueError("Only this public repository may use the free ARM workflow")
    branch = ticket["expected_branch"]
    ref_name = "refs/heads/" + branch
    ref = api("/git/ref/heads/" + branch)
    if (ref.get("ref") != ref_name or ref.get("object", {}).get("type") != "commit"
            or ref.get("object", {}).get("sha") != ticket_commit):
        raise ValueError("Public one-use branch no longer points to the reviewed commit")
    commit = api("/git/commits/" + ticket_commit)
    verification = commit.get("verification", {})
    if commit.get("sha") != ticket_commit or verification.get("verified") is not True or verification.get("reason") != "valid":
        raise ValueError("A GitHub-verified signature on the exact ticket commit is required")
    workflow = api("/actions/workflows/" + Path(WORKFLOW).name)
    if workflow.get("state") != "active" or workflow.get("path") != WORKFLOW:
        raise ValueError("The exact workflow must be active on the default branch")
    part, = ticket["capsule"]["parts"]
    if (not RELEASE_URL.fullmatch(part["url"]) or part["bytes"] != ticket["capsule"]["archive_bytes"]
            or part["sha256"] != ticket["capsule"]["archive_sha256"]):
        raise ValueError("One exact signed public release asset must cover the thin archive")
    tag, asset_name = urllib.parse.urlsplit(part["url"]).path.rsplit("/", 2)[-2:]
    if tag in (".", "..") or asset_name in (".", "..") or tag.startswith("solar-v98-ticket-"):
        raise ValueError("Transport release must be independent of the one-use dispatch branch")
    release = api("/releases/tags/" + tag)
    matches = [row for row in release.get("assets", []) if row.get("name") == asset_name]
    if (release.get("draft") is not False or release.get("tag_name") != tag or not release.get("published_at")
            or len(matches) != 1):
        raise ValueError("The transport release and exact asset must already be public")
    asset = matches[0]
    if (asset.get("state") != "uploaded" or asset.get("size") != part["bytes"]
            or asset.get("digest") != "sha256:" + part["sha256"] or asset.get("browser_download_url") != part["url"]
            or type(asset.get("id")) is not int or asset["id"] <= 0):
        raise ValueError("Published asset is missing the exact uploaded size and SHA256")
    return dict(ticket_commit=ticket_commit, branch=branch, verified_commit=True,
                release_asset_id=asset["id"], archive_bytes=part["bytes"], archive_sha256=part["sha256"])


def verify_ticket_checkout(ticket_path, ticket, head):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != head:
        raise ValueError("The checkout must be the exact reviewed workflow commit")
    raw = subprocess.check_output(["git", "show", head + ":" + ticket_path])
    if raw != Path(ticket_path).read_bytes() or sha(ticket["protocol_path"]) != ticket["protocol_sha256"]:
        raise ValueError("Signed ticket or published protocol bytes differ")
    published(ticket["protocol_path"], read(ticket["protocol_path"]), ticket["source_publication_commit"])


def prepare_browser_dispatch(ticket_path, ticket_publication_commit):
    """Create the one local marker after read-only exact public checks."""
    ticket = validate_ticket(ticket_path)
    _alive_lease(ticket)
    _hex(ticket_publication_commit, HEX40, "Exact ticket publication commit required")
    auth = read(ticket["authorization"]["path"])
    if sha(ticket["authorization"]["path"]) != ticket["authorization"]["sha256"]:
        raise ValueError("Authorization changed after ticket freeze")
    validate_authorization(auth, ticket["protocol_path"], True)
    primary_scope(ticket["protocol_path"], read(ticket["protocol_path"]))
    published(ticket["protocol_path"], read(ticket["protocol_path"]), ticket_publication_commit)
    if subprocess.check_output(["git", "show", ticket_publication_commit + ":" + str(ticket_path)]) != Path(ticket_path).read_bytes():
        raise ValueError("Ticket itself must be committed before browser dispatch preparation")
    evidence = public_dispatch_evidence(ticket, ticket_publication_commit)
    verify_release_download_head(ticket["capsule"]["parts"][0])
    _alive_lease(ticket)
    value = dict(ticket_sha256=sha(ticket_path), nonce=ticket["nonce"],
        ticket_publication_commit=ticket_publication_commit, evidence=evidence,
        dispatched_utc=datetime.now(timezone.utc).isoformat(), authorized_attempt=1, automatic_retry=False,
        mode="prepared_browser_dispatch_once", dispatch_url="https://github.com/" + REPOSITORY + "/actions/workflows/" + Path(WORKFLOW).name,
        reviewed_branch=ticket["expected_branch"], inputs=dict(ticket=str(ticket_path), ticket_sha256=sha(ticket_path),
            ticket_nonce=ticket["nonce"], ticket_commit=ticket_publication_commit),
        note="Use the reviewed branch and inputs once. Do not repeat after an ambiguous UI response.")
    write_new(str(ticket_path) + ".DISPATCHED.json", value)
    return value


def github_api(path):
    """Token is confined to GitHub API; it is never sent to capsule URLs/logs."""
    if not path.startswith("repos/" + REPOSITORY + "/actions/"):
        raise ValueError("Only this public repository's workflow metadata may be requested")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "solar-primary-relay-v75"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(urllib.request.Request("https://api.github.com/" + path, headers=headers), timeout=30) as response:
        return json.load(response)


def workflow_binding(ticket, ticket_commit, environment=None, api=github_api):
    env = os.environ if environment is None else environment
    if (env.get("GITHUB_ACTIONS") != "true" or env.get("GITHUB_REPOSITORY") != REPOSITORY
            or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or env.get("GITHUB_RUN_ATTEMPT") != "1"
            or env.get("RUNNER_OS") != "Linux" or env.get("RUNNER_ARCH") != "ARM64"
            or env.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or env.get("GITHUB_REF") != "refs/heads/" + ticket["expected_branch"]):
        raise ValueError("Attempt1 on the standard Linux ARM free worker is required")
    run_id = env.get("GITHUB_RUN_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("Actual GitHub workflow run ID required")
    head = _hex(env.get("GITHUB_SHA"), HEX40, "Exact workflow source commit required")
    expected_workflow_ref = REPOSITORY + "/" + WORKFLOW + "@" + env["GITHUB_REF"]
    if (head != _hex(ticket_commit, HEX40, "Exact reviewed ticket commit input required")
            or env.get("GITHUB_WORKFLOW_SHA") != head or env.get("GITHUB_WORKFLOW_REF") != expected_workflow_ref):
        raise ValueError("Actual workflow and checkout must come from the exact reviewed one-use branch commit")
    current = api("repos/" + REPOSITORY + "/actions/runs/" + run_id)
    if (str(current.get("id")) != run_id or current.get("event") != "workflow_dispatch" or current.get("run_attempt") != 1
            or current.get("head_sha") != head or current.get("head_branch") != ticket["expected_branch"]
            or current.get("path", "").split("@")[0] != WORKFLOW
            or current.get("display_title") != ticket["expected_run_title"]
            or current.get("repository", {}).get("full_name") != REPOSITORY
            or current.get("head_repository", {}).get("full_name") != REPOSITORY):
        raise ValueError("Actual workflow run/source/ticket bindings differ")
    # Concurrency serializes the same nonce. A second dispatch sees both run
    # records and cannot execute, even on a fresh VM with no local marker.
    hits = []
    for page in range(1, 21):
        response = api("repos/" + REPOSITORY + "/actions/workflows/" + Path(WORKFLOW).name + "/runs?event=workflow_dispatch&per_page=100&page=" + str(page))
        runs = response["workflow_runs"]
        hits.extend(run for run in runs if run.get("display_title") == ticket["expected_run_title"])
        if len(runs) < 100:
            break
    else:
        raise ValueError("Workflow history exceeds bounded uniqueness audit; refuse instead of assuming one-use")
    if len(hits) != 1 or str(hits[0]["id"]) != run_id or hits[0].get("run_attempt") != 1:
        raise ValueError("Duplicate invocation/re-dispatch/rerun of a one-use ticket")
    return dict(workflow_run_id=int(run_id), run_attempt=1, workflow_source_commit=head,
        source_publication_commit=ticket["source_publication_commit"], ticket_nonce=ticket["nonce"],
        reviewed_branch=ticket["expected_branch"], reviewed_ticket_commit=ticket_commit,
        html_url=current.get("html_url"), workflow_started_utc=current.get("run_started_at"))


def restore_precharged_journal(ticket, destination="."):
    """Remote scratch checkout ONLY. Never call this to import returned results."""
    if Path(destination).resolve() != Path.cwd().resolve() or os.environ.get("GITHUB_ACTIONS") != "true":
        raise ValueError("Canonical snapshot may only be installed in the fresh GitHub worker checkout")
    auth_path = ticket["authorization"]["path"]
    if sha(auth_path) != ticket["authorization"]["sha256"]:
        raise ValueError("Capsule authorization hash changed")
    raw = validate_authorization(read(auth_path), ticket["protocol_path"])
    path = Path(CANONICAL_JOURNAL)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Never replace a differing canonical journal, even on the worker")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(raw)
    return read(auth_path)["reservation"]


def consume_pre_reservation(ticket_path, protocol_path):
    """Optional existing-controller hook; returns original entry without reserve()."""
    ticket = validate_ticket(ticket_path)
    auth = read(ticket["authorization"]["path"])
    if ticket["protocol_path"] != protocol_path or sha(ticket["authorization"]["path"]) != ticket["authorization"]["sha256"]:
        raise ValueError("Pre-reserved invocation is bound to another protocol")
    validate_authorization(auth, protocol_path, True)
    context_path = str(ticket_path) + ".WORKER_CONTEXT.json"
    context = read(context_path)
    if (context.get("ticket_sha256") != sha(ticket_path) or context.get("nonce") != ticket["nonce"]
            or str(context.get("workflow_run_id")) != os.environ.get("GITHUB_RUN_ID")
            or context.get("run_attempt") != 1 or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
            or context.get("parent_pid") != os.getppid()):
        raise ValueError("Only the authenticated one-use remote parent controller can consume this charge")
    return auth["reservation"]


def validate_worker_ticket(ticket_path, protocol_path):
    """The inner worker must have the authenticated controller as its parent."""
    ticket = validate_ticket(ticket_path)
    auth = read(ticket["authorization"]["path"])
    if ticket["protocol_path"] != protocol_path or sha(ticket["authorization"]["path"]) != ticket["authorization"]["sha256"]:
        raise ValueError("Worker ticket/authorization belongs to another invocation")
    validate_authorization(auth, protocol_path, True)
    p = read(protocol_path); context = read(str(ticket_path) + ".WORKER_CONTEXT.json")
    watchdog = read(Path(p["output"]) / "watchdog_admission.json")
    controller_parent = int(subprocess.check_output(["ps", "-o", "ppid=", "-p", str(os.getppid())], text=True).strip())
    if (context.get("ticket_sha256") != sha(ticket_path) or context.get("nonce") != ticket["nonce"]
            or str(context.get("workflow_run_id")) != os.environ.get("GITHUB_RUN_ID")
            or os.environ.get("GITHUB_RUN_ATTEMPT") != "1" or context.get("run_attempt") != 1
            or watchdog.get("controller_pid") != os.getppid() or controller_parent != context.get("parent_pid")
            or watchdog.get("protocol_sha256") != sha(protocol_path)):
        raise ValueError("Worker must descend from the actual one-use parent and scientific controller")
    if read(Path(p["output"]) / "reservation.json") != auth["reservation"]:
        raise ValueError("Worker must consume the identical original reservation and UTC")
    return auth["reservation"]


def _alive_lease(ticket, at=None):
    current = at or datetime.now(timezone.utc)
    if current > utc(ticket["original_latest_start_utc"]) or current >= utc(ticket["original_absolute_deadline_utc"]):
        raise TimeoutError("Original worker launch lease expired; queue/setup cannot refresh it")


class _ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in ("github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"):
            raise ValueError("Capsule redirect leaves public GitHub release assets")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ReleaseHeadRedirect(_ReleaseRedirect):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        redirected.method = "HEAD"
        return redirected


def verify_release_download_head(part, opener=None):
    """Reject an asset still unavailable at the public download URL."""
    if not RELEASE_URL.fullmatch(part["url"]):
        raise ValueError("The reviewed release download URL changed")
    if opener is None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _ReleaseHeadRedirect())
    request = urllib.request.Request(part["url"], method="HEAD", headers={
        "Accept-Encoding": "identity", "User-Agent": "solar-browser-gate-v98",
    })
    try:
        with opener.open(request, timeout=15) as response:
            if response.status != 200 or response.headers.get("Content-Length") != str(part["bytes"]):
                raise ValueError("The public capsule download URL is not yet available at the exact size")
    except urllib.error.URLError:
        raise ConnectionError("The public capsule download URL is unavailable") from None


def fetch_capsule(ticket, output, now=lambda: datetime.now(timezone.utc), active_deadline=None):
    capsule = ticket["capsule"]; total = 0; whole = hashlib.sha256()
    if not 1 <= len(capsule["parts"]) <= 16:
        raise ValueError("Bounded original public asset parts required")
    opener = urllib.request.build_opener(_ReleaseRedirect())
    with Path(output).open("xb") as stream:
        for part in capsule["parts"]:
            if not RELEASE_URL.fullmatch(part["url"]) or type(part["bytes"]) is not int or not 0 < part["bytes"] <= MAX_PART_BYTES:
                raise ValueError("Only bounded hash-pinned assets in this public repository")
            _hex(part["sha256"], HEX64, "Public chunk SHA required")
            h = hashlib.sha256(); count = 0
            with opener.open(part["url"], timeout=30) as response:
                while True:
                    if now() >= min(utc(ticket["original_latest_start_utc"]), active_deadline or utc(ticket["original_latest_start_utc"])):
                        raise TimeoutError("Capsule transport consumed the original launch lease")
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk); total += len(chunk)
                    if count > part["bytes"] or total > MAX_ARCHIVE_BYTES:
                        raise ValueError("Capsule download exceeds frozen size")
                    h.update(chunk); whole.update(chunk); stream.write(chunk)
            if count != part["bytes"] or h.hexdigest() != part["sha256"]:
                raise ValueError("Public capsule chunk bytes/hash differ")
    if total != capsule["archive_bytes"] or whole.hexdigest() != capsule["archive_sha256"]:
        raise ValueError("Combined immutable capsule differs")


def terminate_descendants(child):
    """Kill nested process sessions too, including the inner scientific worker."""
    listing = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    children = {}
    for line in listing.splitlines():
        pid, parent = map(int, line.split())
        children.setdefault(parent, []).append(pid)
    descendants = []; queue = [child.pid]
    while queue:
        pid = queue.pop(); descendants.append(pid); queue.extend(children.get(pid, []))
    for pid in reversed(descendants):
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: child.wait(timeout=7)
    except subprocess.TimeoutExpired: pass
    for pid in reversed(descendants):
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass
    child.wait()


def preserve_results(root, protocol, ticket_path, protocol_path):
    """Return scientific rows/logs/arrays/seals, never the input capsule/weights."""
    output = Path(relative(protocol["output"]))
    files = [path for path in output.rglob("*") if path.is_file()] if output.exists() else []
    files += [Path(ticket_path), Path(protocol_path), Path(read(ticket_path)["authorization"]["path"])]
    files = sorted(set(files))
    if len(files) > 30000 or any(path.is_symlink() for path in files):
        raise ValueError("Bounded nonsymlink scientific outputs required")
    total = sum(path.stat().st_size for path in files)
    if total > MAX_RESULT_BYTES:
        raise ValueError("Scientific result archive exceeds predeclared2GiB bound")
    result = Path(root) / "scientific_results.zip"
    with zipfile.ZipFile(result, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, relative(str(path)))
    return dict(path=str(result), sha256=sha(result), expanded_bytes=total, file_count=len(files),
        scientific_output=protocol["output"], includes_failed_partial_outputs=True, input_capsule_included=False, model_weights_included=False)


def execute_remote(ticket_path, expected_sha, expected_nonce, ticket_commit, python=sys.executable):
    """One parent deadline covers transport, source setup and scientific work."""
    ticket = validate_ticket(ticket_path, expected_sha, expected_nonce)
    _alive_lease(ticket)
    binding = workflow_binding(ticket, ticket_commit)
    root = Path("runs/solar-remote-v98") / ticket["nonce"]
    root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc); start_tick = time.monotonic()
    deadline = min(started + timedelta(seconds=ticket["max_calendar_seconds"]), utc(ticket["original_absolute_deadline_utc"]))
    child = None; pending = []; complete = False; failure = None; result_artifact = None
    setup_seconds = None; p = None; phase = "setup"; context = dict(binding, ticket_sha256=sha(ticket_path), nonce=ticket["nonce"], parent_pid=os.getpid())
    def interrupted(signum, frame): pending.append(signum)
    old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    def check():
        if pending or datetime.now(timezone.utc) >= deadline or time.monotonic()-start_tick >= ticket["max_calendar_seconds"]:
            raise TimeoutError("One-use parent allowance exhausted/interrupted; never refresh or retry")
    try:
        _alive_lease(ticket); check()
        verify_ticket_checkout(ticket_path, ticket, ticket_commit)
        context["public_dispatch_evidence"] = public_dispatch_evidence(ticket, ticket_commit)
        _alive_lease(ticket); check()
        downloaded = root / "capsule.zip"; extracted = root / "capsule"
        fetch_capsule(ticket, downloaded, active_deadline=deadline); check()
        from solar_public_transport_v98 import hydrate_thin_zip
        setup_deadline = min(deadline, utc(ticket["original_latest_start_utc"]))

        def check_setup():
            check()
            if datetime.now(timezone.utc) >= setup_deadline:
                raise TimeoutError("Official checkpoint setup consumed the original launch lease")

        def remaining_setup_seconds():
            check_setup()
            return (setup_deadline - datetime.now(timezone.utc)).total_seconds()

        phase = "official_checkpoint_hydration"
        context.update(hydrate_thin_zip(
            downloaded, extracted, ticket["capsule"]["manifest_sha256"], ticket["protocol_sha256"],
            check_setup, remaining_setup_seconds))
        phase = "setup"
        from solar_capsule_v75 import install_capsule
        install_capsule(extracted, Path.cwd()); check_setup()
        restore_precharged_journal(ticket)
        p = read(ticket["protocol_path"])
        primary_scope(ticket["protocol_path"], p); check()
        published(ticket["protocol_path"], p, ticket["source_publication_commit"])
        prepare_modern_sources(p, active_deadline=deadline); check()
        binding = workflow_binding(ticket, ticket_commit); _alive_lease(ticket); check()
        if public_dispatch_evidence(ticket, ticket_commit) != context["public_dispatch_evidence"]:
            raise ValueError("Public dispatch evidence changed during setup")
        _alive_lease(ticket); check()
        setup_seconds = time.monotonic()-start_tick
        context.update(binding, capsule_setup_seconds=setup_seconds, controller_started_utc=datetime.now(timezone.utc).isoformat())
        write_new(str(ticket_path) + ".WORKER_CONTEXT.json", context)
        phase = "evaluation"
        command = [python, "run_solar_evaluation_v98.py", "--protocol", ticket["protocol_path"], "--publication-commit", ticket["source_publication_commit"],
                   "--python", python, "--pre-reserved-ticket", ticket_path]
        with (root / "controller.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1"))
            while child.poll() is None:
                check(); time.sleep(.1)
            check()
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
            completion = read(Path(p["output"]) / "COMPLETE.json")
            if completion.get("complete") is not True or completion.get("protocol_sha256") != ticket["protocol_sha256"] or completion.get("external_watchdog_passed") is not True:
                raise ValueError("Inner scientific watchdog did not complete")
            complete = True
    except BaseException as exc:
        failure = dict(error_type=type(exc).__name__, error=str(exc), phase=phase)
        if child is not None and child.poll() is None:
            terminate_descendants(child)
        raise
    finally:
        # Preserve the actual science directory even after setup/controller
        # failure, never merely a successful-looking outer status JSON.
        archive_failure = None
        try:
            if p is not None:
                result_artifact = preserve_results(root, p, ticket_path, ticket["protocol_path"])
        except BaseException as exc:
            archive_failure = dict(error_type=type(exc).__name__, error=str(exc))
            complete = False
        if complete and (datetime.now(timezone.utc) >= deadline or time.monotonic()-start_tick >= ticket["max_calendar_seconds"]):
            complete = False
            failure = dict(error_type="TimeoutError", error="Result preservation exceeded original remote allowance", phase="result_preservation")
        auth_path = ticket["authorization"]["path"]
        reservation = read(auth_path)["reservation"] if Path(auth_path).is_file() else None
        record = dict(complete=complete, **context, protocol_sha256=ticket["protocol_sha256"], ticket_path=ticket_path,
            relay_source_sha256=sha(__file__), workflow_source_sha256=sha(WORKFLOW),
            authorization_sha256=ticket["authorization"]["sha256"], original_reservation=reservation,
            started_utc=started.isoformat(), absolute_deadline_utc=deadline.isoformat(), cap_seconds=ticket["max_calendar_seconds"],
            calendar_seconds=time.monotonic()-start_tick, setup_seconds=setup_seconds,
            child_returncode=None if child is None else child.returncode, phase=phase, result_artifact=result_artifact,
            failure=failure, archive_failure=archive_failure, automatic_retry=False, reservation_refunded=False, primary_only=True,
            note="Remote scratch journal is evidence only; never replace the live local canonical journal. Outer cap includes capsule transport/source setup; environment installation consumes the original launch lease.")
        write_new(root / "remote_watchdog.json", record)
        for sig, handler in old.items(): signal.signal(sig, handler)
        if archive_failure is not None and failure is None:
            raise RuntimeError("Scientific result preservation failed: " + archive_failure["error"])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pre-reserve"); p.add_argument("--protocol", required=True); p.add_argument("--source-commit", required=True); p.add_argument("--authorization", required=True)
    p = sub.add_parser("make-thin-zip"); p.add_argument("--capsule", required=True); p.add_argument("--archive", required=True)
    p = sub.add_parser("freeze-ticket"); p.add_argument("--authorization", required=True); p.add_argument("--capsule-manifest", required=True); p.add_argument("--archive", required=True); p.add_argument("--parts-json", required=True); p.add_argument("--ticket", required=True)
    p = sub.add_parser("prepare-browser-dispatch"); p.add_argument("--ticket", required=True); p.add_argument("--ticket-publication-commit", required=True)
    p = sub.add_parser("remote"); p.add_argument("--ticket", required=True); p.add_argument("--ticket-sha256", required=True); p.add_argument("--ticket-nonce", required=True); p.add_argument("--ticket-commit", required=True)
    p = sub.add_parser("verify-ticket"); p.add_argument("--ticket", required=True); p.add_argument("--ticket-sha256", required=True); p.add_argument("--ticket-nonce", required=True); p.add_argument("--ticket-commit", required=True)
    args = parser.parse_args()
    if args.command == "pre-reserve": pre_reserve(args.protocol, args.source_commit, args.authorization)
    elif args.command == "make-thin-zip":
        from solar_public_transport_v98 import make_thin_zip
        print(json.dumps(make_thin_zip(args.capsule, args.archive), indent=2))
    elif args.command == "freeze-ticket": freeze_ticket(args.authorization, args.capsule_manifest, args.archive, read(args.parts_json), args.ticket)
    elif args.command == "prepare-browser-dispatch":
        print(json.dumps(prepare_browser_dispatch(args.ticket, args.ticket_publication_commit), indent=2))
    elif args.command == "verify-ticket":
        ticket = validate_ticket(args.ticket, args.ticket_sha256, args.ticket_nonce)
        _alive_lease(ticket)
        workflow_binding(ticket, args.ticket_commit)
        verify_ticket_checkout(args.ticket, ticket, args.ticket_commit)
        public_dispatch_evidence(ticket, args.ticket_commit)
        verify_release_download_head(ticket["capsule"]["parts"][0])
        _alive_lease(ticket)
    else: execute_remote(args.ticket, args.ticket_sha256, args.ticket_nonce, args.ticket_commit)
