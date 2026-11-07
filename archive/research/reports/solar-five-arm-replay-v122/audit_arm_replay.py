"""Run the unchanged v115 verifier on native Linux ARM without model packages."""

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


REPO = "ThomasdeChillaz/lunar-image-matching"
RUN = 41941904269
ARTIFACT = 13386000111
REPO_ID = 1433662384
TICKET_COMMIT = "a95d290d9ecd001b2316e90285ae7bb2bff28c87"
TICKET_SHA = "5394f9ed554357c3c5e20c0a4d9d98132c6e3b5a71ac330e15a43970212b7c36"
NONCE = "f4185e74104c0a62d3b14d64e61cc6b3ef622feec4b28c3191ceba871855dd11"
PROTOCOL_SHA = "c43b7a0a5236b27055a7179caf1cfa1a274b18743a9fd868d0054fa96d23878c"
OUTER_BYTES = 13164199
OUTER_SHA = "813f2c9e6c5986ae711ab885d3a6c25a146cb87fead8261f99984e689f5e538b"
BASE_BYTES = 563675388
BASE_SHA = "2d25dca8b56f86f10444289b7458aeed4ba9a6755bf1ff19fca7014afaa1dee0"
VERIFIER_SHA = "d77bfa4a298eb0497f3f72a9b15acf62a78d97917c23ff3fcf51a3747acaeb5b"
LOCK_SHA = "48ddb80c82a0463c9ad59d4b0dbd54665d20e035847b582fbf0581522e5b0f72"
API = f"https://api.github.com/repos/{REPO}/actions/artifacts/{ARTIFACT}"
BASE = f"https://github.com/{REPO}/releases/download/solar-arm-transport-v98/solar-arm-v98-public.zip"
TOKEN_ENV = "V115_ARTIFACT_READ_TOKEN"
WHEELS = ("numpy", "opencv-python")
RECEIPTS = (
    "artifact_audit_v115.json", "github_v115_metadata.json", "nested_cell_inventory_v115.json",
    "independent_replay_v115.json", "matched_comparison_v115.json", "replay_input_hashes_v115.json",
)
UA = "lunar-v115-native-arm-audit"


def require(condition, error):
    if not condition:
        raise ValueError(error)


def sha(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def exact_file(path, size, checksum):
    require(path.stat().st_size == size and sha(path) == checksum, "Frozen file size or SHA-256 differs")


def wheel_lines(raw):
    selected = [line for line in raw.splitlines() if line.split(" @ ", 1)[0] in WHEELS]
    require(len(selected) == 2 and {line.split(" @ ", 1)[0] for line in selected} == set(WHEELS),
            "Exactly one frozen NumPy and OpenCV ARM wheel line required")
    require(all("aarch64" in line and " --hash=sha256:" in line for line in selected),
            "Native ARM wheels must retain their original hashes")
    return "\n".join(selected) + "\n"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        return None


def github_request(url, token):
    require(url in (API, API + "/zip") and token, "Exact same-repository artifact read token required")
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": UA})
    request.add_unredirected_header("Authorization", "Bearer " + token)
    return request


def allowed_download_url(url):
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    require(parsed.scheme == "https" and parsed.port in (None, 443) and not parsed.username and not parsed.password
            and (host == "github.com" or host.endswith((".githubusercontent.com", ".blob.core.windows.net"))),
            "Artifact transfer must use a GitHub or GitHub-signed Azure HTTPS location")
    return url


class DownloadRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        return Request(allowed_download_url(url), headers={"User-Agent": UA})


def public_download(url, target, size, checksum):
    opener = build_opener(DownloadRedirect())
    request = Request(allowed_download_url(url), headers={"User-Agent": UA})
    with opener.open(request, timeout=40) as response, target.open("xb") as file:
        written = 0
        while chunk := response.read(1024**2):
            written += len(chunk)
            require(written <= size, "Download exceeded the frozen byte count")
            file.write(chunk)
    exact_file(target, size, checksum)


def validate_origin(metadata):
    branch = "solar-five-arm-v115-ticket-" + NONCE
    artifact = {
        "id": ARTIFACT, "name": f"solar-five-arm-v115-{NONCE}-{RUN}-attempt-1",
        "size_in_bytes": OUTER_BYTES, "digest": "sha256:" + OUTER_SHA, "expired": False,
        "archive_download_url": API + "/zip",
    }
    origin = {
        "id": RUN, "repository_id": REPO_ID, "head_repository_id": REPO_ID,
        "head_branch": branch, "head_sha": TICKET_COMMIT,
    }
    origin_metadata = metadata.get("workflow_run") or {}
    require(all(metadata.get(k) == value for k, value in artifact.items())
            and all(origin_metadata.get(k) == value for k, value in origin.items()),
            "Exact artifact REST identity or first-run origin differs")


def official_download(token, target):
    api = build_opener(NoRedirect())
    with api.open(github_request(API, token), timeout=20) as response:
        payload = response.read(65537)
    require(len(payload) <= 65536, "Unexpected REST metadata size")
    validate_origin(json.loads(payload))
    try:
        with api.open(github_request(API + "/zip", token), timeout=20):
            raise ValueError("Expected GitHub's temporary artifact download redirect")
    except HTTPError as error:
        try:
            require(error.code == 302, "GitHub did not issue the expected artifact download redirect")
            location = error.headers["Location"]
        finally:
            error.close()
    public_download(location, target, OUTER_BYTES, OUTER_SHA)


def update_status(scratch, stage, complete):
    status = {
        "schema": "solar_v115_exact_replay_arm_v122", "original_run_id": RUN, "original_artifact_id": ARTIFACT,
        "stage": stage, "complete": complete, "new_inference": False, "model_weights_loaded": False,
        "runtime": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
    }
    target = scratch / "audit_status.json"
    temporary = scratch / "status.next"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(target)
    return target


def run(repository, scratch):
    stage = "runtime_and_source"
    try:
        require(platform.system() == "Linux" and platform.machine() == "aarch64"
                and platform.python_version() == "3.12.15" and platform.libc_ver() == ("glibc", "2.39"),
                "Original Linux aarch64, Python and glibc runtime required")
        verifier = Path(__file__).with_name("verify_five_arm_artifact_v115.py")
        lock = repository / "requirements_arm_cp312_v113.txt"
        require(sha(verifier) == VERIFIER_SHA and sha(lock) == LOCK_SHA, "Frozen verifier or wheel lock differs")
        require(sha(repository / "reports/solar-five-arm-v115/comparison-protocol.json") == PROTOCOL_SHA,
                "Frozen original protocol differs")
        wheels = scratch / "audit-wheels.txt"
        wheels.write_text(wheel_lines(lock.read_text()))
        env = dict(os.environ)
        token = env.pop(TOKEN_ENV, "")
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"):
            env.pop(name, None)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        stage = "locked_dependencies"
        log = scratch / "private-audit.log"
        with log.open("xb") as private:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes", "-r", str(wheels)],
                           env=env, stdout=private, stderr=subprocess.STDOUT, check=True, timeout=100)
            stage = "official_artifact_download"
            outer = scratch / "original-actions.zip"
            official_download(token, outer)
            stage = "public_base_download"
            base = scratch / "public-v98-base.zip"
            public_download(BASE, base, BASE_BYTES, BASE_SHA)
            stage = "frozen_saved_array_replay"
            audited = scratch / "verified"
            subprocess.run([sys.executable, "-B", str(verifier), "--run-id", str(RUN),
                            "--ticket-commit", TICKET_COMMIT, "--ticket-sha256", TICKET_SHA,
                            "--repository", str(repository), "--outer", str(outer), "--base", str(base),
                            "--output", str(audited), "--replay"],
                           env=env, stdout=private, stderr=subprocess.STDOUT, check=True, timeout=480)
        stage = "receipt_publication"
        receipt = json.loads((audited / RECEIPTS[0]).read_bytes())
        replay = json.loads((audited / "independent_replay_v115.json").read_bytes())
        require(receipt["complete_archive_verified"] is True and receipt["independently_replayed"] is True
                and receipt["new_inference"] is False and receipt["protocol_sha256"] == PROTOCOL_SHA
                and receipt["archived_downloads"]["outer"]["digest"] == "sha256:" + OUTER_SHA
                and replay["passed"] is True and replay["replayed_rows"] == 480, "Expected complete no-model replay")
        publish = scratch / "public-success"
        publish.mkdir()
        for name in RECEIPTS:
            source = audited / name
            require(source.is_file() and not source.is_symlink() and source.stat().st_size <= 8 * 1024**2,
                    "Expected bounded original verifier JSON receipt")
            (publish / name).write_bytes(source.read_bytes())
        status = update_status(scratch, "complete", True)
        (publish / status.name).write_bytes(status.read_bytes())
    except Exception:
        update_status(scratch, stage, False)
        print(f"Exact ARM replay incomplete at stage {stage}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.repository.resolve(strict=True), args.scratch.resolve(strict=True)))
