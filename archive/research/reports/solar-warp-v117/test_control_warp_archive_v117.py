import pytest

import control_warp_archive_v117 as control


def stub_run(status, conclusion=None):
    return dict(id=41941904269, head_sha=control.V115_HEAD, run_attempt=1,
                event="workflow_dispatch", status=status, conclusion=conclusion)


def test_live_predecessor_cannot_freeze_dispatch(tmp_path):
    path = tmp_path / "dispatch.json"
    queried = []

    def get(url):
        queried.append(url)
        return stub_run("in_progress")

    with pytest.raises(RuntimeError, match="still live"):
        control.freeze("timing", None, path, get=get)
    assert queried == [control.V115]
    assert not path.exists()


def test_failed_predecessor_with_no_artifact_has_explicit_terminal_record():
    responses = {control.V115: stub_run("completed", "failure"),
                 control.V115 + "/artifacts?per_page=100": dict(total_count=0, artifacts=[])}
    record = control.terminal(None, get=responses.__getitem__)
    assert record["conclusion"] == "failure"
    assert record["artifact_audit_sha256"] is None
    responses[control.V115]["conclusion"] = "success"
    with pytest.raises(ValueError, match="Archive"):
        control.terminal(None, get=responses.__getitem__)


def test_terminal_artifacts_must_be_archived_and_preserved(tmp_path):
    record = dict(id=7, digest="sha256:bounded")
    responses = {control.V115: stub_run("completed", "failure"),
                 control.V115 + "/artifacts?per_page=100": dict(total_count=1, artifacts=[record])}
    with pytest.raises(ValueError, match="Archive"):
        control.terminal(None, get=responses.__getitem__)
    preserved = tmp_path / "watchdog.json"
    preserved.write_bytes(b"archived evidence")
    receipt = tmp_path / "receipt.json"
    control.put(receipt, dict(
        schema="solar_five_arm_official_artifact_audit_v115", run_id=41941904269, conclusion="failure",
        protocol_sha256=control.V115_PROTOCOL, ticket_sha256=control.V115_TICKET,
        new_inference=False, archived_downloads={"outer": record},
        records={"watchdog": dict(path=preserved.name, sha256=control.sha(preserved))}))
    assert control.terminal(receipt, get=responses.__getitem__)["artifact_audit_sha256"] == control.sha(receipt)
    preserved.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="changed"):
        control.terminal(receipt, get=responses.__getitem__)


def test_timing_evidence_must_match_full_seal_and_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "WORK", tmp_path)
    receipt = tmp_path / "timing" / "fixture" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    complete = tmp_path / "timing" / "COMPLETE.json"
    for seconds, valid in ((181., False), (120., True)):
        receipt.unlink(missing_ok=True)
        complete.unlink(missing_ok=True)
        control.put(receipt, dict(passed=True, full_scope_forecast=dict(
            projected_seconds=seconds, admitted=True, projected_uncompressed_evidence_bytes=30_000_000)))
        control.put(complete, dict(phase="timing", complete=True, timing_receipt_sha256=control.sha(receipt)))
        if valid:
            assert control.timing_proof()["projected_archive_seconds"] == seconds
        else:
            with pytest.raises(ValueError, match="timing failed"):
                control.timing_proof()
