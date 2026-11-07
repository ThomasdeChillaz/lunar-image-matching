from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lunar_matching.artifacts import sha256, write_json_exclusive
from lunar_matching.validation import prepare_case, camera_to_world, intrinsics, checked_file


@pytest.fixture
def native_case(tmp_path):
    image = np.indices((512, 512)).sum(axis=0).astype(np.uint16)
    image = np.repeat(image[:, :, None], 3, axis=2)
    assert cv2.imwrite(str(tmp_path / "image.png"), image)
    np.savez(tmp_path / "geometry.npz", valid=np.ones((512, 512), bool),
             axial_depth=np.full((512, 512), 1000.), cam2world=np.eye(4),
             intrinsics=[[440., 0, 255.5], [0, 440., 255.5], [0, 0, 1]])

    def record(name):
        return {"path": name, "sha256": sha256(tmp_path / name)}

    case = dict(id="generated", reference_geometry=record("geometry.npz"),
                query_image=record("image.png"), query_intrinsics=[[440, 0, 255.5], [0, 440, 255.5], [0, 0, 1]],
                catalogue=[dict(light_id=0, sun_world=[1, 0, 0], image=record("image.png"))])
    return tmp_path, case


def save(base, case):
    path = base / "manifest.json"
    path.write_text(json.dumps([case]))
    return path


def test_manifest_paths_are_explicit_and_input_records_are_not_mutated(native_case, monkeypatch, tmp_path):
    base, case = native_case
    original = deepcopy(case)
    manifest = save(base, case)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    prepared = prepare_case(manifest, "generated", base)
    assert Path(prepared["query_image"]["path"]) == base / "image.png"
    assert case == original


@pytest.mark.parametrize("field", ["query_pose", "query_depth", "query_sun", "scoring_labels"])
def test_query_truth_cannot_enter_manifest(native_case, field):
    base, case = native_case
    case[field] = "forbidden"
    with pytest.raises(ValueError, match="query truth"):
        prepare_case(save(base, case), "generated", base)


def test_query_truth_cannot_hide_in_reference_archive(native_case):
    base, case = native_case
    path = base / "geometry.npz"
    with np.load(path) as old:
        content = {name: old[name].copy() for name in old.files}
    np.savez(path, **content, query_pose=np.eye(4))
    case["reference_geometry"]["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="Reference geometry only"):
        prepare_case(save(base, case), "generated", base)


@pytest.mark.parametrize("change", ["bytes", "light_id", "sun", "encoding", "depth"])
def test_changed_or_incompatible_native_inputs_fail_before_inference(native_case, change):
    base, case = native_case
    if change == "bytes":
        (base / "image.png").write_bytes(b"changed")
    elif change == "light_id":
        case["catalogue"] *= 2
    elif change == "sun":
        case["catalogue"][0]["sun_world"] = [3, 0, 0]
    elif change == "encoding":
        assert cv2.imwrite(str(base / "image.png"), np.zeros((512, 512, 3), np.uint8))
        case["query_image"]["sha256"] = sha256(base / "image.png")
        case["catalogue"][0]["image"]["sha256"] = case["query_image"]["sha256"]
    else:
        with np.load(base / "geometry.npz") as z:
            values = {name: z[name].copy() for name in z.files}
        values["axial_depth"][0, 0] = np.nan
        np.savez(base / "geometry.npz", **values)
        case["reference_geometry"]["sha256"] = sha256(base / "geometry.npz")
    with pytest.raises(ValueError):
        prepare_case(save(base, case), "generated", base)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.diag([-1., 1, 1]), np.full((3, 3), np.nan)])
def test_invalid_calibration_is_rejected(matrix):
    with pytest.raises(ValueError):
        intrinsics(matrix)


def test_reflections_and_nonrigid_poses_are_rejected():
    for value in (np.diag([-1., 1, 1, 1]), np.diag([2., 1, 1, 1]), np.eye(3)):
        with pytest.raises(ValueError):
            camera_to_world(value)


def test_exclusive_json_never_overwrites_results_or_leaves_partial_output(tmp_path):
    path = tmp_path / "nested/result.json"
    write_json_exclusive(path, {"pose": None})
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"pose": "changed"})
    assert json.loads(path.read_text()) == {"pose": None}
    assert not list(path.parent.glob(".lunar-result-*"))
    with pytest.raises(ValueError):
        write_json_exclusive(tmp_path / "invalid.json", {"value": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_input_record_cannot_sneak_extra_fields(native_case):
    base, case = native_case
    with pytest.raises(ValueError):
        checked_file(dict(case["query_image"], query_depth="forbidden"), base)


def test_validation_command_runs_no_matcher(native_case, monkeypatch, capsys):
    from lunar_matching import backend
    from lunar_matching.cli import main
    base, case = native_case
    monkeypatch.setattr(backend, "run_backend", lambda *args, **kwargs: pytest.fail("No inference is permitted"))
    assert main(["validate", "--manifest", str(save(base, case)), "--case-id", "generated"]) == 0
    assert json.loads(capsys.readouterr().out)["matching_run"] is False


@pytest.mark.parametrize("wrong_pose", [False, True])
def test_localization_boundary_handles_old_schema_and_refuses_raw_pose_leaks(native_case, monkeypatch, capsys, wrong_pose):
    from lunar_matching import backend
    from lunar_matching.cli import main
    base, case = native_case
    response = dict(pose=np.eye(4).tolist() if wrong_pose else None, acceptance_status="uncalibrated",
                    rejection_reason="candidate_not_calibrated")
    monkeypatch.setattr(backend, "run_backend", lambda *args, **kwargs: response)
    output = base / "result.json"
    code = main(["localize", "--manifest", str(save(base, case)), "--case-id", "generated", "--output", str(output)])
    assert code == (2 if wrong_pose else 0)
    if wrong_pose:
        assert not output.exists()
    else:
        value = json.loads(output.read_text())
        assert value["accepted"] is False and value["pose"] is None
