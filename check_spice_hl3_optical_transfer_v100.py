"""Exact synthetic clock/pose/crop/epipolar checks, plus optical-only fallback."""
import csv
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

import spice_hl3_optical_transfer_v100 as runner


def fails(function, text):
    try:
        function()
    except ValueError as exc:
        if text not in str(exc):
            raise
    else:
        raise AssertionError("Expected rejection: " + text)


def check_quaternions_and_clock():
    r = runner.rotation([0, 0, 2**-.5, 2**-.5])
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    fails(lambda: runner.rotation([0, 0, .55, .55]), "unit-norm")
    a = runner.pose(np.eye(3), [0, 0, 0])
    b = runner.pose(r, [2, 0, 0])
    midpoint = runner.interpolate(a, b, .5)
    np.testing.assert_allclose(midpoint[:3, 3], [1, 1 - 2**.5, 0], atol=1e-12)

    def marker(x, stamp):
        return dict(Time=stamp, PositionX=str(x), PositionY="0", PositionZ="0",
                    OrientationX="0", OrientationY="0", OrientationZ="0", OrientationW="1")

    rows = [marker(4, "1000.080"), marker(0, "1000.000"), marker(2, "1000.040")]
    times = [Decimal(r["Time"]) for r in rows]
    ranks = {1: 0, 2: 1, 0: 2}
    marker_to_camera = runner.pose(np.eye(3), [0, .2, 0])
    frame = dict(timestamp_s="1000.400", marker_bracket_rows=[2, 0])
    c2w = runner.camera_at(frame, rows, times, ranks, marker_to_camera, Decimal("-.340"))
    np.testing.assert_allclose(c2w[:3, 3], [3, -.2, 0], atol=1e-12)
    fails(lambda: runner.camera_at(frame, rows, times, ranks, marker_to_camera, Decimal(".340")), "no qualified")
    skipped = dict(frame, marker_bracket_rows=[1, 0])
    fails(lambda: runner.camera_at(skipped, rows, times, ranks, marker_to_camera, Decimal("-.340")), "adjacent sorted")


def check_epipolar():
    k = np.array([[450., 0, 267.], [0, 460., 148.], [0, 0, 1.]])
    d = np.array([-.19, .04, .002, .001, 0.])
    rng = np.random.default_rng(20261104)
    world = rng.uniform([-1.3, -.9, 4.], [1.4, 1., 12.], (200, 3))
    rod = np.array([.04, -.07, .03])
    r = cv2.Rodrigues(rod)[0]
    t = np.array([.52, -.13, .08])
    p = cv2.projectPoints(world, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
    q = cv2.projectPoints(world, rod, t, k, d)[0].reshape(-1, 2)
    a, b = runner.rectified(p, k, d), runner.rectified(q, k, d)
    moved = b.copy()
    moved[:18] = moved[18:36]
    fitted_r, fitted_t, _, _, _ = runner.fit_relative(a, moved, k)
    known = runner.evaluate_geometry(a, moved, fitted_r, fitted_t, k, np.eye(4), runner.invert(runner.pose(r, t)))
    assert known["strict_recovered"]
    assert known["independent_epipolar"]["within_pixels"]["1"] >= 182
    assert known["independent_epipolar"]["within_pixels"]["1"] < 195
    mirrored = runner.evaluate_geometry(a, b, r, -t, k, np.eye(4), runner.invert(runner.pose(r, t)))
    assert mirrored["signed_translation_direction_error_deg"] > 179. and not mirrored["recovered"]
    exact = runner.evaluate_geometry(a, b, None, None, k, np.eye(4), runner.invert(runner.pose(r, t)))
    assert exact["rotation_error_deg"] is None and exact["independent_epipolar"]["median_px"] < 1e-6
    assert runner.rectified(a, k, np.zeros(5)).tobytes() == a.tobytes()


def check_manifest():
    with TemporaryDirectory() as directory:
        root = Path(directory)

        def record(path):
            return dict(path=path.name, sha256=runner.sha(path))

        intr = root / "intrinsic.json"
        settings = dict(imageWidth=640, imageHeight=360)
        optics = dict(fx=340., fy=350., cx=320.5, cy=180.5)
        runner.write(intr, {"zed2": {"left_cam": dict(settings=settings, intrinsics=optics)}})
        ext = root / "extrinsic.json"
        runner.write(ext, {"zed2-left-to-marker": {"T_c_M": np.eye(4).ravel().tolist(), "t_d": 0.}})
        mk = root / "mocap.csv"
        with mk.open("w", newline="") as stream:
            out = csv.writer(stream)
            out.writerow(["Time", "PositionX", "PositionY", "PositionZ", "OrientationX", "OrientationY", "OrientationZ", "OrientationW"])
            out.writerows([["1000.240", .30, 0, 0, 0, 0, 0, 1], ["1000.000", 0, 0, 0, 0, 0, 0, 1],
                           ["1000.040", .02, 0, 0, 0, 0, 0, 1], ["1000.200", .28, 0, 0, 0, 0, 0, 1]])
        md = root / "metadata.csv"
        with md.open("w", newline="") as stream:
            out = csv.writer(stream)
            out.writerow(["Frame ID", "Timestamp", "Height", "Width", "Encoding"])
            out.writerows([[74, "1000.02", 360, 640, "mono8"], [3, "1000.2200", 360, 640, "mono8"]])
        rng = np.random.default_rng(20261104)
        base = cv2.GaussianBlur(rng.integers(0, 256, (360, 640), dtype=np.uint8), (3, 3), .6)
        second = np.roll(base, -13, axis=1)
        frames = []
        for i, (capture, bracket, pixels) in enumerate((("1000.020", [1, 2], base), ("1000.220", [3, 0], second))):
            path = root / (str(i) + ".png")
            assert cv2.imwrite(str(path), pixels)
            frames.append(dict(id=str(i), **record(path), timestamp_s=capture, stream_id="fast",
                               archive_member=f"fast/stereo/left/stereo_left_{capture}_{i}.png",
                               camera_metadata_row=i, marker_bracket_rows=bracket))
        manifest = dict(schema=runner.SCHEMA, track=runner.POSE_TRACK, trajectory="C", camera_key="zed2:left",
                        prepared_only=False, calibration_or_final_access=False, calibration=record(intr),
                        extrinsics=record(ext), crop=[12, 11, 600, 330], time_convention=runner.CLOCK,
                        mocap_streams={"fast": dict(mocap=record(mk), camera_metadata=record(md))},
                        frames=frames, cases=[dict(id="pair", reference="0", query="1")])
        path = root / "manifest.json"
        runner.write(path, manifest)
        _, _, size, k, d, truth = runner.validate_manifest(path)
        assert size == (640, 360)
        np.testing.assert_array_equal(k[:2, 2], [308.5, 169.5])
        np.testing.assert_array_equal(d, np.zeros(5))
        np.testing.assert_allclose([truth["0"][0, 3], truth["1"][0, 3]], [.01, .29], atol=1e-12)
        result_dir = root / "result"
        runner.run(path, result_dir, 20)
        first = runner.read(result_dir / "00000.json")
        replayed = root / "replay.json"
        runner.replay(path, result_dir, replayed)
        assert runner.read(replayed)["passed"]
        assert first["independent_geometry"]["independent_epipolar"]["within_pixels"]["1"] > 200
        image_only_path = root / "image_only.json"
        runner.write(image_only_path, dict(manifest, track=runner.IMAGE_TRACK))
        result_dir = root / "image_result"
        runner.run(image_only_path, result_dir, 20)
        assert runner.read(result_dir / "00000.json")["independent_geometry"] is None
        frames[0]["camera_metadata_row"] = 1
        bad = root / "bad.json"
        runner.write(bad, manifest)
        fails(lambda: runner.validate_manifest(bad), "archive-member timestamp")


if __name__ == "__main__":
    cv2.setNumThreads(1)
    check_quaternions_and_clock()
    check_epipolar()
    check_manifest()
    print("PASS: official clock sign and transform direction, sorted bracket checks, ZED crop/no double rectification, known E and signed direction, synthetic end-to-end and image-only separation")
