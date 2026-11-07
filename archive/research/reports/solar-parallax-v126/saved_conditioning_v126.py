"""Reproduce the saved v115 correspondence/conditioning audit without fitting a pose."""
import argparse
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


PROTOCOL = "reports/solar-five-arm-v115/comparison-protocol.json"
PUBLIC = "reports/solar-five-arm-results-v123"
OUTPUT = "runs/solar-five-arm-v115/comparison"
REPLAY_ZIP = "f19f372f46bd874e780b3b803e0e846765044968c9e87b555863594ddc5c6715"
BASE_ZIP = "2d25dca8b56f86f10444289b7458aeed4ba9a6755bf1ff19fca7014afaa1dee0"
ACTIONS_ZIP = "813f2c9e6c5986ae711ab885d3a6c25a146cb87fead8261f99984e689f5e538b"
SUMMARY = "3f65b164472e2300d6b5aeb1611a2988d0f5d0155fe850c6a5f229eb567a38bc"
SEAL = "dd108e49fa87be8ac49308a378405c263ede1cc5ed9d41a4b952823a0c0c2c50"
METADATA = {
    PROTOCOL: "c43b7a0a5236b27055a7179caf1cfa1a274b18743a9fd868d0054fa96d23878c",
    f"{PUBLIC}/public-replay-receipt.json": "d4468742704b148c419fb64d5f5dc296a7e4c9cbc24b02e8a6191ebbef67e4b1",
    f"{PUBLIC}/independent_replay_v115.json": "844433a3bcfcb5180da648e446b3a05c089d2b69d5ebcdb8ee2af214f25dd2b5",
    f"{PUBLIC}/replay_input_hashes_v115.json": "bd856d419adb42c40a0f5b4ff7ab452e1918a7280a74d96584382ecd53f54d09",
}
EXPECTED = ("u021-l16-01-positive", "u061-l08-01-positive", "u072-l08-17-positive")
POSE_FIELDS = ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def path_sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pinned(raw, expected):
    need(sha256(raw).hexdigest() == expected, "Frozen evidence digest differs")
    return raw


def member(archive, name, expected):
    need(archive.namelist().count(name) == 1, "Ambiguous replay member: " + name)
    return pinned(archive.read(name), expected)


def arrays(archive, name, expected):
    with np.load(BytesIO(member(archive, name, expected)), allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def saved_evidence(repo, replay, base):
    p, original, exact, selected = [json.loads(pinned((repo / path).read_bytes(), digest))
                                     for path, digest in METADATA.items()]
    need(original["official_benchmark_artifact_sha256"] == ACTIONS_ZIP
         and original["public_replay_asset_digest"] == "sha256:" + REPLAY_ZIP
         and original["benchmark_protocol_sha256"] == METADATA[PROTOCOL]
         and original["sealed_and_independently_replayed_cells"] == exact["replayed_rows"] == 480
         and exact["passed"] is True and exact["summary_sha256"] == SUMMARY
         and len(replay.namelist()) == len(set(replay.namelist())) == 482,
         "Expected exact original five-arm run and public ARM replay")
    summary = json.loads(member(replay, f"{OUTPUT}/summary.json", SUMMARY))
    seal = json.loads(member(replay, f"{OUTPUT}/COMPLETE.json", SEAL))
    need(seal["complete"] is True and seal["summary_sha256"] == SUMMARY
         and summary["protocol_sha256"] == METADATA[PROTOCOL] and summary["accepted_results"] is False
         and summary["calibration_or_final_access"] is False, "Original development seal differs")
    manifest_hash = p["input_sha256"][p["manifest"]]
    need(selected[p["manifest"]] == manifest_hash, "Frozen case manifest binding differs")
    manifest = json.loads(member(base, "payload/" + p["manifest"], manifest_hash))
    case_ids = p["comparison_case_ids"]
    cases = {case["case_id"]: case for case in manifest["cases"] if case["case_id"] in case_ids}
    frozen_grid = [(profile["name"], identity, "estimated_sun") for profile in p["profiles"] for identity in case_ids]
    identity = lambda row: (row["profile"], row["case_id"], row["track"])
    need([identity(r) for r in exact["rows"]] == [identity(r) for r in summary["rows"]] == frozen_grid,
         "Full sealed five-arm ordering differs")
    need(all(all(exact_row[key] is saved_row["metrics"][key] for key in POSE_FIELDS)
             for exact_row, saved_row in zip(exact["rows"], summary["rows"], strict=True)),
         "Original pose outcomes and exact ARM replay differ")
    misses = {}
    for case_id in case_ids:
        case = cases[case_id]
        need(case["split"] == "model_selection", "Expected only frozen development cases")
        if case["positive"] and case["light_a"] != case["light_b"]:
            rows = [r for r in summary["rows"] if r["case_id"] == case_id]
            if not any(r["metrics"]["correct"] for r in rows):
                misses[case_id] = rows
    qualified = {case: rows for case, rows in misses.items()
                 if max(r["metrics"]["correspondence"]["2"]["correct"] for r in rows) >= 6}
    need(tuple(sorted(qualified)) == EXPECTED and len(misses) == 12, "Saved-case screen differs")
    return p, selected, cases, qualified, manifest_hash


def project(world, intrinsics, pose):
    xyz = (world - pose[:3, 3]) @ pose[:3, :3]
    h = xyz @ intrinsics.T
    front = np.isfinite(h).all(axis=1) & (xyz[:, 2] > 0.)
    pixels = np.full((len(world), 2), np.inf)
    pixels[front] = h[front, :2] / h[front, 2, None]
    return pixels, xyz


def hull_fraction(points):
    points = sorted({tuple(p) for p in points})
    if len(points) < 3:
        return 0.
    def half(seq):
        hull = []
        for p in seq:
            while len(hull) >= 2:
                a, b = hull[-2:]
                cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross > 0.:
                    break
                hull.pop()
            hull.append(p)
        return hull[:-1]
    h = half(points) + half(reversed(points))
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(h, h[1:] + h[:1]))) / (2. * 512**2)


def support(pixels):
    sides = pixels >= 256.
    return dict(hull_pct=rounded(100. * hull_fraction(pixels)), span_native_px=rounded(np.ptp(pixels, axis=0)),
                quadrant_counts=np.bincount(2 * sides[:, 1] + sides[:, 0], minlength=4).tolist(),
                exact_unique_points=len(np.unique(pixels, axis=0)),
                occupied_16px_cells=len(np.unique(np.floor(pixels/16.), axis=0)))


def rounded(value):
    array = np.asarray(value)
    return np.round(array, 6).tolist()


def jacobian(camera, intrinsics, reference_range):
    x, y, z = camera.T
    h = camera @ intrinsics.T
    derivative = (intrinsics[None, :2] * z[:, None, None] - h[:, :2, None] * intrinsics[None, 2:]) / z[:, None, None]**2
    twist = np.zeros((len(camera), 3, 6))
    twist[:, 0, 1], twist[:, 0, 2] = z, -y
    twist[:, 1, 0], twist[:, 1, 2] = -z, x
    twist[:, 2, 0], twist[:, 2, 1] = y, -x
    twist[:, :, :3] *= np.deg2rad(2.)
    twist[:, :, 3:] = np.eye(3) * (.01 * reference_range)
    return (derivative @ twist).reshape(-1, 6)


def conditioning(camera, intrinsics, reference_range, true_pose=None, saved_pose=None):
    j = jacobian(camera, intrinsics, reference_range)
    _, singular, axes = np.linalg.svd(j, full_matrices=False)
    result = dict(rank=int(np.linalg.matrix_rank(j)), condition_number=rounded(singular[0]/singular[-1]),
                  weakest_mode_px_per_anchor=rounded(singular[-1]/np.sqrt(len(camera))))
    if saved_pose is not None and saved_pose.shape == (4, 4):
        r = saved_pose[:3, :3].T @ true_pose[:3, :3]
        k = .5 * np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
        norm = np.linalg.norm(k)
        angle = np.arctan2(norm, (np.trace(r)-1.)/2.)
        omega = np.zeros(3) if not norm else k * angle / norm
        delta = saved_pose[:3, :3].T @ (true_pose[:3, 3]-saved_pose[:3, 3])
        step = np.r_[omega/np.deg2rad(2.), delta/(.01*reference_range)]
        result["saved_displacement_fraction_in_weakest_two_subspace"] = rounded(np.linalg.norm(axes[-2:] @ step) / np.linalg.norm(step))
    return result


def inspect_row(replay, row, reference, joint):
    name, digest = row["saved_arrays"], row["saved_arrays_sha256"]
    need(name.startswith(OUTPUT + "/") and name.endswith(".npz"), "Unexpected saved-array path")
    data = arrays(replay, name, digest)
    ref, query = data["reference_xy"], data["query_xy"]
    truth, visible = data["scoring_query_xy"], data["scoring_visible"]
    target, saved = joint["query_cam2world"], data["pose"]
    need(np.array_equal(target, data["scoring_query_cam2world"]), "Exact scoring camera differs")
    accurate = visible & (np.linalg.norm(query-truth, axis=1) <= 2.)
    n = int(accurate.sum())
    need(n == row["metrics"]["correspondence"]["2"]["correct"], "Exact two-pixel count differs")
    distance = row["metrics"]["reference_range_m"]
    answer = dict(profile=row["profile"], saved_arrays=name, saved_arrays_sha256=digest, returned=len(query),
                  accurate_at_2px=n, archived_ap3p_inliers=row["final_estimator"]["inliers"],
                  archived_pose_error=row["metrics"]["pose_error"], native_translation_limit_m=rounded(.01*distance),
                  archived_pose_reason=row["final_estimator"]["reason"])
    if n < 6:
        return answer
    centre = reference["cam2world"]
    x, y = np.rint(ref).astype(int).T
    depth = reference["depth"][y, x]
    need(np.all(np.isfinite(depth) & (depth > 0.)), "Invalid original reference depth")
    rays = np.column_stack((ref, np.ones(len(ref)))) @ np.linalg.inv(reference["intrinsics"]).T
    world = (rays * depth[:, None]) @ centre[:3, :3].T + centre[:3, 3]
    qk = joint["query_intrinsics"]
    projected, camera = project(world, qk, target)
    need(np.allclose(projected[visible], truth[visible], rtol=0, atol=1e-9), "Independent projection differs")
    t_error = np.linalg.norm(projected-query, axis=1)
    score = dict(truth_consensus_at_3px=int((t_error <= 3.).sum()))
    if saved.shape == (4, 4):
        saved_projection, _ = project(world, qk, saved)
        s_error = np.linalg.norm(saved_projection-query, axis=1)
        s3 = s_error <= 3.
        need(int(s3.sum()) == answer["archived_ap3p_inliers"], "Independent saved consensus differs")
        score.update(saved_consensus_at_3px=int(s3.sum()), saved_consensus_outside_truth_3px=int((s3 & (t_error > 3.)).sum()),
                     saved_consensus_in_accurate_2px=int((s3 & accurate).sum()),
                     truth_clipped_squared_cost=rounded(np.minimum(t_error**2, 9.).sum()),
                     saved_clipped_squared_cost=rounded(np.minimum(s_error**2, 9.).sum()),
                     truth_median_px_on_accurate_2px=rounded(np.median(t_error[accurate])),
                     saved_median_px_on_accurate_2px=rounded(np.median(s_error[accurate])))
    else:
        need(saved.shape == (0, 4), "Explicit no-pose sentinel required")
    hits, xyz, p, q = world[accurate], camera[accurate], ref[accurate], query[accurate]
    plane = np.linalg.svd(hits-hits.mean(axis=0), compute_uv=False)
    sight_a, sight_b = hits-centre[:3, 3], hits-target[:3, 3]
    cosine = np.einsum("ni,ni->n", sight_a, sight_b) / (np.linalg.norm(sight_a, axis=1)*np.linalg.norm(sight_b, axis=1))
    parallax = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    _, representatives = np.unique(np.floor(p/16.), axis=0, return_index=True)
    answer.update(
        exact_unique_pairs=len(np.unique(np.column_stack((p, q)), axis=0)),
        reference_support=support(p), query_support=support(q),
        terrain_elevation_p10_p90_m=rounded(np.percentile(hits[:, 2], (10, 90))),
        true_query_axial_depth_p10_p90_m=rounded(np.percentile(xyz[:, 2], (10, 90))),
        world_plane_singular_3_over_2=rounded(plane[2]/plane[1]),
        viewing_ray_angle_p10_p50_p90_deg=rounded(np.percentile(parallax, (10, 50, 90))),
        true_local_pnp=conditioning(xyz, qk, distance, target, saved),
        true_local_pnp_one_per_reference_16px_cell=conditioning(xyz[representatives], qk, distance),
        consensus=score,
    )
    return answer


def report(repo, replay_path, base_path, original_path):
    need(path_sha(replay_path) == REPLAY_ZIP and path_sha(base_path) == BASE_ZIP,
         "Use exact public v115 replay and original v98 input ZIPs")
    if original_path is not None:
        need(path_sha(original_path) == ACTIONS_ZIP, "Original Actions ZIP differs")
    with ZipFile(replay_path) as replay, ZipFile(base_path) as base:
        p, selected, cases, rows, manifest_hash = saved_evidence(repo, replay, base)
        results = []
        for identity in EXPECTED:
            case = cases[identity]
            names = (case["reference_geometry"], case["scoring_geometry"])
            hashes = {name: case["input_sha256"][name] for name in names}
            need(all(p["input_sha256"][k] == selected[k] == h for k, h in hashes.items()),
                 "Frozen v98 geometry identity differs")
            reference, joint = [arrays(base, "payload/" + name, hashes[name]) for name in names]
            need(all(np.array_equal(reference[k], joint["reference_" + k])
                     for k in ("depth", "cam2world", "intrinsics")), "Original reference and scoring copy differ")
            results.append(dict(case_id=identity, component=case["component"], geometry_sha256=hashes,
                                reference_query_camera_baseline_m=rounded(np.linalg.norm(
                                    reference["cam2world"][:3, 3]-joint["query_cam2world"][:3, 3])),
                                rows=[inspect_row(replay, row, reference, joint) for row in rows[identity]]))
    return dict(schema="solar_v115_saved_conditioning_v126", source_sha256=path_sha(Path(__file__).resolve()),
                benchmark_run_id=41941904269, exact_arm_replay_run_id=41964097603,
                official_original_actions_sha256=ACTIONS_ZIP, public_replay_zip_sha256=REPLAY_ZIP,
                base_input_zip_sha256=BASE_ZIP, sealed_summary_sha256=SUMMARY, completion_sha256=SEAL,
                manifest_sha256=manifest_hash, metadata_sha256=METADATA,
                eligible_changed_light_misses=12, measured_cases=results,
                selection="Exposed development cases missed by all five v115 arms with at least six visible 2px correspondences in one arm.",
                conditioning="Analytic pixel Jacobian at the frozen true query camera. Scale angle columns by 2 degrees and translation columns by 1% median reference range. Report all accurate matches and first original observation per fixed reference 16px cell.",
                limitations="Local full rank is not a test for separate global pose solutions. Count/reprojection diagnostics score archived observations only, including duplicates. Truth is used after the original inference seal. No new fit, inference, or reserve access.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay-zip", required=True, type=Path)
    cli.add_argument("--base-zip", required=True, type=Path)
    cli.add_argument("--original-actions-zip", type=Path)
    cli.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = cli.parse_args()
    print(json.dumps(report(args.repo, args.replay_zip, args.base_zip, args.original_actions_zip),
                     indent=2, allow_nan=False))
