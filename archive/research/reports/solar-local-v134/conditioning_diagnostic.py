"""Post-hoc geometry sensitivity of sealed matches; no new pose or image inference."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from replay_saved import saved, summarize


def sensitivity(world, pixels, intrinsics, camera_pose, reference_range):
    camera = (world-camera_pose[:3, 3])@camera_pose[:3, :3]
    homogeneous = camera@intrinsics.T
    projected = homogeneous[:, :2]/homogeneous[:, 2, None]
    keep = (camera[:, 2] > 0)&(np.linalg.norm(projected-pixels, axis=1) <= 3.)
    points, retained = camera[keep], pixels[keep]
    if len(points) < 6:
        return dict(fitted_inliers=int(keep.sum()), rank=None)
    jacobian = []
    for x, y, z in points:
        projection = np.array([[intrinsics[0, 0]/z, 0., -intrinsics[0, 0]*x/z**2],
                               [0., intrinsics[1, 1]/z, -intrinsics[1, 1]*y/z**2]])
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        jacobian.append(projection@np.c_[-skew, np.eye(3)*reference_range])
    jacobian = np.vstack(jacobian)
    _, values, vectors = np.linalg.svd(jacobian, full_matrices=False)
    rank = int((values > values[0]*1e-10).sum())
    position_rms = None
    if rank == 6:
        covariance = (vectors.T/values**2)@vectors
        position_rms = float(reference_range*np.sqrt(np.trace(covariance[3:, 3:])))
    hull = cv2.convexHull(retained.astype(np.float32))
    return dict(fitted_inliers=len(points), rank=rank,
                normalized_pose_jacobian_singular_values=values.tolist(),
                condition_number=None if rank < 6 else float(values[0]/values[-1]),
                query_convex_hull_fraction=float(cv2.contourArea(hull)/(512*512)),
                linearized_position_rms_m_for_independent_one_pixel_noise=position_rms)


def analyze():
    summary = summarize()
    protocol = saved.metric.read("reports/solar-local-v134/protocol.json")
    raw = saved.metric.read("runs/solar-local-v134/comparison/predictions.json")
    scores = saved.metric.read("runs/solar-local-v134/comparison/scores.json")
    methods = ("framed_direct", "framed_native_plane_integer", "framed_native_terrain_subpixel")
    rows = []
    for item, scored in zip(raw["rows"], scores, strict=True):
        if not item["case"]["positive"]:
            continue
        k = np.asarray(protocol["inputs"][item["case"]["query_index"]]["query_intrinsics"])
        record = {}
        for name in methods:
            result = item["result"]["methods"][name]
            metric = scored["methods"][name]
            entry = dict(matches=len(result["query_xy"]), native_one_pixel_correct=metric["correspondence"]["1"]["correct"],
                         native_two_pixel_correct=metric["correspondence"]["2"]["correct"],
                         correct_pose=metric["correct"], pose_error=metric["pose_error"])
            if result["pose"] is not None:
                entry.update(sensitivity(np.asarray(result["reference_world"]), np.asarray(result["query_xy"]), k,
                                         np.asarray(result["pose"]), metric["pose_error"]["reference_range_m"]))
            record[name] = entry
        rows.append(dict(case=item["case"], methods=record))
    return dict(posthoc=True, new_matching_or_pose_estimation=False,
                predictions_sha256=summary["predictions_sha256"], rows=rows,
                caveat="Local sensitivity at the SAVED fitted camera, using its three-pixel inliers and a hypothetical independent one-pixel error model. Not calibrated uncertainty, a causal proof, or a new pose result. Grouped failures remain in the principal denominator.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved.metric.put(args.output, analyze())
