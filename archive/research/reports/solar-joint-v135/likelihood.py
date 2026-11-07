"""Fit one camera to full image-correlation surfaces before choosing endpoints.

This is a correlation-potential model, not calibrated probability. Coordinates
chosen with camera information are explicitly conditional correspondences, never
independent checkpoints. Query pose/depth/Sun are not inference inputs.
"""

from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"solar-local-v134"))
from finite_diagnostics import direction as safe_direction, frame

native = frame.native
REACH = native.RADIUS + native.HALO
KAPPA = 20.
OUTLIER = .2
GEOMETRY_SIGMA = 2.
MAX_STEPS = 40
SCALE = np.array([.02, .02, .02, 50., 50., 50.])


def sample_fields(values, xy):
    """Bilinearly sample each anchor's own field, without mixing anchors."""
    if len(values) != len(xy):
        raise ValueError("One image potential per reference anchor required")
    finite = np.isfinite(xy).all(axis=1)
    safe = np.clip(np.nan_to_num(xy, nan=0., posinf=48., neginf=0.), 0., values.shape[1]-1.000001)
    low = np.floor(safe).astype(int)
    dx, dy = (safe-low).T
    x, y = low.T
    i = np.arange(len(values))
    result = ((1-dx)*(1-dy)*values[i, y, x]+dx*(1-dy)*values[i, y, x+1]
              +(1-dx)*dy*values[i, y+1, x]+dx*dy*values[i, y+1, x+1])
    return np.where(finite, result, 0.)


def prepare_fields(reference, query, depth, k, reference_pose, query_k, initial_pose, *, reference_valid):
    """Compute a single physical reference raster and at most36 image fields."""
    native.prepare(reference, reference_valid)
    observed, observed_valid = native.prepare(query)
    if np.asarray(depth).shape != (512, 512):
        raise ValueError("Native reference axial depth required")
    anchors = native.control.pipeline.previous.ANCHORS
    if initial_pose is None or not (reference_valid & np.isfinite(depth) & (depth > 0)).any():
        return None
    warper = native.control.pipeline.previous.ReferencePatchWarper(
        reference, depth, k, reference_pose, anchors, reference_valid=reference_valid)
    bank = warper.warp(query_k, initial_pose, reference.shape)["geometry_3d"]
    image, available = native.prepare(bank.raster, bank.detector_valid)
    local_world, valid = native.control.pipeline.map_points_z(anchors, depth, k, np.eye(4))
    if not valid.all():
        expanded = np.zeros((len(anchors), 3)); expanded[valid] = local_world
        local_world = expanded
    records, scores, support, skipped = [], [], [], []
    for index, center in enumerate(bank.predicted_xy):
        if not valid[index] or not bank.anchor_visible[index] or not np.isfinite(center).all():
            continue
        template, mask = native.sample(image, available, center, native.PATCH)
        target, target_valid = native.sample(observed, observed_valid, center, native.PATCH+2*REACH)
        values, reason = native.correlation_surface(template, mask, target, target_valid)
        if values is None or not np.isfinite(values).any():
            skipped.append(dict(anchor_index=index, reason=reason or "insufficient_query_support_or_texture"))
            continue
        known = np.isfinite(values)
        scores.append(np.where(known, values, 0.))
        support.append(known)
        records.append(dict(anchor_index=index, center=center.copy(), local_world=local_world[index]))
    if not records:
        return dict(records=[], skipped=skipped, image=image, available=available, query=observed,
                    query_valid=observed_valid, scores=np.empty((0, 49, 49)),
                    support=np.empty((0, 49, 49), bool), initial_pose=initial_pose)
    score = np.asarray(scores)
    known = np.asarray(support)
    # Missing support supplies zero inlier evidence. It is not renormalized
    # into an artificially confident single supported pixel.
    potential = np.where(known, np.exp(KAPPA*(score-native.MIN_CORRELATION)), 0.)
    blurred = np.asarray([cv2.GaussianBlur(v, (13, 13), GEOMETRY_SIGMA,
                                          borderType=cv2.BORDER_CONSTANT) for v in potential])
    return dict(records=records, skipped=skipped, image=image, available=available, query=observed,
                query_valid=observed_valid, scores=score, support=known,
                blurred=blurred, initial_pose=initial_pose)


def optimize_camera(fields, reference_pose, query_k):
    if fields is None or len(fields["records"]) < 6:
        return None, dict(reason="insufficient_image_fields", calls=0)
    world = np.asarray([r["local_world"] for r in fields["records"]])
    centers = np.asarray([r["center"] for r in fields["records"]])
    center = world.mean(axis=0)
    local_initial = np.linalg.inv(reference_pose)@fields["initial_pose"]
    initial = native.control.pose_parameters(local_initial, center)
    calls = 0
    def cost(parameters):
        nonlocal calls
        calls += 1
        pose = native.control.camera_pose(initial+parameters*SCALE, center)
        projected, _, front = native.control.project_z(world, query_k, pose)
        delta = projected-centers
        mass = sample_fields(fields["blurred"], delta+REACH)
        energy = -np.log(OUTLIER+(1.-OUTLIER)*mass)/KAPPA
        excess = np.maximum(np.abs(np.nan_to_num(delta, nan=512.))-native.RADIUS, 0.)
        barrier = np.minimum((excess**2).sum(axis=1)/9., 100.)
        barrier = np.where(front & np.isfinite(delta).all(axis=1), barrier, 100.)
        return float(np.mean(energy+barrier))
    def gradient(x):
        result = np.empty(6)
        for axis in range(6):
            offset = np.zeros(6); offset[axis] = .001
            result[axis] = (cost(x+offset)-cost(x-offset))/.002
        return result
    x = np.zeros(6)
    value = initial_value = cost(x)
    derivative = gradient(x)
    inverse = np.eye(6)
    steps = 0
    for steps in range(MAX_STEPS):
        if np.linalg.norm(derivative, ord=np.inf) < 1e-6:
            break
        delta = -inverse@derivative
        if delta@derivative >= 0:
            delta = -derivative; inverse = np.eye(6)
        delta *= min(1., 1./max(float(np.max(np.abs(delta))), 1e-12))
        accepted = False
        for exponent in range(8):
            rate = .5**exponent
            proposed = x+rate*delta
            observed = cost(proposed)
            if observed <= value+1e-4*rate*(derivative@delta):
                accepted = True
                break
        if not accepted:
            break
        updated = gradient(proposed)
        s, y = proposed-x, updated-derivative
        curvature = float(s@y)
        if curvature > 1e-10:
            rho = 1./curvature
            operator = np.eye(6)-rho*np.outer(s, y)
            inverse = operator@inverse@operator.T+rho*np.outer(s, s)
        else:
            inverse = np.eye(6)
        x, value, derivative = proposed, observed, updated
    pose = reference_pose@native.control.camera_pose(initial+x*SCALE, center)
    return pose, dict(reason=None, calls=calls, iterations=steps+1,
                      initial_energy=initial_value, final_energy=value,
                      energy_improved=value < initial_value, calibrated_probability=False)


def selected_peak(scores, support, predicted_delta):
    """Select an image-grid posterior mode, not a projected camera coordinate."""
    yy, xx = np.indices(scores.shape)
    offsets = np.stack([xx-REACH, yy-REACH], axis=-1)
    allowed = support & (np.abs(offsets) <= native.RADIUS).all(axis=-1)
    posterior = KAPPA*(scores-native.MIN_CORRELATION)-(
        (offsets-predicted_delta)**2).sum(axis=-1)/(2*GEOMETRY_SIGMA**2)
    posterior = np.where(allowed, posterior, -np.inf)
    empty = dict(query_offset=None, reason=None, score=None, margin=None,
                 conditional_on_camera=True)
    if not np.isfinite(posterior).any():
        return dict(empty, reason="unavailable_joint_support")
    y, x = np.unravel_index(np.argmax(posterior), posterior.shape)
    score = float(scores[y, x])
    alternate = support & (np.max(np.abs(offsets-offsets[y, x]), axis=-1) > native.HALO)
    if not alternate.any():
        return dict(empty, reason="insufficient_peak_comparison", score=score)
    margin = float(score-scores[alternate].max())
    report = dict(empty, score=score, margin=margin)
    if score < native.MIN_CORRELATION:
        return dict(report, reason="weak_peak")
    if margin < native.MIN_MARGIN:
        return dict(report, reason="ambiguous_peak")
    correction = native.quadratic_offset(posterior, y, x)
    if correction is None:
        return dict(report, reason="nonconcave_subpixel_peak")
    offset = offsets[y, x]+correction
    if np.max(np.abs(offset)) > native.RADIUS:
        return dict(report, reason="outside_returned_search_radius")
    return dict(report, query_offset=offset.tolist())


def independent_peak(scores, support, center):
    """Replay the frozen hard-peak decision using the already computed field."""
    values = np.where(support, scores, -np.inf)
    y, x = np.unravel_index(np.argmax(values), values.shape)
    offset = np.array([x-REACH, y-REACH], float)
    score = float(values[y, x])
    yy, xx = np.indices(values.shape)
    alternate = support & (np.maximum(np.abs(xx-x), np.abs(yy-y)) > native.HALO)
    margin = None if not alternate.any() else float(score-values[alternate].max())
    report = dict(query_xy=None, reason=None, score=score, margin=margin,
                  integer_offset=offset.tolist(), conditional_on_camera=False)
    if score < native.MIN_CORRELATION:
        return dict(report, reason="weak_peak")
    if margin is None or margin < native.MIN_MARGIN:
        return dict(report, reason="ambiguous_peak")
    if np.any(np.abs(offset) > native.RADIUS) or min(x, y) < 1 or max(x, y) >= 2*REACH:
        return dict(report, reason="outside_returned_search_radius")
    correction = native.quadratic_offset(values, y, x)
    if correction is None:
        return dict(report, reason="nonconcave_subpixel_peak")
    offset += correction
    if np.max(np.abs(offset)) > native.RADIUS:
        return dict(report, reason="outside_returned_search_radius")
    return dict(report, query_xy=(center+offset).tolist())


def reverse(fields, p, observed):
    peak_support = cv2.erode(fields["available"].astype(np.uint8), np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    _, paired = native.sample(fields["image"], peak_support, p, native.PATCH)
    result = safe_direction(fields["query"], fields["query_valid"], fields["image"],
                            fields["available"], observed, p, subpixel=True, template_mask=paired)
    error = None if result["query_xy"] is None else float(np.linalg.norm(np.asarray(result["query_xy"])-p))
    return result, error


def measure(fields, depth, k, reference_pose, query_k, *, camera=None):
    """Hard independent peaks or explicitly camera-conditioned grid modes."""
    anchors = native.control.pipeline.previous.ANCHORS
    p, q, traces, failures = [], [], [], []
    if fields is not None:
        failures.extend(fields["skipped"])
        local = None if camera is None else np.linalg.inv(reference_pose)@camera
        for index, record in enumerate(fields["records"]):
            center = record["center"]
            if camera is None:
                forward = independent_peak(fields["scores"][index], fields["support"][index], center)
                endpoint = forward["query_xy"]
            else:
                prediction, _, front = native.control.project_z(record["local_world"][None], query_k, local)
                if not front[0]:
                    forward = dict(query_offset=None, reason="behind_hypothesized_camera")
                else:
                    forward = selected_peak(fields["scores"][index], fields["support"][index], prediction[0]-center)
                endpoint = None if forward["query_offset"] is None else (center+forward["query_offset"]).tolist()
            backwards = None; error = None
            reason = forward["reason"]
            if endpoint is not None:
                backwards, error = reverse(fields, center, np.asarray(endpoint))
                if error is None:
                    reason = "reverse_"+backwards["reason"]; endpoint = None
                elif error > native.MAX_ROUND_TRIP:
                    reason = "nonreciprocal_peak"; endpoint = None
            traces.append(dict(anchor_index=record["anchor_index"], forward=forward,
                               reverse=backwards, round_trip_error_px=error, reason=reason))
            if endpoint is None:
                failures.append(dict(anchor_index=record["anchor_index"], reason=reason))
            else:
                p.append(anchors[record["anchor_index"]].tolist()); q.append(endpoint)
    result = frame.solve(p, q, depth, k, reference_pose, query_k)
    return dict(result, searches=traces, failures=failures,
                attempted_patches=0 if fields is None else len(fields["records"])+len(fields["skipped"]),
                conditional_correspondences=camera is not None,
                independent_landmark_validation=False)


def run(reference, query, depth, k, reference_pose, query_k, initial_pose, *, reference_valid):
    start = perf_counter()
    fields = prepare_fields(reference, query, depth, k, reference_pose, query_k, initial_pose,
                            reference_valid=reference_valid)
    field_seconds = perf_counter()-start
    start = perf_counter()
    hard = measure(fields, depth, k, reference_pose, query_k)
    hard_seconds = perf_counter()-start
    hard["conditional_inference_seconds"] = field_seconds+hard_seconds
    start = perf_counter()
    camera, fit = optimize_camera(fields, reference_pose, query_k)
    fitted = perf_counter()-start
    start = perf_counter()
    joint = measure(fields, depth, k, reference_pose, query_k, camera=camera)
    joint_seconds = perf_counter()-start
    joint["conditional_inference_seconds"] = field_seconds+fitted+joint_seconds
    proposal = dict(reference_xy=[], query_xy=[], pose=None if camera is None else camera.tolist(),
                    independent_landmark_validation=False, proposal_only=True,
                    conditional_inference_seconds=field_seconds+fitted)
    return dict(methods=dict(hard=hard, joint=joint, joint_camera_proposal=proposal),
                fit=fit, fields=fields, calibrated_acceptance=False,
                timing=dict(fields=field_seconds, hard_measure=hard_seconds,
                            camera_fit=fitted, joint_measure=joint_seconds),
                primary_pose_source="Common AP3P on camera-conditioned IMAGE-GRID correspondences; not independent landmarks",
                patch_budget=36, known_matching_illumination=True)
