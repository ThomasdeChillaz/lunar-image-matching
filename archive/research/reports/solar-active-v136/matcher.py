"""Experimental sequential patch acquisition around a soft camera estimate.

This library has no lunar-data loader or launch authority. Its procedural smoke
is separate from a future admitted comparison. Public poses remain null until
an independent acceptance study. Historical matching code is imported unchanged.
"""

from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"solar-joint-v135"))
import likelihood as joint
from reference_patch_warp import ReferencePatchWarper, _project_surface
from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import map_points_z, project_z
from policy import MAX_EXTRA, POLICIES, Pool, acquire

native = joint.native
AXIS = np.rint(np.linspace(64, 448, 16)).astype(int)
ANCHORS = np.array([(x, y) for y in AXIS for x in AXIS])


def previews(image, valid, centers):
    """Nine cheap REFERENCE preview derivatives, never a query feature bank."""
    axis = np.array([-16., 0., 16.])
    y, x = np.meshgrid(axis, axis, indexing="ij")
    points = centers[:, None, :]+np.c_[x.ravel(), y.ravel()][None]
    def sample(values, delta=(0., 0.)):
        p = points+delta
        return cv2.remap(values.astype(np.float32), p[..., 0].astype(np.float32),
                         p[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    safe = np.isfinite(points).all(axis=-1)
    points = np.nan_to_num(points, nan=-100., posinf=-100., neginf=-100.)
    known = safe.copy()
    for offset in ((0., 0.), (-1., 0.), (1., 0.), (0., -1.), (0., 1.)):
        known &= sample(valid, offset) >= 1.-1e-6
    values = sample(image)
    scale = np.maximum(np.std(values, axis=1), .05)
    gx = .5*(sample(image, (1., 0.))-sample(image, (-1., 0.)))
    gy = .5*(sample(image, (0., 1.))-sample(image, (0., -1.)))
    gradients = np.stack((gx, gy), axis=-1)/scale[:, None, None]
    return np.where(known[..., None], gradients, 0.), known


def prepare_extra(reference, depth, k, reference_pose, query_k, initial, reference_valid):
    """One reference-only 3D raster; full query patch work happens after choice."""
    warper = ReferencePatchWarper(reference, depth, k, reference_pose,
                                  native.control.pipeline.previous.ANCHORS,
                                  reference_valid=reference_valid)
    raster, zbuffer, diagnostics = _project_surface(warper.world, warper.triangles,
        warper.radiance, query_k, initial, (512, 512), physical_visibility=True)
    image, available = native.prepare(raster, np.isfinite(raster))
    local_world, valid = map_points_z(ANCHORS, depth, k, np.eye(4))
    world = np.zeros((len(ANCHORS), 3)); world[valid] = local_world
    local_initial = np.linalg.inv(reference_pose)@initial
    centers, axial, front = project_z(world, query_k, local_initial)
    safe = np.isfinite(centers).all(axis=1)
    index = np.clip(np.floor(np.nan_to_num(centers, nan=-100., posinf=-100., neginf=-100.)+.5), 0, 511).astype(int)
    yy, xx = index[:, 1], index[:, 0]
    eligible = valid & reference_valid[ANCHORS[:, 1], ANCHORS[:, 0]] & front & safe
    eligible &= ((centers >= 33.) & (centers <= 478.)).all(axis=1)
    eligible &= available[yy, xx] & (np.abs(zbuffer[yy, xx]-axial) <= .05+.01*axial)
    # Do not reacquire the same old anchor under a different identifier.
    old = native.control.pipeline.previous.ANCHORS
    eligible &= ((ANCHORS[:, None]-old[None])**2).sum(axis=2).min(axis=1) >= 12.**2
    gradients, known = previews(image, available, centers)
    eligible &= known.all(axis=1)
    pool = Pool(ANCHORS.copy(), world, centers, gradients, known, eligible).validate()
    return pool, dict(image=image, available=available, diagnostics=diagnostics)


def camera_cloud(camera, fields, reference_pose, k):
    """Seven bounded local hypotheses along three weak projection directions.

    The Jacobian-based cloud is a search policy, not a calibrated pose covariance.
    It uses only the current predicted camera and acquired reference 3D points.
    """
    local = np.linalg.inv(reference_pose)@camera
    world = np.array([r["local_world"] for r in fields["records"]]).reshape(-1, 3)
    if len(world) < 6:
        return local[None]
    center = world.mean(axis=0)
    parameters = native.control.pose_parameters(local, center)
    base_xy, _, front = project_z(world, k, local)
    if not (front & np.isfinite(base_xy).all(axis=1)).all():
        return local[None]
    columns = []
    for axis in range(6):
        step = np.zeros(6); step[axis] = joint.SCALE[axis]*.001
        plus = project_z(world, k, native.control.camera_pose(parameters+step, center))[0]
        minus = project_z(world, k, native.control.camera_pose(parameters-step, center))[0]
        columns.append(((plus-minus)/.002).ravel())
    jacobian = np.column_stack(columns)
    if not np.isfinite(jacobian).all():
        return local[None]
    _, _, directions = np.linalg.svd(jacobian, full_matrices=False)
    poses = [local]
    for direction in directions[-3:]:
        rms = np.sqrt(np.mean((jacobian@direction)**2))
        magnitude = min(2., 2./max(rms, 1e-8))
        for sign in (-1., 1.):
            poses.append(native.control.camera_pose(parameters+sign*magnitude*direction*joint.SCALE, center))
    return np.asarray(poses)


def hypothesis_weights(fields, poses, k):
    """Recompute tempered evidence from every acquired field after a camera fit."""
    if not fields["records"]:
        return np.full(len(poses), 1./len(poses))
    world = np.array([r["local_world"] for r in fields["records"]])
    centers = np.array([r["center"] for r in fields["records"]])
    log_weights = []
    for pose in poses:
        xy, _, front = project_z(world, k, pose)
        delta = xy-centers
        inside = front & np.isfinite(delta).all(axis=1) & (np.abs(delta) <= native.RADIUS).all(axis=1)
        mass = np.where(inside, joint.sample_fields(fields["blurred"], delta+joint.REACH), 0.)
        log_weights.append(float(np.mean(np.log(joint.OUTLIER+(1.-joint.OUTLIER)*mass)))/joint.KAPPA)
    w = np.exp(np.array(log_weights)-max(log_weights))
    return w/w.sum()


def acquire_field(index, pool, extra, observed, observed_valid):
    """The sole full-resolution query correlation access for an extra anchor."""
    center = pool.template_centers[index]
    template, mask = native.sample(extra["image"], extra["available"], center, native.PATCH)
    target, valid = native.sample(observed, observed_valid, center, native.PATCH+2*joint.REACH)
    values, reason = native.correlation_surface(template, mask, target, valid)
    if values is None or not np.isfinite(values).any():
        return None, reason or "insufficient_query_support_or_texture"
    support = np.isfinite(values)
    scores = np.where(support, values, 0.)
    potential = np.where(support, np.exp(joint.KAPPA*(scores-native.MIN_CORRELATION)), 0.)
    blurred = cv2.GaussianBlur(potential, (13, 13), joint.GEOMETRY_SIGMA,
                               borderType=cv2.BORDER_CONSTANT)
    return dict(record=dict(anchor_index=36+index, reference_xy=pool.reference_xy[index].copy(),
        center=center.copy(), local_world=pool.local_world[index].copy(), raster_kind="extra"),
        scores=scores, support=support, blurred=blurred), None


def extend(fields, observation, camera):
    if observation is None:
        return fields
    result = dict(fields, records=fields["records"]+[observation["record"]], initial_pose=camera)
    for key in ("scores", "support", "blurred"):
        result[key] = np.concatenate((fields[key], observation[key][None]), axis=0)
    return result


def measure(fields, extra, camera, depth, k, reference_pose, query_k):
    p, q, traces, failures = [], [], [], list(fields["skipped"])
    local = np.linalg.inv(reference_pose)@camera
    for i, record in enumerate(fields["records"]):
        point, _, front = project_z(record["local_world"][None], query_k, local)
        forward = (joint.selected_peak(fields["scores"][i], fields["support"][i], point[0]-record["center"])
                   if front[0] else dict(query_offset=None, reason="behind_hypothesized_camera"))
        endpoint = None if forward["query_offset"] is None else record["center"]+forward["query_offset"]
        reverse = error = None; reason = forward["reason"]
        if endpoint is not None:
            source = fields if record["raster_kind"] == "base" else dict(fields, image=extra["image"], available=extra["available"])
            reverse, error = joint.reverse(source, record["center"], endpoint)
            if error is None:
                reason = "reverse_"+reverse["reason"]; endpoint = None
            elif error > native.MAX_ROUND_TRIP:
                reason = "nonreciprocal_peak"; endpoint = None
        traces.append(dict(anchor_index=record["anchor_index"], forward=forward, reverse=reverse,
                           round_trip_error_px=error, reason=reason))
        if endpoint is None:
            failures.append(dict(anchor_index=record["anchor_index"], reason=reason))
        else:
            p.append(record["reference_xy"].tolist()); q.append(endpoint.tolist())
    result = joint.frame.solve(p, q, depth, k, reference_pose, query_k)
    return dict(result, searches=traces, failures=failures, conditional_correspondences=True,
                independent_landmark_validation=False)


def run(reference, query, depth, k, reference_pose, query_k, initial_pose, *,
        reference_valid, height_direction_reference, policies=POLICIES, budget=MAX_EXTRA):
    """Run equal-extra-search controls; callers must admit any lunar execution."""
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError("Unique declared acquisition controls required")
    if type(budget) is not int or not 0 <= budget <= MAX_EXTRA:
        raise ValueError("Bounded extra patch budget required")
    camera_check(k, reference_pose)
    up = np.asarray(height_direction_reference, float)
    if up.shape != (3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up), 1.):
        raise ValueError("Explicit unit reference-frame terrain-height direction required")
    start = perf_counter()
    fields = joint.prepare_fields(reference, query, depth, k, reference_pose, query_k,
                                  initial_pose, reference_valid=reference_valid)
    if fields is None:
        null = joint.frame.solve([], [], depth, k, reference_pose, query_k)
        return dict(methods={p:dict(null, reason="no_reference_support_or_initializer") for p in ("joint_no_extra", *policies)},
                    calibrated_acceptance=False, native_pixels=512, parameters_added=0, extra_patch_budget=budget)
    fitted_camera, initial_fit = joint.optimize_camera(fields, reference_pose, query_k)
    base = joint.measure(fields, depth, k, reference_pose, query_k, camera=fitted_camera)
    camera = np.asarray(initial_pose) if fitted_camera is None else fitted_camera
    fields = dict(fields, initial_pose=camera,
                  blurred=fields.get("blurred", np.empty((0, 49, 49))),
                  records=[dict(r, reference_xy=native.control.pipeline.previous.ANCHORS[r["anchor_index"]].copy(),
                                raster_kind="base") for r in fields["records"]])
    if budget:
        pool, extra = prepare_extra(reference, depth, k, reference_pose, query_k, np.asarray(initial_pose), reference_valid)
    else:
        pool = extra = None
    shared_seconds = perf_counter()-start
    methods = dict(joint_no_extra=base)
    for policy in policies:
        tick = perf_counter(); state = dict(fields=fields, camera=camera, fit=initial_fit)
        failures = []; fit_calls = 0
        def propose(current):
            poses = camera_cloud(current["camera"], current["fields"], reference_pose, query_k)
            return poses, hypothesis_weights(current["fields"], poses, query_k)
        def observe(index):
            observation, reason = acquire_field(index, pool, extra, fields["query"], fields["query_valid"])
            if reason is not None:
                failures.append(dict(anchor_index=36+index, reason=reason))
            return observation
        def update(current, index, observation):
            nonlocal fit_calls
            if observation is None:
                return current
            expanded = extend(current["fields"], observation, current["camera"])
            camera, fit = joint.optimize_camera(expanded, reference_pose, query_k)
            fit_calls += fit["calls"]
            return dict(fields=expanded, camera=current["camera"] if camera is None else camera, fit=fit)
        if budget:
            state, acquisition = acquire(pool, query_k, up, policy, budget, state, propose, observe, update)
        else:
            acquisition = dict(policy=policy, requested_extra_patches=0, attempted_extra_patches=0,
                               selected_indices=[], cell_schedule=[], trace=[])
        row = (measure(state["fields"], extra, state["camera"], depth, k, reference_pose, query_k)
               if budget else deepcopy(base))
        row["failures"].extend(failures)
        methods[policy] = dict(row, acquisition=acquisition, extra_fit_calls=fit_calls,
            conditional_stage_seconds=shared_seconds+perf_counter()-tick,
            soft_camera_proposal=state["camera"].tolist(), soft_camera_is_primary=False)
    return dict(methods=methods, calibrated_acceptance=False, parameters_added=0,
                native_pixels=512, extra_patch_budget=budget, shared_seconds=shared_seconds,
                initial_camera_fit=initial_fit, initializer_computation_included=False,
                scope="Experimental known-light conditional stage; no independent landmarks or calibrated pose.")


def public_result(result, research_output=False):
    output = dict(pose=None, acceptance_status="uncalibrated", rejection_reason="candidate_not_calibrated")
    if research_output:
        output["raw_research"] = deepcopy(result)
    return output
