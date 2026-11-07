"""Paid native NCC additions to the exact reference-render SIFT control."""

from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np

from diagnose_direct_pose_geometry_v92 import reference_range
from direct_render_adapter_v73 import DirectRenderAdapter, validate_points
from lunar_geometry_v5 import map_points_z, project_z
from solar_assignment_v63 import height_variants
from solar_frontend_v63 import reference_anchors
from solar_translation_acquisition_v93 import choose_translation_patch


POLICIES = ('frozen_direct', 'uniform', 'translation_information')
SEARCH_BUDGETS = (32, 64, 128)
TEMPLATE_RADIUS = 12
SEARCH_RADIUS = 16
CROP_RADIUS = TEMPLATE_RADIUS + SEARCH_RADIUS


def _pixel(point):
    return tuple(np.rint(point).astype(int))


def _full_crop(point, radius=CROP_RADIUS):
    return all(radius <= xy < 512 - radius for xy in point)


def _crop(image, centre, radius):
    x, y = centre
    return image[y - radius:y + radius + 1, x - radius:x + radius + 1]


def _peak(template, search):
    if template.max() == template.min():
        return None, None, 0
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    peak = response.max()
    positions = np.argwhere(response == peak)
    if not np.isfinite(peak) or len(positions) != 1:
        return None, None, 1
    return positions[0, ::-1] - SEARCH_RADIUS, response, 1


def measure_window(render, query, reference_centre, projected_centre):
    """One paid search, at most two NCC calls, with integer reciprocity."""
    a, b = _pixel(reference_centre), _pixel(projected_centre)
    if not _full_crop(a) or not _full_crop(b):
        return None, 0, 'incomplete_native_crop'
    offset, response, calls = _peak(_crop(render, a, TEMPLATE_RADIUS), _crop(query, b, CROP_RADIUS))
    if offset is None:
        return None, calls, 'flat_template_or_ambiguous_forward_peak'
    integer_query = tuple(np.asarray(b) + offset)
    reverse, _, reverse_calls = _peak(_crop(query, integer_query, TEMPLATE_RADIUS), _crop(render, a, CROP_RADIUS))
    calls += reverse_calls
    if reverse is None or np.any(reverse != 0):
        return None, calls, 'nonreciprocal_or_ambiguous_peak'

    # The reverse test uses the integer match. Refine the forward centre
    # afterward using only its paid correlation surface.
    centre = np.asarray(integer_query, float)
    x, y = offset + SEARCH_RADIUS
    for axis, coordinate in enumerate((x, y)):
        if 0 < coordinate < 2 * SEARCH_RADIUS:
            left, middle, right = (response[y, x - 1:x + 2] if axis == 0 else response[y - 1:y + 2, x])
            curvature = float(left) + float(right) - 2 * float(middle)
            if curvature < 0:
                centre[axis] += np.clip(.5 * (float(left) - float(right)) / curvature, -.5, .5)
    return centre, calls, None


@dataclass(frozen=True)
class ProjectedPool:
    reference_centres: np.ndarray
    query_centres: np.ndarray
    world: np.ndarray
    grid_ids: np.ndarray


def projected_pool(seed_p, seed_q, seed_pose, geometry, query_k):
    """Same nominal geometry, crop and duplicate eligibility for both arms."""
    anchors = np.rint(reference_anchors()).astype(float)
    world, depth_valid = map_points_z(anchors, geometry.depth, geometry.intrinsics, geometry.cam2world)
    pixels, _, front = project_z(world, query_k, seed_pose)
    seed_reference, seed_query = ({_pixel(p) for p in points} for points in (seed_p, seed_q))
    centres, proposed, xyz, grid_ids = [], [], [], []
    for i, p, q, w, visible in zip(np.flatnonzero(depth_valid), anchors[depth_valid], pixels, world, front, strict=True):
        if not visible or not np.isfinite(q).all():
            continue
        a, b = _pixel(p), _pixel(q)
        if a in seed_reference or b in seed_query or not _full_crop(a) or not _full_crop(b):
            continue
        seed_reference.add(a)
        seed_query.add(b)
        centres.append(a)
        proposed.append(b)
        xyz.append(w)
        grid_ids.append(int(i))
    return ProjectedPool(np.asarray(centres, float).reshape(-1, 2), np.asarray(proposed, float).reshape(-1, 2),
                         np.asarray(xyz, float).reshape(-1, 3), np.asarray(grid_ids, int))


def _inlier_support(p, q, pose, geometry, query_k):
    world, valid = map_points_z(p, geometry.depth, geometry.intrinsics, geometry.cam2world)
    p, q = p[valid], q[valid]
    projected, _, front = project_z(world, query_k, pose)
    inliers = front & (np.linalg.norm(projected - q, axis=1) <= 3.)
    # Repeated SIFT orientations at the same pixel do not add independent
    # information to the selector. The actual seed matches stay untouched.
    keep, used_p, used_q = [], set(), set()
    for index in np.flatnonzero(inliers):
        a, b = _pixel(p[index]), _pixel(q[index])
        if a in used_p or b in used_q:
            continue
        keep.append(int(index))
        used_p.add(a)
        used_q.add(b)
    return p[keep], q[keep], world[keep]


def _cells(points):
    xy = np.floor(points / 128).astype(int)
    return xy[:, 1] * 4 + xy[:, 0]


def _translation_context(pool, inlier_q, inlier_world, pose, geometry):
    world = np.vstack((inlier_world, pool.world))
    n = len(inlier_world)
    # Grazing or horizontal reference rays do not have useful height-shift
    # variants. They remain in the shared pool, including uniform fallback.
    variant_supported = np.abs(world[:, 2] - geometry.cam2world[2, 3]) >= 1e-8
    variants = np.repeat(world[:, None], 7, axis=1)
    variants[variant_supported] = height_variants(world[variant_supported], geometry.cam2world)
    graph = SimpleNamespace(world_variants=variants, query_xy=np.vstack((inlier_q, pool.query_centres))[:, None].copy())
    beams = [dict(pose=pose, score=0., assignment=dict(
        native_keep=np.flatnonzero(variant_supported[:n]), native_candidate=np.zeros(len(world), int),
    ))]
    actions = np.column_stack((np.arange(n, len(world)), np.zeros(len(pool.world), int)))
    return graph, beams, actions, variant_supported[n:], reference_range(geometry.depth, geometry.intrinsics)


def add_windows(seed, render, query, geometry, query_k, *, policy, budget, final_estimator):
    """Return (result, trace). Failure returns the identical frozen seed object."""
    if policy not in POLICIES or (policy != 'frozen_direct' and budget not in SEARCH_BUDGETS):
        raise ValueError('Explicit direct control or matched native window budget required')
    trace = dict(policy=policy, requested_budget=0 if policy == 'frozen_direct' else budget,
                 eligible_grid_ids=[], attempted_windows=[], accepted_additions=0, ncc_calls=0,
                 additional_common_pnp_calls=0, used_original_seed=True, reason=None)
    if policy == 'frozen_direct' or seed['pose'] is None:
        trace['reason'] = 'frozen_control' if policy == 'frozen_direct' else 'no_direct_seed_pose'
        return seed, trace
    if any(not isinstance(image, np.ndarray) or image.shape != (512, 512) or image.dtype != np.uint8
           for image in (render, query)):
        raise ValueError('Original native uint8 selected render and query required')
    seed_p, seed_q = validate_points(seed['reference_xy'], seed['query_xy'])
    pose = seed['pose']
    pool = projected_pool(seed_p, seed_q, pose, geometry, query_k)
    trace['eligible_grid_ids'] = pool.grid_ids.tolist()
    if not len(pool.grid_ids):
        trace['reason'] = 'no_eligible_projected_windows'
        return seed, trace
    support_p, support_q, support_world = _inlier_support(seed_p, seed_q, pose, geometry, query_k)
    occupied = np.bincount(_cells(support_p), minlength=16)
    cells = _cells(pool.reference_centres)
    unavailable = np.zeros(len(pool.grid_ids), bool)
    if policy == 'translation_information':
        graph, beams, actions, supported, scale = _translation_context(pool, support_q, support_world, pose, geometry)
    used_query = {_pixel(point) for point in seed_q}
    p_add, q_add = [], []
    for _ in range(min(budget, len(unavailable))):
        action = None
        chosen_policy = 'uniform'
        if policy == 'translation_information':
            action = choose_translation_patch(graph, beams, actions, query_k, unavailable, scale, supported)
            if action is not None:
                chosen_policy = policy
        if action is None:
            action = min(np.flatnonzero(~unavailable), key=lambda i: (occupied[cells[i]], cells[i], i))
        unavailable[action] = True
        occupied[cells[action]] += 1
        p, centre = pool.reference_centres[action], pool.query_centres[action]
        q, calls, failure = measure_window(render, query, p, centre)
        trace['ncc_calls'] += calls
        if q is not None and _pixel(q) in used_query:
            q, failure = None, 'duplicate_query_match'
        trace['attempted_windows'].append(dict(grid_id=int(pool.grid_ids[action]), policy=chosen_policy,
                                               reference_xy=p.tolist(), projected_query_xy=centre.tolist(),
                                               query_xy=None if q is None else q.tolist(), reason=failure))
        if q is None:
            continue
        used_query.add(_pixel(q))
        p_add.append(p)
        q_add.append(q)
        if policy == 'translation_information' and supported[action]:
            new_anchor = actions[action, 0]
            graph.query_xy[new_anchor, 0] = q
            assignment = beams[0]['assignment']
            assignment['native_keep'] = np.append(assignment['native_keep'], new_anchor)

    trace['accepted_additions'] = len(p_add)
    if not p_add:
        trace['reason'] = 'no_valid_added_matches'
        return seed, trace
    p, q = np.vstack((seed_p, p_add)), np.vstack((seed_q, q_add))
    trace['additional_common_pnp_calls'] = 1
    proposed, info = final_estimator(p, q)
    if proposed is None:
        trace.update(reason='added_match_pose_failed', refit_reason=info.get('reason'))
        return seed, trace
    trace['used_original_seed'] = False
    return dict(seed, reference_xy=p, query_xy=q, pose=proposed, reason=info.get('reason'), common_estimator=info), trace


class DirectAdditiveWindowsBackend:
    """Opt-in evaluator backend. The parent direct adapter stays frozen."""

    def __init__(self, profile):
        self.policy = profile['configuration']['policy']
        self.budget = profile['configuration']['budget']
        if self.policy not in POLICIES or self.budget not in SEARCH_BUDGETS:
            raise ValueError('Explicit matched additive native-window profile required')
        self.direct = DirectRenderAdapter(profile['direct_profile'])
        self.provenance = dict(parent=self.direct.provenance, policy=self.policy, budget=self.budget,
                               native_features_before_additions='frozen whole-image direct SIFT',
                               incremental_measurement='native 25px NCC template and 33x33 search, reciprocal')

    def infer_with_trace(self, inp):
        seed = self.direct.infer(inp)
        render = None
        if self.policy != 'frozen_direct' and seed['pose'] is not None:
            key = inp.solar_lookup.protocol['reference_geometry_sha256']
            selected = seed['render_search']['selected_light_index']
            entry, = [entry for entry in self.direct.libraries[key]['entries'] if entry['light_index'] == selected]
            render = self.direct.image_loader(entry)
        return add_windows(seed, render, inp.query, inp.geometry, inp.query_intrinsics,
                           policy=self.policy, budget=self.budget,
                           final_estimator=lambda p, q: self.direct.estimator(p, q, inp))

    def infer(self, inp):
        result, self.last_window_trace = self.infer_with_trace(inp)
        return result
