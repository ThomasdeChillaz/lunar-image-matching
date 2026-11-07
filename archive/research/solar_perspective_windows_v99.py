"""Paid reciprocal patch matching in a projected reference-surface frame."""
import cv2
import numpy as np

from direct_render_adapter_v73 import validate_points
from lunar_geometry_v5 import map_points_z, project_z
from solar_direct_additive_windows_v96 import (
    CROP_RADIUS, SEARCH_BUDGETS, SEARCH_RADIUS, TEMPLATE_RADIUS, DirectAdditiveWindowsBackend,
    _cells, _crop, _full_crop, _inlier_support, _peak, _pixel, projected_pool,
)


METHOD = "terrain_perspective_uniform"
GRID_OFFSETS = np.stack(np.meshgrid(np.arange(-CROP_RADIUS, CROP_RADIUS + 1),
                                   np.arange(-CROP_RADIUS, CROP_RADIUS + 1)), axis=-1)
SEARCH_SLICE = slice(TEMPLATE_RADIUS, 2 * CROP_RADIUS + 1 - TEMPLATE_RADIUS)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def surface_map(reference_centre, geometry, query_k, seed_pose):
    """Query image coordinates of the local reference depth grid under the seed pose."""
    source = GRID_OFFSETS + reference_centre
    world, valid = map_points_z(source.reshape(-1, 2), geometry.depth, geometry.intrinsics, geometry.cam2world)
    if not valid.all():
        return None, "incomplete_reference_surface"
    pixels, _, front = project_z(world, query_k, seed_pose)
    if not front.all() or not np.isfinite(pixels).all():
        return None, "surface_behind_query_camera"
    if not ((pixels >= 0) & (pixels <= 511)).all():
        return None, "incomplete_projected_query_support"
    pixels = pixels.reshape(source.shape)
    # Check the two consistently oriented triangles covering each cell.
    # A local fold/reversal is detectable from the reference map alone.
    origin, end = pixels[:-1, :-1], pixels[1:, 1:]
    dx, dy = pixels[:-1, 1:] - origin, pixels[1:, :-1] - origin
    if (_cross(dx, dy) <= 0).any() or (_cross(end - pixels[1:, :-1], end - pixels[:-1, 1:]) <= 0).any():
        return None, "folded_or_degenerate_projected_surface"
    return pixels, None


def _masked_peak(template, search, supported):
    if not supported.any() or template.max() == template.min():
        return None, None, 0
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    masked = np.where(supported, response, -np.inf)
    peak = masked.max()
    positions = np.argwhere(masked == peak)
    if not np.isfinite(peak) or len(positions) != 1:
        return None, None, 1
    return positions[0, ::-1], masked, 1


def _peak_delta(response, x, y):
    step = np.zeros(2)
    for axis, index in enumerate((x, y)):
        if 0 < index < 2 * SEARCH_RADIUS:
            left, centre, right = (response[y, x - 1:x + 2] if axis == 0 else response[y - 1:y + 2, x])
            if np.isfinite([left, centre, right]).all():
                curvature = float(left) + float(right) - 2 * float(centre)
                if curvature < 0:
                    step[axis] = np.clip(.5 * (float(left) - float(right)) / curvature, -.5, .5)
    return step


def _interpolate_grid(grid, point):
    x, y = point
    i, j = int(x), int(y)
    nx, ny = min(i + 1, grid.shape[1] - 1), min(j + 1, grid.shape[0] - 1)
    dx, dy = x - i, y - j
    return (1 - dy) * ((1 - dx) * grid[j, i] + dx * grid[j, nx]) + dy * ((1 - dx) * grid[ny, i] + dx * grid[ny, nx])


def measure_surface_window(render, query, reference_centre, projected_centre, geometry, query_k, seed_pose):
    """One paid window, one 25px template, and at most two NCC calls.

    The 57x57 search is sampled on the reference surface. Candidate centres
    still lie within the original plus/minus 16 native query-pixel bounds.
    """
    a, b = _pixel(reference_centre), _pixel(projected_centre)
    if not _full_crop(a) or not _full_crop(b):
        return None, 0, "incomplete_native_crop"
    pixels, error = surface_map(a, geometry, query_k, seed_pose)
    if pixels is None:
        return None, 0, error
    target_grid = pixels[SEARCH_SLICE, SEARCH_SLICE]
    supported = (np.abs(target_grid - b) <= SEARCH_RADIUS).all(2)
    projected = cv2.remap(query, pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    reference_search = _crop(render, a, CROP_RADIUS)
    reference_template = _crop(render, a, TEMPLATE_RADIUS)
    peak, response, calls = _masked_peak(reference_template, projected, supported)
    if peak is None:
        return None, calls, "flat_template_or_ambiguous_forward_peak"
    x, y = peak
    # Reverse in the same surface coordinates. This is a cycle check, and
    # still cannot distinguish fully repeated or optically wrong textures.
    query_template = projected[y:y + 2 * TEMPLATE_RADIUS + 1, x:x + 2 * TEMPLATE_RADIUS + 1]
    reverse, _, reverse_calls = _peak(query_template, reference_search)
    calls += reverse_calls
    if reverse is None or np.any(reverse != 0):
        return None, calls, "nonreciprocal_or_ambiguous_peak"
    subpixel = _interpolate_grid(target_grid, peak + _peak_delta(response, x, y))
    if (np.abs(subpixel - b) > SEARCH_RADIUS).any():
        subpixel = target_grid[y, x].copy()
    return subpixel, calls, None


def uniform_actions(cells, occupied, budget):
    """Same frozen occupancy and tie/order rule as the v96 uniform control."""
    available = set(range(len(cells)))
    for _ in range(min(budget, len(available))):
        action = min(available, key=lambda i: (occupied[cells[i]], cells[i], i))
        available.remove(action)
        occupied[cells[action]] += 1
        yield action


def add_surface_windows(seed, render, query, geometry, query_k, *, budget, final_estimator):
    if budget not in SEARCH_BUDGETS:
        raise ValueError("Explicit matched incremental patch budget required")
    trace = dict(policy=METHOD, requested_budget=budget, eligible_grid_ids=[], attempted_windows=[],
                 accepted_additions=0, ncc_calls=0, additional_common_pnp_calls=0, used_original_seed=True, reason=None)
    if seed["pose"] is None:
        trace["reason"] = "no_direct_seed_pose"
        return seed, trace
    if any(not isinstance(im, np.ndarray) or im.shape != (512, 512) or im.dtype != np.uint8 for im in (render, query)):
        raise ValueError("Original native uint8 selected render and query required")

    seed_p, seed_q = validate_points(seed["reference_xy"], seed["query_xy"])
    pose = seed["pose"]
    pool = projected_pool(seed_p, seed_q, pose, geometry, query_k)
    trace["eligible_grid_ids"] = pool.grid_ids.tolist()
    if not len(pool.grid_ids):
        trace["reason"] = "no_eligible_projected_windows"
        return seed, trace
    support_p, _, _ = _inlier_support(seed_p, seed_q, pose, geometry, query_k)
    occupied = np.bincount(_cells(support_p), minlength=16)
    used = {_pixel(point) for point in seed_q}
    added_p, added_q = [], []
    for i in uniform_actions(_cells(pool.reference_centres), occupied, budget):
        p, target = pool.reference_centres[i], pool.query_centres[i]
        q, calls, reason = measure_surface_window(render, query, p, target, geometry, query_k, pose)
        trace["ncc_calls"] += calls
        if q is not None and _pixel(q) in used:
            q, reason = None, "duplicate_query_match"
        trace["attempted_windows"].append(dict(
            grid_id=int(pool.grid_ids[i]), policy=METHOD, reference_xy=p.tolist(),
            projected_query_xy=target.tolist(), query_xy=None if q is None else q.tolist(), reason=reason,
        ))
        if q is not None:
            used.add(_pixel(q))
            added_p.append(p)
            added_q.append(q)

    trace["accepted_additions"] = len(added_p)
    if not added_p:
        trace["reason"] = "no_valid_added_matches"
        return seed, trace
    p, q = np.vstack((seed_p, added_p)), np.vstack((seed_q, added_q))
    trace["additional_common_pnp_calls"] = 1
    proposed, info = final_estimator(p, q)
    if proposed is None:
        trace.update(reason="added_match_pose_failed", refit_reason=info.get("reason"))
        return seed, trace
    trace["used_original_seed"] = False
    return dict(seed, reference_xy=p, query_xy=q, pose=proposed, reason=info.get("reason"), common_estimator=info), trace


class TerrainPerspectiveWindowsBackend(DirectAdditiveWindowsBackend):
    """Same frozen direct seed and uniform paid schedule as v96."""

    def __init__(self, profile):
        config = profile["configuration"]
        if config["policy"] != METHOD:
            raise ValueError("Explicit opt-in terrain perspective profile required")
        super().__init__(dict(profile, configuration=dict(config, policy="uniform")))
        self.policy = METHOD
        self.provenance.update(policy=METHOD, incremental_measurement=(
            "Reference axial-depth surface projected under the direct seed; reciprocal NCC and native-coordinate output."
        ))

    def infer_with_trace(self, inp):
        seed = self.direct.infer(inp)
        render = None
        if seed["pose"] is not None:
            key = inp.solar_lookup.protocol["reference_geometry_sha256"]
            selected = seed["render_search"]["selected_light_index"]
            entry, = [entry for entry in self.direct.libraries[key]["entries"] if entry["light_index"] == selected]
            render = self.direct.image_loader(entry)
        return add_surface_windows(
            seed, render, inp.query, inp.geometry, inp.query_intrinsics, budget=self.budget,
            final_estimator=lambda p, q: self.direct.estimator(p, q, inp),
        )
