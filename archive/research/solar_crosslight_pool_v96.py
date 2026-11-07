"""Opt-in cross-light pose rescue with the original eight-plus-one solver calls."""

from collections import defaultdict
from math import floor

import numpy as np

from direct_render_adapter_v73 import (
    DirectRenderAdapter, available_sun_entries, geometry_digest,
)
from historical_compact_adapters_v73 import BaselineUnavailable, validate_images, validate_points


NAME = "reference_camera_render_match_crosslight_rescue"
POLICY = dict(
    version="unique_crosslight_eighth_slot_v96", spatial_conflict_radius_native_px=3.0,
    maximum_pooled_matches=2048, required_minimum_matches=6,
    candidate_pose_calls=8, final_pose_calls=1,
)


class SpatialIndex:
    """Return accepted endpoint IDs inside the fixed native-pixel radius."""

    def __init__(self, radius):
        self.radius = radius
        self.cells = defaultdict(list)

    def cell(self, point):
        return tuple(floor(float(v) / self.radius) for v in point)

    def nearby(self, point):
        x, y = self.cell(point)
        limit = self.radius ** 2
        return {
            index
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for index, previous in self.cells.get((x + dx, y + dy), ())
            if float((point - previous) @ (point - previous)) <= limit
        }

    def insert(self, point, index):
        self.cells[self.cell(point)].append((index, point))


def pool_correspondences(ranked):
    """Greedy one-to-one union in original light rank and SIFT match order."""
    radius = POLICY["spatial_conflict_radius_native_px"]
    limit = POLICY["maximum_pooled_matches"]
    reference_index, query_index = SpatialIndex(radius), SpatialIndex(radius)
    reference, query, lights = [], [], set()
    duplicates = conflicts = beyond_cap = raw_pairs = 0
    for proposal in ranked:
        light = proposal["entry"]["light_index"]
        raw_pairs += len(proposal["p"])
        for p, q in zip(proposal["p"], proposal["q"], strict=True):
            if len(reference) >= limit:
                beyond_cap += 1
                continue
            near_p, near_q = reference_index.nearby(p), query_index.nearby(q)
            if near_p.intersection(near_q):
                duplicates += 1
            elif near_p or near_q:
                conflicts += 1
            else:
                index = len(reference)
                reference_index.insert(p, index)
                query_index.insert(q, index)
                reference.append(p)
                query.append(q)
                lights.add(int(light))
    audit = dict(
        raw_pairs=raw_pairs, distinct_pairs=len(reference), duplicate_pairs_removed=duplicates,
        conflicting_endpoint_pairs_removed=conflicts, capped_pairs_removed=beyond_cap,
        contributing_reference_light_indices=sorted(lights),
    )
    return np.asarray(reference, float).reshape(-1, 2), np.asarray(query, float).reshape(-1, 2), audit


def _score(proposal, estimator, inp):
    pose, info = estimator(proposal["p"], proposal["q"], inp)
    return dict(proposal, pose=pose, info=info)


def _rank_original(proposal):
    return (-len(proposal["p"]), -proposal["appearance"], proposal["entry"]["light_index"])


def _rank_solved(proposal):
    return (proposal["pose"] is None, -int(proposal["info"].get("inliers", 0)),
            float(proposal["info"].get("clipped_squared_reprojection_cost", float("inf"))),
            -len(proposal["p"]), proposal["entry"]["light_index"])


def select_and_finalize(proposals, estimator, inp):
    """Spend one otherwise incapable slot only after the first seven miss."""
    trials = POLICY["candidate_pose_calls"]
    minimum = POLICY["required_minimum_matches"]
    ranked = sorted(proposals, key=_rank_original)
    if len(ranked) < trials:
        raise ValueError("At least eight declared reference templates required")
    top = ranked[:trials]
    solved = [_score(proposal, estimator, inp) for proposal in top[:-1]]
    eighth = top[-1]
    can_replace = len(eighth["p"]) < minimum and all(row["pose"] is None for row in solved)
    audit = dict(
        eligible=can_replace, attempted=False, selected_pooled=False,
        original_eighth_pairs=len(eighth["p"]), original_eighth_light_index=eighth["entry"]["light_index"],
        candidate_pose_calls=trials, final_pose_calls=POLICY["final_pose_calls"],
        original_match_counts={str(row["entry"]["light_index"]): len(row["p"]) for row in ranked},
    )
    pooled = None
    if can_replace:
        p, q, evidence = pool_correspondences(ranked)
        audit["evidence"] = evidence
        if len(p) >= minimum and len(evidence["contributing_reference_light_indices"]) >= 2:
            pooled = _score(dict(p=p, q=q, pooled=True), estimator, inp)
            audit["attempted"] = True
    if pooled is None:
        solved.append(_score(eighth, estimator, inp))
    else:
        # The real shared estimator unconditionally returns null with zero
        # inliers below six raw pairs. This sentinel participates only in the
        # original tie-break. It does not count as a measured solver call.
        solved.append(dict(eighth, pose=None, info={"inliers": 0}))
    audit["pooled_trial_pose_returned"] = bool(pooled is not None and pooled["pose"] is not None)
    best = pooled if audit["pooled_trial_pose_returned"] else min(solved, key=_rank_solved)
    final_pose, info = estimator(best["p"], best["q"], inp)
    if best.get("pooled") and (final_pose is None or not np.allclose(final_pose, best["pose"], rtol=0.0, atol=1e-7)):
        raise ValueError("The unchanged final estimator did not reproduce its pooled pose")
    audit["selected_pooled"] = bool(best.get("pooled"))
    return best, final_pose, info, audit


class CrossLightDirectAdapter(DirectRenderAdapter):
    """Same direct render and common pose solver with one optional rescue."""

    def __init__(self, profile):
        if (profile.get("name") != NAME or profile.get("backend") != "direct_render"
                or profile.get("crosslight_pooling") != POLICY
                or profile["configuration"]["top_pose_candidates"] != POLICY["candidate_pose_calls"]):
            raise ValueError("Exact opt-in cross-light direct profile required")
        super().__init__(profile)
        self.provenance = dict(self.provenance, crosslight_pooling=dict(POLICY))

    def infer(self, inp):
        validate_images(inp)
        if inp.known_query_sun is not None:
            raise ValueError("This cross-light profile requires the primary unknown-Sun track")
        if inp.solar_lookup is None or not hasattr(inp.solar_lookup, "protocol"):
            raise BaselineUnavailable("Identical reference-map identity/coverage is unavailable")
        physical = inp.solar_lookup.protocol
        key = physical["reference_geometry_sha256"]
        if key not in self.libraries:
            raise BaselineUnavailable("No declared library for this qualified reference geometry")
        library = self.libraries[key]
        if (physical["mesh_sha256"] != library["map_mesh_sha256"]
                or physical["scene_mesh_sha256"] != library["scene_mesh_sha256"]):
            raise BaselineUnavailable("Render library map differs from primary map/height perturbation")
        if geometry_digest(inp.geometry.depth, inp.geometry.intrinsics, inp.geometry.cam2world) != library["reference_geometry_array_sha256"]:
            raise ValueError("Native reference depth/camera differs from library rendering camera")
        entries = available_sun_entries(library["entries"], None)
        features = self.matcher.query_features(inp.query)
        proposals = []
        for entry in entries:
            render = self.image_loader(entry)
            p, q, appearance = self.matcher.match(render, features)
            p, q = validate_points(p, q)
            proposals.append(dict(entry=entry, p=p, q=q, appearance=float(appearance)))
        best, pose, info, audit = select_and_finalize(proposals, self.estimator, inp)
        entry = best.get("entry")
        return dict(
            reference_xy=best["p"], query_xy=best["q"], pose=pose, reason=info.get("reason"), common_estimator=info,
            estimated_sun=None if entry is None else np.asarray(entry["sun_world"]), query_truth_used=False,
            render_search=dict(
                reference_viewpoint_only=True, library_entries_considered=len(entries),
                common_pnp_candidates=POLICY["candidate_pose_calls"], final_common_pnp_calls=POLICY["final_pose_calls"],
                exact_known_sun=False, selected_light_index=None if entry is None else entry["light_index"],
                original_reference_preparation_seconds=library["original_reference_preparation_seconds"],
                incremental_render_seconds=0.0,
                preparation_cost_scope=library.get("preparation_cost_scope", "conservative whole-pair outer wall-time bound"),
                original_render_source="synthetic physical reference-role images, never query-role images",
                optical_accuracy_verified=False, crosslight_pooling=audit,
            ),
        )


class CrossLightBackend:
    """Backend factory for an explicitly frozen v91-aligned comparison arm."""

    def __init__(self, profile):
        self.profile = profile
        self.model = CrossLightDirectAdapter(profile)
        self.provenance = self.model.provenance

    def infer(self, inp):
        return self.model.infer(inp)
