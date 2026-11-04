"""Reference-only physical relighting bootstrap for the v84 lazy solver."""
from time import perf_counter

import cv2
import numpy as np
import torch

from contrastive_terrain import sample_descriptors
from direct_render_adapter_v73 import DirectRenderAdapter, geometry_digest, validate_images, validate_points
from historical_compact_adapters_v73 import BaselineUnavailable
from localization_robust_pose_v4 import estimate
from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import map_points_z
from solar_acquisition_cache_v84 import AcquisitionProjectionCacheV84, acquisition_predictions_v84
from solar_assignment_v63 import (
    CandidateGraph, _lm_update, _matched, assignments, choose_action, evidence_tensor,
    initial_beams, project,
)
from solar_coarse_training_adapter_v86 import coarse_support, native_to_coarse
from solar_distributional_evaluation_v90 import distributional_candidates
from solar_frontend_v63 import candidates
from solar_native_v63 import SparseNativeRefiner, state_digest
from solar_physics_v63 import soft_shadow_observation, solar_log_evidence
from solar_response_cache_v84 import GeometryOwnerGuardV84
from solar_rotation_v71 import QUARTER_ANGLES, RotationBank

REVISION = "render_bootstrap_coarse128_v92"
RENDER_SLOTS = 4
BEAMS = 8
INITIAL_SAMPLES_PER_BEAM = 128


def resize128(image):
    return cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)


def lift_coarse_xy(points):
    return (np.asarray(points, dtype=float) + 0.5) * 4.0 - 0.5


def _exact_sun_index(catalogue, direction):
    matches = np.flatnonzero(np.linalg.norm(catalogue - direction, axis=1) <= 1e-6)
    if len(matches) > 1:
        raise ValueError("Ambiguous render Sun in the physical catalogue")
    return int(matches[0]) if len(matches) else None


class RenderBootstrap:
    """The same sealed library and SIFT settings as the v91 direct control.

    SIFT runs only on area-resized 128px views. The full native reference render
    is retained for later paid patches, and no native descriptors are computed
    here. Only four of the eight existing initial AP3P slots are allocated.
    """

    def __init__(self, direct_profile):
        self.direct = DirectRenderAdapter(direct_profile)

    def propose(self, inp):
        validate_images(inp)
        if inp.known_query_sun is not None:
            raise ValueError("This primary bootstrap does not take a query Sun label")
        if inp.solar_lookup is None or not hasattr(inp.solar_lookup, "protocol"):
            raise BaselineUnavailable("Qualified same-map render library required")
        physical = inp.solar_lookup.protocol
        library = self.direct.libraries.get(physical["reference_geometry_sha256"])
        if library is None:
            raise BaselineUnavailable("No sealed library for the exact reference geometry")
        geometry = inp.geometry
        if (physical["mesh_sha256"] != library["map_mesh_sha256"]
                or physical["scene_mesh_sha256"] != library["scene_mesh_sha256"]
                or geometry_digest(geometry.depth, geometry.intrinsics, geometry.cam2world)
                != library["reference_geometry_array_sha256"]):
            raise BaselineUnavailable("Reference depth, pose and full map must match the direct library")
        entries = sorted(library["entries"], key=lambda row: row["light_index"])
        if len(entries) != 24 or [row["light_index"] for row in entries] != list(range(24)):
            raise ValueError("All 24 original reference-only relights are required")
        query_features = self.direct.matcher.query_features(resize128(inp.query))
        ranked = []
        for entry in entries:
            render = self.direct.image_loader(entry)
            a, b, appearance = self.direct.matcher.match(resize128(render), query_features)
            p, q = validate_points(lift_coarse_xy(a), lift_coarse_xy(b))
            sun = _exact_sun_index(geometry.sun_vectors, np.asarray(entry["sun_world"]))
            ranked.append(dict(entry=entry, image=render, p=p, q=q, appearance=appearance, sun_index=sun))
        ranked.sort(key=lambda row: (-len(row["p"]), -row["appearance"], row["entry"]["light_index"]))
        selected = [row for row in ranked if row["sun_index"] is not None][:RENDER_SLOTS]
        seeds = []
        for row in selected:
            world, support = map_points_z(row["p"], geometry.depth, geometry.intrinsics, geometry.cam2world)
            pose, info = estimate(world, row["q"][support], inp.query_intrinsics, dict(
                solver="ap3p", iterations=INITIAL_SAMPLES_PER_BEAM, reclassify=True,
            ))
            seeds.append(dict(
                pose=pose, sun_index=row["sun_index"], light_index=row["entry"]["light_index"],
                requested_samples=INITIAL_SAMPLES_PER_BEAM, raw_matches=len(row["p"]),
                supported_matches=len(world), inliers=info["inliers"], reason=info["reason"],
                cost=info.get("clipped_squared_reprojection_cost"),
            ))
        seeds += [dict(pose=None, requested_samples=0, reason="no_above_horizon_catalogue_seed")] * (RENDER_SLOTS - len(seeds))
        live = [seed for seed in seeds if seed["pose"] is not None]
        if live:
            best = min(live, key=lambda seed: (-seed["inliers"], seed["cost"], seed["light_index"]))
            source = next(row for row in selected if row["entry"]["light_index"] == best["light_index"])
            reference, reference_sun = source["image"], np.asarray(source["entry"]["sun_world"])
        else:
            best = None
            reference, reference_sun = inp.reference, inp.reference_sun
        trace = dict(
            reference_views_considered=len(ranked), coarse_image_size=128, selected_render=None if best is None else best["light_index"],
            original_reference_preparation_seconds=library["original_reference_preparation_seconds"],
            coarse_match_counts={str(row["entry"]["light_index"]): len(row["p"]) for row in ranked},
            requested_render_samples=sum(seed["requested_samples"] for seed in seeds),
            seeds=seeds,
        )
        return reference, reference_sun, seeds, trace


def _seed_rotation(world, projected, pose, intrinsics, reference_pose):
    horizontal, front = project(world + reference_pose[:3, 0], pose, intrinsics)
    delta = horizontal - projected
    usable = front & np.isfinite(delta).all(axis=1)
    turns = np.zeros(len(world), dtype=int)
    turns[usable] = np.rint(np.rad2deg(np.arctan2(delta[usable, 1], delta[usable, 0])) / 90).astype(int) % 4
    return turns, usable


def insert_seed_windows(coarse, reference, bank, p, q, appearance, rotations, world, intrinsics, geometry, seeds):
    """Keep best four appearance edges and insert up to four valid seed windows."""
    q_new, score_new, rotation_new = q.copy(), appearance.copy(), rotations.copy()
    render_proposals = []
    for seed in seeds:
        pose = seed["pose"]
        if pose is None:
            continue
        predicted, front = project(world, pose, intrinsics)
        turns, tangent = _seed_rotation(world, predicted, pose, intrinsics, geometry.cam2world)
        render_proposals.append((predicted, front & tangent, turns))
    if not render_proposals:
        return q_new, score_new, rotation_new, 0
    images = np.stack([resize128(image) for image in [reference] + bank.images])
    with torch.inference_mode():
        dense, _ = coarse(torch.from_numpy(images).float()[:, None] / 255.0)
        ref = sample_descriptors(dense[:1], torch.from_numpy(native_to_coarse(p))[None])[0]
        used = np.zeros(len(p), dtype=int)
        for predicted, visible, turns in render_proposals:
            for index in range(len(bank.angles)):
                ids = np.flatnonzero(visible & (turns == index))
                if not len(ids):
                    continue
                rotated = bank.transform(predicted[ids], index)
                supported = coarse_support(bank.valid[index], native_to_coarse(rotated))
                supported &= np.asarray([bank.native_support(point, index) for point in rotated])
                ids = ids[supported]
                if not len(ids):
                    continue
                pixels = native_to_coarse(rotated[supported])
                query = sample_descriptors(dense[index + 1:index + 2], torch.from_numpy(pixels)[None])[0]
                scores = (ref[ids] * query).sum(dim=-1).cpu().numpy()
                for anchor, score in zip(ids, scores):
                    slot = RENDER_SLOTS + used[anchor]
                    # Duplicate centres under the same rotation add no window.
                    same = (rotation_new[anchor, :slot] == index) & (
                        np.linalg.norm(q_new[anchor, :slot] - predicted[anchor], axis=1) <= 1e-6)
                    if same.any():
                        continue
                    q_new[anchor, slot] = predicted[anchor]
                    score_new[anchor, slot] = score
                    rotation_new[anchor, slot] = index
                    used[anchor] += 1
    return q_new, score_new, rotation_new, int(used.sum())


def _render_seed_beams(graph, intrinsics, ensure_suns, seeds, response_enabled):
    regular = initial_beams(graph, "shared", BEAMS, response_enabled)
    if not regular:
        return []
    initial = [dict(regular[j % len(regular)], render_seed=None) for j in range(RENDER_SLOTS)]
    for j, seed in enumerate(seeds):
        if seed["pose"] is None:
            initial.append(dict(regular[(j + RENDER_SLOTS) % len(regular)], render_seed=seed))
            continue
        sun = seed["sun_index"]
        ensure_suns([sun])
        assignment = assignments(
            graph, sun, "shared", seed["pose"], intrinsics,
            measurement_state=np.zeros(graph.appearance.shape, dtype=np.uint8),
        )
        initial.append(dict(
            sun_index=sun, assignment=assignment, pose=None, score=assignment["total"],
            initial_candidate_rank=0, render_seed=seed,
        ))
    return initial


# Copy of v84 solve; the two initialization substitutions above its native rounds
# preserve the sampling, acquisition, Sun-refinement and final-estimator logic.
def solve_render_bootstrap(graph, intrinsics, measure, render_seeds, solar_mode='shared', active=False, policy='active', budget=128, response_enabled=False, response_refresh=None, final_estimator=None, solar_lookup=None, pose_initialization='strict3'):
    """Joint research inference without query depth, true pose or Sun labels.

    Each local search counts once. Responses are refreshed on the selected
    patches under every CURRENT trial pose, never carried over from an old
    canonicalization. After every measurement, correspondence assignment and
    hypothesis weights update before the next active decision. Actual pose
    RANSAC/LM updates occur at the three common round boundaries, identically
    across ablations. The proposal allocation stays <=4096 plus one common
    final2000iteration estimator. final_estimator(p,q) can supply the exact same
    reference-depth geometry/estimator used by all competing matchers.
    """
    if budget not in (32, 64, 128):
        raise ValueError('Declared32/64/128patch budget required')
    if pose_initialization not in ('strict3', 'quantization_aware'):
        raise ValueError('Explicit supported initial-pose policy required')
    if response_enabled and (not callable(response_refresh)):
        raise ValueError('Pair-specific current-pose response refresh callback required')
    graph = CandidateGraph(**{name: getattr(graph, name).copy() for name in graph.__dataclass_fields__})

    def ensure_suns(suns):
        missing = np.unique(np.asarray(suns, int))
        missing = missing[~graph.solar_materialized_mask[missing]]
        if not len(missing):
            return
        if not callable(solar_lookup):
            raise ValueError('Uncomputed solar columns require a source-pinned physical lookup')
        table = solar_lookup(graph.sun_vectors[missing])
        states = np.asarray(table['solar_states'])
        variants = np.asarray(table['solar_variants'])
        if states.shape != (len(graph.world), len(missing)) or variants.shape != (len(graph.world), len(missing), 7) or (not np.isin(states, [0, 1, 2]).all()) or (not np.isin(variants, [0, 1, 2]).all()):
            raise ValueError('Physical lookup returned invalid explicit solar states')
        graph.solar_states[:, missing] = states
        graph.solar_variants[:, missing] = variants
        graph.solar_materialized_mask[missing] = True
    if solar_mode != 'off':
        ensure_suns(np.flatnonzero(graph.coarse_sun_mask))
    if solar_mode != 'shared' or pose_initialization != 'quantization_aware' or len(render_seeds) != RENDER_SLOTS:
        raise ValueError('Experimental render bootstrap requires four slots, shared Sun and quantization-aware support')
    beams = _render_seed_beams(graph, intrinsics, ensure_suns, render_seeds, response_enabled)
    measurements = {}
    history = []
    incremental = []
    if not beams:
        return dict(pose=None, reason='no_candidate_assignments', measurements=[], hypotheses=[])
    n, k = graph.appearance.shape
    actions = np.array([(i, c) for i in range(n) for c in range(k)], int)
    acquisition_cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics, capacity=4)
    try:
        points = np.repeat(graph.reference_xy, k, axis=0)
        used = np.zeros(len(actions), bool)
        appearances = graph.appearance.reshape(-1)
        evidence = evidence_tensor(graph)
        response_calls = 0
        response_edges = set()
        measurement_state = np.zeros((n, k), np.uint8) if pose_initialization == 'quantization_aware' else None

        def response_cube(beam, suns):
            nonlocal response_calls
            if not response_enabled or beam['pose'] is None:
                return np.repeat(graph.appearance[:, None, :], len(suns), axis=1)
            token = getattr(response_refresh, 'cache_token', None)
            token = token() if callable(token) else None
            key = (beam['pose'].tobytes(), tuple(suns), token)
            cache = beam.setdefault('_response_matrices', {})
            if key not in cache:
                if len(cache) >= 4:
                    cache.clear()
                cache[key] = (np.repeat(graph.appearance[:, None, :], len(suns), axis=1), set())
            result, done = cache[key]
            for (i, c), measurement in measurements.items():
                edge = (i, c)
                if edge in done:
                    continue
                done.add(edge)
                if measurement.get('query_xy') is None:
                    continue
                response_calls += 1
                values = response_refresh(i, c, measurement['query_xy'], beam['pose'], suns)
                if values is None:
                    continue
                values = np.asarray(values, float)
                if values.shape != (len(suns),) or not np.isfinite(values).all():
                    raise ValueError('Finite one-response-score-per-requested-Sun required')
                result[i, :, c] = values
                response_edges.add(edge)
            return result

        def update_beam(beam, suns):
            suns = np.asarray(suns, int)
            if solar_mode != 'off':
                ensure_suns(suns)
                evidence[:, suns] = solar_log_evidence(graph.solar_states[:, suns, None], graph.observed_shadow[:, None, :])
            responses = None if solar_mode == 'independent' and (not response_enabled) else response_cube(beam, suns)
            pose = beam['pose'] if beam['native_pose'] or pose_initialization == 'quantization_aware' else None
            if solar_mode == 'independent':
                a = assignments(graph, beam['sun_index'], solar_mode, pose, intrinsics, None, evidence, response_by_sun=responses, independent_sun_indices=suns, measurement_state=measurement_state, native_reprojection_gate=3.0 if measurement_state is not None and beam['native_pose'] else None)
                beam['assignment'] = a
                beam['score'] = a['total']
                return
            choices = []
            for j, si in enumerate(suns):
                a = assignments(graph, int(si), solar_mode, pose, intrinsics, responses[:, j, :], evidence, measurement_state=measurement_state, native_reprojection_gate=3.0 if measurement_state is not None and beam['native_pose'] else None)
                choices.append((a['total'], -int(si), int(si), a))
            _, _, beam['sun_index'], beam['assignment'] = max(choices, key=lambda x: x[:2])
            beam['score'] = beam['assignment']['total']

        def take(action, beam):
            i, c = map(int, actions[action])
            measurement = measure(i, c, beam['pose'], beam['sun_index'])
            used[action] = True
            if not isinstance(measurement, dict):
                raise ValueError('Patch measurement must be an explicit record')
            q = measurement.get('query_xy')
            if q is not None and (np.asarray(q).shape != (2,) or not np.isfinite(q).all()):
                raise ValueError('Finite native query correspondence required')
            if q is not None:
                if ((np.asarray(q) < 0) | (np.asarray(q) > 511)).any():
                    raise ValueError('Measured correspondence outside native image')
                graph.query_xy[i, c] = q
                if 'observed_shadow' in measurement:
                    graph.observed_shadow[i, c] = float(measurement['observed_shadow'])
                computed = graph.solar_materialized_mask
                evidence[i, computed, c] = solar_log_evidence(graph.solar_states[i, computed], graph.observed_shadow[i, c])
            measurements[i, c] = measurement
            history.append(dict(anchor=i, candidate=c, trial_sun_index=beam['sun_index']))
            if measurement_state is not None:
                measurement_state[i, c] = 1 if q is not None else 2
            for current in beams:
                if solar_mode == 'independent':
                    suns = current.get('_independent_search', np.flatnonzero(graph.coarse_sun_mask))
                else:
                    suns = np.array([current['sun_index']])
                update_beam(current, suns)
            scores = np.array([b['score'] for b in beams])
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            incremental.append(dict(after_search=len(history), weights=weights.tolist(), candidate_assignments=[b['assignment']['candidate'].copy() for b in beams]))
        proposal_requested = 0
        for b in beams:
            a = b['assignment']
            ids = a['keep']
            q = graph.query_xy[ids, a['candidate'][ids]]
            allocation = 1024 // len(beams)
            proposal_requested += allocation
            render_seed = b.pop('render_seed')
            if render_seed is not None:
                b['pose'] = render_seed['pose']
                b['pose_info'] = dict(render_seed, stage='coarse128_render_seed_slot')
            elif pose_initialization == 'quantization_aware':
                from solar_pose_proposals_v73 import propose
                b['pose'], b['pose_info'] = propose(graph.world[ids], q, intrinsics, allocation)
            else:
                b['pose'], b['pose_info'] = estimate(graph.world[ids], q, intrinsics, dict(solver='ap3p', iterations=allocation, reclassify=True))
            b['native_pose'] = False
        initialization = dict(policy=pose_initialization, initial_hypothesis_count=len(beams), initial_live_poses=sum((b['pose'] is not None for b in beams)), proposal_info=[dict(b['pose_info']) for b in beams], final_reprojection_threshold_native_px=3.0)
        seed_count = min(budget // 2, 8 * len(beams))
        orders = []
        for b in beams:
            a = b['assignment']
            order = sorted(a['keep'], key=lambda i: (int(graph.reference_xy[i, 1] // 128) * 4 + int(graph.reference_xy[i, 0] // 128), i))
            positions = np.rint(np.linspace(0, max(0, len(order) - 1), 8)).astype(int)
            orders.append([int(order[j]) * k + int(a['candidate'][order[j]]) for j in positions] if order else [])
        best_seed = max(range(len(beams)), key=lambda j: beams[j]['score'])
        for position in range(8):
            order = orders[best_seed]
            if len(history) >= seed_count:
                break
            if order and (not used[order[position]]):
                take(order[position], beams[best_seed])
        for turn in range(8 * len(beams)):
            if len(history) >= seed_count:
                break
            j = turn % len(beams)
            position = (turn // len(beams) + j) % 8
            order = orders[j]
            if order and (not used[order[position]]):
                take(order[position], beams[j])
        rounds = []
        for round_index in range(3):
            target = min(budget, len(history) + int(np.ceil((budget - len(history)) / (3 - round_index))))
            while len(history) < target and (not used.all()):
                predictions, weights, valid = acquisition_predictions_v84(graph, beams, actions, intrinsics, solar_mode, cache=acquisition_cache)
                chosen_policy = policy if active and any((b['pose'] is not None for b in beams)) else 'uniform'
                visited = {i for i, c in measurements}
                unavailable = used.copy()
                if len(visited) < n:
                    unavailable |= np.array([int(i) in visited for i, c in actions])
                action = choose_action(predictions, weights, points, unavailable, chosen_policy, appearances, valid)
                if action is None:
                    break
                take(action, max(beams, key=lambda b: b['score']))
            for b in beams:
                if solar_mode == 'independent':
                    centres = np.unique(b['assignment']['independent_sun_indices'])
                    suns = np.unique(np.concatenate([graph.search_indices(si, round_index) for si in centres]))
                    b['_independent_search'] = suns
                elif solar_mode == 'shared' or response_enabled:
                    suns = graph.search_indices(b['sun_index'], round_index)
                else:
                    suns = np.array([b['sun_index']])
                update_beam(b, suns)
                _, q, w, _ = _matched(graph, b, measurements)
                allocation = 1024 // len(beams)
                proposal_requested += allocation
                native_pose, info = estimate(w, q, intrinsics, dict(solver='ap3p', iterations=allocation, reclassify=True))
                if native_pose is not None:
                    b['pose'] = _lm_update(native_pose, w, q, intrinsics)
                    b['native_pose'] = True
                    b['pose_info'] = info
                update_beam(b, np.array([b['sun_index']]) if solar_mode != 'independent' else b['_independent_search'])
            rounds.append(dict(round=round_index + 1, queries=len(history), live_poses=sum((b['pose'] is not None for b in beams)), sun_indices=[b['sun_index'] for b in beams]))
        candidates = [b for b in beams if b['pose'] is not None]
        final_only_rescue = not candidates
        for b in candidates:
            if solar_mode == 'independent':
                suns = b.get('_independent_search', np.flatnonzero(graph.coarse_sun_mask))
            elif solar_mode == 'shared' or response_enabled:
                suns = graph.search_indices(b['sun_index'], 2)
            else:
                suns = np.array([b['sun_index']])
            update_beam(b, suns)
        if pose_initialization == 'quantization_aware':
            supported = [b for b in candidates if len(_matched(graph, b, measurements)[0]) >= 6]
            verified = [b for b in supported if b['native_pose'] and len(b['assignment'].get('native_inlier_keep', [])) >= 6]
            if verified:
                candidates = verified
            elif supported:
                candidates = supported
        best = max(candidates or beams, key=lambda b: b['score'])
        p, q, w, _ = _matched(graph, best, measurements)
        if len(p) < 6:
            pose = None
            info = dict(reason='insufficient_supported_native_matches', inliers=0)
            final_geometry = 'not_run_insufficient_native_measurements'
            final_iterations = 0
        elif final_estimator is None:
            pose, info = estimate(w, q, intrinsics, dict(solver='ap3p', iterations=2000, reclassify=True))
            final_geometry = 'exact_reference_map_anchors'
            final_iterations = 2000
        else:
            pose, info = final_estimator(p, q)
            final_geometry = 'caller_supplied_common_reference_geometry'
            final_iterations = info.get('iterations_requested')
        for b in beams:
            b.pop('_response_matrices', None)
            b.pop('_independent_search', None)
        for record in history:
            record['response_used'] = (record['anchor'], record['candidate']) in response_edges
        return dict(pose=pose, reason=info['reason'], reference_xy=p, query_xy=q, initialization=initialization, sun_enu=None if solar_mode != 'shared' else graph.sun_vectors[best['sun_index']], solar_supported=best['assignment']['solar_supported'], measurements=history, hypotheses=beams, proposal_samples_requested=proposal_requested, final_estimator_iterations=final_iterations, final_geometry=final_geometry, final_only_rescue=final_only_rescue, rounds=rounds, incremental_updates=incremental, response_measurements=len(response_edges), response_refresh_calls=response_calls, fine_sun_search_available=graph.has_fine_sun, solar_columns_materialized=int(graph.solar_materialized_mask.sum()), estimated_sun_parameters=None if not len(graph.sun_parameters) or solar_mode != 'shared' else graph.sun_parameters[best['sun_index']], acquisition_projection_cache=acquisition_cache.stats)
    finally:
        acquisition_cache.finish()


class RenderBootstrapBackend:
    """Normal evaluator backend: InferenceInput in, raw native pairs and pose out."""

    def __init__(self, profile):
        self.profile = profile
        self.bootstrap = RenderBootstrap(profile["direct_profile"])
        self.coarse_response = None
        if profile.get("coarse_response_checkpoint") is not None:
            from solar_distributional_evaluation_v90 import load_distributional_checkpoint

            self.coarse_response = load_distributional_checkpoint(**profile["coarse_response_checkpoint"])
            self.coarse = self.coarse_response.coarse
        else:
            from compact_lunar import load_checkpoint

            self.coarse, _ = load_checkpoint(profile["teacher_directory"])
            self.coarse.requires_grad_(False)
        if state_digest(self.coarse) != profile["frozen_descriptor_sha256"]:
            raise ValueError("Exact original compact proposal/refinement encoder required")
        self.provenance = dict(
            revision=REVISION, library=self.bootstrap.direct.provenance,
            original_descriptor_sha256=profile["frozen_descriptor_sha256"],
            coarse_response=profile.get("coarse_response_checkpoint"),
            parameters=sum(parameter.numel() for parameter in (self.coarse_response or self.coarse).parameters()),
            patch_budget=profile["configuration"]["budget"], bootstrap_ap3p_samples=512,
            standard_initial_ap3p_samples=512, native_pose_refinement_samples=3072,
        )
        if self.provenance["parameters"] >= 100000:
            raise ValueError("Combined experimental matcher exceeds 100,000 parameters")

    def infer(self, inp):
        from solar_evaluation_v70 import common_pose

        tick = perf_counter()
        cfg = self.profile["configuration"]
        feedback = cfg.get("render_feedback", "full")
        if feedback not in ("full", "pose_only", "reference_only"):
            raise ValueError("Use full, pose-only or matched reference-only render feedback")
        if cfg["budget"] not in (32, 64, 128) or cfg["policy"] not in ("active", "uniform", "appearance", "geometry"):
            raise ValueError("Original explicit patch budget and policy required")
        query_k, _ = camera_check(inp.query_intrinsics, np.eye(4))
        owner = GeometryOwnerGuardV84(inp.geometry)
        try:
            reference, reference_sun, seeds, bootstrap_trace = self.bootstrap.propose(inp)
            geometry = inp.geometry.with_reference_sun(reference_sun)
            bootstrap_trace["render_feedback"] = feedback
            bootstrap_trace["observed_reference_std"] = float(inp.reference.std())
            bootstrap_trace["measurement_reference_std"] = float(reference.std())
            bootstrap_trace["query_std"] = float(inp.query.std())
            bootstrap_trace["bootstrap_seconds"] = perf_counter() - tick
            if min(reference.std(), inp.query.std()) < 1:
                return dict(
                    pose=None, reason="blank_or_uninformative_image", render_bootstrap=bootstrap_trace,
                    reference_xy=np.empty((0, 2)), query_xy=np.empty((0, 2)),
                    elapsed_seconds=perf_counter() - tick, inference_revision=REVISION,
                )
            if self.coarse_response is None:
                p, q, appearance, rotation = candidates(self.coarse, reference, inp.query, QUARTER_ANGLES)
                response_trace = None
            else:
                p, q, appearance, rotation, response_trace = distributional_candidates(
                    self.coarse_response, reference, inp.query, geometry, QUARTER_ANGLES,
                    geometry.sun_vectors[geometry.coarse_sun_mask],
                )
            _, use = map_points_z(p, geometry.depth, geometry.intrinsics, geometry.cam2world)
            if use.sum() < 6:
                return dict(
                    pose=None, reason="insufficient_reference_surface", render_bootstrap=bootstrap_trace,
                    reference_xy=np.empty((0, 2)), query_xy=np.empty((0, 2)),
                    elapsed_seconds=perf_counter() - tick, inference_revision=REVISION,
                )
            p, q, appearance, rotation = p[use], q[use], appearance[use], rotation[use]
            world = geometry.world_variants[use, 0]
            reprojected, front = project(world, geometry.cam2world, geometry.intrinsics)
            if not front.all() or np.max(np.linalg.norm(reprojected - p, axis=1)) > 0.02:
                raise ValueError("Reference-map anchor geometry violates native pixel centres")
            bank = RotationBank(inp.query, QUARTER_ANGLES)
            inserted = 0
            if feedback == "full":
                q, appearance, rotation, inserted = insert_seed_windows(
                    self.coarse, reference, bank, p, q, appearance, rotation, world, query_k, geometry, seeds,
                )
            bootstrap_trace["seed_windows_inserted"] = inserted
            # The reference-only control spends the same render proposal slots
            # selecting the same relight, then withholds their poses from v84.
            solver_seeds = seeds if feedback != "reference_only" else [
                dict(seed, pose=None, reason="matched_render_pose_withheld") for seed in seeds
            ]
            shadow, observation = soft_shadow_observation(inp.query)
            shadow = shadow.astype(np.float32)
            observed = cv2.remap(shadow, q[:, :, 0].astype(np.float32), q[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
            graph = CandidateGraph(
                p, world, q, appearance, observed, geometry.solar_states[use], geometry.sun_vectors,
                geometry.world_variants[use], geometry.solar_variants[use], geometry.sun_parameters,
                geometry.coarse_sun_mask, geometry.solar_materialized_mask,
            )
            refiner = SparseNativeRefiner(self.coarse)

            def measure(anchor, candidate, trial_pose, sun):
                r = int(rotation[anchor, candidate])
                position = bank.transform(q[anchor, candidate][None], r)[0]
                if not bank.native_support(position, r):
                    return dict(query_xy=None, reason="incomplete_rotated_native_patch_support")
                measured, meta = refiner.refine(reference, bank.images[r], p[anchor], position)
                if measured is None:
                    return dict(query_xy=None, reason=meta["reason"])
                measured = bank.transform(measured[None], r, True)[0]
                if not ((measured >= 0) & (measured <= 511)).all():
                    return dict(query_xy=None, reason="refinement_outside_image")
                x, y = np.float32(measured)
                probability = cv2.remap(shadow, np.array([[x]], np.float32), np.array([[y]], np.float32), cv2.INTER_LINEAR)
                return dict(query_xy=measured, observed_shadow=float(probability[0, 0]))

            def lookup(suns):
                physical = inp.solar_lookup(suns)
                return dict(solar_states=physical["solar_states"][use], solar_variants=physical["solar_variants"][use])

            result = solve_render_bootstrap(
                graph, query_k, measure, solver_seeds, solar_mode="shared", active=cfg["active"],
                policy=cfg["policy"], budget=cfg["budget"], pose_initialization="quantization_aware",
                final_estimator=lambda a, b: common_pose(a, b, inp), solar_lookup=lookup,
            )
            result.update(
                elapsed_seconds=perf_counter() - tick, inference_revision=REVISION, observation=observation,
                render_bootstrap=bootstrap_trace, coarse_response=response_trace, native_encoder_forwards=refiner.forward_calls,
                coarse_reference_xy=p.copy(), coarse_query_candidates=q.copy(), coarse_rotation_indices=rotation.copy(),
                proposal_bank=dict(angles_degrees=list(QUARTER_ANGLES), candidates_per_anchor=8, native_image_size=512),
            )
            return result
        finally:
            owner.check()
