"""Exact-cache execution version; original learned model and solver flow retained.

Frozen49/42 source files are imported unchanged. Only pure repeated response
and acquisition projections are memoized. This is a new implementation version,
not a new learned technique or an established speedup/localization improvement.
"""
from dataclasses import replace
from time import perf_counter
import hashlib
import cv2
import numpy as np
from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import map_points_z
from solar_frontend_v63 import SolarMatcher, candidates
from solar_native_v63 import SparseNativeRefiner
from solar_physics_v63 import soft_shadow_observation, solar_log_evidence
from solar_rotation_v71 import RotationBank
from solar_assignment_v63 import CandidateGraph, initial_beams, evidence_tensor, assignments, _matched, _lm_update, estimate, choose_action
from solar_response_cache_v84 import PairResponseCacheV84, GeometryOwnerGuardV84
from solar_acquisition_cache_v84 import AcquisitionProjectionCacheV84, acquisition_predictions_v84
CACHE_REVISIONS = {'strict3': 'solar-v84-exact-cache', 'quantization_aware': 'solar-v84-quantized-init-exact-cache'}
FROZEN_COPY_SOURCE_SHA256 = {'solar_assignment_v63.py': 'ebb84f4e04a1b037f427bdba4266fa776fdb1c32f7ba0518eaa24c5dbc592861', 'solar_frontend_v63.py': 'cdbd20fd07b3d8582e82d27e56b10295e7f2c5671b21927aa74790a72a4407c0'}

def solve_v84(graph, intrinsics, measure, solar_mode='shared', active=False, policy='active', budget=128, response_enabled=False, response_refresh=None, final_estimator=None, solar_lookup=None, pose_initialization='strict3'):
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
    beams = initial_beams(graph, solar_mode, 8, response_enabled)
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
            if pose_initialization == 'quantization_aware':
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

class SolarMatcherV84(SolarMatcher):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_revision = CACHE_REVISIONS[self.pose_initialization]

    def localize(self, reference, query, geometry, query_intrinsics, solar_mode='shared', response=False, active=False, budget=128, policy='active', known_sun=None, reference_sun=None, final_estimator=None, solar_lookup=None):
        tick = perf_counter()
        original_geometry_guard = GeometryOwnerGuardV84(geometry)
        try:
            for image in (reference, query):
                if np.asarray(image).shape != (512, 512) or np.asarray(image).dtype != np.uint8:
                    raise ValueError('Native512uint8grayscale images required')
            query_intrinsics, _ = camera_check(query_intrinsics, np.eye(4))
            if reference_sun is not None:
                geometry = geometry.with_reference_sun(reference_sun)
            if response and self.response is None:
                raise ValueError('Learned response requested without a frozen response checkpoint')
            if known_sun is not None:
                known_sun = np.asarray(known_sun, float)
                if known_sun.shape != (3,) or not np.isfinite(known_sun).all() or (not np.isclose(np.linalg.norm(known_sun), 1, atol=1e-06)):
                    raise ValueError('Finite known-Sun metadata required')
                matches = np.flatnonzero(np.linalg.norm(geometry.sun_vectors - known_sun, axis=1) < 1e-06)
                if len(matches) > 1:
                    raise ValueError('Ambiguous exact known-Sun catalogue direction')
                if len(matches) == 0:
                    if not callable(solar_lookup):
                        raise ValueError('Exact known Sun outside catalogue requires a source-pinned physical lookup; snapping forbidden')
                    table = solar_lookup(known_sun[None])
                    geometry = replace(geometry, sun_vectors=known_sun[None].copy(), solar_states=np.asarray(table['solar_states']), solar_variants=np.asarray(table['solar_variants']), sun_parameters=np.empty((0, 2)), coarse_sun_mask=np.ones(1, bool), solar_materialized_mask=np.ones(1, bool))
                    ids = np.array([0])
                else:
                    ids = matches
            else:
                ids = np.arange(len(geometry.sun_vectors))
            if min(reference.std(), query.std()) < 1:
                return dict(pose=None, reason='blank_or_uninformative_image', reference_xy=np.empty((0, 2)), query_xy=np.empty((0, 2)), elapsed_seconds=perf_counter() - tick, inference_revision=self.inference_revision)
            p, q, appearance, rotation = candidates(self.coarse, reference, query, self.proposal_angles)
            world, use = map_points_z(p, geometry.depth, geometry.intrinsics, geometry.cam2world)
            anchors = np.flatnonzero(use)
            if len(anchors) < 6:
                return dict(pose=None, reason='insufficient_reference_surface', reference_xy=np.empty((0, 2)), query_xy=np.empty((0, 2)), elapsed_seconds=perf_counter() - tick, inference_revision=self.inference_revision)
            p, q, appearance, rotation = (p[use], q[use], appearance[use], rotation[use])
            world = geometry.world_variants[use, 0]
            from solar_assignment_v63 import project
            reprojection, front = project(world, geometry.cam2world, geometry.intrinsics)
            if not front.all() or np.max(np.linalg.norm(reprojection - p, axis=1)) > 0.02:
                raise ValueError('Reference-map anchor geometry violates native pixel centres')
            shadow, observation = soft_shadow_observation(query)
            observed = cv2.remap(shadow.astype(np.float32), q[:, :, 0].astype(np.float32), q[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
            graph = CandidateGraph(p, world, q, appearance, observed, geometry.solar_states[use][:, ids], geometry.sun_vectors[ids], geometry.world_variants[use], geometry.solar_variants[use][:, ids], geometry.sun_parameters[ids] if len(geometry.sun_parameters) else None, np.ones(len(ids), bool) if known_sun is not None else geometry.coarse_sun_mask[ids], geometry.solar_materialized_mask[ids])
            refiner = SparseNativeRefiner(self.fine)
            query_bank = RotationBank(query, self.proposal_angles)
            responder = PairResponseCacheV84(self.response, reference, query, geometry, query_intrinsics, anchors, ids, self.pooled) if response else None
            try:

                def measure(i, c, trial_pose, sun):
                    r = int(rotation[i, c])
                    proposal = query_bank.transform(q[i, c][None], r, False)[0]
                    if not query_bank.native_support(proposal, r):
                        return dict(query_xy=None, reason='incomplete_rotated_native_patch_support')
                    measured, meta = refiner.refine(reference, query_bank.images[r], p[i], proposal)
                    if measured is None:
                        return dict(query_xy=None, reason=meta['reason'])
                    measured = query_bank.transform(measured[None], r, True)[0]
                    if not ((measured >= 0) & (measured <= 511)).all():
                        return dict(query_xy=None, reason='refinement_outside_image')
                    record = dict(query_xy=measured, observed_shadow=float(cv2.remap(shadow.astype(np.float32), np.float32([[measured[0]]]), np.float32([[measured[1]]]), cv2.INTER_LINEAR)[0, 0]))
                    return record
                if final_estimator is None:
                    from native_baseline_pose_v7 import propose_pose

                    def final_estimator(native_p, native_q):
                        pose, info = propose_pose(native_p, native_q, geometry.depth, geometry.intrinsics, geometry.cam2world, query_intrinsics)
                        return (pose, dict(info, iterations_requested=2000, reprojection_threshold_native_px=3.0, solver='ap3p_reclassify'))
                    estimator_label = 'common_reference_axial_depth_AP3P_3px'
                else:
                    estimator_label = 'caller_supplied_common_final_estimator'

                def graph_lookup(suns):
                    table = solar_lookup(suns)
                    return dict(solar_states=np.asarray(table['solar_states'])[use], solar_variants=np.asarray(table['solar_variants'])[use])
                result = solve_v84(graph, query_intrinsics, measure, solar_mode, active, policy, budget, response, response_refresh=responder, final_estimator=final_estimator, solar_lookup=graph_lookup if callable(solar_lookup) else None, pose_initialization=self.pose_initialization)
                result['final_geometry'] = estimator_label
                result.update(coarse_reference_xy=p.copy(), coarse_query_candidates=q.copy(), coarse_rotation_indices=rotation.copy())
                result['proposal_bank'] = dict(angles_degrees=list(self.proposal_angles), anchors=256, candidates_per_anchor=8, native_image_size=512, engineering_control=True, validity='Entire coarse25px low-resolution receptive field and native57px refinement crop must have valid rotated source support; inverse affine returns native pixel centres.')
                if responder is not None:
                    responder.finish()
                result.update(elapsed_seconds=perf_counter() - tick, inference_revision=self.inference_revision, total_parameters=self.total_parameters, native_encoder_forwards=refiner.forward_calls, observation=observation, input_digest=hashlib.sha256(reference.tobytes() + query.tobytes()).hexdigest(), inference_inputs='reference image/depth/pose/normals/solar ray table; query image/intrinsics; optional known-Sun metadata; no query truth', known_sun_supplied=known_sun is not None, response_cost=None if responder is None else responder.cost, solar_lookup_cost=None if solar_lookup is None else dict(getattr(solar_lookup, 'stats', {})), sun_search_limit='Coarse5degree/.5degree initialization, then bounded1.25degree/.25degree catalogue refinement.' if graph.has_fine_sun else 'Legacy discrete catalogue; no fine columns available.', reference_sun_world=geometry.reference_sun.copy(), native_refinement_encoder=self.native_refinement_encoder)
                return result
            finally:
                if responder is not None and (not responder.closed):
                    responder.finish()
        finally:
            original_geometry_guard.check()
