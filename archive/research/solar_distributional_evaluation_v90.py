"""Opt-in coarse response proposal arm over the v84 lazy native solver.

This is an experimental development backend, not a checkpoint-selection,
calibration or job-admission mechanism. Reference Sun is observed metadata.
The primary proposal distribution marginalizes the training view prior and
uniformly moment-matches the declared coarse trial Suns. Shared physical Sun
inference continues in the unchanged solver. Native descriptors are extracted
only by that solver's paid patch callback.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import load_checkpoint
from contrastive_terrain import sample_descriptors
from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import map_points_z
from solar_assignment_v63 import CandidateGraph
from solar_coarse_training_adapter_v86 import (
    coarse_support,
    native_to_coarse,
    reference_normal_field,
)
from solar_distributional_response_v86 import (
    TrainingViewPrior,
    build_distributional_response,
)
from solar_frontend_v63 import reference_anchors
from solar_inference_cache_v84 import SolarMatcherV84, solve_v84
from solar_native_v63 import SparseNativeRefiner, state_digest
from solar_physics_v63 import soft_shadow_observation
from solar_response_cache_v84 import GeometryOwnerGuardV84, PairResponseCacheV84
from solar_rotation_v71 import RotationBank


BASELINE_FRONTEND_SHA256 = "732627a68f9422533e0e633df78526600c6703d75e6674f603b1cfd0c102b769"
PROPOSAL_REVISION = "distributional-coarse128-moment-mixture-v90"
FUSION_WEIGHT = 0.5
TRAINING_TEMPERATURE = 0.10


def marginal_sun_moments(model, base, condition, suns):
    """Match first two moments of the equal-weight trial-Sun Gaussian mixture."""
    if suns.ndim != 2 or suns.shape[1:] != (3,) or len(suns) == 0:
        raise ValueError("Nonempty reference-camera trial Sun catalogue required")
    n = len(base)
    first = torch.zeros((n, 32), dtype=torch.float64)
    second = torch.zeros((n, 32, 32), dtype=torch.float64)
    # Keep the output covariance bank bounded, even for the full catalogue.
    for block in suns.split(8):
        count = len(block)
        distribution = model.marginal_distribution(
            base.expand(count, -1, -1).reshape(-1, 32),
            condition.expand(count, -1, -1).reshape(-1, 10),
            block[:, None].expand(-1, n, -1).reshape(-1, 3),
            diagonal_only=False,
        )
        mean = distribution["mean"].reshape(count, n, 32).double()
        covariance = distribution["covariance"].reshape(count, n, 32, 32).double()
        first += mean.sum(dim=0)
        second += (covariance + mean[..., :, None] * mean[..., None, :]).sum(dim=0)
    mean = first / len(suns)
    covariance = second / len(suns) - mean[..., :, None] * mean[..., None, :]
    return mean, (covariance + covariance.transpose(-1, -2)) * 0.5


def _learned_similarity(mean, covariance, query_features):
    """Within-anchor relative likelihood mapped to [0,1] on the training scale."""
    if not len(query_features):
        raise ValueError("At least one valid native-view coarse feature required")
    rows = []
    features = query_features.double().T
    for mu, sigma in zip(mean.split(8), covariance.split(8)):
        factor = torch.linalg.cholesky(sigma)
        residual = features[None] - mu[..., None]
        whitened = torch.linalg.solve_triangular(factor, residual, upper=False)
        energy = 0.5 * whitened.square().sum(dim=1)
        # A Gaussian's log determinant is constant over candidates in one row.
        relative = (energy - energy.amin(dim=1, keepdim=True)) / (32 * TRAINING_TEMPERATURE)
        rows.append(1.0 - torch.tanh(relative))
    return torch.cat(rows, dim=0).float()


def _learned_conditions(model, reference128, geometry, p, trial_suns):
    geometry_view = {
        "axial_depth": geometry.depth,
        "valid": np.isfinite(geometry.depth) & (geometry.depth > 0),
        "intrinsics": geometry.intrinsics,
        "cam2world": geometry.cam2world,
    }
    normals = reference_normal_field(geometry_view)
    low_xy = native_to_coarse(p)
    support = coarse_support(geometry_view["valid"], low_xy, normals["valid"])
    ids = np.flatnonzero(support)
    if not len(ids):
        return ids, None, None
    rotation = geometry.cam2world[:3, :3]
    los = geometry.cam2world[:3, 3] - geometry.world_variants[ids, 0]
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    condition = np.column_stack(
        (
            np.broadcast_to(geometry.reference_sun @ rotation, los.shape),
            geometry.normals[ids] @ rotation,
            los @ rotation,
            np.zeros(len(ids)),
        )
    ).astype(np.float32)
    field = np.concatenate((reference128[None].astype(np.float32) / 255.0, normals["normal_xy"]))
    base = model.coarse_reference_features(torch.from_numpy(field)[None], torch.from_numpy(low_xy[ids])[None])[0]
    mean, covariance = marginal_sun_moments(
        model, base, torch.from_numpy(condition), torch.from_numpy((trial_suns @ rotation).astype(np.float32))
    )
    return ids, mean, covariance


def distributional_candidates(model, reference, query, geometry, proposal_angles, trial_suns):
    """Return the original eight proposals contract with learned coarse ranks."""
    p = reference_anchors()
    yy, xx = np.mgrid[12:116:4, 12:116:4]
    low = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    native = (low + 0.5) * 4 - 0.5
    bank = RotationBank(query, proposal_angles)
    common_valid = bank.coarse_mask(low).reshape(-1)
    q_native = np.concatenate([bank.transform(native, r, True) for r in range(len(bank.angles))])
    q_native_low = native_to_coarse(q_native)
    query_support = coarse_support(np.ones((512, 512), bool), q_native_low)
    query_ids = np.flatnonzero(common_valid & query_support)
    resized = np.stack(
        [cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA) for image in [reference] + bank.images]
    )
    with torch.inference_mode():
        dense, _ = model.coarse(torch.from_numpy(resized).float()[:, None] / 255.0)
        frozen_reference = sample_descriptors(dense[:1], torch.from_numpy(native_to_coarse(p))[None])[0]
        grid = torch.from_numpy(low)[None].expand(len(bank.angles), -1, -1)
        frozen_rotated_query = sample_descriptors(dense[1:], grid).reshape(-1, 32)
        frozen_score = frozen_reference @ frozen_rotated_query.T
        score = frozen_score.clone()
        anchor_ids, mean, covariance = _learned_conditions(model, resized[0], geometry, p, trial_suns)
        valid_query_count = 0
        if len(anchor_ids) and len(query_ids):
            # index 1 is the native-view query (bank angle 0). Never use
            # rotated teacher vectors in the learned likelihood.
            teacher = sample_descriptors(dense[1:2], torch.from_numpy(q_native_low[query_ids])[None])[0]
            nonzero = teacher.norm(dim=1) > 1e-8
            query_ids = query_ids[nonzero.cpu().numpy()]
            valid_query_count = len(query_ids)
            if valid_query_count:
                learned = _learned_similarity(mean, covariance, teacher[nonzero])
                supported = np.ix_(anchor_ids, query_ids)
                score[supported] = (1 - FUSION_WEIGHT) * frozen_score[supported] + FUSION_WEIGHT * learned
        score[:, ~torch.from_numpy(common_valid)] = -torch.inf
        rank = torch.argsort(score, dim=1, descending=True, stable=True)[:, :8]
        values = score.gather(1, rank).cpu().numpy()
        indices = rank.cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Insufficient fully supported rotation proposals")
    return p, q_native[indices], values, indices // len(low), {
        "coarse_only": True,
        "learned_supported_anchors": int(len(anchor_ids)),
        "learned_supported_query_positions": int(valid_query_count),
        "trial_suns": int(len(trial_suns)),
        "native_view_128_teacher": True,
        "target_view": "training_prior_marginal",
        "sun_pooling": "uniform_trial_catalogue_first_and_second_moments",
        "fusion_weight": FUSION_WEIGHT,
    }


def load_distributional_checkpoint(checkpoint, *, checkpoint_sha256, expected_step,
                                   prior_path, prior_sha256, teacher_directory,
                                   frozen_descriptor_sha256):
    """Load a pinned candidate and preserve exact frozen teacher/prior buffers."""
    def pinned_bytes(path, expected):
        value = Path(path).read_bytes()
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError("Pinned inference input changed: " + str(path))
        return value

    prior = TrainingViewPrior.from_frozen_payload(json.loads(pinned_bytes(prior_path, prior_sha256)))
    frozen, _ = load_checkpoint(teacher_directory)
    if state_digest(frozen) != frozen_descriptor_sha256:
        raise ValueError("Unqualified original frozen descriptor")
    model = build_distributional_response(frozen.eval().requires_grad_(False), prior)
    from io import BytesIO
    snapshot = torch.load(BytesIO(pinned_bytes(checkpoint, checkpoint_sha256)), map_location="cpu", weights_only=True)
    if snapshot.get("step") != expected_step or snapshot.get("prior_provenance") != prior.provenance:
        raise ValueError("Checkpoint step or training-view provenance differs")
    frozen_buffers = {name: value.clone() for name, value in model.named_buffers()}
    model.load_state_dict(snapshot["model"], strict=True)
    if state_digest(model.coarse) != frozen_descriptor_sha256 or any(
        not torch.equal(value, frozen_buffers[name]) for name, value in model.named_buffers()
    ):
        raise ValueError("Checkpoint altered the frozen descriptor or prior")
    return model.eval().requires_grad_(False)


class DistributionalCoarseMatcher(SolarMatcherV84):
    """v84 pose/refinement control with a single opt-in coarse proposal change."""

    def __init__(self, distribution, *, proposal_angles=None, pose_initialization="strict3"):
        super().__init__(
            distribution.coarse,
            distribution.coarse,
            proposal_angles=proposal_angles,
            pose_initialization=pose_initialization,
        )
        self.distribution = distribution.eval().requires_grad_(False)
        self.total_parameters = sum(p.numel() for p in distribution.parameters())
        if self.total_parameters != 95116:
            raise ValueError("Exact declared distributional architecture required")
        self.inference_revision = PROPOSAL_REVISION + "/" + self.inference_revision

    # Protected v84 localize flow, with only proposal construction and metadata changed.
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
            proposal_sun_ids = ids if known_sun is not None else np.flatnonzero(geometry.coarse_sun_mask)
            p, q, appearance, rotation, distributional_cost = distributional_candidates(
                self.distribution, reference, query, geometry, self.proposal_angles,
                geometry.sun_vectors[proposal_sun_ids],
            )
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
                result['coarse_response'] = distributional_cost
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


class CoarseResponseBackend:
    """Development adapter to the existing evaluate_case truth/scoring boundary."""

    def __init__(self, profile):
        self.profile = profile
        config = profile["configuration"]
        expected = {"proposal_angles", "pose_initialization", "solar_mode", "active", "policy", "budget"}
        if set(config) != expected:
            raise ValueError("Declare the exact matched v84 proposal and pose controls")
        keys = ("checkpoint_sha256", "expected_step", "prior_path", "prior_sha256",
                "teacher_directory", "frozen_descriptor_sha256")
        self.provenance = {key: profile[key] for key in keys}
        model = load_distributional_checkpoint(profile["checkpoint"], **self.provenance)
        self.model = DistributionalCoarseMatcher(
            model,
            proposal_angles=config["proposal_angles"],
            pose_initialization=config["pose_initialization"],
        )
        self.provenance.update(proposal_revision=PROPOSAL_REVISION, parameters=self.model.total_parameters)

    def infer(self, inp):
        from solar_evaluation_v70 import common_pose

        config = self.profile["configuration"]
        return self.model.localize(
            inp.reference,
            inp.query,
            inp.geometry,
            inp.query_intrinsics,
            solar_mode=config["solar_mode"],
            response=False,
            active=config["active"],
            budget=config["budget"],
            policy=config["policy"],
            known_sun=inp.known_query_sun,
            reference_sun=inp.reference_sun,
            final_estimator=lambda p, q: common_pose(p, q, inp),
            solar_lookup=inp.solar_lookup,
        )
