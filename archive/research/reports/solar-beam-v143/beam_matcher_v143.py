"""Persistent shared-light camera branches through native patch acquisition.

Library only: no lunar loader, experiment admission, query truth or confidence
fitting. Frozen historical matching modules remain byte-identical.
"""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/"solar-blind-v138"))
from catalogue_v138 import rank
from blind_matcher_v138 import active
from policy import cell_schedule as frozen_cell_schedule, choose as frozen_choose, separation as frozen_separation
from beam_policy_v143 import (MAX_BRANCHES, MAX_SEARCHES, choose_diverse_seeds,
    normalized_weights, preview_values, search_schedule, selection_scores, candidate_queue, common_field_budget)

joint = active.joint
native = active.native
POLICIES = ("single_uniform", "beam_uniform", "beam_geometry", "beam_active")


def initialize(catalogue, query, depth, k, reference_pose, query_k, reference_valid,
               *, poses=None, cache=None, lights=4):
    started = perf_counter()
    coarse = rank(catalogue, query, depth, k, reference_pose, query_k,
                  reference_valid=reference_valid, poses=poses, cache=cache)
    world, valid = active.map_points_z(active.ANCHORS, depth, k, np.eye(4))
    seeds = [] if not valid.any() else choose_diverse_seeds(coarse["scores"], catalogue.light_ids,
        coarse["cameras"], world, query_k, reference_pose, lights=lights)
    hypotheses, unavailable = [], []
    for slot in sorted({s["light_slot"] for s in seeds}):
        selected = [s for s in seeds if s["light_slot"] == slot]
        refined = native.control.refine(catalogue.images[slot], query, depth, k, reference_pose, query_k,
                         coarse["cameras"][[s["pose_index"] for s in selected]], reference_valid=reference_valid)
        returned = {h["seed_index"] for h in refined}
        unavailable.extend(dict(seed, reason="frozen_photometric_refinement_unavailable")
                           for i, seed in enumerate(selected) if i not in returned)
        for h in refined:
            hypotheses.append(dict(h, light_slot=slot, light_id=catalogue.light_ids[slot],
                coarse_pose_index=selected[h["seed_index"]]["pose_index"],
                catalogue_sun_world=catalogue.sun_world[slot].tolist()))
    hypotheses.sort(key=lambda h: (h["score"], h["light_id"], h["coarse_pose_index"]))
    for i, h in enumerate(hypotheses):
        h["branch_id"] = i
    return hypotheses, dict(coarse=coarse["diagnostics"], seeds=seeds,
        light_budget=lights, maximum_branches=MAX_BRANCHES, unavailable_before_acquisition=unavailable,
        coarse_scores=coarse["scores"],
        photometric_hypotheses=[dict(h, pose=h["pose"].tolist()) for h in hypotheses],
        initialization_seconds=perf_counter()-started, query_truth_used=False,
        global_light_per_branch=True, pre_acquisition_winner_chosen=False)


def prepare_branches(catalogue, hypotheses, depth, k, reference_pose, query_k, reference_valid, observed, observed_valid):
    branches = []
    side = 2*joint.REACH+1
    for h in hypotheses:
        pool, extra = active.prepare_extra(catalogue.images[h["light_slot"]], depth, k,
            reference_pose, query_k, np.asarray(h["pose"]), reference_valid)
        preview, known = preview_values(extra["image"], extra["available"], pool.template_centers)
        fields = dict(records=[], skipped=[], image=extra["image"], available=extra["available"],
            query=observed, query_valid=observed_valid, scores=np.empty((0, side, side)),
            support=np.empty((0, side, side), bool), blurred=np.empty((0, side, side)), initial_pose=np.asarray(h["pose"]))
        branches.append(dict(branch_id=h["branch_id"], light_slot=h["light_slot"], light_id=h["light_id"],
            sun=h["catalogue_sun_world"], pool=pool, extra=extra, preview=preview, preview_known=known,
            camera=np.asarray(h["pose"]), fields=fields, attempted=0, log_evidence=0., fit_calls=0))
    return branches


def branch_evidence(branch, reference_pose, query_k):
    """Common attempted-measurement denominator; missing support is an outlier.

    Values are tempered NCC evidence, not a probability or acceptance confidence.
    A branch with one lucky supported measurement cannot discard its failed ones.
    """
    attempted, fields = branch["attempted"], branch["fields"]
    if not attempted:
        return 0.
    total = (attempted-len(fields["records"]))*np.log(.05)
    if fields["records"]:
        world = np.array([r["local_world"] for r in fields["records"]])
        centers = np.array([r["center"] for r in fields["records"]])
        xy, _, front = active.project_z(world, query_k, np.linalg.inv(reference_pose) @ branch["camera"])
        delta = xy-centers
        supported = front & np.isfinite(delta).all(axis=1) & (np.abs(delta) <= native.RADIUS).all(axis=1)
        supported &= joint.sample_fields(fields["support"].astype(float), delta+joint.REACH) >= .9999
        correlation = joint.sample_fields(fields["scores"], delta+joint.REACH)
        mass = np.where(supported, np.clip(correlation, 0., 1.), 0.)
        total += np.log(.05+.95*mass).sum()
    return float(total/attempted)


def advance(branches, index, observed, observed_valid, reference_pose, query_k):
    """Acquire one charged field per retained branch, then update every camera."""
    result, events = [], []
    identities = [(b["branch_id"], b["light_id"]) for b in branches]
    for branch in branches:
        observation, reason = (active.acquire_field(index, branch["pool"], branch["extra"], observed, observed_valid)
            if branch["pool"].eligible[index] else (None, "unknown_reference_support_for_branch"))
        fields = active.extend(branch["fields"], observation, branch["camera"])
        camera, fit = joint.optimize_camera(fields, reference_pose, query_k)
        updated = dict(branch, fields=fields, camera=branch["camera"] if camera is None else camera,
                       attempted=branch["attempted"]+1, fit_calls=branch["fit_calls"]+fit["calls"])
        updated["log_evidence"] = branch_evidence(updated, reference_pose, query_k)
        result.append(updated)
        events.append(dict(branch_id=branch["branch_id"], light_id=branch["light_id"],
            observation_present=observation is not None, reason=reason,
            attempted_fields=updated["attempted"], log_evidence=updated["log_evidence"],
            retained_camera=updated["camera"].tolist()))
    if identities != [(b["branch_id"], b["light_id"]) for b in result]:
        raise ValueError("Camera/light branches cannot be replaced or pruned during acquisition")
    return result, events


def run_policy(branches, observed, observed_valid, depth, k, reference_pose, query_k, up, mode, field_budget):
    if mode not in POLICIES:
        raise ValueError("Declared equal-field-budget acquisition control required")
    tick = perf_counter()
    beam = [dict(b) for b in (branches[:1] if mode == "single_uniform" else branches)]
    policy = mode.removeprefix("beam_").removeprefix("single_")
    # Every control receives the same reference-only candidate universe. A
    # single retained camera can attempt an unsupported candidate, which is
    # charged and explicitly fails; it cannot silently shrink the comparison.
    pool = candidate_queue(branches)
    cells, schedule = search_schedule(pool, field_budget, len(beam), frozen_cell_schedule)
    selected, traces = [], []
    weights = normalized_weights([b["log_evidence"] for b in beam])
    for cell in schedule:
        scores, evidence = selection_scores(beam, query_k, reference_pose, up, weights, policy, frozen_separation)
        if mode == "single_uniform":
            # Prefer currently supported anchors inside each common spatial slot.
            scores = beam[0]["pool"].eligible.astype(float)
        index = frozen_choose(pool, cells, cell, selected, scores)
        selected.append(index)
        previous = weights.tolist()
        beam, events = advance(beam, index, observed, observed_valid, reference_pose, query_k)
        weights = normalized_weights([b["log_evidence"] for b in beam])
        traces.append(dict(index=index, cell=cell, score=float(scores[index]), informative_branch_pairs=int(evidence[index]),
            weights_before=previous, weights_after=weights.tolist(), branches=events))
    candidates = []
    for branch, weight in zip(beam, weights, strict=True):
        measured = active.measure(branch["fields"], branch["extra"], branch["camera"], depth, k, reference_pose, query_k)
        candidates.append(dict(measured, branch_id=branch["branch_id"], light_id=branch["light_id"],
            estimated_catalogue_sun_proposal=branch["sun"], branch_weight=float(weight),
            attempted_fields=branch["attempted"], log_evidence=branch["log_evidence"],
            soft_camera_proposal=branch["camera"].tolist(), fit_calls=branch["fit_calls"]))
    candidates.sort(key=lambda c: (c["pose"] is None, -c["estimator"].get("inliers", 0),
        c["estimator"].get("clipped_squared_reprojection_cost", float("inf")), -c["log_evidence"], c["branch_id"]))
    chosen = candidates[0]
    return dict(chosen, conditional_stage_seconds=perf_counter()-tick,
        branch_candidates=candidates, acquisition=dict(policy=mode, requested_native_fields=field_budget,
            attempted_native_fields=sum(b["attempted"] for b in beam), retained_branch_ids=[b["branch_id"] for b in beam],
            retained_light_ids=[b["light_id"] for b in beam], selected_indices=selected, cell_schedule=schedule,
            rounds=len(selected), trace=traces, query_fields_before_selection=0,
            native_fields_per_round=len(beam), pruning=False, branch_probabilities_calibrated=False,
            unused_remainder_fields=field_budget-len(selected)*len(beam)))


def run(catalogue, query, depth, k, reference_pose, query_k, *, reference_valid,
        height_direction_reference, policies=POLICIES, field_budget=128, poses=None,
        cache=None, light_budget=4, research_output=False):
    started = perf_counter()
    if (type(field_budget) is not int or not 0 <= field_budget <= MAX_SEARCHES or not policies
        or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies)
        or type(light_budget) is not int or not 1 <= light_budget <= 4):
        raise ValueError("Unique declared controls and at most128 actual patch searches required")
    catalogue.validate()
    if catalogue.images.shape[1:] != (512, 512) or np.asarray(query).shape != (512, 512):
        raise ValueError("Native512 reference catalogue and query required")
    up = np.asarray(height_direction_reference, float)
    if up.shape != (3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up), 1.):
        raise ValueError("Explicit unit reference-frame height direction required")
    hypotheses, initialization = initialize(catalogue, query, depth, k, reference_pose, query_k,
        reference_valid, poses=poses, cache=cache, lights=light_budget)
    observed, observed_valid = native.prepare(query)
    branches = prepare_branches(catalogue, hypotheses, depth, k, reference_pose, query_k,
        reference_valid, observed, observed_valid)
    preparation_seconds = perf_counter()-started
    effective_fields, capacity = (0, 0) if not branches else common_field_budget(branches, field_budget)
    if not branches:
        null = joint.frame.solve([], [], depth, k, reference_pose, query_k)
        methods = {p: dict(null, reason="no_camera_light_hypotheses", estimated_catalogue_sun_proposal=None,
            acquisition=dict(requested_native_fields=field_budget, attempted_native_fields=0, retained_branch_ids=[],
                retained_light_ids=[], rounds=0, trace=[]), conditional_stage_seconds=0.) for p in policies}
    else:
        methods = {p: run_policy(branches, observed, observed_valid, depth, k, reference_pose, query_k, up, p, effective_fields)
                   for p in policies}
    for row in methods.values():
        row["online_seconds"] = preparation_seconds+row["conditional_stage_seconds"]
    raw = dict(methods=methods, initialization=initialization, shared_preparation_seconds=preparation_seconds,
        measured_whole_call_seconds=perf_counter()-started, parameters_added=0, native_pixels=512,
        requested_field_cap=field_budget, common_effective_native_fields=effective_fields,
        shared_reference_candidate_capacity=capacity,
        query_truth_used=False, known_query_sun_input=False, calibrated_acceptance=False,
        continuous_sun_estimation=False, solar_visibility_certified=False,
        scope="Persistent discrete shared-light camera branches; all native correlation searches counted")
    return active.public_result(raw, research_output=research_output)
