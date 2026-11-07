"""Unreferenced pose-free candidate-retrieval math prototype, not a backend.

All vectors/conditioning use ONE reference-camera frame. The response provider
predicts same-reference-view relighting in the frozen teacher's 32D basis; it
does not synthesize a query viewpoint. Native query features are not surface
canonicalized, a deliberate unverified representation mismatch. Extraction,
checkpoint loading, cameras, rays, pose fitting and experiment admission are
NOT implemented. This module takes no query pose/depth/true Sun argument.
"""
from dataclasses import dataclass

import numpy as np

from solar_physics_v63 import solar_log_evidence
from solar_sun_search_v68 import bounded_sun_indices


DIM = 32
MAX_ANCHORS = 256
MAX_QUERY_ENTRIES = 10404
MAX_CATALOGUE = 1885
MAX_COARSE_SUNS = 259
MAX_SCORED_SUNS = 292
MAX_DESCRIPTOR_MACS = 24887033856
CANDIDATES_PER_ANCHOR = 8
SUPPRESSION_RADIUS_NATIVE_PX = 8.
PAIR_NORMALIZER = 256.


@dataclass(frozen=True)
class Limits:
    max_descriptor_macs: int = MAX_DESCRIPTOR_MACS
    max_array_bytes: int = 256*1024*1024
    max_scored_suns: int = MAX_SCORED_SUNS


def admit_workload(n_anchors, n_query_entries, n_catalogue, n_coarse,
                   *, input_array_bytes=0, limits=Limits()):
    """Preflight bounds for THIS NumPy math, not complete inference latency/RAM.

    Decoder/model/extraction and BLAS workspace are unknown external costs;
    future scientific execution needs separate measured wall-time admission.
    Eighty bytes per NxM pair conservatively cover retained/current score
    matrices, reductions and temporary array copies. No SxNxM tensor is allowed.
    """
    values = (n_anchors, n_query_entries, n_catalogue, n_coarse, input_array_bytes,
              limits.max_descriptor_macs, limits.max_array_bytes, limits.max_scored_suns)
    if any(type(v) is not int for v in values):
        raise ValueError("Integer workload sizes and finite integer limits required")
    if not (1 <= n_anchors <= MAX_ANCHORS and 8 <= n_query_entries <= MAX_QUERY_ENTRIES
            and 1 <= n_catalogue <= MAX_CATALOGUE and 1 <= n_coarse <= MAX_COARSE_SUNS
            and n_coarse <= n_catalogue and input_array_bytes >= 0):
        raise ValueError("Workload exceeds the 256-anchor/10404-entry/catalogue bounds")
    if min(limits.max_descriptor_macs, limits.max_array_bytes, limits.max_scored_suns) <= 0:
        raise ValueError("Positive finite operation/memory limits required")
    logical_suns = n_coarse+min(15, n_catalogue)+2*min(9, n_catalogue)
    descriptor_macs = n_anchors*n_query_entries*DIM*logical_suns
    peak_bytes = 2*input_array_bytes+80*n_anchors*n_query_entries+64*n_query_entries
    if logical_suns > min(MAX_SCORED_SUNS, limits.max_scored_suns):
        raise ValueError("Declared logical Sun evaluation allowance exceeded")
    if descriptor_macs > min(MAX_DESCRIPTOR_MACS, limits.max_descriptor_macs):
        raise ValueError("Declared descriptor multiply-accumulate allowance exceeded")
    if peak_bytes > limits.max_array_bytes:
        raise ValueError("Declared NxM working-array memory allowance exceeded")
    return dict(logical_sun_bound=logical_suns, descriptor_macs_bound=descriptor_macs,
                compatibility_elements_bound=n_anchors*n_query_entries*logical_suns,
                estimated_peak_array_bytes=peak_bytes, array_budget_bytes=limits.max_array_bytes,
                scope="NumPy retrieval arrays/dot products only; excludes provider, extraction and pose costs")


def _array(value, shape, name):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in 'fiu':
        raise ValueError(name+" must be a real numeric ndarray of the declared shape")
    if not np.isfinite(value).all():
        raise ValueError(name+" must be finite")
    return value


def _unit_rows(value, name):
    a = np.asarray(value, np.float32)
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(a).all() or not np.isfinite(norm).all() or (norm <= 1e-8).any():
        raise ValueError(name+" needs finite nonzero descriptor rows")
    return a/norm


def _unit_directions(value, name):
    a = np.asarray(value, float)
    if not np.allclose(np.linalg.norm(a, axis=-1), 1., rtol=0, atol=1e-6):
        raise ValueError(name+" must contain unit directions in the reference-camera frame")
    return a


def _pair_score(cosine, compatibility, xy):
    """Frozen top-candidate native-location uniqueness; mutual support is global.

    Primary key is unchanged appearance+physical compatibility /256. Mutual
    cosine sum then mutual count are deterministic secondary tie breakers;
    there is no tuned blend and no per-anchor Sun maximization.
    """
    best = compatibility.argmax(1)
    values = compatibility[np.arange(len(best)), best]
    owners = compatibility.argmax(0)  # anchor-index tie break
    keep = []
    seen = set()
    for i in sorted(range(len(best)), key=lambda i: (-values[i], i)):
        location = tuple(xy[best[i]])
        if location not in seen:
            seen.add(location)
            keep.append(i)
    keep = np.asarray(sorted(keep), int)
    mutual = keep[owners[best[keep]] == keep]
    appearance = float(cosine[keep, best[keep]].sum(dtype=np.float64)/PAIR_NORMALIZER)
    total = float(values[keep].sum(dtype=np.float64)/PAIR_NORMALIZER)
    mutual_cosine = float(cosine[mutual, best[mutual]].sum(dtype=np.float64)/PAIR_NORMALIZER)
    return dict(score=total, appearance=appearance, physical=total-appearance,
                unique_anchors=int(len(keep)), mutual_count=int(len(mutual)),
                mutual_cosine=mutual_cosine)


def _key(record):
    return (record['score'], record['mutual_cosine'], record['mutual_count'], -record['sun_index'])


def _suppress(scores, xy, rotation, original_ids):
    """One winning Sun already fixed; exact deterministic cross-rotation NMS."""
    n = len(scores)
    ids = np.full((n, CANDIDATES_PER_ANCHOR), -1, np.int64)
    suppressions = np.zeros(n, np.int64)
    for i in range(n):
        # Bank is preordered by native y/x/rotation/original index. Stable
        # descending-score sort retains that exact tie break.
        order = np.argsort(-scores[i], kind='stable')
        chosen = []
        for j in order:
            if any(float(np.sum((xy[j]-xy[k])**2)) <= SUPPRESSION_RADIUS_NATIVE_PX**2
                   for k in chosen):
                suppressions[i] += 1
                continue
            chosen.append(int(j))
            if len(chosen) == CANDIDATES_PER_ANCHOR:
                break
        if len(chosen) != CANDIDATES_PER_ANCHOR:
            raise ValueError("Fewer than eight spatially distinct candidates for anchor "+str(i))
        ids[i] = chosen
    return dict(query_native_xy=xy[ids], rotation_indices=rotation[ids],
                query_bank_indices=original_ids[ids],
                compatibility=scores[np.arange(n)[:, None], ids],
                suppressed_before_eighth=suppressions)


def retrieve_candidates(reference_response_features, frozen_reference_descriptors,
                        known_reference_sun, reference_normals, reference_views,
                        query_descriptors, query_native_xy, query_rotation_indices,
                        query_valid, sun_vectors, sun_parameters, coarse_sun_mask,
                        *, arm='sh', response_predictor=None, solar_states=None,
                        query_soft_shadow=None, limits=Limits()):
    """Select ONE global Sun, then eight query candidates per reference anchor.

    response_predictor(base[N,32], condition[N,9], target_sun[3], pooled=bool)
    returns N x32 descriptors in the frozen teacher coordinate system. Its
    extraction/head implementation remains external and unqualified. The
    reference condition never pretends to be a query viewing direction.

    Optional physical solar states[N,S] and image soft-shadow measurements[M]
    must be provided together. They use frozen solar_log_evidence exactly.
    Brightness measurements are not ground-truth visibility/shadow labels.
    """
    if arm not in ('native', 'sh', 'pooled'):
        raise ValueError("Explicit native/sh/pooled retrieval arm required")
    if not isinstance(reference_response_features, np.ndarray) or reference_response_features.ndim != 2:
        raise ValueError("Reference learned features must be Nx32 ndarray")
    n = len(reference_response_features)
    if not isinstance(query_descriptors, np.ndarray) or query_descriptors.ndim != 2:
        raise ValueError("Frozen query descriptors must be Mx32 ndarray")
    m = len(query_descriptors)
    if not isinstance(sun_vectors, np.ndarray) or sun_vectors.ndim != 2:
        raise ValueError("Generic Sun catalogue must be Sx3 ndarray")
    s = len(sun_vectors)
    _array(reference_response_features, (n, DIM), 'reference_response_features')
    _array(frozen_reference_descriptors, (n, DIM), 'frozen_reference_descriptors')
    _array(known_reference_sun, (3,), 'known_reference_sun')
    _array(reference_normals, (n, 3), 'reference_normals')
    _array(reference_views, (n, 3), 'reference_views')
    _array(query_native_xy, (m, 2), 'query_native_xy')
    if not isinstance(query_valid, np.ndarray) or query_valid.shape != (m,) or query_valid.dtype != bool:
        raise ValueError("Explicit M-entry query validity mask required")
    if query_descriptors.shape != (m, DIM) or query_descriptors.dtype.kind != 'f':
        raise ValueError("Frozen query descriptors must be floating Mx32")
    if (not isinstance(query_rotation_indices, np.ndarray) or query_rotation_indices.shape != (m,)
            or query_rotation_indices.dtype.kind not in 'iu'
            or not np.isin(query_rotation_indices, [0, 1, 2, 3]).all()):
        raise ValueError("Four declared quarter-turn rotation indices required")
    if ((query_native_xy < 0) | (query_native_xy > 511)).any():
        raise ValueError("Query locations must use native 512-image pixel centres")
    _array(sun_vectors, (s, 3), 'sun_vectors')
    _array(sun_parameters, (s, 2), 'sun_parameters')
    if not isinstance(coarse_sun_mask, np.ndarray) or coarse_sun_mask.shape != (s,) or coarse_sun_mask.dtype != bool:
        raise ValueError("Explicit generic coarse-Sun mask required")
    if (np.abs(sun_parameters[:, 0]) > 90+1e-9).any() or (np.abs(sun_parameters[:, 1]) > 1.5+1e-9).any():
        raise ValueError("Sun parameter catalogue exceeds the declared generic bounds")
    lattice = sun_parameters/np.array([1.25, .25])
    if not np.allclose(lattice, np.rint(lattice), atol=1e-9, rtol=0) or len(np.unique(sun_parameters, axis=0)) != s:
        raise ValueError("Unique declared 1.25/.25-degree generic catalogue required")
    expected_coarse = np.isclose(sun_parameters[:, 0]/5, np.rint(sun_parameters[:, 0]/5), atol=1e-9) & np.isclose(sun_parameters[:, 1]/.5, np.rint(sun_parameters[:, 1]/.5), atol=1e-9)
    if not np.array_equal(coarse_sun_mask, expected_coarse):
        raise ValueError("Coarse mask must retain exactly the declared 5/.5-degree catalogue subset")
    if (solar_states is None) != (query_soft_shadow is None):
        raise ValueError("Physical states and soft-shadow measurements must be supplied together")
    physical = solar_states is not None
    if physical:
        if not isinstance(solar_states, np.ndarray) or solar_states.shape != (n, s) or not np.isin(solar_states, [0, 1, 2]).all():
            raise ValueError("Explicit unknown/clear/shadow map states NxS required")
        _array(query_soft_shadow, (m,), 'query_soft_shadow')
        if ((query_soft_shadow < 0) | (query_soft_shadow > 1)).any():
            raise ValueError("Soft-shadow observations must lie in[0,1]")
    arrays = [reference_response_features, frozen_reference_descriptors, known_reference_sun,
              reference_normals, reference_views, query_descriptors, query_native_xy,
              query_rotation_indices, query_valid, sun_vectors, sun_parameters, coarse_sun_mask]
    if physical:
        arrays += [solar_states, query_soft_shadow]
    bound = admit_workload(n, m, s, int(coarse_sun_mask.sum()),
                           input_array_bytes=sum(a.nbytes for a in arrays), limits=limits)
    if arm != 'native' and not callable(response_predictor):
        raise ValueError("SH/pooled require an explicit teacher-space response provider")
    _unit_directions(known_reference_sun[None], 'known_reference_sun')
    _unit_directions(reference_normals, 'reference_normals')
    _unit_directions(reference_views, 'reference_views')
    _unit_directions(sun_vectors, 'sun_vectors')
    selected = np.flatnonzero(query_valid)
    if len(selected) < 8:
        raise ValueError("At least eight valid native query entries required")
    # Native y/x/rotation/original-index ties, shared by every arm and Sun.
    order = np.lexsort((selected, query_rotation_indices[selected],
                        query_native_xy[selected, 0], query_native_xy[selected, 1]))
    selected = selected[order]
    xy = np.asarray(query_native_xy[selected], float)
    rotation = query_rotation_indices[selected]
    query = _unit_rows(query_descriptors[selected], 'valid query descriptors')
    condition = np.column_stack([np.broadcast_to(known_reference_sun, (n, 3)),
                                 reference_normals, reference_views]).astype(np.float32)
    features = np.asarray(reference_response_features, np.float32).copy()
    features.setflags(write=False)
    condition.setflags(write=False)
    shadow = None if not physical else query_soft_shadow[selected]
    coarse = np.flatnonzero(coarse_sun_mask)
    appearance_cache = None
    response_calls = dot_products = descriptor_macs = 0
    logical_count = 0
    records = {}
    trace = []
    winner = None
    winning_scores = None

    def evaluate(si, phase):
        nonlocal appearance_cache, response_calls, dot_products, descriptor_macs
        nonlocal logical_count, winner, winning_scores
        logical_count += 1
        if logical_count > min(MAX_SCORED_SUNS, limits.max_scored_suns):
            raise ValueError("Logical Sun evaluation allowance exhausted")
        if si in records:
            trace.append(dict(sun_index=si, phase=phase, score_reused=True))
            return records[si]
        if appearance_cache is None or arm == 'sh':
            if arm == 'native':
                response = _unit_rows(frozen_reference_descriptors, 'frozen reference descriptors')
            else:
                response_calls += 1
                raw = response_predictor(features, condition, sun_vectors[si].copy(), pooled=arm == 'pooled')
                _array(raw, (n, DIM), 'predicted teacher-space response')
                response = _unit_rows(raw, 'predicted teacher-space response')
            next_macs = n*len(selected)*DIM
            if descriptor_macs+next_macs > min(MAX_DESCRIPTOR_MACS, limits.max_descriptor_macs):
                raise ValueError("Descriptor multiply-accumulate allowance exhausted")
            cosine = response @ query.T
            descriptor_macs += next_macs
            dot_products += n*len(selected)
            if arm != 'sh':
                appearance_cache = cosine
        else:
            cosine = appearance_cache
        compatibility = cosine.astype(np.float64)
        if physical:
            # Stream physical rows; never allocate an SxNxM or NxM evidence cube.
            for i in range(n):
                compatibility[i] += solar_log_evidence(solar_states[i, si], shadow)
        record = dict(sun_index=si, **_pair_score(cosine, compatibility, xy))
        records[si] = record
        trace.append(dict(sun_index=si, phase=phase, score_reused=False))
        if winner is None or _key(record) > _key(winner):
            winner = record
            winning_scores = compatibility.copy()
        return record

    for si in coarse:
        evaluate(int(si), 'coarse')
    centre = winner['sun_index']
    for round_index in range(3):
        local = bounded_sun_indices(sun_parameters, sun_parameters[centre], round_index)
        local_records = [evaluate(int(si), 'fine_'+str(round_index)) for si in local]
        centre = max(local_records, key=_key)['sun_index']
    # The centre remains a local/global maximum because each window includes it.
    if centre != winner['sun_index']:
        raise ValueError("Unexpected disagreement in the declared global-Sun search")
    candidates = _suppress(winning_scores, xy, rotation, selected)
    alternatives = []
    for si, record in sorted(records.items()):
        angle = float(np.degrees(np.arccos(np.clip(np.dot(sun_vectors[si], sun_vectors[centre]), -1., 1.))))
        if angle >= 10.-1e-9:
            alternatives.append(dict(record, separation_from_winner_degrees=angle))
    alternative = None if not alternatives else max(alternatives, key=_key)
    return dict(prototype_only=True, arm=arm, shared_sun_index=int(centre),
                shared_sun_vector=sun_vectors[centre].copy(),
                shared_sun_parameters=sun_parameters[centre].copy(), global_score=dict(winner),
                candidates=candidates, sun_score_curve=[records[i] for i in sorted(records)],
                strongest_alternative_at_least_ten_degrees=alternative,
                evaluation_trace=trace, scoring_track='appearance_plus_physical' if physical else 'appearance_only',
                diagnostics=dict(logical_sun_evaluations=logical_count, unique_scored_suns=len(records),
                                 response_provider_calls=response_calls, descriptor_dot_products=dot_products,
                                 descriptor_macs=descriptor_macs, workload_bound=bound,
                                 physical_evidence_values=0 if not physical else n*len(selected)*len(records),
                                 query_entries_supplied=m, valid_query_entries=len(selected),
                                 logical_search_ceiling_common_across_arms=True,
                                 constant_appearance_arm=arm != 'sh',
                                 sun_identifiability="Not established by score/support alone"),
                limitations=['External feature extraction/checkpoint provenance unimplemented',
                             'External producer must verify above-horizon catalogue/frame/parameter correspondence',
                             'Native query descriptors are not surface-canonicalized teacher targets',
                             'Reference-view response has no independent target-view conditioning',
                             'Catalogue latitude directions beyond training are unsupervised extension',
                             'Sun/score/mutual support are not calibrated confidence or nonoverlap proof'])


def extract_reference_and_query_features(*args, **kwargs):
    raise NotImplementedError("Image extraction, native/crop equivalence, rotation mapping and checkpoint qualification are not implemented by this math prototype")
