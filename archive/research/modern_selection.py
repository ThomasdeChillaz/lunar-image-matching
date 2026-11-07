"""Image-only pool ablations with common complete structural-feature support.

All four EfficientLoFTR/MatchAnything variants share one architecture-family vote.
The legacy control uses exactly the same image evidence as the expanded pool;
only admitted candidate identities differ. Truth never enters these functions.
"""
from candidate_selection import Candidate, MODES, choose, transform_distance
from structural_refinement import AlignmentObjective


FAMILIES = dict(ncc='intensity', sift='sift', self_similarity_ncc='self_similarity',
    orientation_ncc='orientation', self_similarity_search='self_similarity',
    loftr_outdoor='loftr', disk_lightglue='disk', research_adapter='disk',
    efficient_loftr_2x_mps='eloftr', efficient_loftr_4x_mps='eloftr',
    matchanything_2x_mps='eloftr', matchanything_4x_mps='eloftr')
LEGACY = tuple(list(FAMILIES)[:8])
MODERN = tuple(list(FAMILIES)[8:])
POOLS = dict(legacy=LEGACY, modern=MODERN, expanded=tuple(FAMILIES))
METHODS = [f'{pool}_{mode}' for pool in POOLS for mode in MODES]


def pool_scores(candidates, qualities, shape):
    scores = {}
    for candidate in candidates:
        quality = qualities.get(candidate.name)
        if quality is None:
            continue
        families = {c.family for c in candidates if c.family != candidate.family and c.matrix is not None}
        agree = {c.family for c in candidates if c.family in families and c.matrix is not None
                 and transform_distance(candidate.matrix, c.matrix, shape) <= 3.}
        scores[candidate.name] = dict(quality, agreement_fraction=len(agree)/max(len(families), 1),
            agreeing_other_families=sorted(agree))
    return scores


def infer_pools(reference, query, reference_mask, query_mask, candidates):
    if len(candidates) != len(FAMILIES) or {c.name for c in candidates} != set(FAMILIES):
        raise ValueError('Every declared predictor must be present, including explicit failed predictions')
    if any(not isinstance(c, Candidate) or c.family != FAMILIES[c.name] for c in candidates):
        raise ValueError('Candidate type or declared architecture family differs')
    objective = AlignmentObjective(reference, query, reference_mask, query_mask)
    qualities = {c.name: objective.quality(c.matrix) for c in candidates}
    by_name = {c.name: c for c in candidates}
    outputs = {}
    for pool, names in POOLS.items():
        admitted = [by_name[name] for name in names]
        scores = pool_scores(admitted, qualities, reference.shape)
        for mode in MODES:
            selected, quality = choose(admitted, scores, mode)
            outputs[f'{pool}_{mode}'] = dict(candidate=selected, quality=quality, candidate_quality=scores)
    return outputs
