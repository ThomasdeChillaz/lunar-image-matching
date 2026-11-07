"""Controlled addition of our learned local representation to strong predictors.

Initial CNN weights appear only in their explicit control pool. All decisions
use native image evidence; correctness, motion and errors are not arguments.
"""
from candidate_selection import Candidate, choose
from modern_selection import FAMILIES as BASE_FAMILIES, LEGACY, MODERN, pool_scores
from structural_refinement import AlignmentObjective


FAMILIES = dict(BASE_FAMILIES, contrastive_initial='terrain_cnn', contrastive_selected='terrain_cnn')
MODES = ('global',)
POOLS = dict(modern=MODERN, modern_terrain=MODERN+('contrastive_selected',),
    expanded=tuple(BASE_FAMILIES), expanded_terrain=tuple(BASE_FAMILIES)+('contrastive_selected',),
    terrain=('contrastive_selected',), initial=('contrastive_initial',))
METHODS = [f'{pool}_global' for pool in POOLS]


def infer_pools(reference, query, reference_mask, query_mask, candidates):
    if len(candidates) != len(FAMILIES) or {c.name for c in candidates} != set(FAMILIES):
        raise ValueError('Every declared predictor must be present, including explicit failures')
    if any(not isinstance(c, Candidate) or c.family != FAMILIES[c.name] for c in candidates):
        raise ValueError('Candidate type or architecture family differs')
    objective = AlignmentObjective(reference, query, reference_mask, query_mask)
    qualities = {c.name: objective.quality(c.matrix) for c in candidates}
    by_name = {c.name: c for c in candidates}
    outputs = {}
    for pool, names in POOLS.items():
        admitted = [by_name[name] for name in names]
        scores = pool_scores(admitted, qualities, reference.shape)
        selected, quality = choose(admitted, scores, 'global')
        outputs[f'{pool}_global'] = dict(candidate=selected, quality=quality, candidate_quality=scores)
    return outputs
