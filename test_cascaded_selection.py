import numpy as np
from candidate_selection import Candidate
from cascaded_selection import InferenceEvidence, select


def evidence(name, score, reverse=True):
    matrix = np.array([[1., 0, 0], [0, 1, 0]])
    return InferenceEvidence(Candidate(name, name, matrix), matrix if reverse else None,
        dict(global_correlation=score, patch_median_correlation=score,
             geometric_support_fraction=1., joint_nondark_fraction=.5, agreement_fraction=0.))


def policy():
    rule = dict(min_global=.4, min_patch=-1., min_lit=0., min_support=.2, min_agreement=0., max_cycle=4.)
    return dict(primary=rule, fallback=rule)


def test_confidence_cascade_uses_supported_fallback_after_primary_rejection():
    chosen, stage = select(evidence('primary', .1), evidence('fallback', .8), policy())
    assert stage == 'fallback' and chosen.candidate.name == 'fallback'
    chosen, stage = select(evidence('primary', .8), evidence('fallback', .9), policy())
    assert stage == 'primary' and chosen.candidate.name == 'primary'


def test_cascade_rejects_without_independent_reverse_evidence():
    chosen, stage = select(evidence('primary', .8, False), evidence('fallback', .8, False), policy())
    assert chosen is None and stage == 'rejected'
