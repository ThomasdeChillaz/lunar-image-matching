"""Small score-matrix counterexample, not a neural or lunar experiment."""
import argparse
import json

import torch

from dataset import sha256
from lunargeo_dataset import write_new
from mutual_assignment_v23 import window_correction
from search_window_learning_v18 import direction, window_mask


def mutual_pairs(scores):
    index = scores.argmax(1); reverse = scores.argmax(0)
    return [[i, int(j)] for i, j in enumerate(index) if int(reverse[j]) == i and scores[i, j] >= .5]


def diagnose(output):
    torch.set_num_threads(1)
    initial = torch.tensor([[.8, .7, .2], [.7, .8, .2], [.2, .2, .1]], dtype=torch.float64)
    competing = initial.clone(); competing[0, 1] = .95
    grid = torch.tensor([[12., 12.], [16., 12.], [24., 12.]], dtype=torch.float64)
    # The first two forward/reverse targets correspond to a half-original-pixel
    # translation. A third, unmatched candidate supplies a distractor. This
    # defines a loss-algebra fixture, not a full rendered/occlusion geometry.
    forward = grid.clone(); forward[:2, 0] += .125
    reverse = grid.clone(); reverse[:2, 0] -= .125
    visible = torch.tensor([True, True, False]); records = []
    for label, scores in [('initial', initial), ('competing_reachable_candidate', competing)]:
        old = scores.sum()*0; correction = scores.sum()*0; anchors = []
        for matrix, target, baseline in [(scores, forward, initial), (scores.T, reverse, initial.T)]:
            loss, counts = direction(matrix, target, visible, grid, baseline, .5)
            old += .5*loss; anchors.append(counts['anchors'])
            correction += .5*window_correction(matrix, window_mask(target, visible, grid))
        records.append(dict(case=label, scores=scores.tolist(), full_v18_toy_objective=float(old),
            coupled_toy_objective=float(old+correction), mutual_pairs=mutual_pairs(scores),
            protected_anchors_by_direction=anchors))
    before, after = records
    passed = (len(before['mutual_pairs']) == 2 and len(after['mutual_pairs']) == 1
        and after['full_v18_toy_objective'] < before['full_v18_toy_objective']
        and after['coupled_toy_objective'] > before['coupled_toy_objective']
        and before['protected_anchors_by_direction'] == after['protected_anchors_by_direction'] == [2, 2])
    if not passed:raise ValueError('Declared full-objective counterexample did not hold')
    result = dict(passed=True, scope='Small loss-algebra fixture only, not a lunar/model causal diagnosis or localization result.',
        records=records, grid_coarse_xy=grid.tolist(), forward_target_coarse_xy=forward.tolist(),
        reverse_target_coarse_xy=reverse.tolist(), visible=visible.tolist(),
        source_sha256={p: sha256(p) for p in ['diagnose_mutual_assignment_v23.py', 'mutual_assignment_v23.py', 'search_window_learning_v18.py']},
        model_inferences=0, training_updates=0,
        interpretation='The executed v18 window loss and genuine-proposal anchor can prefer a reachable competitor that reduces mutual match cardinality. Coupled normalization penalizes this fixture. Whether this occurs in current lunar fitting and whether pose improves remain untested.')
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    a = p.parse_args(); diagnose(a.output)
