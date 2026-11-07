"""Matched-control audit plus the unchanged v54 twenty-snapshot replay."""
import argparse
import json
from pathlib import Path

from north_optimization_rate_v55 import BASELINE, baseline_evidence, check_group_settings
from north_optimization_probe_v54 import read, sha


def only_rate_changed(original, candidate, rate):
    if len(original) != 2 or len(candidate) != 2:
        raise ValueError('Two unchanged optimizer groups required')
    expected = [dict(g) for g in original]
    if expected[0]['lr'] != 1e-4 or expected[1]['lr'] != 1e-5:
        raise ValueError('Original learning rates differ')
    expected[1]['lr'] = rate
    if expected != candidate:
        raise ValueError('An optimizer setting other than coarse rate changed')
    check_group_settings(candidate, rate)


def audit(run, output, numeric_output):
    root, out = Path(run), Path(output)
    if out.exists() or Path(numeric_output).exists():
        raise FileExistsError('Preserve every replay attempt')
    baseline = baseline_evidence(); protocol = read(root / 'protocol.json')
    if protocol['baseline_v54_evidence'] != baseline:
        raise ValueError('Matched baseline provenance differs')
    rate = protocol['coarse_learning_rate']
    if protocol['max_calendar_seconds'] != 900 or protocol['context_learning_rate'] != 1e-4:
        raise ValueError('Declared time and context settings changed')
    import numpy as np
    import torch
    fields = ('loss', 'labels', 'global_unclipped_gradient_norm',
              'parameter_group_unclipped_gradient_norm')
    for spec in protocol['fixtures']:
        folder, original = root / spec['name'], BASELINE / spec['name']
        with np.load(folder / 'fixture.npz', allow_pickle=False) as a, np.load(original / 'fixture.npz', allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a.files):
                raise ValueError('Matched fixture inputs differ')
        load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
        a, b = load(folder / 'initial.pt'), load(original / 'initial.pt')
        if a.keys() != b.keys() or any(not torch.equal(v, b[k]) for k, v in a.items()):
            raise ValueError('Matched initialization differs')
        latest, control = load(folder / 'latest.pt'), load(original / 'latest.pt')
        only_rate_changed(control['optimizer']['param_groups'], latest['optimizer']['param_groups'], rate)
        if any(latest['history'][0][k] != control['history'][0][k] for k in fields):
            raise ValueError('First objective/gradient differs before rate intervention')
        first, old = read(folder / 'evaluation_0.json'), read(original / 'evaluation_0.json')
        if any(first[k] != old[k] for k in ('objective', 'labels', 'metrics')):
            raise ValueError('Step-zero scores/decoding differ')
    from verify_north_optimization_v54 import audit as snapshot_replay
    numeric = snapshot_replay(root, numeric_output)
    if not numeric['passed'] or numeric['neural_snapshot_forwards_replayed'] != 20:
        raise ValueError('Complete actual forward/objective replay required')
    result = dict(passed=True, coarse_learning_rate=rate, only_coarse_rate_changed=True,
        matched_fixtures=4, matched_initializations=4, matched_first_update_gradients=4,
        neural_snapshot_forwards_replayed=20, logical_recorded_updates=400,
        numeric_replay_sha256=sha(numeric_output), summary_sha256=sha(root / 'summary.json'),
        baseline_evidence=baseline, source_sha256=sha(__file__), model_promoted=False,
        full_optimizer_retraining=False, calibration_or_final_pixels_loaded=0,
        scope='Fixed TRAINING-example learning-rate control and twenty actual snapshot replays; no held-out localization or candidate promotion.')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    p.add_argument('--numeric-output', required=True)
    a=p.parse_args(); audit(a.run,a.output,a.numeric_output)
