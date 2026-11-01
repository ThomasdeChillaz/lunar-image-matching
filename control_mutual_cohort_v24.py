"""Actual selected-checkpoint controls and exact72case primary forward replay."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_geometry_warp_v11 import control
from lunargeo_dataset import write_new
from search_window_learning_v18 import SearchWindowMatcher
from train_balanced_localization_v3 import load_bundle
from train_mutual_cohort_v24 import CONFIGURATION, SEEDS, published
from train_search_window_v18 import configure, evaluate, make_model, read_rows


def run(cohort, output, audit_path):
    cohort, root = Path(cohort), Path(output)
    if root.exists():raise FileExistsError('Preserve all selected-checkpoint control attempts')
    configure(); protocol = json.loads((cohort/'protocol.json').read_text()); published(protocol['publication_commit'])
    proof = json.loads(Path(audit_path).read_text()); summary = json.loads((cohort/'summary.json').read_text())
    if (not proof['passed'] or proof['recorded_updates'] != 6000 or proof['completed_runs'] != 3
            or proof['protocol_sha256'] != sha256(cohort/'protocol.json') or len(summary['results']) != 3):
        raise ValueError('Completed independent cohort audit required')
    root.mkdir(parents=True); bundle = load_bundle('reports/localization_training_data_v5.json')
    coarse = CompactMatcher(bundle[0]['coarse_model']); coarse.model.eval().requires_grad_(False); records = []
    from synthetic import texture
    low = texture(994, 128)
    for saved in summary['results']:
        folder = cohort/f'{CONFIGURATION}-{saved["seed"]}'; meta = json.loads((folder/'training.json').read_text())
        complete = json.loads((folder/'COMPLETE.json').read_text())
        if sha256(folder/'selected.pt') != complete['sha256']['selected.pt']:raise ValueError('Selected weights changed')
        model = make_model(coarse, CONFIGURATION); model.load_state_dict(torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)); model.eval()
        matcher = SearchWindowMatcher(coarse, model); actual = control(matcher)
        invalid = matcher.points(low, low, np.zeros_like(low, bool), np.ones_like(low, bool))
        record = dict(seed=saved['seed'], selected_step=saved['selected_step'], checkpoint_sha256=sha256(folder/'selected.pt'),
            controls=actual, invalid_mask_empty=not any(len(v) for v in invalid))
        if saved['seed'] == SEEDS[0]:
            counts, rows = evaluate(model, coarse, bundle, {}, False)
            expected = read_rows(folder/f'selection_step_{meta["selected_step"]}.json.gz')
            if rows != expected or counts != saved['evaluation']:raise ValueError('Actual selected72-case forward differs')
            with gzip.open(root/'primary_forward.json.gz', 'wt') as stream:json.dump(rows, stream, allow_nan=False)
            record.update(actual_primary_forward_cases=72, actual_primary_forward_exact=True, evaluation=counts,
                primary_forward_sha256=sha256(root/'primary_forward.json.gz'))
        records.append(record); print('mutual selected controls', saved['seed'], actual['passed'], flush=True)
    result = dict(passed=all(r['controls']['passed'] and r['invalid_mask_empty'] for r in records), selected_models=3,
        records=records, source_sha256=sha256(__file__), cohort_summary_sha256=sha256(cohort/'summary.json'),
        scope='Actual selected-model controls/primary forward replay on exposed development data; not fresh geographic validation.')
    write_new(root/'summary.json', result)
    if not result['passed']:raise ValueError('Selected model software control failed')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--cohort', required=True); p.add_argument('--output', required=True); p.add_argument('--audit', required=True)
    a = p.parse_args(); run(a.cohort, a.output, a.audit)
