"""Re-run a selected model on every saved development case and compare predictions."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from lunargeo_dataset import write_new
from native_refinement_v7 import NativeRefiner
from train_balanced_localization_v3 import load_bundle
from train_surface_learning_v13 import evaluate


def replay(root, model_name, output):
    root = Path(root); folder = root/model_name
    if folder.resolve().parent != root.resolve():
        raise ValueError('Choose one direct model folder from the completed cohort')
    protocol = json.loads((root/'protocol.json').read_text()); meta = json.loads((folder/'training.json').read_text())
    completed = json.loads((folder/'COMPLETE.json').read_text())
    for source, digest in protocol['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Executed matching/data source differs: '+source)
    if (sha256(folder/'selected.pt') != completed['selected.pt_sha256']
            or sha256(folder/'development_results.json.gz') != completed['development_results.json.gz_sha256']
            or sha256(root/'initial_proposals.json.gz') != protocol['initial_proposals_sha256']):
        raise ValueError('Selected model/proposal/result artifact differs')
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    bundle = load_bundle('reports/localization_training_data_v5.json')
    if sha256('reports/localization_training_data_v5.json') != protocol['manifest_sha256']:
        raise ValueError('Frozen development manifest differs')
    coarse = CompactMatcher(bundle[0]['coarse_model']); model = NativeRefiner(meta['variant'])
    model.load_state_dict(torch.load(folder/'selected.pt', map_location='cpu', weights_only=True), strict=True)
    with gzip.open(root/'initial_proposals.json.gz', 'rt') as stream:
        proposals = {v['pair_id']: v for v in json.load(stream)}
    counts, rows = evaluate(model, bundle, proposals, protocol['pilot'])
    with gzip.open(folder/'development_results.json.gz', 'rt') as stream:
        saved = json.load(stream)
    if rows != saved or counts != meta['evaluation']:
        raise ValueError('Actual model predictions do not exactly reproduce saved rows')
    result = dict(passed=True, model=model_name, cases=len(rows), evaluation=counts,
        checkpoint_sha256=sha256(folder/'selected.pt'), results_sha256=sha256(folder/'development_results.json.gz'),
        source_sha256=sha256(__file__),
        scope='Actual selected-model native forward replay. Consumed development cases, not a fresh geographic test or independent reimplementation.')
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True)
    parser.add_argument('--model', default='normalized_pair-20261101'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); replay(args.run, args.model, args.output)
