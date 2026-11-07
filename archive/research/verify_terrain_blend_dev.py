"""Replay all controlled prediction-pool decisions and audit saved geometry."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from candidate_selection import Candidate
from dataset import sha256
from terrain_selection import FAMILIES, METHODS, infer_pools
from moon_dataset import load_image
from research_geometry import prepare
from verify_research_dev import verify as verify_geometry


def verify(run):
    run = Path(run)
    audit = verify_geometry(run)
    frozen = json.loads((run/'frozen_inputs.json').read_text())
    complete = json.loads((run/'COMPLETE.json').read_text())
    if frozen['methods'] != METHODS or sha256(run/'upstream_completion.json') != complete['upstream_completion_sha256']:
        raise ValueError('Pool design or terminal bindings differ')
    terminals = json.loads((run/'upstream_completion.json').read_text())
    predictions, cache = {}, {}
    for method, binding in frozen['predictors'].items():
        root = Path(binding['source_run'])
        terminal = terminals[str(root)]
        if json.loads((root/'COMPLETE.json').read_text()) != terminal or sha256(root/'frozen_inputs.json') != binding['frozen_inputs_sha256']:
            raise ValueError('Upstream terminal/design changed')
        if sha256(root/'results.json') != terminal['results_sha256']:
            raise ValueError('Upstream prediction bytes changed')
        if str(root) not in cache:
            values = json.loads((root/'results.json').read_text())
            cache[str(root)] = {(r['pair_id'], r['method']): r for r in values}
            if len(cache[str(root)]) != len(values):
                raise ValueError('Duplicate predictor outcome')
        predictions[method] = cache[str(root)]
    rows = {(r['pair_id'], r['method']): r for r in json.loads((run/'results.json').read_text())}
    tiles = {t['tile_id']: t for t in frozen['manifest']['tiles'] if t['split'] == 'dev'}
    images = {}
    def image(tile, index):
        key = tile, index
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    count = 0
    for index, case in enumerate(frozen['cases'], 1):
        a, ma = image(case['tile_id'], case['reference_light_index'])
        base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
        if case['geometry_family'] == 'nonoverlap_negative':
            b, qm = base, mb
        else:
            b, qm, _, _, _, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
        candidates, reverses = [], []
        for method in FAMILIES:
            row = predictions[method][(case['pair_id'], method)]
            candidates.append(Candidate(method, FAMILIES[method], row['transform']))
            reverses.append(Candidate(method, FAMILIES[method], row['reverse_transform']))
        forward = infer_pools(a, b, ma, qm, candidates)
        reverse = infer_pools(b, a, qm, ma, reverses)
        for method in METHODS:
            saved = rows[(case['pair_id'], method)]
            for value, transform_key, quality_key, candidate_key in [
                (forward[method], 'transform', 'selection_quality', 'candidate_quality'),
                (reverse[method], 'reverse_transform', 'reverse_selection_quality', 'reverse_candidate_quality')]:
                selected = value['candidate']
                if selected is None:
                    if saved[transform_key] is not None:
                        raise ValueError('Invented supported transform')
                elif saved[transform_key] is None or not np.allclose(saved[transform_key], selected.matrix, atol=1e-10, rtol=0):
                    raise ValueError('Chosen transform differs from image-only pool selection')
                if value['quality'] != saved[quality_key] or value['candidate_quality'] != saved[candidate_key]:
                    raise ValueError('Replayed image evidence or pool-specific votes differ')
            count += 1
        if index % 100 == 0:
            print(f'Pool replay {index}/{len(frozen["cases"])}', flush=True)
    audit.update(image_only_choices_replayed=count, separate_reverse_choices_replayed=True,
        common_feature_support_radius=7, untrained_weights_excluded_from_trained_pools=True,
        modern_variants_share_one_architecture_family=True, all_fourteen_predictors_bound=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.wait:
        print('Waiting for terminal controlled-pool experiment; no reserve evaluation scheduled.', flush=True)
        while not (root/'COMPLETE.json').exists():
            time.sleep(2)
    result = verify(root)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
