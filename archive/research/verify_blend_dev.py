"""Replay every image-only blend decision and independently audit its errors."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from candidate_selection import Candidate, MODES, choose, score_candidates
from dataset import sha256
from moon_dataset import load_image
from research_geometry import prepare
from verify_research_dev import verify as verify_geometry


def verify(run):
    run = Path(run)
    result = verify_geometry(run)
    frozen = json.loads((run/'frozen_inputs.json').read_text())
    complete = json.loads((run/'COMPLETE.json').read_text())
    if sha256(run/'upstream_completion.json') != complete['upstream_completion_sha256']:
        raise ValueError('Upstream terminal fingerprints changed')
    terminals = json.loads((run/'upstream_completion.json').read_text())
    sources, predictions = {}, {}
    for method, source in frozen['predictors'].items():
        root = Path(source['source_run'])
        terminal = terminals[str(root)]
        if sha256(root/'frozen_inputs.json') != source['frozen_inputs_sha256'] or terminal['frozen_inputs_sha256'] != source['frozen_inputs_sha256']:
            raise ValueError('Upstream design identity differs')
        if sha256(root/'results.json') != terminal['results_sha256'] or json.loads((root/'COMPLETE.json').read_text()) != terminal:
            raise ValueError('Upstream prediction identity differs')
        if str(root) not in sources:
            values = json.loads((root/'results.json').read_text())
            sources[str(root)] = {(r['pair_id'], r['method']): r for r in values}
            if len(sources[str(root)]) != len(values):
                raise ValueError('Duplicate upstream prediction')
        predictions[method] = sources[str(root)]
    rows = {(r['pair_id'], r['method']): r for r in json.loads((run/'results.json').read_text())}
    tiles = {t['tile_id']: t for t in frozen['manifest']['tiles'] if t['split'] == 'dev'}
    images = {}
    def image(tile, index):
        key = (tile, index)
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    decisions = 0
    for case in frozen['cases']:
        a, ma = image(case['tile_id'], case['reference_light_index'])
        base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
        if case['geometry_family'] == 'nonoverlap_negative':
            b, qm = base, mb
        else:
            b, qm, _, _, _, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
        candidates, reverses = [], []
        for method, source in frozen['predictors'].items():
            prediction = predictions[method][(case['pair_id'], method)]
            # Only these three fields are admitted to selection. Correctness,
            # known motion and even the geometry-family label remain excluded.
            candidates.append(Candidate(method, source['family'], prediction['transform']))
            reverses.append(Candidate(method, source['family'], prediction['reverse_transform']))
        scores = score_candidates(a, b, ma, qm, candidates)
        reverse_scores = score_candidates(b, a, qm, ma, reverses)
        for mode in MODES:
            row = rows[(case['pair_id'], f'blend_{mode}')]
            selected, quality = choose(candidates, scores, mode)
            back, reverse_quality = choose(reverses, reverse_scores, mode)
            if selected is None:
                if row['estimated'] or row['transform'] is not None:
                    raise ValueError('Invented supported candidate')
            elif row['selection_quality']['selected_predictor'] != selected.name or not np.allclose(row['transform'], selected.matrix, atol=1e-10):
                raise ValueError('Saved transform was not selected from image evidence')
            if back is None:
                if row['reverse_transform'] is not None:
                    raise ValueError('Invented reverse candidate')
            elif row['reverse_selection_quality']['selected_predictor'] != back.name or not np.allclose(row['reverse_transform'], back.matrix, atol=1e-10):
                raise ValueError('Reverse choice differs from independent image-order replay')
            for actual, saved in [(quality, row['selection_quality']), (reverse_quality, row['reverse_selection_quality'])]:
                if set(actual) != set(saved):
                    raise ValueError('Image-quality fields differ')
                for name, value in actual.items():
                    if isinstance(value, (int, float)):
                        if not np.isclose(value, saved[name], atol=1e-8):
                            raise ValueError('Image-only diagnostic differs')
                    elif value != saved[name]:
                        raise ValueError('Selected candidate provenance differs')
            decisions += 1
    result.update(image_only_choices_replayed=decisions, all_planned_predictors_bound=True,
        separate_reverse_selection_replayed=True, truth_fields_excluded_from_selector=True,
        upstream_completion_sha256=complete['upstream_completion_sha256'])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.wait:
        print('Waiting for terminal blend results before exhaustive selector replay.', flush=True)
        while not (Path(args.run)/'COMPLETE.json').exists():
            time.sleep(2)
    result = verify(args.run)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
