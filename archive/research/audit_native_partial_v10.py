"""Replay timed-out native jobs without calling their64/72 coverage complete."""
import argparse
import json
from pathlib import Path

import numpy as np

from combine_native_baselines_v8 import check_control
from dataset import sha256
from lunargeo_dataset import write_new
from modern_baselines_v2 import SOURCES
from verify_coarse_training_v5 import replay_row


def audit(root, output):
    root = Path(root); protocol = json.loads((root/'protocol.json').read_text())
    timeout = json.loads((root/'TIMEOUT.json').read_text()); control = json.loads((root/'controls.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text()); method = protocol['method']
    family = 'romav2' if method == 'romav2' else 'loma'
    if method not in ('romav2', 'loma_r') or protocol['profile'] != 'common' or protocol['cpu_threads'] != 2:
        raise ValueError('Unplanned comparison')
    if provenance['revision'] != SOURCES[family]['revision'] or provenance['method'] != method or provenance['profile'] != 'common':
        raise ValueError('Official author configuration differs')
    lock = json.loads(Path('reports/compact_baseline_lock_v2.json').read_text())['modern'][method]['weights']
    if any(provenance['weights'].get(p) != h for p, h in lock.items()):
        raise ValueError('Pretrained weight fingerprint differs')
    for p, h in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if sha256(p) != h:
            raise ValueError('Executed partial source/input differs')
    check_control(control)
    manifest_path = Path('reports/localization_training_data_v5.json')
    if sha256(manifest_path) != protocol['manifest_sha256']:
        raise ValueError('Frozen common manifest differs')
    manifest = json.loads(manifest_path.read_text()); cases = [c for c in manifest['cases'] if c['split'] == 'model_selection']
    if protocol['case_ids'] != [c['pair_id'] for c in cases] or timeout['eligible_pairs'] != 72:
        raise ValueError('Partial eligible denominator differs')
    n = timeout['completed_pairs']
    if not 0 < n < 72:
        raise ValueError('This auditor only accepts incomplete original jobs')
    expected = {(c['pair_id'], m) for c in cases[:n] for m in (method, 'compact_orientation')}
    rows = [json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
    if len(rows) != 2*n or {(r['pair_id'], r['method']) for r in rows} != expected:
        raise ValueError('Missing/duplicated timed-out prefix case')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}; data = {}
    for key, t in tiles.items():
        with np.load(t['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
    for row in rows:
        if row['component'] != tiles[row['tile_id']]['component']:
            raise ValueError('Partial terrain identity differs')
        replay_row(row, tiles[row['tile_id']], data[row['tile_id']])
    result = dict(passed=True, scope='Partial-job native metric/control replay; no complete modern comparison or ranking.',
        method=method, completed_pairs=n, eligible_pairs=72, replayed_rows=len(rows),
        public_execution_id=protocol['public_execution_id'], source_sha256=sha256(__file__),
        protocol_sha256=sha256(root/'protocol.json'),
        case_sha256={p.name: sha256(p) for p in sorted((root/'cases').glob('*.json'))},
        complete_comparison=False)
    write_new(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'case_sha256'}, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
