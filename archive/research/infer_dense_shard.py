"""Blind original-RoMa inference on exact packaged development arrays."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np

from confidence import diagnostics
from dataset import sha256
from roma_matcher import RoMaMatcher


def load_inputs(archive, case):
    values = {name: np.load(io.BytesIO(archive.read(path)), allow_pickle=False) for name, path in case['files'].items()}
    actual = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in values.items()}
    if actual != case['input_array_sha256'] or set(values) != {'reference','query','reference_mask','query_mask'}:
        raise ValueError('Remote native input bytes differ')
    return values


def run(package, output, index, count):
    package, output = Path(package), Path(output)
    if output.exists() or not 0 <= index < count:
        raise ValueError('Fresh output and valid shard coordinates required')
    expected = package.with_suffix('.sha256').read_text().split()[0]
    if sha256(package) != expected:
        raise ValueError('Input archive differs from published checksum')
    with zipfile.ZipFile(package) as archive:
        contract = json.loads(archive.read('contract.json'))
        if contract['split']!='dev' or any(c['split']!='dev' for c in contract['cases']):
            raise ValueError('Only development inputs permitted')
        cases = [c for i, c in enumerate(contract['cases']) if i % count == index]
        if not cases:
            raise ValueError('Empty inference shard')
        files = ['infer_dense_shard.py','roma_matcher.py','roma_model_lock.json','requirements-roma-linux.txt',
            'matchers.py','loftr_matcher.py','confidence.py','geometry.py','dataset.py']
        frozen = dict(kind='roma_dense_prediction_shard', split='dev', input_archive_sha256=expected,
            source_sha256={f: sha256(f) for f in files}, model_lock=json.loads(Path('roma_model_lock.json').read_text()),
            shard_index=index, shard_count=count, cases=cases, method='roma_outdoor_cpu_fp32',
            execution=dict(github_run_id=os.environ.get('GITHUB_RUN_ID'), github_sha=os.environ.get('GITHUB_SHA'),
                github_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT')),
            runtime={p: importlib.metadata.version(p) for p in ['torch','torchvision','numpy','opencv-python','Pillow','einops','loguru']},
            status='Blind predictions only; no accuracy truth supplied. Separate local scoring required.')
        output.mkdir(parents=True)
        (output/'frozen_inputs.json').write_text(json.dumps(frozen, sort_keys=True, allow_nan=False))
        (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
        model, rows, warmed = RoMaMatcher(), [], False
        for number, case in enumerate(cases, 1):
            data = load_inputs(archive, case)
            a, b, ma, mb = (data[k] for k in ['reference','query','reference_mask','query_mask'])
            if not warmed and ma.any() and a[ma].std() >= 1:
                model(a, a, ma, ma)
                warmed = True
            started = perf_counter()
            forward = model(a, b, ma, mb)
            elapsed = (perf_counter()-started)*1000
            reverse = model(b, a, mb, ma)
            coords = lambda v: None if v is None else np.asarray(v).tolist()
            row = dict(pair_id=case['pair_id'], method='roma_outdoor_cpu_fp32', split='dev',
                estimated=forward.matrix is not None, transform=coords(forward.matrix), reverse_transform=coords(reverse.matrix),
                reference_points=coords(forward.reference_points), query_points=coords(forward.query_points),
                support=forward.support, reason=forward.reason, forward_ms=elapsed,
                input_array_sha256=case['input_array_sha256'], **diagnostics(forward, reverse, a.shape))
            rows.append(row)
            with (output/'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            print(f'{number}/{len(cases)} {case["pair_id"]}', flush=True)
        (output/'predictions.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        (output/'COMPLETE.json').write_text(json.dumps(dict(kind='roma_dense_prediction_shard', rows=len(rows),
            predictions_sha256=sha256(output/'predictions.json'), frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default='reports/dense_dev_inputs_v1.zip')
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard-index', required=True, type=int)
    parser.add_argument('--shard-count', required=True, type=int)
    args = parser.parse_args()
    run(args.package, args.output, args.shard_index, args.shard_count)
