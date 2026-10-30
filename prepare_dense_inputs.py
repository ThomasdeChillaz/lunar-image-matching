"""Package exact public development pixels for blind remote dense inference.

Only development input arrays are exported. The package contains no transform
truth, evaluation points, correctness labels, matcher predictions or reserve
images. Accuracy is scored locally after independent remote inference.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from dataset import sha256
from moon_dataset import load_image
from research_geometry import prepare


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    frozen = json.loads((source/'frozen_inputs.json').read_text())
    complete = json.loads((source/'COMPLETE.json').read_text())
    if frozen['split'] != 'dev' or sha256(source/'results.json') != complete['results_sha256'] or sha256(source/'frozen_inputs.json') != complete['frozen_inputs_sha256']:
        raise ValueError('Only completed, unchanged development sources may be exported')
    rows = {r['pair_id']: r for r in json.loads((source/'results.json').read_text()) if r['method']=='loftr_outdoor'}
    tiles = {t['tile_id']: t for t in frozen['manifest']['tiles'] if t['split']=='dev'}
    images, members, references, records = {}, set(), {}, []
    def image(tile, index):
        key = tile, index
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        def write(name, array):
            if name in members:
                raise ValueError('Duplicate input member')
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            archive.writestr(name, stream.getvalue())
            members.add(name)
        for number, case in enumerate(frozen['cases']):
            a, ma = image(case['tile_id'], case['reference_light_index'])
            base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
            if case['geometry_family']=='nonoverlap_negative':
                b, qm = base, mb
            else:
                b, qm, _, _, _, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
            if hashes != rows[case['pair_id']]['input_array_sha256']:
                raise ValueError('Exported pixels differ from audited common benchmark')
            key = case['tile_id'], case['reference_light_index']
            if key not in references:
                index = len(references)
                references[key] = dict(reference=f'r{index:03d}.npy', reference_mask=f'rm{index:03d}.npy')
                write(references[key]['reference'], a)
                write(references[key]['reference_mask'], ma)
            files = dict(references[key], query=f'q{number:04d}.npy', query_mask=f'qm{number:04d}.npy')
            write(files['query'], b)
            write(files['query_mask'], qm)
            records.append(dict(**case, files=files, input_array_sha256=hashes,
                observation_group=tiles[case['tile_id']]['observation_group'],
                query_observation_group=tiles[case.get('query_tile_id', case['tile_id'])]['observation_group']))
        contract = dict(schema_version=1, split='dev', cases=records, source_completion=complete,
            excluded=['ground_truth_transform', 'evaluation_points', 'correctness', 'all_reserve_images'],
            source_sha256={p: sha256(p) for p in ['prepare_dense_inputs.py','research_geometry.py','moon_dataset.py']},
            attribution='MoonAnything/LunarPhoto, Grethen et al.2026, Zenodo18415119, CC BY4.0. Exact benchmark uint8/mask derivatives of physical lighting renders; artificial geometric warps are separately declared.')
        archive.writestr('contract.json', json.dumps(contract, sort_keys=True, allow_nan=False))
        archive.writestr('ATTRIBUTION.txt', contract['attribution']+'\nAccuracy truth is excluded; these are development inputs only.\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Input package CRC failure')
    output.with_suffix('.sha256').write_text(sha256(output)+'  '+output.name+'\n')
    print(json.dumps(dict(cases=len(records), bytes=output.stat().st_size, sha256=sha256(output), split='dev')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/geometry-dev-v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.source, args.output)
