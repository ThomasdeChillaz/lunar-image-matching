"""Check a saved run's hashes, grouping and summary against its measurements."""
import argparse,json
from pathlib import Path
from dataset import sha256
from evaluate import summarize


def verify(directory):
    path=Path(directory);meta=json.loads((path/'environment.json').read_text());config=json.loads((path/'frozen_config.json').read_text());rows=json.loads((path/'results.json').read_text())
    if sha256(path/'frozen_config.json')!=meta['config_sha256'] or sha256(path/'frozen_manifest.json')!=meta['manifest_sha256']:
        raise ValueError('Frozen input hash mismatch')
    expected=summarize(rows,config['thresholds_px']);actual=json.loads((path/'summary.json').read_text())
    if expected!=actual:raise ValueError('Summary does not match raw measurements')
    for filename,digest in meta['source_sha256'].items():
        if sha256(Path(__file__).with_name(filename))!=digest:raise ValueError(f'Executed source differs: {filename}')
    pairs=json.loads((path/'frozen_manifest.json').read_text())['pairs']
    regions=len({p['region_id'] for p in pairs});n=len(pairs)*len(config['methods'])*len(config['preprocessing'])
    n+=regions*len(config['controls'])*len(config['methods'])*len(config['preprocessing'])
    if len(rows)!=n:raise ValueError('Missing or duplicate experiment rows')
    unique={(r['pair_id'],r['method'],r['preprocessing']) for r in rows}
    if len(unique)!=len(rows):raise ValueError('Duplicate evaluations')
    print(f'Verified {len(rows)} evaluations, frozen inputs, source hashes and grouped summaries.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');a=p.parse_args();verify(a.directory)
