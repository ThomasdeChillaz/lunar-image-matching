"""Partition all original CNSFM pairs across Windows jobs, preserving failures."""
import argparse
import itertools
import json
from pathlib import Path
from run_cnsfm import run


def partition(products, shard_index, shard_count):
    if not 0 <= shard_index < shard_count or shard_count <= 0:
        raise ValueError('Invalid shard')
    return [(index, pair) for index, pair in enumerate(itertools.combinations(products, 2)) if index % shard_count == shard_index]


def execute(source, output, region, shard_index, shard_count, bidirectional=False, data_source=None):
    source, output = Path(source), Path(output)
    folder = (Path(data_source) if data_source else source/'Dataset')/region
    products = [Path(line.strip().replace('\\', '/')).stem for line in (folder/'list_images.txt').read_text().splitlines() if line.strip()]
    pairs = partition(products, shard_index, shard_count)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = dict(region=region, shard_index=shard_index, shard_count=shard_count, bidirectional=bidirectional,
        planned_pairs=[dict(index=index, reference=pair[0], query=pair[1]) for index, pair in pairs])
    (output/'shard_plan.json').write_text(json.dumps(plan, indent=2))
    outcomes = []
    for index, pair in pairs:
        for direction, ordered in [('forward', pair), *([('reverse', pair[::-1])] if bidirectional else [])]:
            destination = output/f'pair_{index:03}_{direction}'
            try:
                result = run(source, destination, region, products=ordered, data_source=data_source)
                outcome = dict(index=index, reference=pair[0], query=pair[1], direction=direction, status=result['status'])
            except Exception as exc:
                outcome = dict(index=index, reference=pair[0], query=pair[1], direction=direction,
                    status='execution_error', error=f'{type(exc).__name__}: {exc}')
            outcomes.append(outcome)
            (output/'outcomes.json').write_text(json.dumps(outcomes, indent=2))
            print(json.dumps(outcome), flush=True)
    if any(o['status'] != 'executed' for o in outcomes):
        raise RuntimeError('Some original executions failed; inspect outcomes and retry affected pairs')
    (output/'COMPLETE.json').write_text(json.dumps(dict(executions=len(outcomes), pairs=len(pairs))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--region', choices=['S1', 'S2', 'S3'], required=True)
    p.add_argument('--shard-index', type=int, required=True)
    p.add_argument('--shard-count', type=int, required=True)
    p.add_argument('--bidirectional', action='store_true')
    p.add_argument('--dataset-root')
    args = p.parse_args()
    execute(args.source, args.output, args.region, args.shard_index, args.shard_count, args.bidirectional, args.dataset_root)
