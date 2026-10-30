"""Bind every blind RoMa shard and independently score the local known geometry."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from evaluate import endpoint_statistics
from moon_dataset import load_image
from research_geometry import prepare
from verify_roma_probe import verify as verify_pretrained_control


def run(executions, probe, output, source='runs/geometry-dev-v1', package='reports/dense_dev_inputs_v1.zip'):
    executions, probe, output, source, package = map(Path, [executions, probe, output, source, package])
    if output.exists():
        raise FileExistsError(output)
    verified_probe=verify_pretrained_control(probe)
    design = json.loads((source/'frozen_inputs.json').read_text())
    complete = json.loads((source/'COMPLETE.json').read_text())
    if design['split']!='dev' or sha256(source/'results.json')!=complete['results_sha256'] or sha256(source/'frozen_inputs.json')!=complete['frozen_inputs_sha256']:
        raise ValueError('Original development measurements changed')
    package_hash = package.with_suffix('.sha256').read_text().split()[0]
    if sha256(package)!=package_hash:
        raise ValueError('Input package changed')
    with zipfile.ZipFile(package) as archive:
        packet = json.loads(archive.read('contract.json'))
    if packet['source_completion']!=complete:
        raise ValueError('Package source binding differs')
    proof_complete = json.loads((probe/'COMPLETE.json').read_text())
    if sha256(probe/'result.json')!=proof_complete['result_sha256'] or sha256(probe/'frozen_inputs.json')!=proof_complete['frozen_inputs_sha256']:
        raise ValueError('Dense software proof changed')
    proof, proof_design = json.loads((probe/'result.json').read_text()), json.loads((probe/'frozen_inputs.json').read_text())
    if not proof['passed'] or proof['error_px'] is None or proof['error_px']>3:
        raise ValueError('A passing actual pretrained control is required')
    shards, predictions, bound, reference = {}, {}, {}, None
    for marker in sorted(executions.rglob('COMPLETE.json')):
        root = marker.parent
        terminal = json.loads(marker.read_text())
        frozen = json.loads((root/'frozen_inputs.json').read_text())
        if terminal.get('kind')!='roma_dense_prediction_shard' or frozen.get('kind')!=terminal['kind']:
            raise ValueError('Unexpected artifact in inference-shard directory')
        if sha256(root/'predictions.json')!=terminal['predictions_sha256'] or sha256(root/'frozen_inputs.json')!=terminal['frozen_inputs_sha256']:
            raise ValueError('Remote prediction artifact changed')
        if frozen['split']!='dev' or frozen['input_archive_sha256']!=package_hash:
            raise ValueError('Remote inference uses different input data')
        if reference is None:
            reference=frozen
        for field in ['source_sha256','runtime','model_lock','execution','shard_count']:
            if frozen[field]!=reference[field]:
                raise ValueError('Inference shards use different sources/runtime/execution')
        for name,digest in proof_design['source_sha256'].items():
            if name in frozen['source_sha256'] and frozen['source_sha256'][name]!=digest:
                raise ValueError('Pretrained proof and inference implementation differ')
        if frozen['execution']['github_run_id']!=proof_design['execution']['github_run_id']:
            raise ValueError('Proof and predictions came from different workflow runs')
        index,count=frozen['shard_index'],frozen['shard_count']
        if index in shards or not 0<=index<count:
            raise ValueError('Duplicate/invalid inference shard')
        expected=[c for i,c in enumerate(packet['cases']) if i%count==index]
        if frozen['cases']!=expected:
            raise ValueError('Changed remote case plan')
        values=json.loads((root/'predictions.json').read_text())
        if len(values)!=len(expected) or terminal['rows']!=len(values) or {r['pair_id'] for r in values}!={c['pair_id'] for c in expected}:
            raise ValueError('Incomplete or unplanned shard coverage')
        for row in values:
            if row['pair_id'] in predictions or 'correct' in row or 'error_px' in row:
                raise ValueError('Duplicate prediction or accuracy truth inside blind inference')
            predictions[row['pair_id']]=row
        shards[index]=root
        bound[str(root)]=terminal
    if reference is None or set(shards)!=set(range(reference['shard_count'])) or len(predictions)!=len(design['cases']):
        raise ValueError('Every planned inference shard is required before scoring')
    for name,digest in reference['source_sha256'].items():
        if sha256(name)!=digest:
            raise ValueError(f'Current source differs from frozen remote implementation: {name}')
    cases={c['pair_id']:c for c in packet['cases']}
    original={r['pair_id']:r for r in json.loads((source/'results.json').read_text()) if r['method']=='loftr_outdoor'}
    tiles={t['tile_id']:t for t in design['manifest']['tiles'] if t['split']=='dev'}
    images={}
    def image(tile,index):
        key=tile,index
        if key not in images:images[key]=load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    rows=[]
    for case in design['cases']:
        prediction=predictions[case['pair_id']]
        if prediction['method']!='roma_outdoor_cpu_fp32' or prediction['input_array_sha256']!=cases[case['pair_id']]['input_array_sha256'] or prediction['input_array_sha256']!=original[case['pair_id']]['input_array_sha256']:
            raise ValueError('Remote prediction identity or native bytes differ')
        negative=case['geometry_family']=='nonoverlap_negative'
        if negative:
            error,rmse=None,None
        else:
            a,ma=image(case['tile_id'],case['reference_light_index'])
            base,mb=image(case['tile_id'],case['query_light_index'])
            _,_,_,points,targets,_=prepare(a,base,ma,mb,case['pair_id'],case['geometry_family'])
            error,rmse=endpoint_statistics(prediction['transform'],points,targets)
        rows.append(dict(prediction,**case, observation_group=cases[case['pair_id']]['observation_group'],
            query_observation_group=cases[case['pair_id']]['query_observation_group'],
            dataset_kind='physical_render_research_development',label_source='known_nonoverlap' if negative else 'exact_geometry',
            correct=bool(not negative and error is not None and error<=3),error_px=error,rmse_px=rmse,applicability='within_model'))
    files=set(reference['source_sha256'])|{'score_dense_shards.py','verify_roma_probe.py','research_geometry.py','moon_dataset.py','evaluate.py','research_dataset.py'}
    frozen=dict(design,methods=['roma_outdoor_cpu_fp32'],source_sha256={f:sha256(f) for f in files},
        dense_input_archive_sha256=package_hash,dense_model_lock=reference['model_lock'],dense_runtime=reference['runtime'],
        remote_execution=reference['execution'],upstream_prediction_artifacts=bound,
        pretrained_control_sha256=proof_complete,pretrained_control_audit=verified_probe,
        status='Complete blind remote predictions, scored locally on exact development geometry; no reserve access or SOTA claim.')
    output.mkdir(parents=True)
    (output/'frozen_inputs.json').write_text(json.dumps(frozen,sort_keys=True,allow_nan=False))
    summary=[]
    for family in sorted({c['geometry_family'] for c in rows}):
        subset=[r for r in rows if r['geometry_family']==family]
        summary.append(dict(method='roma_outdoor_cpu_fp32',geometry_family=family,cases=len(subset),returned=sum(r['estimated'] for r in subset),correct=sum(r['correct'] for r in subset)))
    (output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (output/'results.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),results_sha256=sha256(output/'results.json'),frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executions',required=True)
    parser.add_argument('--probe',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.executions,args.probe,args.output)
