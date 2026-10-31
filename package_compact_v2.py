"""Package completed study outcomes without changing models or operating points."""
import json
from pathlib import Path
import zipfile

from dataset import sha256
from lunargeo_dataset import write_new


def package():
    protocol_path='reports/compact_freeze_v2.json';protocol=json.loads(Path(protocol_path).read_text())
    selected=json.loads(Path('reports/compact_selection_v2.json').read_text());variant=selected['selected_variant']
    two=[];confidence=[];full={};common={};extra=[]
    for v in ('expanded','reliability','hard_negatives','multiscale'):
        two.append(f'runs/compact-v2-{v}-selection-canonical')
        if v!='expanded':
            cal=f'runs/compact-v2-{v}-calibration-canonical';two.append(cal)
            confidence.append(dict(selection=f'runs/compact-v2-{v}-selection-canonical',calibration=cal,
                policy=f'reports/compact_ablation_policy_v2_{v}.json'))
    for seed in selected['all_seeds']:
        model=f'runs/compact-v2-{variant}-{seed}'
        sel=f'runs/compact-v2-{variant}-selection-canonical' if seed==20261030 else model+'-selection-canonical'
        cal=model+'-calibration-canonical';test=model+'-test';two.extend([sel,cal,test])
        confidence.append(dict(selection=sel,calibration=cal,policy=protocol['policy_index'][model]))
        full[f'compact-{seed}']=dict(run=test,policy_key=model)
    primary=f'runs/compact-v2-{variant}-20261030'
    common['compact']=dict(run=primary+'-test',policy_key=primary)
    for method in ('original','sift'):
        sel=f'runs/compact-v2-{method}-evidence-model_selection-canonical'
        cal=f'runs/compact-v2-{method}-evidence-calibration-canonical';test=f'runs/compact-v2-{method}-test'
        two.extend([sel,cal,test]);confidence.append(dict(selection=sel,calibration=cal,policy=protocol['policy_index'][method]))
        full[method]=dict(run=test,policy_key=method);common[method]=dict(run=test,policy_key=method)
    full['original-frozen']=dict(run='runs/compact-v2-original-test',policy_key='original',original_gate=True)
    for method in ('loma_b','loma_r','romav2','roma','matchanything'):
        dev=Path('runs/compact-v2-modern-development')/method
        if method in protocol['policy_index']:
            sel=str(dev/'model_selection');cal=str(dev/'calibration');two.extend([sel,cal])
            confidence.append(dict(selection=sel,calibration=cal,policy=protocol['policy_index'][method]))
        test=Path('runs/compact-v2-modern-test')/method/'test'
        if (test/'COMPLETE.json').exists():two.append(str(test));common[method]=dict(run=str(test),policy_key=method)
        elif test.exists():extra.append(str(test))
    geometry=[f'runs/compact-v2-{m}-3d-test' for m in ('compact','original','sift')]
    sensor=[f'runs/compact-v2-{m}-sensor' for m in ('compact','original','sift')]
    index=dict(protocol=protocol_path,results_report='reports/compact_results_v2.json',two_dimensional=sorted(set(two)),
        three_dimensional=geometry,sensor=sensor,confidence=confidence,
        sensor_models={sensor[0]:primary,sensor[1]:'original',sensor[2]:'sift'},aggregate_full=full,aggregate_common=common)
    write_new('reports/compact_run_index_v2.json',index)
    paths=set()
    for root in index['two_dimensional']+geometry+sensor+extra:
        for p in Path(root).rglob('*'):
            if p.is_file():paths.add(p)
    # Preserve every raw compute attempt, including superseded-coordinate records.
    for root in Path('runs').glob('compact-v2-modern-*'):
        if root.is_dir():
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.jsonl','.txt'):paths.add(p)
    for model in Path('runs').glob('compact-v2-*-2026103*'):
        if model.is_dir() and (model/'training.json').exists():
            for name in ('training.json','COMPLETE.json','frozen_config.json'):
                if (model/name).exists():paths.add(model/name)
    for root in Path('runs').glob('compact-v2-cli-controls*'):
        if root.is_dir():
            for p in root.rglob('*'):
                if p.is_file():paths.add(p)
    from compact_research import SOURCES
    snapshot=Path('runs/compact-v2-source-snapshot')
    for name in SOURCES+['report_compact_v2.py','package_compact_v2.py','COMPACT_RESEARCH.md','COMPACT_REPRODUCTION.md']:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());paths.add(target)
    files={str(p):sha256(p) for p in sorted(paths)}
    archive=Path('reports/compact-v2-experiments.zip')
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,str(p))
        z.writestr('CAPSULE.json',json.dumps(dict(schema_version=2,protocol_sha256=sha256(protocol_path),files=files,
            source='Frozen south-pole study; earlier coordinate-v1 and failed compute records explicitly retained as historical diagnostics.'),indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Packaged experiment CRC failed')
    Path('reports/compact_experiment_assets_v2.sha256').write_text(sha256(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(files=len(files),archive_bytes=archive.stat().st_size,sha256=sha256(archive))))


if __name__=='__main__':package()
