"""Install the checksummed study capsule and independently replay saved metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import zipfile

from compact_confidence import accepts
from dataset import sha256
from geometry import endpoint_error,evaluation_points
from lunargeo_dataset import write_new
import numpy as np


def install_capsule(path,expected):
    if sha256(path)!=expected:raise ValueError('Experiment capsule checksum differs')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('Experiment capsule CRC differs')
        index=json.loads(archive.read('CAPSULE.json'))
        expected_names=set(index['files'])|{'CAPSULE.json'}
        if set(archive.namelist())!=expected_names:raise ValueError('Unexpected/missing capsule members')
        for name,digest in index['files'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('runs','data'):
                raise ValueError('Capsule member is outside experiment data')
            if stat.S_ISLNK(archive.getinfo(name).external_attr>>16):raise ValueError('Capsule symlinks are not supported')
            content=archive.read(name)
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Capsule file checksum differs')
            if p.exists():
                if sha256(p)!=digest:raise ValueError('Existing experiment file differs; refusing replacement: '+name)
            else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    return index


def install_assets(path,expected):
    if sha256(path)!=expected:raise ValueError('Published asset checksum differs')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('Asset CRC differs')
        for item in archive.infolist():
            if item.filename=='DATASET.json':continue
            p=Path(item.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('runs','data'):
                raise ValueError('Asset member outside experiment data')
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Asset symlinks are not supported')
            content=archive.read(item)
            digest=hashlib.sha256(content).hexdigest()
            if p.exists():
                if sha256(p)!=digest:raise ValueError('Existing file differs; refusing replacement: '+str(p))
            else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)


def replay(output,index_path='reports/compact_run_index_v2.json'):
    from compact_research import audit,confidence
    from verify_compact_3d import audit as audit_3d
    index=json.loads(Path(index_path).read_text())
    protocol=json.loads(Path(index['protocol']).read_text())
    for name,digest in {**protocol['source_sha256'],**protocol['policies'],**protocol['comparison_assets'],
                        protocol['manifest_path']:protocol['manifest_sha256']}.items():
        if sha256(name)!=digest:raise ValueError('Frozen source/data/policy differs: '+name)
    for directory,digest in protocol['models'].items():
        if sha256(Path(directory)/'selected.pt')!=digest:raise ValueError('Frozen model differs')
    completed=[]
    with tempfile.TemporaryDirectory() as tmp:
        for kind,runs in [('2d',index['two_dimensional']),('3d',index['three_dimensional'])]:
            for i,run in enumerate(runs):
                result=(audit if kind=='2d' else audit_3d)(run,Path(tmp)/f'{kind}-{i}.json')
                if not result['passed']:raise ValueError('Metric replay failed')
                completed.append(dict(track=kind,run=run,rows=result['rows']))
        for i,item in enumerate(index['confidence']):
            path=Path(tmp)/f'policy-{i}.json'
            confidence(item['selection'],item['calibration'],path)
            if json.loads(path.read_text())!=json.loads(Path(item['policy']).read_text()):
                raise ValueError('Independent confidence fit/calibration replay differs')
    sensor=json.loads(Path(protocol['sensor_protocol']).read_text())
    for name,digest in sensor['files_sha256'].items():
        if sha256(name)!=digest:raise ValueError('POLAR photograph differs')
    identity=np.array([[1.,0,0],[0,1.,0]])
    for run in index['sensor']:
        rows=json.loads(Path(run,'results.json').read_text())
        completion=json.loads(Path(run,'COMPLETE.json').read_text())
        if sha256(Path(run,'results.json'))!=completion['results_sha256']:raise ValueError('Sensor outcomes differ')
        if [r['pair_id'] for r in rows]!=[c['pair_id'] for c in sensor['cases']]:raise ValueError('Sensor coverage differs')
        policy=json.loads(Path(protocol['policy_index'][index['sensor_models'][run]]).read_text())
        for row in rows:
            error=endpoint_error(row['raw_transform'],identity,evaluation_points((128,128),identity))
            if (error is None)!=(row['nominal_error_px'] is None) or (error is not None and not np.isclose(error,row['nominal_error_px'])):
                raise ValueError('Sensor nominal reference replay differs')
            accepted=accepts(policy,{'features':row['features']})
            if accepted!=row['accepted'] or (row['deployed_transform'] is None)!=(not accepted):
                raise ValueError('Sensor frozen rejection behavior differs')
        completed.append(dict(track='nominal_sensor',run=run,rows=len(rows)))
    result=dict(passed=True,protocol_sha256=sha256(index['protocol']),runs=completed,
        confidence_policies_replayed=len(index['confidence']),new_model_inference=False,
        verified_real_navigation_accuracy=None)
    write_new(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capsule');p.add_argument('--expected-sha256');p.add_argument('--output',required=True)
    p.add_argument('--models',help='Downloaded compact-v2-models.zip release asset')
    p.add_argument('--data',help='Downloaded compact-v2-data.zip release asset')
    p.add_argument('--index',default='reports/compact_run_index_v2.json')
    a=p.parse_args()
    for name in ('models','data'):
        path=getattr(a,name)
        if path:
            sidecar=Path('reports')/('compact_model_assets_v2.sha256' if name=='models' else 'compact_data_assets_v2.sha256')
            install_assets(path,sidecar.read_text().split()[0])
    if a.capsule:
        if not a.expected_sha256:p.error('--capsule requires --expected-sha256')
        install_capsule(a.capsule,a.expected_sha256)
    print(json.dumps(replay(a.output,a.index),indent=2))
