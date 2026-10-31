"""Distribute development-only own weights, supervised caches and measured results."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from package_localization_v3 import install


def write_archive(kind, files, note):
    path=Path('reports')/('localization-v3-development-'+kind+'.zip')
    if path.exists():raise FileExistsError(path)
    files=sorted(set(Path(p) for p in files))
    manifest=dict(schema_version=3,scope='Development-only training artifacts; no new final test.',
                  files={p.as_posix():sha256(p) for p in files})
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info=zipfile.ZipInfo(p.as_posix(),date_time=(2026,10,31,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
        z.writestr('CAPSULE_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('ARTIFACT_NOTICE.txt',note)
    path.with_suffix('.sha256').write_text(sha256(path)+'  '+path.name+'\n')
    return dict(path=path.as_posix(),files=len(files),bytes=path.stat().st_size,sha256=sha256(path))


def package(root, balanced='runs/localization-v3-balanced-development'):
    roots=[Path(root),Path(balanced)]
    for folder,audit_name in zip(roots,('localization_training_audit_v3.json','localization_balanced_audit_v3.json')):
        summary=json.loads((folder/'summary.json').read_text())
        audit=json.loads((Path('reports')/audit_name).read_text())
        if not audit['passed'] or len(summary['results'])!=5:
            raise ValueError('Both five-run cohorts and independent audits are required')
    runs=sorted(p.parent for folder in roots for p in folder.glob('*/training.json'))
    pilot=Path('runs/localization-v3-pilot')
    if pilot.exists():runs.append(pilot/'descriptor_3d-20261101')
    model_files=[]
    for run in runs:
        model_files.extend(run/name for name in ('initial.pt','selected.pt','final.pt','frozen_config.json','training.json','COMPLETE.json'))
    coarse=Path('runs/compact-v2-expanded-20261030')
    model_files.extend(coarse/name for name in ('selected.pt','training.json','COMPLETE.json'))
    artifacts={}
    artifacts['models']=write_archive('models',model_files,
        'Own structural CNN/refiner weights and code are MIT. Includes the frozen own coarse model. '
        'These are development candidates, not validated navigation models. No third-party pretrained weights are included.')
    manifest=Path('reports/localization_training_data_v3.json');d=json.loads(manifest.read_text())
    artifacts['data']=write_archive('data',[manifest,Path(d['origin_manifest'])]+[Path(t['native_cache']) for t in d['tiles']],
        'Derived from MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0. '
        'Native grayscale renders/ray-depth/camera metadata are repackaged into NPZ caches; image hashes and transformations are documented. '
        'Geometric labels are camera-projected simulated references, not independently verified lunar observations. '
        'Contains only18 training and8 model-selection geometries. No calibration/test pixels or private correspondence.')
    experiments=[p for folder in roots for p in (folder/'protocol.json',folder/'summary.json')]
    for run in runs:
        experiments.extend(p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.gz','.py'))
    experiments.extend(p for p in Path('runs/localization-v3-training-controls').rglob('*') if p.is_file())
    experiments.extend(p for p in Path('runs/localization-v3-balanced-controls').rglob('*') if p.is_file())
    experiments.extend(p for p in Path('runs/localization-v3-training-smoke').rglob('*') if p.is_file())
    experiments.extend(Path('reports').glob('localization_training_*.json'))
    experiments.extend(Path('reports').glob('localization_training_*.png'))
    experiments.extend(Path('reports').glob('localization_balanced_*.json'))
    experiments.extend(Path('reports').glob('localization_balanced_*.png'))
    experiments.extend([Path('LOCALIZATION_TRAINING.md'),Path('reports/LOCALIZATION_TRAINING_V3.md')])
    experiments.append(Path('reports/LOCALIZATION_BALANCED_V3.md'))
    experiments.extend(Path('.').glob('*localization*training*v3.py'))
    experiments.extend(Path('.').glob('*balanced*localization*v3.py'))
    experiments.extend([Path('localization_refiner_v3.py'),Path('train_localization_v3.py'),Path('refine_lunar_development.py')])
    baseline=Path('reports/localization_pose_refit_v3.json')
    experiments.extend([baseline,Path(json.loads(baseline.read_text())['source_run'])/'results.json.gz'])
    artifacts['experiments']=write_archive('experiments',experiments,
        'Measured development outcomes, exact executed source snapshots and procedural software controls. '
        'Initial checkpoint selection, failed heads and all predefined seeds are retained. '
        'No final/generalization, accepted-error guarantee or real orbital accuracy claim.')
    result=dict(schema_version=3,scope='Development training only',artifacts=artifacts)
    Path('reports/localization_training_assets_v3.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/localization-v3-development')
    p.add_argument('--balanced',default='runs/localization-v3-balanced-development')
    p.add_argument('--install');p.add_argument('--destination',default='.')
    a=p.parse_args()
    if a.install:install(a.install,a.destination)
    else:package(a.run,a.balanced)
