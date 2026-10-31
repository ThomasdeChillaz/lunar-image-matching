"""Self-contained corrected-geometry replay capsule; live training weights excluded."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from package_localization_v3 import install


def package(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    audit=json.loads(Path('reports/localization_v5_rescore_audit.json').read_text())
    if not audit['passed']:raise ValueError('Corrected independent replay required')
    files=set()
    for manifest in ('reports/localization_training_data_v3.json','reports/localization_training_data_v5.json'):
        files.add(Path(manifest));meta=json.loads(Path(manifest).read_text());files.add(Path(meta['origin_manifest']))
        for t in meta['tiles']:files.add(Path(t['native_cache']))
    for run in ('runs/localization-v4-solver-diagnosis','runs/localization-v4-proposal-coverage',
                'runs/localization-v4-native-sift','runs/localization-v4-rotation-bank',
                'runs/localization-v5-axial-correction','runs/localization-v5-controlled-smoke'):
        files.update(p for p in Path(run).rglob('*') if p.is_file())
    v5=json.loads(Path('runs/localization-v5-axial-correction/protocol.json').read_text())
    files.update(Path(p) for p in v5['inputs'])
    coarse=Path('runs/compact-v2-expanded-20261030')
    files.update(coarse/p for p in ('selected.pt','training.json','COMPLETE.json'))
    for pattern in ('*v4.py','*v5.py'):files.update(Path('.').glob(pattern))
    files.update(p for p in Path('reports').glob('*v5*.json') if p.name!='localization_v5_capsule.json')
    files.update(Path('reports').glob('localization_v4*.json'))
    files.add(Path('reports/lusnar_release_inventory_v4.json'))
    files.update([Path('reports/GEOMETRY_CORRECTION_V5.md'),Path('reports/localization_v5_geometry_correction.png')])
    files.add(Path('runs/localization-v4-coarse-development/INTERRUPTED_FOR_GEOMETRY_AUDIT.json'))
    # Only immutable launch/config/source files from the live cohort are included.
    live=Path('runs/localization-v5-coarse-development');files.add(live/'protocol.json')
    for run in live.glob('*/frozen_config.json'):
        files.add(run);files.update((run.parent/'source').glob('*.py'))
    manifest=dict(schema_version=5,scope='Corrected axial-depth development replay. No new trained model or final-test claim.',
        files={p.as_posix():sha256(p) for p in sorted(files)})
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            info=zipfile.ZipInfo(p.as_posix(),date_time=(2026,11,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,p.read_bytes())
        archive.writestr('CAPSULE_MANIFEST.json',json.dumps(manifest,indent=2))
        archive.writestr('ARTIFACT_NOTICE.txt',
            'Correction: audited LunarGeo EXRs contain optical-axis z depth. Legacy ray-based3D labels/results are superseded. '
            'V4 files are retained as explicitly obsolete diagnostics; v5 axial metrics are the corrected development readout. '
            'Images/depth derive from MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0. '
            'Own code and the frozen own coarse CNN weights are MIT. No third-party pretrained weights or copied upstream source. '
            'Live training checkpoint weights and all calibration/test pixels are excluded. No independently verified orbital accuracy claim.')
    output.with_suffix('.sha256').write_text(sha256(output)+'  '+output.name+'\n')
    record=dict(path=str(output),files=len(files),bytes=output.stat().st_size,sha256=sha256(output),scope=manifest['scope'])
    Path('reports/localization_v5_capsule.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/axial-geometry-v5-capsule.zip')
    p.add_argument('--install');p.add_argument('--destination',default='.')
    a=p.parse_args();install(a.install,a.destination) if a.install else package(a.output)
