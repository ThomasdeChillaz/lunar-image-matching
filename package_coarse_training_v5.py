"""Immutable corrected-training artifacts; install the geometry capsule first."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from lunargeo_dataset import write_new
from package_localization_v3 import install


def package(root,output):
    root=Path(root);output=Path(output)
    if output.exists() or output.with_suffix('.sha256').exists():raise FileExistsError(output)
    audit_path=Path('reports/coarse_training_audit_v5.json');audit=json.loads(audit_path.read_text())
    if not audit['passed'] or audit['trained_runs']!=4 or audit['summary_sha256']!=sha256(root/'summary.json'):
        raise ValueError('Completed independently replayed four-run cohort required')
    files=[p for p in root.rglob('*') if p.is_file()]
    files.extend([audit_path,Path('reports/coarse_training_comparison_v5.json'),Path('reports/COARSE_TRAINING_V5.md'),
        Path('reports/coarse_training_outcomes_v5.png'),Path('reports/localization_training_data_v5.json')])
    files.extend(Path(p) for p in ('train_coarse_localization_v5.py','coarse_training_v5.py','coarse_pipeline_v5.py',
        'lunar_geometry_v5.py','verify_coarse_training_v5.py','report_coarse_training_v5.py','package_coarse_training_v5.py',
        'combine_pose_backends_v5.py','test_pose_backends_v5.py','control_coarse_candidates_v5.py','verify_procedural_smoke.py',
        'test_coarse_training_replay_v5.py','diagnose_axial_pose_controls_v5.py'))
    files.extend(p for p in Path('runs/localization-v5-backend-fallback').rglob('*') if p.is_file())
    files.extend(p for p in Path('runs/localization-v5-training-smoke').rglob('*') if p.is_file())
    files.extend([Path('reports/coarse_training_controls_v5.json'),Path('reports/coarse_training_smoke_audit_v5.json')])
    files.extend([Path('runs/localization-v5-axial-pose-controls.json'),Path('reports/axial_pose_controls_summary_v5.json')])
    files=sorted(set(files))
    for p in files:
        if p.is_absolute() or '..' in p.parts or not p.is_file():raise ValueError('Unsafe or absent capsule member')
    manifest=dict(schema_version=5,scope='Four corrected-depth development runs; no calibrated acceptance or final claim.',
        prerequisite='Install the axial-geometry-v5 release capsule and matching SHA256 sidecar first.',
        files={p.as_posix():sha256(p) for p in files})
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            info=zipfile.ZipInfo(p.as_posix(),date_time=(2026,11,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,p.read_bytes())
        archive.writestr('CAPSULE_MANIFEST.json',json.dumps(manifest,indent=2))
        archive.writestr('ARTIFACT_NOTICE.txt','Own compact CNN code/weights: MIT. No third-party pretrained weights. '
            'Source dataset: MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0. '
            'Requires the separate audited axial-geometry-v5 data capsule. '
            'Includes every initial/selected/final/optimizer checkpoint and failed run outcome. '
            'No calibration/test pixels, private correspondence, paid compute or human review. '
            'This is development evidence, not SOTA or independently verified orbital navigation.')
    digest=sha256(output);output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n')
    result=dict(path=output.as_posix(),sha256=digest,bytes=output.stat().st_size,files=len(files),scope=manifest['scope'])
    write_new('reports/coarse_training_capsule_v5.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/localization-v5-coarse-development')
    p.add_argument('--output',default='reports/coarse-training-v5-capsule.zip');p.add_argument('--install')
    a=p.parse_args()
    if a.install:print(json.dumps(install(a.install),indent=2))
    else:package(a.run,a.output)
