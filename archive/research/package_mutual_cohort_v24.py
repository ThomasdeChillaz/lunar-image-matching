"""Package the complete new-loss pilot/cohort/selected controls, without external weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def package(output):
    out=Path(output)
    if out.exists()or out.with_suffix('.sha256').exists():raise FileExistsError('Preserve prior capsules')
    report=read('reports/mutual_cohort_complete_v24.json');audit=read('reports/native_mutual_complete_audit_v24.json')
    if (not audit['passed']or audit['recorded_updates']!=6000 or audit['replayed_pose_rows']!=1080
            or report['audit_sha256']!=sha('reports/native_mutual_complete_audit_v24.json')
            or not report['actual_primary_forward_exact']or report['actual_selected_model_controls']!=3):
        raise ValueError('Complete measured report/audit/actual controls required')
    roots=[Path('runs/localization-v23-mutual-cost-pilot'),Path('runs/localization-v24-mutual-development'),
        Path('runs/localization-v24-mutual-selected-controls'),Path('runs/mutual-v23-software-smoke')]
    files={p for root in roots for p in root.rglob('*')if p.is_file()}
    for root in roots:
        if not root.is_dir():raise ValueError('Missing complete study dependency')
        protocol=root/'protocol.json'
        if protocol.exists():
            for name,digest in read(protocol).get('source_sha256',{}).items():
                if sha(name)!=digest:raise ValueError('Executed source changed: '+name)
                files.add(Path(name))
    extras=['mutual_assignment_v23.py','train_mutual_pilot_v23.py','verify_mutual_pilot_v23.py',
        'train_mutual_cohort_v24.py','verify_mutual_cohort_v24.py','control_mutual_cohort_v24.py',
        'report_mutual_cohort_v24.py','render_mutual_cohort_v24.py','package_mutual_cohort_v24.py',
        'diagnose_mutual_assignment_v23.py','test_mutual_assignment_v23.py','test_mutual_training_v24.py',
        'verify_window_cohort_v18.py','verify_procedural_smoke.py','MUTUAL_ASSIGNMENT_V23.md',
        'reports/native_mutual_pilot_audit_v23.json','reports/native_mutual_pilot_summary_v23.json',
        'reports/native_mutual_complete_audit_v24.json','reports/native_mutual_smoke_v23.json',
        'reports/mutual_assignment_counterexample_v23.json','reports/mutual_initialization_reuse_v23.json',
        'reports/mutual_cohort_complete_v24.json','reports/mutual_cohort_complete_v24.md','reports/mutual_cohort_complete_v24.png']
    files.update(Path(p)for p in extras)
    for p in files:
        if p.is_absolute()or '..'in p.parts or not p.is_file()or p.suffix in ('.part','.temporary'):
            raise ValueError('Unsafe/missing capsule member: '+str(p))
    manifest=dict(schema_version=24,scope=report['scope'],
        prerequisites='Install axial-geometry-v5, native-training-v7, context-diagnostics-v15 and native-window-v18 capsules with checksums. The v18 capsule supplies the matched historical source/initial controls and exact-replay dependencies.',
        full_optimizer_retraining_performed=False,exact_independent_pilot_retraining_steps=100,
        files={p.as_posix():sha(p)for p in sorted(files)})
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED)as z:
        for p in sorted(files):
            info=zipfile.ZipInfo(p.as_posix(),date_time=(2026,11,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
        z.writestr('CAPSULE_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('ARTIFACT_NOTICE.txt','Own code/compact checkpoints MIT. LunarGeo source data, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0 via prerequisites. '
            'All predefined seeds and failed trained endpoints retained. No external pretrained weights, private correspondence, new calibration/test pixels or northern terrain. '
            'This negative exposed-development result is not a model upgrade, calibrated acceptance, fresh-test or SOTA claim. Full optimizer retraining is not claimed;100-step pilot was exactly retrained.')
    with zipfile.ZipFile(out)as z:
        if z.testzip()is not None:raise ValueError('New capsule CRC failed')
    digest=sha(out);out.with_suffix('.sha256').write_text(digest+'  '+out.name+'\n')
    result=dict(path=str(out),sha256=digest,bytes=out.stat().st_size,payload_files=len(files),zip_crc_passed=True,scope=manifest['scope'])
    p=Path('reports/mutual_cohort_capsule_v24.json')
    if p.exists():raise FileExistsError('Preserve earlier capsule index')
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/mutual-assignment-v24-capsule.zip');p.add_argument('--install');p.add_argument('--destination',default='.')
    a=p.parse_args()
    if a.install:
        from package_localization_v3 import install
        install(a.install,a.destination)
    else:package(a.output)
