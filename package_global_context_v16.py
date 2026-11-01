"""Package the complete negative context study and independently audited diagnosis."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from lunargeo_dataset import write_new
from package_localization_v3 import install


def package(output):
    output=Path(output)
    if output.exists() or output.with_suffix('.sha256').exists():raise FileExistsError('Preserve earlier capsules')
    cohort=Path('runs/localization-v16-global-context-development')
    audit=json.loads(Path('reports/native_context_complete_audit_v16.json').read_text())
    controls=Path('runs/localization-v16-selected-controls')
    actual=json.loads((controls/'summary.json').read_text())
    if (not audit['passed'] or audit['partial'] or audit['completed_runs']!=5 or audit['recorded_updates']!=10000
            or not actual['passed'] or actual['selected_models']!=5
            or actual['cohort_summary_sha256']!=sha256(cohort/'summary.json')):
        raise ValueError('Complete five-run audit and actual selected-weight controls required')
    if not any(r.get('actual_primary_forward_exact') and r.get('actual_primary_forward_cases')==72 for r in actual['records']):
        raise ValueError('Actual72-case selected forward replay required')
    roots=[cohort,controls,Path('runs/localization-v17-pointwise-proposal-diagnosis'),
        Path('runs/localization-v17-oracle-filter-diagnosis'),Path('runs/localization-v17-matchanything-native-control'),
        Path('runs/localization-v17-diagnostics-smoke-01')]
    files={p for root in roots for p in root.rglob('*') if p.is_file()}
    for root in roots:
        protocol_path=root/'protocol.json'
        if protocol_path.exists():
            for source,digest in json.loads(protocol_path.read_text()).get('source_sha256',{}).items():
                if sha256(source)!=digest:raise ValueError('Executed source changed: '+source)
                files.add(Path(source))
    extras=['control_selected_global_v16.py','verify_global_cohort_v16.py','report_global_cohort_v16.py',
        'verify_context_diagnosis_v17.py','verify_oracle_filter_v17.py','report_context_failure_v17.py',
        'package_global_context_v16.py','verify_surface_learning_v13.py','verify_coarse_training_v5.py',
        'report_surface_learning_v13.py','report_coarse_training_v5.py','verify_procedural_smoke.py',
        'GLOBAL_COHORT_V16.md','test_global_cohort_v16.py','test_context_diagnosis_v17.py',
        'test_native_matchanything_v17.py','reports/native_context_complete_audit_v16.json',
        'reports/global_context_complete_v16.json','reports/global_context_complete_v16.md','reports/global_context_complete_v16.png',
        'reports/native_context_proposal_audit_v17.json','reports/native_oracle_filter_audit_v17.json',
        'reports/context_failure_diagnosis_v17.json','reports/context_failure_diagnosis_v17.md','reports/context_failure_diagnosis_v17.png',
        'reports/native_matchanything_control_v17.json','reports/stereolunar_source_overlap_v17.json',
        'reports/native_diagnostics_smoke_v17.json']
    files.update(Path(p) for p in extras)
    for path in files:
        if path.is_absolute() or '..' in path.parts or not path.is_file() or path.suffix in ('.temporary','.part'):
            raise ValueError('Unsafe/incomplete capsule member: '+str(path))
    manifest=dict(schema_version=16,scope='Complete five-run development context study with no learned gain; audited proposal/oracle diagnosis and pretrained native software controls.',
        prerequisite='Install axial-geometry-v5, native-training-v7 and context-diagnostics-v15 capsules into the tagged public source checkout. These provide native development pixels, original checkpoints/controls and cost-pilot prerequisites.',
        files={p.as_posix():sha256(p) for p in sorted(files)})
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info=zipfile.ZipInfo(path.as_posix(),date_time=(2026,11,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
        archive.writestr('CAPSULE_MANIFEST.json',json.dumps(manifest,indent=2))
        archive.writestr('ARTIFACT_NOTICE.txt','Own code and compact checkpoints MIT. MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0. '
            'Development pixels come from prerequisites. Context training did not improve localization. Oracle filters/branches use query truth and are not deployed results. '
            'The MatchAnything software gate is not a lunar comparison. No external pretrained weights, new calibration/test pixels, private correspondence, paid compute, human review or email.')
    digest=sha256(output);output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n')
    result=dict(path=str(output),sha256=digest,bytes=output.stat().st_size,files=len(files),scope=manifest['scope'])
    write_new('reports/global_context_capsule_v16.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='reports/global-context-v16-capsule.zip')
    parser.add_argument('--install');parser.add_argument('--destination',default='.');args=parser.parse_args()
    if args.install:install(args.install,args.destination)
    else:package(args.output)
