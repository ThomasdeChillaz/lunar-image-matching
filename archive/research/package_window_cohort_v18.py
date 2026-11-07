"""Package all completed native-window ablations and common native baseline evidence."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from lunargeo_dataset import write_new
from package_localization_v3 import install


def package(output):
    output=Path(output)
    if output.exists() or output.with_suffix('.sha256').exists():raise FileExistsError('Preserve all previous capsules')
    cohort=Path('runs/localization-v18-window-development');audit_path=Path('reports/native_window_complete_audit_v18.json')
    audit=json.loads(audit_path.read_text());summary=json.loads((cohort/'summary.json').read_text())
    controls=Path('runs/localization-v18-selected-controls');actual=json.loads((controls/'summary.json').read_text())
    if (not audit['passed'] or audit['partial'] or audit['completed_runs']!=6 or audit['recorded_updates']!=12000
        or len(summary['results'])!=6 or not actual['passed'] or actual['selected_models']!=6
        or actual['cohort_summary_sha256']!=sha256(cohort/'summary.json')):raise ValueError('Complete audited cohort/actual selected controls required')
    if not any(r.get('actual_primary_forward_exact') and r.get('actual_primary_forward_cases')==72 for r in actual['records']):
        raise ValueError('Actual selected primary72-case replay required')
    baseline=Path('runs/localization-v17-matchanything-native-development');baseline_audit=Path('reports/native_matchanything_audit_v17.json')
    ba=json.loads(baseline_audit.read_text())
    if not ba['passed'] or ba['records']!=144 or ba['summary_sha256']!=sha256(baseline/'summary.json'):raise ValueError('Complete common native baseline audit required')
    roots=[cohort,controls,Path('runs/localization-v18-window-initial-control'),
        Path('runs/localization-v18-window-cost-pilot'),baseline,Path('runs/native-window-v18-software-smoke')]
    files={p for root in roots for p in root.rglob('*') if p.is_file()}
    for root in roots:
        protocol=root/'protocol.json'
        if protocol.exists():
            for p,h in json.loads(protocol.read_text()).get('source_sha256',{}).items():
                if sha256(p)!=h:raise ValueError('Executed study source changed: '+p)
                files.add(Path(p))
    extras=['control_selected_window_v18.py','package_window_cohort_v18.py','verify_window_pilot_v18.py',
        'verify_window_cohort_v18.py','report_window_cohort_v18.py','verify_native_matchanything_v17.py',
        'report_native_matchanything_v17.py','verify_surface_learning_v13.py','verify_coarse_training_v5.py',
        'verify_axial_results_v5.py','report_surface_learning_v13.py','report_coarse_training_v5.py',
        'report_global_cohort_v16.py','verify_procedural_smoke.py','SEARCH_WINDOW_V18.md',
        'reports/native_window_initial_control_v18.json','reports/native_window_pilot_audit_v18.json',
        'reports/native_window_complete_audit_v18.json','reports/window_cohort_complete_v18.json',
        'reports/window_cohort_complete_v18.md','reports/window_cohort_complete_v18.png',
        'reports/native_matchanything_audit_v17.json','reports/native_matchanything_comparison_v17.json',
        'reports/native_matchanything_comparison_v17.md','reports/native_matchanything_comparison_v17.png',
        'reports/native_window_smoke_v18.json','test_window_runner_v18.py','test_search_window_learning_v18.py']
    files.update(Path(p) for p in extras)
    for p in files:
        if p.is_absolute() or '..' in p.parts or not p.is_file() or p.suffix in ('.temporary','.part'):raise ValueError('Unsafe/incomplete capsule member: '+str(p))
    manifest=dict(schema_version=18,scope='Complete native-window development cohort and independently replayed native MatchAnything comparison. No fresh final, acceptance or SOTA claim.',
        prerequisites='Install axial-geometry-v5, native-training-v7 and context-diagnostics-v15 capsules in the public tagged source checkout. These provide original native pixels/frozen checkpoints and dependency evidence.',
        files={p.as_posix():sha256(p) for p in sorted(files)})
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info=zipfile.ZipInfo(p.as_posix(),date_time=(2026,11,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
        z.writestr('CAPSULE_MANIFEST.json',json.dumps(manifest,indent=2))
        z.writestr('ARTIFACT_NOTICE.txt','Own code and compact checkpoints MIT. Source LunarGeo data, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0, provided by prerequisites. '
            'All failed checkpoints/seeds are retained. Native MatchAnything uses external pinned weights which are not redistributed. '
            'Four previously exposed selection components, not fresh validation. No new calibration/test pixels, private correspondence, paid compute, human review or email. '
            'Actual study outcomes and limitations are in window_cohort_complete_v18.md; software completion does not establish a learned gain.')
    digest=sha256(output);output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n')
    result=dict(path=str(output),sha256=digest,bytes=output.stat().st_size,files=len(files),scope=manifest['scope'])
    write_new('reports/window_cohort_capsule_v18.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='reports/native-window-v18-capsule.zip')
    parser.add_argument('--install');parser.add_argument('--destination',default='.');args=parser.parse_args()
    if args.install:install(args.install,args.destination)
    else:package(args.output)
