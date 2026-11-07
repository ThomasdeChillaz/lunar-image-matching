"""Completed recovery failures and context cost/control artifacts; no live fitting."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from lunargeo_dataset import write_new
from package_localization_v3 import install


def package(output):
    output = Path(output)
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('Preserve previous capsules')
    roots = [Path(p) for p in ('runs/localization-v14-broad-recovery-bank',
        'runs/localization-v14-broad-recovery-development', 'runs/localization-v14-missing-branch-diagnosis',
        'runs/localization-v15-context-initial-control', 'runs/localization-v15-context-cost-pilot')]
    audits = ['reports/native_recovery_bank_audit_v14.json', 'reports/native_recovery_audit_v14.json',
        'reports/native_missing_branch_audit_v14.json', None, 'reports/native_context_pilot_audit_v15.json']
    files = set()
    for root, audit_path in zip(roots, audits):
        summary = root/'summary.json'; protocol = json.loads((root/'protocol.json').read_text())
        if audit_path:
            audit = json.loads(Path(audit_path).read_text())
            if not audit['passed'] or audit['summary_sha256'] != sha256(summary):
                raise ValueError('Complete independently replayed artifact required')
        elif not json.loads(summary.read_text())['passed']:
            raise ValueError('Actual complete initial-control gate required')
        for name, digest in protocol['source_sha256'].items():
            if sha256(name) != digest:
                raise ValueError('Executed source changed: '+name)
            files.add(Path(name))
        files.update(p for p in root.rglob('*') if p.is_file())
    for folder in ('runs/localization-v14-recovery-smoke', 'runs/localization-v15-context-smoke', 'runs/localization-v15-context-smoke-02'):
        files.update(p for p in Path(folder).rglob('*') if p.is_file())
    extras = [p for p in audits if p]+['verify_pose_recovery_v14.py', 'verify_missing_poses_v14.py',
        'verify_context_pilot_v15.py', 'report_pose_recovery_v14.py', 'package_context_diagnostics_v15.py',
        'verify_coarse_training_v5.py', 'verify_axial_results_v5.py', 'verify_procedural_smoke.py',
        'test_pose_recovery_v14.py', 'test_global_context_v15.py', 'test_context_supervision_v15.py',
        'reports/native_recovery_comparison_v14.json', 'reports/POSE_RECOVERY_V14.md', 'reports/native_recovery_failure_v14.png',
        'reports/native_recovery_smoke_v14.json', 'reports/native_context_initial_control_v15.json',
        'reports/native_context_smoke_v15.json', 'reports/native_context_supervision_smoke_v15.json',
        'reports/native_context_supervision_gate_v15.json', 'reports/native_context_pilot_progress_v15.json',
        'reports/CONTEXT_PILOT_V15.md']
    files.update(Path(p) for p in extras)
    for p in files:
        if p.is_absolute() or '..' in p.parts or not p.is_file() or p.suffix == '.temporary':
            raise ValueError('Unsafe/incomplete member')
    manifest = dict(schema_version=15, scope='Completed negative broad-pose recovery, exact initial context controls and100-update train-only cost pilot. No context localization claim.',
        prerequisite='Install axial-geometry-v5 and native-training-v7 capsules into this tagged public source checkout first.',
        files={p.as_posix(): sha256(p) for p in sorted(files)})
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            info = zipfile.ZipInfo(p.as_posix(), date_time=(2026, 11, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
        archive.writestr('CAPSULE_MANIFEST.json', json.dumps(manifest, indent=2))
        archive.writestr('ARTIFACT_NOTICE.txt', 'Own code and small model checkpoints MIT. MoonAnything LunarGeo, '
            'Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0; obtain development pixels from prerequisite capsule. '
            'Recovery failed and is not adopted. Context cost pilot is not a localization result. No final/calibration '
            'pixels, third-party pretrained weights, paid compute, human review, private correspondence or automatic email.')
    digest = sha256(output); output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n')
    result = dict(path=str(output), sha256=digest, files=len(files), bytes=output.stat().st_size, scope=manifest['scope'])
    write_new('reports/context_diagnostics_capsule_v15.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/context-diagnostics-v15-capsule.zip')
    parser.add_argument('--install'); parser.add_argument('--destination', default='.')
    args = parser.parse_args()
    if args.install:
        install(args.install, args.destination)
    else:
        package(args.output)
