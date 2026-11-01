"""Package completed native/orientation studies; never include a live cohort."""
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
    roots = [Path(p) for p in ('runs/localization-v7-native-development', 'runs/localization-v6-orientation-development',
        'runs/localization-v7-native-pilot', 'runs/localization-v8-uniform-native-pilot')]
    audits = ['reports/native_training_audit_v7.json', 'reports/orientation_training_audit_v6.json',
        'reports/native_pilot_audit_v7.json', 'reports/native_uniform_pilot_audit_v8.json']
    for root, audit_path, count in zip(roots, audits, (5, 4, 1, 1)):
        audit = json.loads(Path(audit_path).read_text())
        if not audit['passed'] or audit['trained_runs'] != count or audit['summary_sha256'] != sha256(root/'summary.json'):
            raise ValueError('Completed independently replayed study required: '+str(root))
        if len(list(root.glob('*/COMPLETE.json'))) != count:
            raise ValueError('Partial study cannot enter completed capsule')
    files = {p for root in roots for p in root.rglob('*') if p.is_file()}
    # Include exactly the executed source closure, never downloaded author code/weights.
    for root in roots:
        protocol = json.loads((root/'protocol.json').read_text())
        for name, digest in protocol['source_sha256'].items():
            p = Path(name)
            if sha256(p) != digest:
                raise ValueError('Executed source changed: '+name)
            files.add(p)
    extra_roots = ('runs/localization-v7-native-initial-control', 'runs/localization-v7-window-controls',
                   'runs/localization-v9-progress-smoke', 'runs/localization-v10-affine-bank',
                   'runs/localization-v10-affine-development', 'runs/localization-v11-surface-warp-bank',
                   'runs/localization-v11-surface-warp-development', 'runs/localization-v11-surface-smoke')
    files.update(p for folder in extra_roots for p in Path(folder).rglob('*') if p.is_file())
    names = audits+['reports/native_training_controls_v7.json', 'reports/native_training_comparison_v7.json',
        'reports/NATIVE_TRAINING_V7.md', 'reports/native_training_outcomes_v7.png',
        'reports/native_pilot_controls_v7.json', 'reports/native_uniform_pilot_controls_v8.json',
        'reports/native_initial_equivalence_v7.json', 'reports/native_sampling_coverage_v7.json',
        'reports/native_window_full_audit_v7.json', 'reports/native_window_summary_v7.json',
        'reports/native_no_match_diagnosis_v9.json', 'reports/native_no_match_generalization_v9.json',
        'reports/native_progress_smoke_v9.json', 'reports/orientation_training_comparison_v6.json',
        'reports/orientation_training_outcomes_v6.png', 'reports/ORIENTATION_TRAINING_V6.md',
        'reports/orientation_plot_render_v6.json', 'reports/localization_training_data_v5.json',
        'reports/axial_pose_controls_summary_v5.json', 'runs/localization-v5-axial-pose-controls.json',
        'verify_native_training_v7.py', 'verify_native_training_v8.py', 'verify_coarse_training_v5.py',
        'control_native_candidates_v7.py', 'report_native_training_v7.py', 'report_coarse_training_v5.py',
        'report_orientation_v6.py', 'render_orientation_v6.py', 'diagnose_native_windows_v7.py',
        'verify_native_windows_v7.py', 'diagnose_no_match_v9.py', 'diagnose_no_match_generalization_v9.py',
        'verify_procedural_smoke.py', 'package_native_training_v7.py', 'NATIVE_REFINEMENT_V7.md',
        'affine_native_v10.py', 'evaluate_affine_native_v10.py', 'verify_affine_native_v10.py',
        'report_affine_native_v10.py', 'test_affine_native_v10.py',
        'reports/native_affine_bank_full_audit_v10.json', 'reports/native_affine_audit_v10.json',
        'reports/native_affine_comparison_v10.json', 'reports/AFFINE_NATIVE_V10.md',
        'geometry_warp_native_v11.py', 'evaluate_geometry_warp_v11.py', 'verify_geometry_warp_v11.py',
        'report_geometry_warp_v11.py', 'render_native_geometry_v11.py', 'test_geometry_warp_v11.py',
        'reports/native_surface_bank_audit_v11.json', 'reports/native_surface_audit_v11.json',
        'reports/native_surface_comparison_v11.json', 'reports/SURFACE_WARP_V11.md',
        'reports/native_surface_smoke_v11.json', 'reports/native_completed_software_v11.json',
        'reports/native_geometry_controls_v11.png', 'reports/native_geometry_figure_v11.json']
    files.update(Path(p) for p in names)
    for p in files:
        if p.is_absolute() or '..' in p.parts or not p.is_file() or p.suffix == '.temporary':
            raise ValueError('Unsafe or incomplete capsule member: '+str(p))
    files = sorted(files)
    for folder, evidence in [('runs/localization-v10-affine-development', 'reports/native_affine_audit_v10.json'),
                             ('runs/localization-v11-surface-warp-development', 'reports/native_surface_audit_v11.json')]:
        audit = json.loads(Path(evidence).read_text())
        if not audit['passed'] or audit['summary_sha256'] != sha256(Path(folder)/'summary.json'):
            raise ValueError('Completed geometry control replay required')
    manifest = dict(schema_version=7, scope='Completed development native/orientation studies, including all failed runs and pilots. No consistent learned pose gain, calibrated acceptance or SOTA claim.',
        prerequisite='Install audited axial-geometry-v5 data capsule first. No live v8 full-training files are included.',
        files={p.as_posix(): sha256(p) for p in files})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo(p.as_posix(), date_time=(2026, 11, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
        z.writestr('CAPSULE_MANIFEST.json', json.dumps(manifest, indent=2))
        z.writestr('ARTIFACT_NOTICE.txt', 'Own code and compact checkpoints: MIT. Source data: MoonAnything LunarGeo, Grethen et al., '
            'doi:10.5281/zenodo.18415119, CC BY4.0, separately released in axial-geometry-v5. '
            'No third-party pretrained weights, calibration/test pixels, human annotations or private correspondence. '
            'Includes initial/selected/final and optimizer checkpoints for every completed ablation/seed and both pilots. '
            'Earlier ray-depth3D results are superseded; these use corrected optical-axis z. '
            'Development only; no orbital navigation or global SOTA claim.')
    digest = sha256(output); output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n')
    result = dict(path=str(output), sha256=digest, bytes=output.stat().st_size, files=len(files), scope=manifest['scope'])
    write_new('reports/native_training_capsule_v7.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default='reports/native-training-v7-capsule.zip')
    p.add_argument('--install'); p.add_argument('--destination', default='.')
    a = p.parse_args()
    if a.install:
        install(a.install, a.destination)
    else:
        package(a.output)
