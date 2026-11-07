"""Package a complete audited surface-supervision study, preserving failures."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from lunargeo_dataset import write_new
from package_localization_v3 import install


def package(output):
    output = Path(output); root = Path('runs/localization-v13-surface-development')
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('Preserve prior capsules')
    audit = json.loads(Path('reports/native_surface_learning_audit_v13.json').read_text())
    control = json.loads(Path('reports/native_surface_learning_controls_v13.json').read_text())
    if (not audit['passed'] or not control['passed'] or audit['trained_runs'] != 5
            or audit['summary_sha256'] != sha256(root/'summary.json')
            or len(list(root.glob('*/COMPLETE.json'))) != 5):
        raise ValueError('Complete independently audited five-run study required')
    protocol = json.loads((root/'protocol.json').read_text())
    roots = [root, Path('runs/localization-v13-surface-prepare-failed-01'), Path('runs/localization-v13-surface-smoke-02')]
    files = {p for folder in roots for p in folder.rglob('*') if p.is_file()}
    for name, digest in protocol['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Pinned fitting source changed')
        files.add(Path(name))
    files.update(Path(p) for p in ('verify_surface_learning_v13.py', 'control_surface_learning_v13.py',
        'report_surface_learning_v13.py', 'package_surface_learning_v13.py', 'test_surface_learning_v13.py',
        'SURFACE_LEARNING_V13.md', 'verify_coarse_training_v5.py', 'verify_axial_results_v5.py',
        'report_coarse_training_v5.py', 'verify_procedural_smoke.py',
        'reports/native_surface_learning_audit_v13.json', 'reports/native_surface_learning_controls_v13.json',
        'reports/native_surface_learning_initial_audit_v13.json', 'reports/native_surface_learning_protocol_v13.json',
        'reports/native_surface_learning_smoke_v13_corrected.json',
        'reports/surface_learning_v13.json', 'reports/surface_learning_v13.md', 'reports/surface_learning_v13.png'))
    for p in files:
        if p.is_absolute() or '..' in p.parts or not p.is_file() or p.suffix == '.temporary':
            raise ValueError('Unsafe/incomplete member: '+str(p))
    manifest = dict(schema_version=13, scope='Completed development-only matched surface-supervision cohort; every failed and selected outcome retained.',
        prerequisite='Install axial-geometry-v5 and native-training-v7 capsules first. These supply native development data, coarse checkpoint, exact-pose gate and historical controls. Report SIFT comparison additionally uses the earlier corrected study artifacts.',
        files={p.as_posix(): sha256(p) for p in sorted(files)})
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            info = zipfile.ZipInfo(p.as_posix(), date_time=(2026, 11, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
        archive.writestr('CAPSULE_MANIFEST.json', json.dumps(manifest, indent=2))
        archive.writestr('ARTIFACT_NOTICE.txt', 'Own code and compact checkpoints MIT. MoonAnything LunarGeo, '
            'Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0; obtain data through prerequisite capsule. '
            'Development only. True query geometry is training/scoring only. No calibration/test pixels, '
            'third-party pretrained weights, private correspondence, paid compute, human review or automatic email.')
    digest = sha256(output); output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n')
    result = dict(path=str(output), sha256=digest, files=len(files), bytes=output.stat().st_size, scope=manifest['scope'])
    write_new('reports/surface_learning_capsule_v13.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/surface-learning-v13-capsule.zip')
    parser.add_argument('--install'); parser.add_argument('--destination', default='.')
    args = parser.parse_args()
    if args.install:
        install(args.install, args.destination)
    else:
        package(args.output)
