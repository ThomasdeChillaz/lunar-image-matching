"""The separate completed uniform cohort, excluding every live experiment."""
import argparse
import json
from pathlib import Path
import zipfile

from dataset import sha256
from lunargeo_dataset import write_new
from package_localization_v3 import install


def package(output):
    output = Path(output); root = Path('runs/localization-v8-uniform-native-development')
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('Preserve previous capsules')
    audit = json.loads(Path('reports/native_uniform_training_audit_v8.json').read_text())
    if not audit['passed'] or audit['trained_runs'] != 5 or audit['summary_sha256'] != sha256(root/'summary.json'):
        raise ValueError('Complete independently replayed five-run cohort required')
    protocol = json.loads((root/'protocol.json').read_text())
    files = {p for p in root.rglob('*') if p.is_file()}
    for name, digest in protocol['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Executed uniform source changed')
        files.add(Path(name))
    files.update(Path(p) for p in ('verify_native_training_v8.py', 'report_uniform_native_v8.py',
        'package_uniform_native_v8.py', 'control_native_candidates_v7.py', 'report_coarse_training_v5.py',
        'reports/native_uniform_training_audit_v8.json', 'reports/native_uniform_training_controls_v8.json',
        'reports/native_uniform_training_comparison_v8.json', 'reports/UNIFORM_NATIVE_V8.md', 'reports/native_uniform_training_v8.png'))
    for p in files:
        if p.is_absolute() or '..' in p.parts or not p.is_file() or p.suffix == '.temporary':
            raise ValueError('Unsafe or incomplete uniform capsule member')
    manifest = dict(schema_version=8, scope='Completed development uniform-positive cohort: all five runs and failures, no large learned gain or calibrated acceptance.',
        prerequisite='Install axial-geometry-v5 and the native-training-v7 capsule first. The latter supplies prior comparisons for the publication reporter.',
        files={p.as_posix(): sha256(p) for p in sorted(files)})
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info = zipfile.ZipInfo(p.as_posix(), date_time=(2026, 11, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
        z.writestr('CAPSULE_MANIFEST.json', json.dumps(manifest, indent=2))
        z.writestr('ARTIFACT_NOTICE.txt', 'Own code and compact checkpoints MIT. MoonAnything LunarGeo source data '
            'Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0, separately obtained from the prerequisite capsule. '
            'Every initial/selected/final/optimizer state and failed head is retained. No third-party pretrained weights, '
            'new final/calibration pixels, human review, paid compute or private correspondence. Development only.')
    digest = sha256(output); output.with_suffix('.sha256').write_text(digest+'  '+output.name+'\n')
    result = dict(path=str(output), sha256=digest, bytes=output.stat().st_size, files=len(files), scope=manifest['scope'])
    write_new('reports/native_uniform_capsule_v8.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default='reports/native-uniform-v8-capsule.zip')
    p.add_argument('--install'); p.add_argument('--destination', default='.')
    a = p.parse_args()
    if a.install:
        install(a.install, a.destination)
    else:
        package(a.output)
