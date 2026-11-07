"""Package the measured diagnostic stage, excluding weights and source imagery."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from dataset import sha256


def package(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    files = []
    for folder in ('runs/localization-diagnostics-v3', 'runs/localization-diagnostics-v3-validated',
                   'runs/localization-v3-controlled-smoke'):
        files.extend(p for p in Path(folder).rglob('*') if p.is_file())
    files.extend(p for p in Path('runs/localization-v3').rglob('*') if p.is_file())
    files.extend(Path('reports').glob('localization_*.json'))
    files.extend(Path('reports').glob('localization_*.png'))
    files.extend([Path('reports/localization_diagnostic_brief_v3.pdf'),
                  Path('reports/LOCALIZATION_DIAGNOSTICS_V3.md'), Path('LOCALIZATION_RESEARCH.md'),
                  Path('LOCALIZATION_REPRODUCTION.md')])
    files.extend(Path('.').glob('*localization*v3.py'))
    manifest = dict(schema_version=3, claim='Development diagnostics only; data gate failed before training.',
                    files={p.as_posix(): sha256(p) for p in sorted(set(files))})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(set(files)):
            info = zipfile.ZipInfo(p.as_posix(), date_time=(2026, 10, 31, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, p.read_bytes())
        info = zipfile.ZipInfo('CAPSULE_MANIFEST.json', date_time=(2026, 10, 31, 0, 0, 0))
        archive.writestr(info, json.dumps(manifest, indent=2).encode())
    output.with_suffix('.sha256').write_text(sha256(output)+'  '+output.name+'\n')
    print(json.dumps(dict(files=len(files), bytes=output.stat().st_size, sha256=sha256(output)), indent=2))


def install(archive_path, destination):
    destination = Path(destination).resolve()
    archive_path = Path(archive_path)
    sidecar = archive_path.with_suffix('.sha256')
    if not sidecar.exists() or sidecar.read_text().split()[0] != sha256(archive_path):
        raise ValueError('Missing or invalid archive checksum')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt capsule')
        manifest = json.loads(archive.read('CAPSULE_MANIFEST.json'))
        pending = []
        for name, digest in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError('Unsafe capsule path')
            path = destination.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(destination):
                raise ValueError('Capsule path escapes destination')
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Capsule member checksum differs')
            if path.exists() and (not path.is_file() or sha256(path) != digest):
                raise FileExistsError('Refusing to overwrite different existing file: '+name)
            pending.append((path, content))
        # Validate every member before writing anything.
        for path, content in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(content)
    print('Verified and installed', len(pending), 'diagnostic files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/localization-diagnostics-v3.zip')
    parser.add_argument('--install')
    parser.add_argument('--destination', default='.')
    args = parser.parse_args()
    if args.install:
        install(args.install, args.destination)
    else:
        package(args.output)
