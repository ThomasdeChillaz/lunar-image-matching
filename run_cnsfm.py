"""Run the authors' pinned CNSFM Windows executable; preserve all raw outputs.

No replacement matcher is silently substituted. The probe executes one original
image pair per region before a full run. This wrapper does not infer reference
accuracy from the program's console 'right matches' messages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

REVISION = '94cebaaa7fd3e070f8040620975554c3067651ef'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            result.update(chunk)
    return result.hexdigest()


def run(source, output, region, probe=False, prepare_only=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('CNSFM checkout differs from pinned revision')
    folder = source/'Dataset'/region
    images = [folder/Path(line.strip().replace('\\', '/')).name for line in (folder/'list_images.txt').read_text().splitlines() if line.strip()]
    if probe:
        images = images[:2]
    for image in images:
        if not image.is_file() or not image.with_suffix('.craters').is_file():
            raise FileNotFoundError(image)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    image_list = output/'list_images.txt'
    image_list.write_text('\n'.join(p.as_posix() for p in images)+'\n')
    executable = source/'CNSFM/CNSFM.exe'
    binaries = [executable, *sorted(executable.parent.glob('*.dll'))]
    metadata = dict(source_revision=revision, platform=platform.platform(), region=region, probe=probe,
        input_sha256={str(p.relative_to(source)):digest(p) for im in images for p in (im, im.with_suffix('.craters'))},
        binaries_sha256={p.name:digest(p) for p in binaries},
        coordinate_system='Native published GeoTIFF pixel coordinates; no 512px resampling before original executable',
        status='prepared', warning='Raw original outputs require audited parsing and common-grid conversion before comparison.')
    (output/'provenance.json').write_text(json.dumps(metadata, indent=2))
    if prepare_only:
        return metadata
    if platform.system() != 'Windows':
        raise RuntimeError('The original CNSFM executable requires Windows x86-64. Use the Windows workflow; no substitute baseline is claimed.')
    results = output/'original_outputs'
    results.mkdir()
    command = [str(executable), str(image_list), results.as_posix()+'/']
    start = time.perf_counter()
    with (output/'stdout.txt').open('w') as stdout, (output/'stderr.txt').open('w') as stderr:
        completed = subprocess.run(command, cwd=executable.parent, stdout=stdout, stderr=stderr, timeout=9000)
    metadata.update(status='executed' if completed.returncode == 0 else 'failed',
        command=command, returncode=completed.returncode, elapsed_seconds=time.perf_counter()-start,
        outputs={p.relative_to(output).as_posix():dict(sha256=digest(p), size=p.stat().st_size)
                 for p in results.rglob('*') if p.is_file()})
    (output/'provenance.json').write_text(json.dumps(metadata, indent=2))
    if completed.returncode:
        raise RuntimeError(f'CNSFM failed with return code {completed.returncode}; inspect captured logs')
    if not metadata['outputs']:
        raise RuntimeError('CNSFM returned no output files; inspect logs before claiming execution')
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--region', choices=['S1', 'S2', 'S3'], required=True)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    print(json.dumps(run(args.source, args.output, args.region, args.probe, args.prepare_only), indent=2))
