"""Acquire a checksummed subset of MoonAnything using exact ZIP byte ranges.

The public release nests deflated ZIPs. Inner archives must be downloaded as a
whole; outer 11 GB archives never are. Original assets remain outside version
control. Preserve CRC, SHA256 and the Zenodo record for provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from remote_archive import HTTPRangeReader
import zipfile
from dataset import sha256

RECORD = 'https://zenodo.org/api/records/18415119'


def fetch(output, members):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD, timeout=60) as response:
        record = json.load(response)
    (output/'zenodo_record.json').write_text(json.dumps(record, indent=2))
    file = next(f for f in record['files'] if f['key'] == 'LunarPhoto.zip')
    reader = HTTPRangeReader(file['links']['self'], file['size'], block_size=16*1024*1024)
    archive = zipfile.ZipFile(reader)
    logs = []
    for member in members:
        info = archive.getinfo('LunarPhoto/'+member)
        destination = output/Path(member).name
        provenance = destination.with_suffix(destination.suffix+'.provenance.json')
        if destination.exists() and provenance.exists():
            previous = json.loads(provenance.read_text())
            if sha256(destination) == previous['sha256'] and destination.stat().st_size == info.file_size:
                logs.append(previous)
                continue
            raise ValueError('Cached member differs from provenance')
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.with_suffix(destination.suffix+'.part')
        with archive.open(info) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target, length=1024*1024)
        digest = sha256(temporary)
        temporary.replace(destination)
        log = dict(record_doi=record['doi'], record_url=RECORD, archive=file,
            member=info.filename, member_crc32=info.CRC, member_size=info.file_size,
            sha256=digest, verification='zipfile decompression and CRC32 plus saved SHA256; whole outer archive MD5 not checked')
        provenance.write_text(json.dumps(log, indent=2))
        logs.append(log)
        print(member, info.file_size, digest, flush=True)
    return logs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/moonanything')
    p.add_argument('--members', nargs='+', default=['README.md', 'train_dataset.csv', 'val_dataset.csv',
        'tst_dataset.csv', 'metadata.zip', 'multilighting_hapke_rendering.zip'])
    args = p.parse_args()
    fetch(args.output, args.members)
