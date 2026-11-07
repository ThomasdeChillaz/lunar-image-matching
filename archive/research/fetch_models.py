"""Explicit, checksummed pretrained weight download; no weights in Git."""
import argparse,json,urllib.request
from pathlib import Path
from dataset import sha256


def verify_weights(cache, lock):
    for entry in lock['weights']:
        f=Path(cache)/'hub/checkpoints'/entry['filename']
        if not f.is_file() or sha256(f)!=entry['sha256']:
            raise RuntimeError(f'Missing/corrupt weights: {entry["filename"]}. Run python fetch_models.py --cache {cache}')


def fetch(cache,lock):
    directory=Path(cache)/'hub/checkpoints';directory.mkdir(parents=True,exist_ok=True)
    for entry in lock['weights']:
        dest=directory/entry['filename']
        if dest.is_file() and sha256(dest)==entry['sha256']:continue
        temp=dest.with_suffix('.part')
        try:
            with urllib.request.urlopen(entry['url'],timeout=120) as response,temp.open('wb') as out:
                while chunk:=response.read(1024*1024):out.write(chunk)
            if sha256(temp)!=entry['sha256']:raise ValueError('Weight checksum mismatch')
            temp.replace(dest)
        finally:temp.unlink(missing_ok=True)
    verify_weights(cache,lock)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',default='data/models');a=p.parse_args()
    fetch(a.cache,json.loads(Path(__file__).with_name('model_lock.json').read_text()));print('Both pinned weight files verified.')
