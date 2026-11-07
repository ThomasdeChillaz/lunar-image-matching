"""Bounded acquisition of exact official pretrained assets, before model loading."""
import argparse
import json
from pathlib import Path
import socket
import time
import urllib.request

from dataset import sha256

URLS={
 'dad.pth':'https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth',
 'dinov2_vitl14_pretrain.pth':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
 'loma_B.pt':'https://github.com/davnords/storage/releases/download/loma/loma_B.pt',
 'loma_R.pth':'https://github.com/davnords/storage/releases/download/loma/loma_R.pth',
 'romav2.0.1.pt':'https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt',
}


def acquire(method,output):
    import torch
    socket.setdefaulttimeout(60)
    expected=json.loads(Path('reports/compact_baseline_lock_v2.json').read_text())['modern'][method]['weights']
    root=Path(torch.hub.get_dir())/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    attempts=[]
    try:
        for name,digest in expected.items():
            path=root/name
            if path.exists():
                if sha256(path)!=digest:raise ValueError('Existing official weight differs: '+name)
                continue
            success=False
            for number in range(3):
                url=URLS[name]+('?download=1' if number==1 else '')
                partial=path.with_suffix(path.suffix+'.acquiring')
                start=time.monotonic();record=dict(name=name,url=url,attempt=number+1)
                try:
                    with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as f:
                        while True:
                            blob=response.read(1024*1024)
                            if not blob:break
                            f.write(blob)
                            if time.monotonic()-start>300:raise TimeoutError('Official asset exceeded five-minute download budget')
                    if sha256(partial)!=digest:raise ValueError('Official weight checksum differs')
                    partial.replace(path);record['status']='verified';success=True
                except Exception as exc:
                    record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                    if partial.exists():partial.unlink()
                record['elapsed_seconds']=time.monotonic()-start;attempts.append(record)
                print(name,record['status'],round(record['elapsed_seconds'],1),flush=True)
                if success:break
            if not success:raise RuntimeError('Exact official asset unavailable within bounded acquisition budget: '+name)
    finally:
        out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(method=method,attempts=attempts,expected_weights=expected),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();acquire(a.method,a.output)
