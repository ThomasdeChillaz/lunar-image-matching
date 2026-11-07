"""Acquire the complete unchanged official TIFF after a failed remote window read.

This copies bytes only. No source raster is decoded, including reserve terrain.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request,urlopen

from north_render_plan_v31 import SOURCE

EXPECTED_BYTES=5222983620
EXPECTED_ETAG='W/"1375067c4-18c940352d8"'


def download(output):
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve existing full or partial download')
    root.mkdir(parents=True);start=datetime.now(timezone.utc);tick=time.monotonic()
    protocol=dict(source_url=SOURCE,expected_bytes=EXPECTED_BYTES,expected_etag=EXPECTED_ETAG,
        started_utc=start.isoformat(),calendar_cap_seconds=600,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Complete local byte copy of same source after remote strip-array read failed. No raster decode; unchanged public site/split plan.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    state_path=root/'state.json';count=0;digest=hashlib.sha256()
    def state(status,**extra):
        state_path.write_text(json.dumps(dict(state=status,bytes=count,updated_utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n')
    state('running')
    try:
        with urlopen(Request(SOURCE,headers={'Accept-Encoding':'identity','User-Agent':'lunar-terrain-source/33'}),timeout=60)as response:
            if (response.status!=200 or int(response.headers.get('Content-Length',-1))!=EXPECTED_BYTES
                    or response.headers.get('ETag')!=EXPECTED_ETAG):
                raise ValueError('Complete source bytes/version differ from fixed window-read protocol')
            with (root/'source.tif.part').open('xb')as out:
                while True:
                    wall=(datetime.now(timezone.utc)-start).total_seconds()
                    if wall<0 or wall>600 or time.monotonic()-tick>600:raise TimeoutError('Original600second download cap exceeded')
                    chunk=response.read(min(4*1024*1024,EXPECTED_BYTES-count+1))
                    if not chunk:break
                    count+=len(chunk)
                    if count>EXPECTED_BYTES:raise ValueError('Source longer than fixed header')
                    out.write(chunk);digest.update(chunk)
                    if count%(256*1024*1024)==0:state('running');print('SOURCE BYTES',count,'/',EXPECTED_BYTES,flush=True)
            if count!=EXPECTED_BYTES:raise ValueError('Incomplete full source; preserve partial')
        (root/'source.tif.part').rename(root/'source.tif')
        result=dict(complete=True,bytes=count,sha256=digest.hexdigest(),source_etag=EXPECTED_ETAG,
            elapsed_seconds=(datetime.now(timezone.utc)-start).total_seconds(),
            publisher_checksum_verified=False,source_height_pixels_decoded=0,
            scope='Complete unchanged source byte copy only; no reserve inspection or benchmark.')
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');state('complete');print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:
        state('failed; preserve partial',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='data/north-render-v31/full-source-v33')
    download(p.parse_args().output)
