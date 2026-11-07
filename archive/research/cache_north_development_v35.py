"""Strict same-source range cache for the original24development supports only.

The local sparse TIFF is intentionally incomplete. It must never be presented
as a complete source download or used for any other windows.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import struct
from urllib.request import Request,urlopen

from download_north_source_v33 import EXPECTED_BYTES,EXPECTED_ETAG
from north_render_plan_v31 import SOURCE
from north_terrain_pilot_v32 import PLAN_SHA,sha
from north_development_plan_v35 import development_plan,prerequisite_gate


def strict_range(start,length):
    request=Request(SOURCE,headers={'Range':f'bytes={start}-{start+length-1}',
        'Accept-Encoding':'identity','Cache-Control':'no-cache','User-Agent':'lunar-development-cache/35'})
    with urlopen(request,timeout=60)as response:
        if (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{start+length-1}/{EXPECTED_BYTES}'
                or response.headers.get('ETag')!=EXPECTED_ETAG
                or int(response.headers.get('Content-Length',-1))!=length):
            raise ValueError('Source ignored/misserved exact bounded byte range')
        data=response.read(length+1)
        if len(data)!=length:raise ValueError('Truncated/overlong source range')
    return data


def tile_ranges(header,sites):
    if header[:8]!=b'II+\x00\x08\x00\x00\x00':raise ValueError('Pinned little-endian BigTIFF required')
    offset=struct.unpack_from('<Q',header,8)[0];count=struct.unpack_from('<Q',header,offset)[0]
    tags={}
    for i in range(count):
        tag,kind,n,value=struct.unpack_from('<HHQQ',header,offset+8+i*20);tags[tag]=(kind,n,value)
    for tag,expected in {256:(3,1,43280),257:(3,1,43280),258:(3,1,32),259:(3,1,8),322:(3,1,512),323:(3,1,512)}.items():
        if tags[tag]!=expected:raise ValueError('Source dimensions/compression/tiling differ')
    if tags[324][:2]!=(16,7225)or tags[325][:2]!=(4,7225):raise ValueError('Source tile index shape differs')
    offsets=struct.unpack_from('<7225Q',header,tags[324][2]);lengths=struct.unpack_from('<7225I',header,tags[325][2])
    if sites!=development_plan()[1]:raise ValueError('Only exact original development windows may be cached')
    indices=set()
    for site in sites:
        if site['split']not in ('training','model_selection'):raise ValueError('Calibration/final/pilot supports forbidden')
        r,c,h,w=site['native_window']
        for row in range(r//512,(r+h-1)//512+1):
            for col in range(c//512,(c+w-1)//512+1):indices.add(row*85+col)
    result=[dict(tile=i,offset=offsets[i],length=lengths[i])for i in sorted(indices)]
    if any(r['length']<=0 or r['length']>2*1024*1024 or r['offset']<len(header)
           or r['offset']+r['length']>EXPECTED_BYTES for r in result):raise ValueError('Unsafe source tile extent')
    return result


def cache(output):
    gates=prerequisite_gate();_,sites=development_plan();root=Path(output)
    if root.exists():raise FileExistsError('Preserve previous range cache/failure')
    root.mkdir(parents=True);start=datetime.now(timezone.utc);records=[]
    header=strict_range(0,131072);ranges=tile_ranges(header,sites)
    protocol=dict(source_url=SOURCE,etag=EXPECTED_ETAG,source_total_bytes=EXPECTED_BYTES,
        plan_sha256=PLAN_SHA,header_bytes=len(header),tiles=ranges,components=[s['component']for s in sites],
        source_sha256=sha(__file__),started_utc=start.isoformat(),io_workers=2,maximum_calendar_seconds=1200,prerequisite_sha256=gates,
        scope='Incomplete sparse cache for exact development windows only; no raster decode or full-source checksum.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        with (root/'development-only.tif').open('xb')as out:
            out.write(header);out.truncate(EXPECTED_BYTES)
            def fetch(r):return r,strict_range(r['offset'],r['length'])
            with ThreadPoolExecutor(max_workers=2)as pool:
                for r,payload in pool.map(fetch,ranges):
                    age=(datetime.now(timezone.utc)-start).total_seconds()
                    if age<0 or age>1200:raise TimeoutError('Original bounded range-cache allowance exceeded')
                    out.seek(r['offset']);out.write(payload)
                    records.append(dict(**r,sha256=hashlib.sha256(payload).hexdigest()))
                    if len(records)%20==0:print('CACHED DEVELOPMENT TILES',len(records),'/',len(ranges),flush=True)
        result=dict(complete_for_declared_development_windows=True,complete_source_download=False,
            source_total_bytes=EXPECTED_BYTES,etag=EXPECTED_ETAG,header_sha256=hashlib.sha256(header).hexdigest(),
            downloaded_bytes=len(header)+sum(r['length']for r in records),tile_records=records,
            protocol_sha256=sha(root/'protocol.json'),elapsed_seconds=(datetime.now(timezone.utc)-start).total_seconds(),
            source_height_pixels_decoded=0,publisher_full_checksum_verified=False)
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='tile_records'}),flush=True)
    except BaseException as exc:
        (root/'FAILED.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),completed_tiles=len(records)),indent=2)+'\n');raise


def verify_cache(root):
    root=Path(root);report=json.loads((root/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if (not report['complete_for_declared_development_windows'] or report['complete_source_download']
            or report['protocol_sha256']!=sha(root/'protocol.json') or protocol['source_sha256']!=sha(__file__)):
        raise ValueError('Verified partial cache manifest required')
    _,sites=development_plan()
    with (root/'development-only.tif').open('rb')as source:
        header=source.read(131072)
        if hashlib.sha256(header).hexdigest()!=report['header_sha256']:raise ValueError('Cached header changed')
        required=tile_ranges(header,sites)
        if [{k:r[k]for k in ('tile','offset','length')}for r in report['tile_records']]!=required:
            raise ValueError('Cache does not cover exactly the four development windows')
        for record in report['tile_records']:
            source.seek(record['offset']);payload=source.read(record['length'])
            if hashlib.sha256(payload).hexdigest()!=record['sha256']:raise ValueError('Cached encoded tile changed')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='data/north-render-v31/range-cache-dev-v35')
    cache(p.parse_args().output)
