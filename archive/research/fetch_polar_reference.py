"""Fetch selected NASA POLAR sensor/reference bytes, preserving CRC and hashes.

Original photographs and laser scans stay outside Git. No matcher or outcome is
used to choose the four prespecified laboratory scenes or acquisition settings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

from dataset import sha256
from remote_archive import HTTPRangeReader


PAGE='https://ti.arc.nasa.gov/dataset/IRG_PolarDB/'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if href:
                self.links.append(urllib.parse.urljoin(PAGE,href))


def fetch_scene(url,root):
    root=Path(root)
    terrain=re.search(r'Terrain\d+',url).group()
    folder=root/terrain
    folder.mkdir(parents=True,exist_ok=True)
    registry=folder/'provenance.json'
    if registry.exists():
        old=json.loads(registry.read_text())
        for entry in old['members']:
            if sha256(folder/entry['local_name'])!=entry['sha256']:
                raise ValueError('Cached POLAR member changed')
        return old
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as response:
        size=int(response.headers['Content-Length'])
        headers={k:response.headers.get(k) for k in ('ETag','Last-Modified','Accept-Ranges')}
    reader=HTTPRangeReader(url,size,block_size=4*1024*1024,cache_dir=root/'.range-cache')
    with zipfile.ZipFile(reader) as archive:
        selected=[i for i in archive.infolist() if i.filename.endswith('/stereo.calibration')
            or i.filename.endswith('/GroundTruth_ascii/PosA.xyz')
            or re.search(r'/PosA_1500_Loff/Sun_(30|180)/Cam[LR]_\d+_1024\.png$',i.filename)]
        if len(selected)!=6:
            raise ValueError(f'Expected one calibration, one laser cloud and four fixed-setting photos: {terrain}, {len(selected)}')
        entries=[]
        for info in selected:
            name=Path(info.filename).name
            if any(e['local_name']==name for e in entries):
                raise ValueError('Duplicate flattened POLAR source name')
            content=archive.read(info)  # ZIP decompression verifies original CRC.
            path=folder/name
            digest=hashlib.sha256(content).hexdigest()
            if path.exists() and sha256(path)!=digest:
                raise ValueError('Existing POLAR member differs')
            path.write_bytes(content)
            entries.append(dict(source_member=info.filename,local_name=name,bytes=len(content),
                sha256=digest,zip_crc32=info.CRC,compressed_bytes=info.compress_size))
            print(terrain,name,len(content),flush=True)
    result=dict(dataset='NASA Ames POLAR Stereo Dataset',source_page=PAGE,archive_url=url,
        archive_size=size,archive_headers=headers,terrain=terrain,members=entries,
        selection='Terrains01,06,11,12; camera PosA at1500mm, rover lights off, Sun30 and180, both cameras at1024ms. Metadata-only fixed selection, no image-quality or matcher screening.',
        verification='Exact HTTP ranges, original ZIP member CRC32, saved SHA256; whole outer archive not downloaded or independently checksummed.',
        reference='External laser XYZ and stereo/camera-to-laser calibration. Measurement and calibration uncertainties must be distinguished. This is laboratory analogue data, not lunar orbital navigation.')
    registry.write_text(json.dumps(result,indent=2)+'\n')
    return result


def fetch(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(PAGE,timeout=45) as response:
        html=response.read()
    (root/'source_page.html').write_bytes(html)
    parser=Links()
    parser.feed(html.decode())
    urls=[url for url in parser.links if re.search(r'Terrain(01|06|11|12)_.*\.zip$',url)]
    if len(urls)!=4:
        raise ValueError('Four original NASA scene links required')
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(lambda url:fetch_scene(url,root),urls))
    (root/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/polar')
    args=p.parse_args()
    fetch(args.output)
