"""Source solar direction from LROC subsolar coordinates, never north azimuth.

Compute geometric Sun azimuth clockwise from local north and elevation above
the spherical local horizon at each raster crop center. The public product page
rounds subsolar coordinates to 0.01 degree. This is not terrain illumination or
an independent SPICE reprocessing; it inherits archive geometry uncertainty.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform as project
from dataset import sha256


class Cells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag == 'td':
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        # The archive page closes some label td tags with th.
        if tag in ('td', 'th') and self.current is not None:
            self.cells.append(''.join(self.current).strip())
            self.current = None


def parse_product_page(text, product):
    if f'/{product}' not in text:
        raise ValueError('Archive product identity mismatch')
    parser = Cells()
    parser.feed(text)
    output = {}
    for label in ['Sub solar latitude', 'Sub solar longitude', 'Incidence angle', 'North azimuth', 'Sub solar azimuth']:
        index = parser.cells.index(label)
        output[label] = float(parser.cells[index+1])
    if not all(np.isfinite(v) for v in output.values()):
        raise ValueError('Nonfinite geometry')
    return output


def local_sun(latitude, longitude, sun_latitude, sun_longitude):
    lat, lon, slat, slon = np.deg2rad([latitude, longitude, sun_latitude, sun_longitude])
    sun = np.array([np.cos(slat)*np.cos(slon), np.cos(slat)*np.sin(slon), np.sin(slat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    azimuth = np.rad2deg(np.arctan2(sun@east, sun@north)) % 360
    elevation = np.rad2deg(np.arcsin(np.clip(sun@up, -1, 1)))
    return float(azimuth), float(elevation)


def enrich(manifest, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    def one(asset):
        product = asset['product_id']
        if not re.fullmatch(r'M\d+[LR]E', product):
            raise ValueError('Unexpected LROC product ID')
        url = f'https://data.lroc.imldi.com/lroc/view_lroc/LRO-L-LROC-2-EDR-V1.0/{product}'
        target = cache/(product+'.html')
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            parse_product_page(data.decode(), product)
            target.write_bytes(data)
        values = parse_product_page(target.read_text(), product)
        left, bottom, right, top = asset['bounds_m']
        longitude, latitude = project(CRS.from_wkt(asset['crs_wkt']),
            CRS.from_proj4('+proj=longlat +R=1737400 +no_defs'), [(left+right)/2], [(bottom+top)/2])
        az, el = local_sun(latitude[0], longitude[0], values['Sub solar latitude'], values['Sub solar longitude'])
        asset.update(solar_azimuth_deg=az, solar_elevation_deg=el, crop_center_latitude_deg=latitude[0],
            crop_center_longitude_deg=longitude[0], solar_geometry=dict(source_url=url,
                source_sha256=sha256(target), source_subsolar_latitude_deg=values['Sub solar latitude'],
                source_subsolar_longitude_deg=values['Sub solar longitude'],
                source_coordinate_resolution_deg=0.01, archive_north_azimuth_deg=values['North azimuth'],
                archive_image_subsolar_azimuth_deg=values['Sub solar azimuth'],
                convention='Clockwise from local north at crop center; spherical geometric horizon; distant Sun approximation',
                warning='Rounded archive subsolar metadata; not SPICE recomputation or terrain-shadow geometry'))
        print(product, f'azimuth={az:.2f}, elevation={el:.2f}', flush=True)
    list(ThreadPoolExecutor(max_workers=4).map(one, manifest['assets']))
    manifest['solar_metadata_retrieved_utc'] = datetime.now(timezone.utc).isoformat()
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', default='data/metadata/lroc')
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = enrich(json.loads(Path(args.manifest).read_text()), args.cache)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
