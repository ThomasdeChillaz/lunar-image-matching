"""Strict LOLA PDS3 cylindrical HEIGHT metadata and bounded byte-range reads.

This reads a reference terrain map, not camera ray depth or an optical image.
A complete source-file checksum, renderer registration and visibility remain
separate gates. No matcher or camera-pose estimator is imported here.
"""
from dataclasses import dataclass
import hashlib
import math
import re
import struct
from urllib.request import Request, urlopen

BASE = 'https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/'


def field(text, name):
    matches = re.findall(r'^\s*'+re.escape(name)+r'\s*=\s*([^\n]+)', text, re.M)
    if len(matches) != 1:
        raise ValueError('Expected exactly one label field '+name)
    return matches[0].split('<',1)[0].strip().strip('"')


@dataclass(frozen=True)
class HeightTile:
    filename: str
    rows: int
    columns: int
    pixels_per_degree: float
    north: float
    south: float
    west: float
    east: float
    scale_m: float
    reference_radius_m: float
    frame: str

    @property
    def bytes(self):
        return self.rows*self.columns*2

    def pixel(self, latitude, longitude):
        """Integer-centered native map pixels; no latitude clamping or tile wrap."""
        if not all(math.isfinite(v) for v in (latitude,longitude)):
            raise ValueError('Finite coordinates required')
        if not (self.south < latitude < self.north and self.west < longitude < self.east):
            raise ValueError('Coordinate must be inside the explicit tile; route boundaries separately')
        return ((self.north-latitude)*self.pixels_per_degree-.5,
                (longitude-self.west)*self.pixels_per_degree-.5)

    def offset(self, row, column, width=1):
        if any(type(v) is not int for v in (row,column,width)) or not (0<=row<self.rows and 0<=column<self.columns and 1<=width<=self.columns-column):
            raise ValueError('Valid integer row/column/contiguous width required')
        return 2*(row*self.columns+column)

    def radii(self, data):
        if not data or len(data)%2:
            raise ValueError('Complete nonempty signed16-bit samples required')
        return [dn*self.scale_m+self.reference_radius_m for (dn,) in struct.iter_unpack('<h',data)]


def parse_label(text):
    if (field(text,'PDS_VERSION_ID')!='PDS3'
            or field(text,'MAP_PROJECTION_TYPE')!='SIMPLE CYLINDRICAL'
            or field(text,'SAMPLE_TYPE')!='LSB_INTEGER'
            or field(text,'SAMPLE_BITS')!='16' or field(text,'UNIT')!='METER'
            or field(text,'POSITIVE_LONGITUDE_DIRECTION')!='EAST'
            or field(text,'COORDINATE_SYSTEM_TYPE')!='BODY-FIXED ROTATING'
            or field(text,'MAP_PROJECTION_ROTATION')!='0.0'):
        raise ValueError('Unsupported map/sample/coordinate convention')
    filename=field(text,'FILE_NAME')
    if not re.fullmatch(r'LDEM_[A-Z0-9_]+\.IMG', filename) or field(text,'^IMAGE')!=filename:
        raise ValueError('Expected explicit detached LDEM image filename')
    frame=field(text,'COORDINATE_SYSTEM_NAME')
    if frame!='MEAN EARTH/POLAR AXIS OF DE421':
        raise ValueError('Unverified lunar reference frame')
    tile=HeightTile(filename,int(field(text,'LINES')),int(field(text,'LINE_SAMPLES')),
        float(field(text,'MAP_RESOLUTION')),float(field(text,'MAXIMUM_LATITUDE')),
        float(field(text,'MINIMUM_LATITUDE')),float(field(text,'WESTERNMOST_LONGITUDE')),
        float(field(text,'EASTERNMOST_LONGITUDE')),float(field(text,'SCALING_FACTOR')),
        float(field(text,'OFFSET')),frame)
    if (not all(math.isfinite(v) for v in (tile.pixels_per_degree,tile.north,tile.south,tile.east,tile.west,tile.scale_m,tile.reference_radius_m))
            or tile.rows<=0 or tile.columns<=0 or tile.pixels_per_degree<=0
            or tile.scale_m!=.5 or tile.reference_radius_m!=1737400.
            or not (-90<=tile.south<tile.north<=90 and 0<=tile.west<tile.east<=360)
            or (tile.north-tile.south)*tile.pixels_per_degree!=tile.rows
            or (tile.east-tile.west)*tile.pixels_per_degree!=tile.columns
            or int(field(text,'FILE_RECORDS'))!=tile.rows
            or int(field(text,'RECORD_BYTES'))!=tile.columns*2):
        raise ValueError('Inconsistent tile dimensions or radius units')
    for axis in ('A_AXIS_RADIUS','B_AXIS_RADIUS','C_AXIS_RADIUS'):
        if float(field(text,axis))!=1737.4:raise ValueError('Unsupported reference sphere')
    centre_lat,centre_lon=float(field(text,'CENTER_LATITUDE')),float(field(text,'CENTER_LONGITUDE'))
    if (centre_lat!=0. or centre_lon!=180.
            or float(field(text,'LINE_PROJECTION_OFFSET'))!=(tile.north-centre_lat)*tile.pixels_per_degree-.5
            or float(field(text,'SAMPLE_PROJECTION_OFFSET'))!=(centre_lon-tile.west)*tile.pixels_per_degree-.5
            or int(field(text,'LINE_FIRST_PIXEL'))!=1 or int(field(text,'SAMPLE_FIRST_PIXEL'))!=1
            or int(field(text,'LINE_LAST_PIXEL'))!=tile.rows or int(field(text,'SAMPLE_LAST_PIXEL'))!=tile.columns):
        raise ValueError('Unsupported map-origin or pixel-registration convention')
    return tile


def bounded_range(url, start, length, total, opener=urlopen):
    """Reject ignored ranges before reading the body; bind every returned byte."""
    if any(type(v) is not int for v in (start,length,total)) or not (0<=start<total and 1<=length<=65536 and start+length<=total):
        raise ValueError('Explicit valid range <=64KiB required')
    request=Request(url,headers={'Range':f'bytes={start}-{start+length-1}',
        'Accept-Encoding':'identity','User-Agent':'lunar-reference-geometry/30'})
    with opener(request,timeout=30) as response:
        expected=f'bytes {start}-{start+length-1}/{total}'
        if response.status!=206 or response.headers.get('Content-Range')!=expected:
            raise ValueError('Server did not honor the exact declared range')
        if response.headers.get('Content-Encoding','identity')!='identity':
            raise ValueError('Compressed HTTP transfer changes byte coordinates')
        if response.headers.get('Content-Length') is not None and int(response.headers['Content-Length'])!=length:
            raise ValueError('Range length header differs')
        data=response.read(length+1)
        if len(data)!=length:raise ValueError('Truncated/oversized map range')
        trace=dict(url=url,start=start,length=length,total_bytes=total,
            sha256=hashlib.sha256(data).hexdigest(),content_range=expected,
            etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'),
            complete_source_checksum_verified=False)
    return data,trace


def row_samples(tile,row,column,width=1,opener=urlopen):
    offset=tile.offset(row,column,width)
    data,trace=bounded_range(BASE+tile.filename,offset,2*width,tile.bytes,opener)
    return tile.radii(data),data,trace
