"""Reject bad range replies before reading, and cover only assigned pilot tiles."""
import struct
import pytest

import cache_north_pilot_tiles_v33 as cache
from north_terrain_pilot_v32 import pilot_plan


def header():
    b=bytearray(131072);b[:8]=b'II+\x00\x08\x00\x00\x00';struct.pack_into('<Q',b,8,200)
    tags={256:(3,1,43280),257:(3,1,43280),258:(3,1,32),259:(3,1,8),
          322:(3,1,512),323:(3,1,512),324:(16,7225,4346),325:(4,7225,62146)}
    struct.pack_into('<Q',b,200,len(tags))
    for i,(tag,(kind,n,value))in enumerate(tags.items()):struct.pack_into('<HHQQ',b,208+20*i,tag,kind,n,value)
    struct.pack_into('<7225Q',b,4346,*[200000+i*600000 for i in range(7225)])
    struct.pack_into('<7225I',b,62146,*([500000]*7225))
    return bytes(b)


def test_tile_ranges_cover_exact_original_native_windows():
    _,sites=pilot_plan();r=cache.tile_ranges(header(),sites)
    expected=set()
    for site in sites:
        row,col,h,w=site['native_window']
        # Independently enumerate every source row/column boundary in window.
        rr={x//512 for x in range(row,row+h)};cc={x//512 for x in range(col,col+w)}
        expected.update(a*85+b for a in rr for b in cc)
    assert {a['tile']for a in r}==expected and len(r)==100


def test_reserve_tile_cache_is_forbidden():
    p,_=pilot_plan()
    with pytest.raises(ValueError):cache.tile_ranges(header(),[next(s for s in p['sites']if s['split']=='final_test')])


@pytest.mark.parametrize('status,content_range,etag',[(200,'bytes 0-3/5222983620',cache.EXPECTED_ETAG),
    (206,'bytes 4-7/5222983620',cache.EXPECTED_ETAG),(206,'bytes 0-3/5222983620','changed')])
def test_bad_range_replies_never_read_body(monkeypatch,status,content_range,etag):
    class Reply:
        headers={'Content-Range':content_range,'Content-Length':'4','ETag':etag}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):raise AssertionError('Bad response body must not be read')
    reply=Reply();reply.status=status;monkeypatch.setattr(cache,'urlopen',lambda *a,**k:reply)
    with pytest.raises(ValueError):cache.strict_range(0,4)


def test_truncated_exact_range_is_rejected(monkeypatch):
    class Reply:
        status=206;headers={'Content-Range':'bytes 0-3/5222983620','Content-Length':'4','ETag':cache.EXPECTED_ETAG}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):return b'abc'
    monkeypatch.setattr(cache,'urlopen',lambda *a,**k:Reply())
    with pytest.raises(ValueError):cache.strict_range(0,4)
