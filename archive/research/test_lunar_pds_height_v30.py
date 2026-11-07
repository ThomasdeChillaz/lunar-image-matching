"""No-network/no-neural controls for reference-map metadata, bytes and units."""
import io
from pathlib import Path
import struct
import unittest

from lunar_pds_height_v30 import HeightTile, bounded_range, parse_label


class Response:
    def __init__(self,data=b'\x02\x00',status=206,headers=None):
        self.data=io.BytesIO(data);self.status=status
        self.headers={'Content-Range':'bytes 4-5/100','Content-Length':'2'} if headers is None else headers
        self.read_called=False
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):self.read_called=True;return self.data.read(n)


class PdsHeightControls(unittest.TestCase):
    def setUp(self):
        self.tile=HeightTile('LDEM_64.IMG',11520,23040,64,90,-90,0,360,.5,1737400.,'MEAN EARTH/POLAR AXIS OF DE421')

    def test_half_pixel_and_east_longitude(self):
        self.assertEqual(self.tile.pixel(90-.5/64,.5/64),(0.,0.))
        self.assertEqual(self.tile.pixel(-90+.5/64,360-.5/64),(11519.,23039.))
        self.assertEqual(self.tile.pixel(0,180),(5759.5,11519.5))

    def test_no_silent_wrap_or_clamp(self):
        for lat,lon in [(90,180),(-90,180),(0,-1),(0,360),(float('nan'),0)]:
            with self.assertRaises(ValueError):self.tile.pixel(lat,lon)

    def test_signed_little_endian_radius_not_height(self):
        self.assertEqual(self.tile.radii(struct.pack('<hhh',-2000,0,2000)),[1736400.,1737400.,1738400.])
        with self.assertRaises(ValueError):self.tile.radii(b'\x00')

    def test_exact_byte_offsets_and_bounds(self):
        self.assertEqual(self.tile.offset(1,2,3),46084)
        for args in [(0,23040),(0,-1),(11520,0),(0,1,23040),(0.,0)]:
            with self.assertRaises(ValueError):self.tile.offset(*args)

    def test_honored_range_binds_all_bytes(self):
        response=Response()
        def opener(request,timeout):
            self.assertEqual(request.headers['Range'],'bytes=4-5');self.assertEqual(timeout,30)
            return response
        data,trace=bounded_range('https://example.invalid/map',4,2,100,opener)
        self.assertEqual(data,b'\x02\x00');self.assertFalse(trace['complete_source_checksum_verified'])

    def test_ignored_or_wrong_range_rejected_before_body(self):
        for status,header in [(200,{}),(206,{'Content-Range':'bytes 0-1/100'}),(206,{'Content-Range':'bytes 4-5/101'}),(206,{'Content-Range':'bytes 4-5/100','Content-Encoding':'gzip'})]:
            response=Response(status=status,headers=header)
            with self.assertRaises(ValueError):bounded_range('https://example.invalid/map',4,2,100,lambda *a,**k:response)
            self.assertFalse(response.read_called)

    def test_truncation_oversize_and_budget(self):
        for data in [b'\x00',b'\x00'*3]:
            with self.assertRaises(ValueError):bounded_range('https://example.invalid/map',4,2,100,lambda *a,**k:Response(data))
        for args in [(0,65537,100000),(-1,1,100),(99,2,100)]:
            with self.assertRaises(ValueError):bounded_range('https://example.invalid/map',*args)




class OfficialLabelAndRangeControls(unittest.TestCase):
    def setUp(self):
        import json
        self.fixture=json.loads(Path('pds_height_fixtures_v30.json').read_text())

    def test_official_native_grid_and_range_samples(self):
        import base64,hashlib,math
        for source in self.fixture['official_labels'].values():
            self.assertEqual(hashlib.sha256(source['label'].encode()).hexdigest(),source['label_sha256'])
            tile=parse_label(source['label']);r,c=tile.pixel(self.fixture['protocol']['latitude'],self.fixture['protocol']['longitude'])
            self.assertEqual(source['samples'][0]['row'],math.floor(r));self.assertEqual(source['samples'][0]['column'],math.floor(c))
            for sample in source['samples']:
                raw=base64.b64decode(sample['data_base64'],validate=True)
                self.assertEqual(tile.radii(raw),sample['planetary_radius_m'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),sample['trace']['sha256'])
                self.assertEqual(tile.offset(sample['row'],sample['column'],2),sample['trace']['start'])

    def test_real_labels_reject_wrong_datum_and_registration(self):
        for source in self.fixture['official_labels'].values():
            label=source['label']
            for original,replacement in [('LSB_INTEGER','MSB_INTEGER'),('SCALING_FACTOR        = 0.5','SCALING_FACTOR        = 1.0'),('OFFSET                = 1737400.','OFFSET                = 0.'),('CENTER_LONGITUDE             = 180','CENTER_LONGITUDE             = 0'),('LINE_PROJECTION_OFFSET       =','LINE_PROJECTION_OFFSET       = 99 #')]:
                changed=label.replace(original,replacement)
                self.assertNotEqual(changed,label)
                with self.assertRaises(ValueError):parse_label(changed)
        extra=self.fixture['unused_label']['label']
        t=parse_label(extra);self.assertEqual(t.pixel(45-.5/512,.5/512),(0.,0.))


class RimMetadataGeometryControls(unittest.TestCase):
    def test_independent_cardinal_crater_rim_directions(self):
        import math
        from probe_cresent_lola_rims_v30 import rim_points
        radius=1737400.
        rr,points=rim_points([radius,0.,0.,1000.,500.,0.],4)
        self.assertEqual(rr,radius)
        self.assertAlmostEqual(points[0][0],0.);self.assertAlmostEqual(points[0][1],math.degrees(math.atan2(1000.,radius)))
        self.assertAlmostEqual(points[1][0],math.degrees(math.atan2(500.,radius)));self.assertAlmostEqual(points[1][1],0.)
        _,turned=rim_points([radius,0.,0.,1000.,500.,math.pi/2],4)
        self.assertAlmostEqual(turned[0][0],math.degrees(math.atan2(1000.,radius)))
        self.assertAlmostEqual(turned[1][1],360-math.degrees(math.atan2(500.,radius)))
        with self.assertRaises(ValueError):rim_points([0.,0.,radius,1000.,500.,0.])

    def test_bilinear_radius_and_explicit_tile_edge(self):
        from probe_cresent_lola_rims_v30 import bilinear_radius
        tile=HeightTile('LDEM_TEST.IMG',2,2,1.,2.,0.,0.,2.,.5,1737400.,'fixture')
        data=struct.pack('<hhhh',0,2,4,6)
        self.assertEqual(bilinear_radius(tile,data,1.,1.),1737401.5)
        self.assertEqual(bilinear_radius(tile,data,1.5,.5),1737400.)
        with self.assertRaises(ValueError):bilinear_radius(tile,data,.1,1.9)


if __name__=='__main__':unittest.main()
