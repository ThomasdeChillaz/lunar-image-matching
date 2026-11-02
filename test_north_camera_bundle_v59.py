import unittest
from north_camera_bundle_v59 import make_cases


def tiles():
 return [dict(tile_id=f'{c}/{family}-{i}',component=c,split='pilot_camera_control',family=family,pair_index=i,support=dict(x=x,y=0))for c,x in (('a',0),('b',30000),('c',60000))for family in ('training_motion','unseen_roll')for i in (0,1)]


class CameraCaseControls(unittest.TestCase):
 def test_complete_deterministic_case_lists_and_distinct_endpoints(self):
  t=tiles();positive,negative=make_cases(t);self.assertEqual(len(positive),48);self.assertEqual(len(negative),48);self.assertEqual((positive,negative),make_cases(t[::-1]));index={x['tile_id']:x for x in t}
  for a,b in zip(positive,negative):
   self.assertNotEqual(a['component'],b['query_component']);self.assertEqual(a['family'],index[b['query_tile_id']]['family']);self.assertEqual(a['light_b'],b['light_b'])
 def test_forbids_shared_terrain_support(self):
  t=tiles()
  for r in t:r['support']=dict(x=0,y=0)
  with self.assertRaisesRegex(ValueError,'share source support'):make_cases(t)
 def test_rejects_training_and_single_endpoint(self):
  t=tiles();t[0]['split']='train'
  with self.assertRaisesRegex(ValueError,'Pilot-only'):make_cases(t)
  with self.assertRaisesRegex(ValueError,'distinct terrain'):make_cases([r for r in tiles()if r['component']=='a'])


if __name__=='__main__':unittest.main()
