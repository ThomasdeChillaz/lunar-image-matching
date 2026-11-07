import unittest
from compare_north_fine_v58 import paired_difference,pair_rows


def row(i,group,correct):
 return dict(pair_id=str(i),component=group,tile_id=group,light_a=0,light_b=0,query_view=1,positive=True,pose_correct=correct)


class PairedChecks(unittest.TestCase):
 def test_no_change_exact_zero(self):
  a=[row(i,str(i//3),bool(i%2))for i in range(12)];r=paired_difference(a,a,100)
  self.assertEqual(r['difference_pp'],0);self.assertEqual(r['terrain_bootstrap_95_interval_pp'],[0,0])
 def test_constant_effect(self):
  a=[row(i,str(i//3),False)for i in range(12)];b=[dict(r,pose_correct=True)for r in a];r=paired_difference(a,b,100)
  self.assertEqual(r['difference_pp'],100);self.assertEqual(r['terrain_bootstrap_95_interval_pp'],[100,100]);self.assertEqual(r['gained_cases'],12)
 def test_swapping_is_antisymmetric(self):
  a=[row(i,str(i//3),i%3==0)for i in range(12)];b=[dict(r,pose_correct=i%2==0)for i,r in enumerate(a)]
  x=paired_difference(a,b,100);y=paired_difference(b,a,100)
  self.assertEqual(x['difference_pp'],-y['difference_pp']);self.assertEqual(x['gained_cases'],y['lost_cases'])
 def test_rejects_missing_duplicate_and_reordered_cases(self):
  a=[row(i,'a',True)for i in range(3)]
  for b in (a[:2],a[::-1],[a[0],a[0],a[2]]):
   with self.assertRaises(ValueError):pair_rows(a,b)


if __name__=='__main__':unittest.main()
