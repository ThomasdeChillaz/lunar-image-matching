import copy,unittest
from north_camera_variation_v59 import protocol
from north_camera_contract_v59 import verify_schedule


class CameraContractControls(unittest.TestCase):
 def setUp(self):self.plan=protocol([dict(component='a',split='renderer_cost_pilot')],{'a':0})
 def test_tolerates_only_negligible_rotation_basis_roundoff(self):
  frozen=copy.deepcopy(self.plan);frozen['pairs'][0]['query']['cam2world'][0][0]+=2e-16
  self.assertTrue(verify_schedule(frozen,self.plan)['passed'])
 def test_rejects_translation_even_below_rotation_tolerance(self):
  frozen=copy.deepcopy(self.plan);frozen['pairs'][0]['query']['cam2world'][0][3]+=1e-13
  with self.assertRaisesRegex(ValueError,'translation'):verify_schedule(frozen,self.plan)
 def test_rejects_changed_seeded_draw(self):
  frozen=copy.deepcopy(self.plan);frozen['pairs'][0]['query']['roll_degrees']+=1e-10
  with self.assertRaisesRegex(ValueError,'roll'):verify_schedule(frozen,self.plan)
 def test_rejects_rotation_corruption_or_nonfinite_value(self):
  for value in (1.,float('nan')):
   frozen=copy.deepcopy(self.plan);frozen['pairs'][0]['query']['cam2world'][0][0]=value
   with self.assertRaisesRegex(ValueError,'rotation basis'):verify_schedule(frozen,self.plan)
 def test_rejects_missing_pair_or_undeclared_field(self):
  frozen=copy.deepcopy(self.plan);frozen['pairs'].pop()
  with self.assertRaisesRegex(ValueError,'count'):verify_schedule(frozen,self.plan)
  frozen=copy.deepcopy(self.plan);frozen['unapproved_setting']=True
  with self.assertRaisesRegex(ValueError,'fields'):verify_schedule(frozen,self.plan)


if __name__=='__main__':unittest.main()
