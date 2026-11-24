import unittest
import json
from unittest.mock import patch
from copy import deepcopy
import numpy as np
import torch
from agent_apis.neural import fresh_cohorts as f
from tests.test_compact_iterations import CompactTests
from agent_apis.neural.models import build_model
class FreshTests(unittest.TestCase):
 def test_thousands_of_games_before_updates_and_sampling_bound(self):
  p=json.loads(f.PROTOCOL.read_text());self.assertGreaterEqual(p['expert_games'],2000);self.assertGreaterEqual(p['completed_games_per_iteration'],2000)
  self.assertEqual(p['positions_per_game'],64);self.assertTrue(p['select_best_validation']);self.assertEqual(p['weighting'],'W4')
 def test_demonstration_sampling_is_repeatable_and_pre_action(self):
  r=CompactTests().journal()
  for e in r['events']:e.pop('v',None);e['demonstration']=True
  def capture(rows,winner):return rows
  with patch.object(f.selfplay,'completed_rows',side_effect=capture):
   a=f.derive(r,5,11);b=f.derive(r,5,11)
  self.assertEqual(len(a),5)
  for x,y in zip(a,b):
   np.testing.assert_array_equal(x['features'],y['features']);np.testing.assert_array_equal(x['policy'],y['policy'])
   self.assertEqual(x['policy'].sum(),1);self.assertTrue(x['legal'][x['policy'].argmax()])
 def test_random_initialization_is_seeded_without_loading_state_dict(self):
  torch.manual_seed(901);a=build_model('XD-CNN')
  torch.manual_seed(901);b=build_model('XD-CNN')
  for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k])
if __name__=='__main__':unittest.main()
