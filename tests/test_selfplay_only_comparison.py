import unittest
import numpy as np
import torch
from agent_apis.neural.selfplay_only_comparison import corrected_evaluate
from agent_apis.neural.train import evaluate

class Cache:
 manifest={'families':['example']}
 def __init__(self,soft):
  self.arrays={'features':np.zeros((2,681),np.float32),'legal':np.ones((2,2),bool),'policy':np.tile(np.array(soft,np.float32),(2,1)),'action':np.zeros(2,np.int64),'value':np.array([0.,1.],np.float32),'phase':np.ones(2,np.uint8),'family':np.zeros(2,np.int64)}
 def indices(self,split):return np.arange(2)
 def batch(self,ids):return {k:v[ids].copy() for k,v in self.arrays.items()}
class Model(torch.nn.Module):
 def forward(self,x):return torch.tensor([[2.,0.]]).repeat(len(x),1),torch.zeros(len(x))
class MetricTests(unittest.TestCase):
 def test_onehot_preserves_historical_loss(self):
  a=evaluate(Model(),Cache([1.,0.]),'validation','cpu');b=corrected_evaluate(Model(),Cache([1.,0.]),'validation','cpu')
  self.assertAlmostEqual(a['selection_loss'],b['selection_loss'])
 def test_soft_targets_are_not_reduced_to_argmax(self):
  r=corrected_evaluate(Model(),Cache([.6,.4]),'validation','cpu')
  expected=float(-(torch.tensor([.6,.4])*torch.log_softmax(torch.tensor([2.,0.]),0)).sum())
  self.assertAlmostEqual(r['policy_nll'],expected,places=6)
  self.assertGreater(r['policy_nll'],r['hard_policy_nll'])
  self.assertAlmostEqual(r['selection_loss'],r['policy_nll']+r['value_bce'])
if __name__=='__main__':unittest.main()
