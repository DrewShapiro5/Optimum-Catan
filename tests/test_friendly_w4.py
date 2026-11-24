import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from agent_apis.neural import friendly_w4 as w4
from agent_apis.neural.recipes import balanced_mass
from agent_apis.observation import feature_names

class W4Tests(unittest.TestCase):
 def data(self,policy):
  directory=tempfile.TemporaryDirectory(dir=w4.ROOT/'runs/catan-ai');self.addCleanup(directory.cleanup)
  path=Path(directory.name)/'rows.npz';n=len(policy)
  features=np.zeros((n,681),np.float32);features[:,feature_names().index('RULE_FRIENDLY_ROBBER')]=1
  np.savez_compressed(path,features=features,legal=np.ones((n,332),bool),policy=policy,value=np.arange(n,dtype=np.float32)%2)
  return w4.Data([{'dataset':w4.selfplay.reference(path),'split':'train'},{'dataset':None,'split':'train'}],'train')
 def test_one_hot_exactly_matches_existing_W4_and_value_correction(self):
  labels=np.array([0]*90+[100]*9+[200])
  # Pick three different action families independent of vocabulary numbering.
  ids=[np.flatnonzero(w4.ACTION_FAMILY==f)[0] for f in np.unique(w4.ACTION_FAMILY)[:3]]
  labels=np.array([ids[0]]*90+[ids[1]]*9+[ids[2]])
  p=np.eye(332,dtype=np.float32)[labels];d=self.data(p)
  counts=np.array([90,9,1]);mass=balanced_mass(counts)
  expected=np.array([.5/100+.5*mass[i]/counts[i] for i,c in enumerate(counts) for _ in range(c)])
  np.testing.assert_allclose(d.q,expected)
  self.assertLessEqual((d.q*100).max(),5+1e-6)
  np.testing.assert_allclose(np.sum(d.q*d.arrays['value']/(100*d.q)),d.arrays['value'].mean())
  sampled=d.sample(np.random.default_rng(4),300)
  np.testing.assert_allclose(sampled['policy'].sum(1),1,atol=1e-6)
  self.assertTrue((sampled['policy'].max(1)==1).all())
 def test_soft_policy_joint_mass_is_balanced_and_normalized(self):
  ids=[np.flatnonzero(w4.ACTION_FAMILY==f)[0] for f in np.unique(w4.ACTION_FAMILY)[:2]]
  p=np.zeros((3,332),np.float32);p[:,ids[0]]=[.9,.8,.2];p[:,ids[1]]=1-p[:,ids[0]]
  d=self.data(p);adjusted=p*d.factors[w4.ACTION_FAMILY][None,:]/d.q[:,None]
  np.testing.assert_allclose(adjusted.sum(1),1)
  expected=p*d.factors[w4.ACTION_FAMILY][None,:]
  np.testing.assert_allclose(d.q[:,None]*adjusted,expected)
  self.assertAlmostEqual(float(np.sum(d.q/(3*d.q))),1)
 def test_value_correction_applied_once_without_sample_renormalization(self):
  logits=torch.zeros((2,3),requires_grad=True);value=torch.tensor([.2,-.7],requires_grad=True)
  legal=torch.tensor([[True,True,False],[True,False,False]])
  target=torch.tensor([[.3,.7,0],[1.,0,0]])
  outcome=torch.tensor([1.,0.]);weights=torch.tensor([.4,1.8])
  pl,vl=w4.losses(logits,value,legal,target,outcome,weights)
  expected=(torch.nn.functional.binary_cross_entropy_with_logits(value,outcome,reduction='none')*weights).mean()
  torch.testing.assert_close(vl,expected);(pl+vl).backward();self.assertEqual(logits.grad[0,2],0)
 def test_cutoff_outcomes_cannot_enter_dataset(self):
  p=np.eye(332,dtype=np.float32)[:2];d=self.data(p)
  self.assertEqual(len(d.q),2)

if __name__=='__main__':unittest.main()
