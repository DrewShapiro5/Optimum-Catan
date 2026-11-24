from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from catanatron.serialization import action_record_to_json
from agent_apis import benchmark
from agent_apis.neural import compact_iterations as c
from agent_apis.stochastic_search import ReferenceEvaluator,HeuristicOpponent
from agent_apis.observation import feature_names

class CompactTests(unittest.TestCase):
 def journal(self):
  pair={'pair_id':'test','board_seed':719,'simulator_seed':720,'agent_seed':721}
  g=benchmark.make_game(pair,friendly_robber=True)
  r={'pair':pair,'initial_sha256':c.snapshot_digest(g),'events':[]}
  for _ in range(30):
   a=g.playable_actions[0];i=c.selfplay.action_id(a)
   v=[[i,4]] if len(g.playable_actions)>1 else None
   e={'a':i,'event':action_record_to_json(g.execute(a))}
   if v:e['v']=v
   r['events'].append(e)
  r.update(final_sha256=c.snapshot_digest(g),winner=None,status='cutoff')
  return r
 def test_full_replay_exact_chance_rng_and_no_state_snapshots(self):
  r=self.journal();g=c.replay_journal(r)
  self.assertEqual(c.snapshot_digest(g),r['final_sha256'])
  self.assertNotIn('initial_state',r);self.assertNotIn('final_state',r)
  self.assertTrue(all(set(e)<= {'a','event','v'} for e in r['events']))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'game.json.gz';c.save_journal(p,r)
   with gzip.open(p,'rt') as f:loaded=json.load(f)
   c.replay_journal(loaded)
   self.assertLess(p.stat().st_size,len(json.dumps(r)))
 def test_chance_tampering_and_illegal_targets_are_rejected(self):
  r=self.journal();bad=deepcopy(r);bad['events'][-1]['event'][1]='tampered'
  with self.assertRaises(ValueError):c.replay_journal(bad)
  bad=deepcopy(r);bad['events'][0]['v']=[[999,3]]
  with self.assertRaises(ValueError):c.replay_journal(bad)
 def test_partial_and_cutoff_journals_cannot_supply_outcomes(self):
  r=self.journal();self.assertIsNone(c.replay_journal(r,derive=True))
  r.pop('final_sha256');r.pop('winner');r['status']='in-progress'
  c.replay_journal(r)
 def test_memory_W4_has_same_probabilities_and_value_correction(self):
  ids=[np.flatnonzero(c.w4.ACTION_FAMILY==f)[0] for f in np.unique(c.w4.ACTION_FAMILY)[:2]]
  p=np.zeros((3,332),np.float32);p[:,ids[0]]=[.9,.8,.2];p[:,ids[1]]=1-p[:,ids[0]]
  features=np.zeros((3,681),np.float32);features[:,feature_names().index('RULE_FRIENDLY_ROBBER')]=1
  chunk={'features':features,'legal':np.ones((3,332),bool),'policy':p,'value':np.array([1,0,1],np.float32)}
  memory=c.MemoryData([chunk],[{'test':'memory'}])
  with tempfile.TemporaryDirectory(dir=c.ROOT/'runs/catan-ai') as d:
   f=Path(d)/'rows.npz';np.savez_compressed(f,**chunk)
   disk=c.w4.Data([{'dataset':c.selfplay.reference(f)}])
   np.testing.assert_array_equal(memory.q,disk.q)
   self.assertAlmostEqual(float((memory.q/(3*memory.q)).sum()),1)
 def test_deadline_preserves_replayable_partial_game(self):
  w=c.Controller.__new__(c.Controller);w.core={};w.state={};w.persist=lambda:None
  w.protocol=json.loads(c.PROTOCOL.read_text());w.protocol['simulations']=4
  w.frozen=HeuristicOpponent();calls=0
  def budget():
   nonlocal calls
   calls+=1
   if calls==4:raise c.ReviewDue()
  w.budget=budget
  pair={'pair_id':'deadline','board_seed':19,'simulator_seed':20,'agent_seed':21}
  with tempfile.TemporaryDirectory(dir=c.ROOT/'runs/catan-ai') as d:
   path=Path(d)/'game.json.gz'
   with self.assertRaises(c.ReviewDue):
    w.play(path,pair,benchmark.COLORS[0],ReferenceEvaluator(),ReferenceEvaluator(),'current',[],mode='search')
   self.assertFalse(path.exists())
   with gzip.open(path.with_suffix('.partial.gz'),'rt') as f:r=json.load(f)
   self.assertEqual(r['status'],'interrupted');self.assertEqual(len(r['events']),3)
   self.assertIsNone(c.replay_journal(r,derive=True))

if __name__=='__main__':unittest.main()
