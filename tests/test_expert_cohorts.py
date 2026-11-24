import unittest
from unittest.mock import Mock, patch
from collections import Counter
from pathlib import Path
from agent_apis.neural.expert_cohorts import ExpertCohorts,split_for
from agent_apis.neural.parallel_cohorts import CohortLedger

class ExpertTests(unittest.TestCase):
    def test_splits_survive_out_of_order_completions_and_cutoffs(self):
        ledger=CohortLedger(2000,4000)
        make=lambda attempt,slot:{'split':split_for(slot),'seed':attempt}
        active=set()
        for _ in range(12):active.add(ledger.reserve(active,make)['attempt'])
        for attempt in reversed(sorted(active)):
            ledger.finish(attempt,'cutoff' if attempt==0 else 'completed')
        while len(ledger.completed_slots)<2000:
            a=ledger.reserve(set(),make);ledger.finish(a['attempt'],'completed')
        counts=Counter(a['split'] for a in ledger.assignments.values() if a['status']=='completed')
        self.assertEqual(counts,{'train':1800,'validation':100,'test':100})
        self.assertIsNone(ledger.reserve(set(),make))

    def test_no_training_on_incomplete_cohort(self):
        w=ExpertCohorts.__new__(ExpertCohorts)
        w.state={'iterations':{'1':{'status':'collecting','parent':{},'completed_games':1999}}}
        w.parent={};w.guard=Mock();w.smoke=False
        w.collect=Mock(return_value=(Path('/tmp'),{}));w.train_w4=Mock();w.load_samples=Mock()
        with self.assertRaisesRegex(ValueError,'2000 completed'):w.execute()
        w.train_w4.assert_not_called();w.load_samples.assert_not_called()

    def test_completed_cohort_resolves_paths_and_excludes_test_rows(self):
        from agent_apis.experiment import ROOT
        from agent_apis.neural import friendly_w4 as w4
        w=ExpertCohorts.__new__(ExpertCohorts)
        w.state={'iterations':{'1':{'status':'collecting','parent':{},'completed_games':2000}}}
        w.parent={};w.guard=Mock();w.smoke=False;w.persist=Mock()
        groups={k:[{'path':k+'.json.gz','sha256':'sha'}] for k in ('train','validation','test')}
        w.collect=Mock(return_value=(ROOT/'unused',groups))
        data=object();cache=object();w.load_samples=Mock(side_effect=[data,cache])
        w.train_w4=Mock(side_effect=RuntimeError('training reached'))
        with patch('agent_apis.neural.expert_cohorts.file_digest',return_value='sha'), patch.object(w4,'CORPUS'), patch.object(w4,'OUTPUT'):
            with self.assertRaisesRegex(RuntimeError,'training reached'):w.execute()
        self.assertEqual(w.load_samples.call_args_list[0].args[0],[ROOT/'train.json.gz'])
        self.assertEqual(w.load_samples.call_args_list[1].args[0],[ROOT/'validation.json.gz'])
        self.assertEqual(w.load_samples.call_count,2)
        w.train_w4.assert_called_once_with(data,data)

if __name__=='__main__':unittest.main()
