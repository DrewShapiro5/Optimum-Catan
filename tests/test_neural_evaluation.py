import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from agent_apis.neural import evaluation as e
from agent_apis.experiment import ROOT


class EvaluationTests(unittest.TestCase):
    def test_unequal_schedules_use_complete_common_pairs_only(self):
        a=[{'pair_id':str(i),'candidate_color':c} for i in range(5) for c in ('BLUE','RED')]
        b=a[:7]
        shared=e.common_prefix([a,b])
        self.assertEqual([len(x) for x in shared],[6,6])
        self.assertNotIn('3',{x['pair_id'] for x in shared[0]})
    def test_all_futile_finishes_negative_result_and_still_tests_every_search(self):
        with TemporaryDirectory(dir=ROOT/'runs/catan-ai') as temp:
            folder=Path(temp);calls=[];trained=[];written={}
            class Fixture(e.Evaluation):
                def __init__(self):self.core={};self.state={'selection_rule':'fixture'}
                def persist(self):pass
                def trained(self,model,seed,*args,**kwargs):
                    trained.append((model,seed));return folder/'training.json',{
                        'checkpoint':None,'elapsed_seconds':1,'metrics':{'training':{'parameters':100}}}
                def measure(self,model,seed,mode,final=False):
                    calls.append((model,seed,mode,final));path=folder/f'{model}-{mode}'/'run.json';path.parent.mkdir()
                    path.write_text('{}')
                    rows=[{'pair_id':str(i),'candidate_color':c,'board_sha256':str(i),'simulator_seed':i,'match_score':0.}
                          for i in range(20) for c in ('BLUE','RED')]
                    (path.parent/'games.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                    return path,{'status':'stopped-early','historical_import':{'fixture':True},'metrics':{'benchmark':{
                        'futility':{'stopped':True},'failure_gate_passed':True,'timing_gate_passed':True,
                        'cutoff_gate_passed':True,'games':40,'wins':0,'overall_match_score':0.,'match_score_ci95':[None,None]}}}
                def report(self,*args):pass
            def capture(path,value):written[path.name]=value
            with patch.object(e,'write',side_effect=capture):Fixture().finish()
            result=written['PHASE5_RESULTS.json']
            self.assertTrue(result['evaluation_complete']);self.assertEqual(result['finalists'],[])
            self.assertIsNone(result['provisional_choice']);self.assertFalse(result['confirmation_performed'])
            self.assertEqual(len(calls),8);self.assertTrue(all(seed==11 for _,seed in trained))
            self.assertEqual(calls[:4],[(m,11,'policy',False) for m in e.MODEL_IDS])
            self.assertEqual(calls[4:],[(m,11,'search',False) for m in e.MODEL_IDS])
            self.assertFalse(result['final_evaluation_seeds_used'])

    def test_strong_search_survives_futile_policy_and_gets_confirmation(self):
        with TemporaryDirectory(dir=ROOT/'runs/catan-ai') as temp:
            folder=Path(temp);calls=[];trained=[];written={}
            strengths={'MLP':(0.,0.),'XD-CNN':(.75,.75),'H-GNN':(0.,1.),'TRANSFORMER':(0.,0.)}
            class Fixture(e.Evaluation):
                def __init__(self):self.core={};self.state={'selection_rule':'fixture'}
                def persist(self):pass
                def trained(self,model,seed,*args,**kwargs):
                    trained.append((model,seed));return folder/'training.json',{
                        'checkpoint':None,'elapsed_seconds':1,'configuration':{'content':{'search':{}}},
                        'metrics':{'training':{'parameters':{'MLP':10,'XD-CNN':20,'H-GNN':30,'TRANSFORMER':40}[model]}}}
                def measure(self,model,seed,mode,final=False):
                    calls.append((model,seed,mode,final));path=folder/f'{model}-{seed}-{mode}-{final}'/'run.json';path.parent.mkdir()
                    path.write_text('{}');score=strengths[model][mode=='search']
                    rows=[{'pair_id':str(i),'candidate_color':c,'board_sha256':str(i),'simulator_seed':i,'match_score':score}
                          for i in range(20) for c in ('BLUE','RED')]
                    (path.parent/'games.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                    return path,{'status':'stopped-early' if score==0 else 'completed','historical_import':{'fixture':True},
                        'metrics':{'benchmark':{'futility':{'stopped':score==0},'failure_gate_passed':True,
                        'timing_gate_passed':True,'cutoff_gate_passed':True,'games':40,'wins':int(score*40),
                        'overall_match_score':score,'match_score_ci95':[None,None] if score==0 else [score,score]}}}
                def report(self,*args):pass
            with patch.object(e,'write',side_effect=lambda path,value:written.update({path.name:value})):Fixture().finish()
            result=written['PHASE5_RESULTS.json']
            self.assertEqual(result['finalists'],['H-GNN','XD-CNN'])
            self.assertEqual(result['provisional_choice']['model'],'H-GNN')
            self.assertEqual(result['provisional_choice']['mode'],'search')
            for model in ('H-GNN','XD-CNN'):
                self.assertTrue(all((model,seed) in trained for seed in (11,23,47)))
                self.assertTrue(all((model,11,mode,True) in calls for mode in ('policy','search')))
            self.assertFalse(any(seed!=11 for model,seed in trained if model in ('MLP','TRANSFORMER')))
            self.assertEqual(result['search_promotion_comparison']['ci95'],[1.,1.])


if __name__=='__main__':unittest.main()

class CompatibilityTests(unittest.TestCase):
    def test_only_orchestration_hashes_can_differ(self):
        import copy
        a={'implementation_sha256':{'agent_apis/neural/evaluation.py':'old','agent_apis/neural/pipeline.py':'old',
             'agent_apis/neural/provider.py':'same','agent_apis/futility.py':'same'},
           'candidate':{'checkpoint':'fixed'},'evaluation':{'budget_ms':500,'tier':'smoke'}}
        b=copy.deepcopy(a);b['implementation_sha256']['agent_apis/neural/evaluation.py']='new'
        b['implementation_sha256']['agent_apis/neural/pipeline.py']='new'
        self.assertTrue(e.equivalent_decision_configuration(a,b))
        b['implementation_sha256']['agent_apis/neural/provider.py']='different'
        self.assertFalse(e.equivalent_decision_configuration(a,b))
        b=copy.deepcopy(a);b['candidate']['checkpoint']='different'
        self.assertFalse(e.equivalent_decision_configuration(a,b))
        b=copy.deepcopy(a);b['evaluation']['budget_ms']=2000
        self.assertFalse(e.equivalent_decision_configuration(a,b))


class BeliefCompatibilityTests(unittest.TestCase):
    def test_belief_fix_can_reuse_policy_but_never_search(self):
        import copy
        a={'implementation_sha256':{'agent_apis/development_belief.py':'old'},
           'candidate':{'key':'NEURAL_POLICY','checkpoint':'same'},'evaluation':{'mode':'policy'}}
        b=copy.deepcopy(a);b['implementation_sha256']['agent_apis/development_belief.py']='new'
        self.assertTrue(e.equivalent_decision_configuration(a,b))
        for c in (a,b):c['candidate']['key']='NEURAL_SEARCH';c['evaluation']['mode']='time'
        self.assertFalse(e.equivalent_decision_configuration(a,b))
