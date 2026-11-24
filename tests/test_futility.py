import copy
import io
import json
from contextlib import nullcontext,redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from agent_apis.futility import DEFAULT,check,upper_bound
from agent_apis.benchmark import CONFIG,run
from agent_apis.neural.pipeline import Pipeline,eligible
from agent_apis.experiment import ROOT


def pairs(n,score=0.):
    return [{'pair_id':str(i),'candidate_color':c,'match_score':score,'status':'win' if score else 'loss'}
            for i in range(n) for c in ('BLUE','RED')]


class FutilityTests(unittest.TestCase):
    def test_bounds_are_nonzero_at_zero_wins_and_corrected_for_all_looks(self):
        one=check(pairs(20),20);five=check(pairs(20),100)
        self.assertAlmostEqual(one['upper_match_score'],1-.05**(1/20))
        self.assertAlmostEqual(five['upper_match_score'],1-.01**(1/20))
        self.assertGreater(five['upper_match_score'],one['upper_match_score'])
        self.assertTrue(one['stopped']);self.assertTrue(five['stopped'])
        self.assertGreater(one['upper_match_score'],0)
        self.assertFalse(check(pairs(19),100)['stopped'])
        self.assertFalse(check(pairs(20,.5),100)['stopped'])
        self.assertGreater(upper_bound(.2,20,.01),.2)
        self.assertEqual(upper_bound(1,20,.01),1)
    def test_complete_pairs_required_and_errors_never_misclassified_as_futility(self):
        with self.assertRaisesRegex(ValueError,'complete unique seat pairs'):check(pairs(20)[:-1],100)
        errors=pairs(20);errors[0]['status']='error'
        self.assertFalse(check(errors,100)['stopped'])
        self.assertFalse(check(pairs(20),100,{**DEFAULT,'enabled':False})['stopped'])
    def test_benchmark_stops_after_40_games_and_cannot_extend_futile_smoke(self):
        config=json.loads(CONFIG.read_text());config['candidate']['key']=config['opponent']['key']='RANDOM'
        calls=[]
        def fixture(pair,color,settings,stream):
            calls.append(pair['pair_id'])
            return {**pair,'candidate_color':color.value,'candidate_start_seat':int(color.value=='RED'),
                    'match_score':0.,'status':'loss','illegal_actions':0,'turns':10,'board_sha256':pair['pair_id']},[]
        with TemporaryDirectory() as temporary,patch('agent_apis.benchmark.play_game',side_effect=fixture),redirect_stdout(io.StringIO()):
            path=run(config,Path(temporary));record=json.loads(path.read_text())
            self.assertEqual(len(calls),40);self.assertEqual(record['status'],'stopped-early')
            self.assertEqual(record['metrics']['benchmark']['match_score_ci95'],[None,None])
            self.assertFalse(eligible(record))
            config['evaluation']['tier']='screening'
            with self.assertRaisesRegex(ValueError,'early-stopped'):run(config,Path(temporary),path)
            path=run(config,Path(temporary));record=json.loads(path.read_text())
            self.assertEqual(len(calls),80);self.assertEqual(record['metrics']['benchmark']['games'],40)
            self.assertEqual(record['metrics']['benchmark']['futility']['planned_looks'],[20,40,60,80,100])
    def test_pipeline_does_not_extend_smoke_or_launch_confirmations_after_futility(self):
        config=json.loads((ROOT/'experiments/catan-ai/phase5-configs/MLP-seed11.json').read_text())
        stopped={'metrics':{'benchmark':{'futility':{'stopped':True}}}}
        class Fixture(Pipeline):
            def __init__(self):self.core={};self.state={'benchmarks':{}}
            def trained(self,*args,**kwargs):
                return ROOT/'fixture/run.json',{'configuration':{'content':copy.deepcopy(config)},'checkpoint':None}
            def persist(self):pass
            def guard(self):pass
            def benchmark_record(self,*args):return None
        with TemporaryDirectory(dir=ROOT/'runs/catan-ai') as temporary:
            path=Path(temporary)/'run.json';path.write_text(json.dumps(stopped))
            with patch('agent_apis.neural.pipeline.locked',return_value=nullcontext()),patch('agent_apis.neural.pipeline.write'),patch('agent_apis.neural.pipeline.benchmark.run',return_value=path) as measured:
                with self.assertRaisesRegex(ValueError,'Futility stop'):Fixture().evaluate('MLP',11,'policy')
                self.assertEqual(measured.call_count,1)
                self.assertEqual(measured.call_args.args[0]['evaluation']['tier'],'smoke')

    def test_investigated_cutoff_can_continue_comparison_without_becoming_eligible(self):
        config=json.loads((ROOT/'experiments/catan-ai/phase5-configs/MLP-seed11.json').read_text())
        record={'status':'completed','metrics':{'benchmark':{'futility':{'stopped':False},
                'failure_gate_passed':True,'timing_gate_passed':True,'cutoff_gate_passed':False}}}
        class Fixture(Pipeline):
            def __init__(self):self.core={};self.state={'benchmarks':{},'continue_after_investigated_cutoffs':True}
            def trained(self,*args,**kwargs):return ROOT/'fixture/run.json',{'configuration':{'content':copy.deepcopy(config)},'checkpoint':None}
            def persist(self):pass
            def guard(self):pass
            def benchmark_record(self,*args):return None
        with TemporaryDirectory(dir=ROOT/'runs/catan-ai') as temporary:
            path=Path(temporary)/'run.json';path.write_text(json.dumps(record))
            with patch('agent_apis.neural.pipeline.locked',return_value=nullcontext()),patch('agent_apis.neural.pipeline.write'),patch('agent_apis.neural.pipeline.benchmark.run',return_value=path) as measured:
                _,returned=Fixture().evaluate('MLP',11,'policy')
                self.assertFalse(eligible(returned));self.assertEqual(measured.call_count,1)


if __name__=='__main__':unittest.main()
