"""Search instrumentation, controlled value replacement and exact baseline reuse."""
import json
import math
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from agent_apis.experiment import EXPERIMENTS,ROOT
from agent_apis.neural import phase7
from agent_apis.neural.phase7_runtime import TelemetrySearch,HeuristicValue,measured_search,Phase7Runtime,entropy
from agent_apis.stochastic_search import StochasticSearch,ReferenceEvaluator,HeuristicOpponent
from agent_apis.benchmark import DecisionContext
from tests.test_stochastic_search import game_position,root_view,buy_position


class Phase7Tests(unittest.TestCase):
    def test_instrumentation_preserves_serial_search_decision_and_visits(self):
        view=root_view(buy_position()[0])
        a=StochasticSearch(view,seed=41).run(20)
        b=TelemetrySearch(view,seed=41).run(20)
        self.assertEqual(a['action'],b['action'])
        self.assertEqual(a['root_statistics'],b['root_statistics'])
        self.assertEqual(b['metrics']['simulations'],20)
        for key in ('observation_seconds','simulator_transition_seconds','leaf_evaluation_seconds'):
            self.assertGreaterEqual(b['metrics'][key],0)
        self.assertGreater(b['metrics']['observation_cache_misses'],0)
        self.assertEqual(b['metrics']['neural_output_cache_hits'],0)

    def test_heuristic_control_changes_only_value(self):
        view=root_view(game_position(True));ref=ReferenceEvaluator()
        class Neural:
            def evaluate_batch(self,views):return [(ref.policy.probabilities(v),.999) for v in views]
        model=Neural();control=HeuristicValue(model)
        priors,value=control.evaluate(view)
        self.assertEqual(priors,model.evaluate_batch([view])[0][0])
        self.assertAlmostEqual(value,ref.value(view))
        self.assertNotEqual(value,.999)

    def test_parallel_exact_fixed_simulation_count_and_entropy(self):
        view=root_view(buy_position()[0]);context=SimpleNamespace(mode='simulations')
        r=measured_search(view,42,ReferenceEvaluator(),HeuristicOpponent(),4,13,context,1.5)
        self.assertEqual(r['metrics']['simulations'],13)
        self.assertEqual(sum(row['visits'] for row in r['root_statistics']),13)
        self.assertIn(r['action'],view['current_playable_actions'])
        self.assertGreater(r['metrics']['inference_requests'],0)
        self.assertGreater(r['metrics']['simulations_per_second'],0)
        self.assertGreaterEqual(r['metrics']['raw_outcomes'],r['metrics']['merged_classes'])
        self.assertLessEqual(r['metrics']['root_visit_entropy'],math.log(len(r['root_statistics']))+1e-12)
        self.assertEqual(entropy([{'visits':0}]),0)

    def test_schedule_has_all_planned_budgets_and_only_xd_cnn(self):
        v=phase7.variants()
        self.assertEqual({x['budget_ms'] for x in v.values() if x['mode']=='time'},{100,500,2000})
        self.assertEqual({x['simulations'] for x in v.values() if x['mode']=='simulations'},{128,512,2048})
        self.assertEqual({x['c_puct'] for x in v.values()},{.75,1.5,3.})
        protocol=json.loads(phase7.PROTOCOL.read_text())
        self.assertEqual(protocol['architectures'],['XD-CNN'])
        self.assertEqual(protocol['opponents'],['JSETTLERS'])
        self.assertFalse(protocol['new_training']);self.assertFalse(protocol['final_seeds_used'])

    def test_configs_preserve_tuning_seeds_checkpoint_and_control(self):
        work=object.__new__(phase7.Phase7);work.core={}
        work.phase6=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
        for variant in ('time100','sim2048','heuristic500','time500-c075'):
            c=work.configuration(11,variant,'RANDOM','smoke')
            self.assertEqual(c['model']['id'],'XD-CNN')
            self.assertEqual(c['data']['weighting'],'W0')
            self.assertEqual(c['candidate']['checkpoint'],work.phase6['training']['XD-CNN:11:D2-W0']['checkpoint'])
            self.assertEqual(c['candidate']['key'],'PHASE7_SEARCH')
            self.assertEqual(c['evaluation']['seed_set'],'tuning')
            self.assertEqual(c['evaluation']['smoke_pairs'],5)
            self.assertEqual(c['search']['leaf_evaluator'],'heuristic' if variant=='heuristic500' else 'neural')
        self.assertEqual(work.phase6['training']['XD-CNN:11:D2-W0']['metrics']['updates'],20000)

    def test_real_checkpoint_instrumented_search_and_heuristic_runtime(self):
        r=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
        checkpoint=ROOT/r['training']['XD-CNN:11:D2-W0']['checkpoint']['path']
        work=object.__new__(phase7.Phase7);work.core={};work.phase6=r
        runtime=Phase7Runtime(checkpoint,'search');view=root_view(buy_position()[0])
        for variant in ('sim128','heuristic500'):
            c=work.configuration(11,variant,'RANDOM','smoke')
            context=DecisionContext(__import__('time').perf_counter(),500,'simulations',8,c)
            action=runtime.benchmark_choose(view,context,random.Random(42))
            self.assertIn(action,view['current_playable_actions'])
            self.assertEqual(runtime.last_search_metrics['simulations'],8)

    def test_historical_reference_reuse_is_checksum_verified(self):
        work=object.__new__(phase7.Phase7)
        work.phase6=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
        for mode in ('policy','search'):
            path,record=work.reference(11,mode,'RANDOM')
            self.assertTrue(path.exists())
            self.assertEqual(record['metrics']['benchmark']['games'],50)
        with patch.object(phase7,'file_digest',return_value='changed'):
            with self.assertRaises(ValueError):work.reference(11,'search','RANDOM')

if __name__=='__main__':unittest.main()
