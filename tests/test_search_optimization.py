import json
import random
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from catanatron.serialization import client_view,state_to_json,action_to_json,action_from_json
from catanatron.models.enums import ActionType,ActionRecord
from agent_apis.stochastic import (_production_table,enumerate_classes,canonical_successor,successor,
                                  observation,policy_observation,action_id)
from agent_apis.stochastic_search import StochasticSearch,HeuristicOpponent,ReferenceEvaluator
from agent_apis.observation import from_web_view
from agent_apis.neural.parallel_search import parallel_search,BatchedEvaluator
from tests.test_stochastic_search import game_position,set_hands,root_view,refresh


class SearchOptimizationTests(unittest.TestCase):
    def test_cached_rolls_match_all_36_engine_outcomes_and_dynamic_bank(self):
        game=game_position(True);actor=game.state.current_color()
        for bank in ([19]*5,[0]*5,[0,1,2,3,4]):
            game.state.resource_freqdeck=list(bank)
            roll=next(a for a in game.playable_actions if a.action_type==ActionType.ROLL)
            expected=defaultdict(float)
            for group in enumerate_classes(game,roll,False):
                child=successor(game,roll,group.members[0]);expected[(canonical_successor(child),group.bank_shortages)]+=group.probability
            actual=defaultdict(float)
            for group in enumerate_classes(game,roll,True):
                child=successor(game,roll,group.members[0]);actual[(canonical_successor(child),group.bank_shortages)]+=group.probability
            self.assertEqual(expected.keys(),actual.keys())
            for key in expected:self.assertAlmostEqual(expected[key],actual[key],places=12)
        before=_production_table.cache_info()
        enumerate_classes(game,roll,True)
        self.assertGreater(_production_table.cache_info().hits,before.hits)
        # Robber and city upgrade change the structural signature.
        game.state.board.robber_coordinate=next(c for c in game.state.board.map.land_tiles if c!=game.state.board.robber_coordinate)
        enumerate_classes(game,roll,True);self.assertGreater(_production_table.cache_info().misses,before.misses)
        before=_production_table.cache_info()
        edge=game.state.board.buildable_edges(actor)[0]
        game.state.board.build_road(actor,edge)
        enumerate_classes(game,roll,True)
        self.assertEqual(_production_table.cache_info().misses,before.misses)
        node,(owner,kind)=next((n,p) for n,p in game.state.board.buildings.items() if p[1]=='SETTLEMENT')
        game.state.board.build_city(owner,node)
        enumerate_classes(game,roll,True)
        self.assertGreater(_production_table.cache_info().misses,before.misses)
    def test_fast_observations_match_audited_serializer_after_multiple_actions(self):
        game=game_position(True)
        for _ in range(25):
            for color in game.state.colors:
                document=state_to_json(game);view=client_view(document,color)
                view['current_playable_actions']=[action_to_json(a) for a in game.playable_actions] if game.state.current_color()==color else []
                view['state_index']=len(game.state.action_records)
                expected=policy_observation(from_web_view(view,color.value))
                self.assertEqual(observation(game,color.value),expected)
            action=game.playable_actions[0]
            # Search uses explicit outcomes, preserving the no-future-RNG guard.
            from agent_apis.stochastic import elementary_outcomes
            game=successor(game,action,elementary_outcomes(game,action)[0])
    def test_no_default_four_action_cutoff_and_parallel_root_accounting(self):
        game=game_position();set_hands(game,{game.state.current_color():[4,0,0,0,0]})
        game.state.player_state[f'P{game.state.current_player_index}_HAS_ROLLED']=True;refresh(game)
        class Focused:
            def evaluate(self,view):
                actions=[action_from_json(a) for a in view['current_playable_actions']]
                priority={'END_TURN':4,'ROLL':3,'MOVE_ROBBER':2}
                best=max(actions,key=lambda a:priority.get(a.action_type.value,0))
                return {action_id(a):float(a==best) for a in actions},.5
        view=root_view(game);model=Focused()
        result=StochasticSearch(view,seed=11,evaluator=model).run(100)
        self.assertGreater(result['metrics']['max_root_depth'],4)
        parallel=parallel_search(view,11,model,HeuristicOpponent(),2,12,None)
        self.assertEqual(parallel['metrics']['simulations'],12)
        self.assertIn(parallel['action'],view['current_playable_actions'])
        self.assertEqual(sum(r['visits'] for r in parallel['root_statistics']),12)
    def test_resampled_templates_match_full_world_decoder(self):
        from tests.test_stochastic_search import hidden_card_view
        from agent_apis.development_belief import DevelopmentBelief
        from agent_apis.stochastic import world_from_observation,world_from_template
        view=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None),('other','END_TURN',None),
                               ('other','BUY_DEVELOPMENT_CARD',None)],other_total=2)
        belief=DevelopmentBelief(view);rng=random.Random(42)
        template=world_from_observation(view,belief.sample(rng))
        for _ in range(30):
            allocation=belief.sample(rng)
            expected=world_from_observation(view,allocation)
            actual=world_from_template(template,view,allocation)
            self.assertEqual(state_to_json(expected),state_to_json(actual))
            self.assertEqual(expected.playable_actions,actual.playable_actions)

    def test_batched_models_match_separate_masked_predictions(self):
        import torch
        from agent_apis.neural.models import MODEL_IDS,build_model
        from agent_apis.neural.provider import NeuralCheckpoint
        torch.set_num_threads(2)
        game=game_position(True);first=root_view(game)
        from agent_apis.stochastic import elementary_outcomes
        action=game.playable_actions[0]
        child=successor(game,action,elementary_outcomes(game,action)[0]);second=root_view(child)
        for name in MODEL_IDS:
            model=object.__new__(NeuralCheckpoint);model.model=build_model(name).eval();model.device='cpu'
            batch=model.evaluate_batch([first,second])
            for view,(probability,value) in zip((first,second),batch):
                expected,expected_value=model.evaluate(view)
                self.assertEqual(probability.keys(),expected.keys())
                self.assertAlmostEqual(sum(probability.values()),1,places=12)
                self.assertAlmostEqual(value,expected_value,places=5)
                for key in probability:self.assertAlmostEqual(probability[key],expected[key],places=5)

    def test_batch_service_propagates_exceptions_without_hanging(self):
        class Bad:
            def evaluate_batch(self,views):raise ValueError('inference failed')
        service=BatchedEvaluator(Bad(),2)
        try:
            with ThreadPoolExecutor(2) as pool:
                tasks=[pool.submit(service.evaluate,{}) for _ in range(2)]
                for task in tasks:
                    with self.assertRaisesRegex(ValueError,'inference failed'):task.result(timeout=2)
        finally:service.close()
