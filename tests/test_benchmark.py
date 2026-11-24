import copy
import io
import json
import random
import time
import unittest
from unittest.mock import patch

from catanatron.models.player import Color
from catanatron.serialization import state_to_json, action_to_json

from agent_apis.benchmark import (Agent, CONFIG, SEED_FILE, DecisionContext,
    corpus_seeds, frozen_seeds, hierarchical_bootstrap, make_game,
    paired_bootstrap, paired_difference, play_game, summarize, validate_seeds)
from agent_apis.observation import from_web_view


def result(pair, color, score, status=None):
    return {'pair_id':pair,'candidate_color':color,'candidate_start_seat':int(color=='RED'),
            'match_score':score,'status':status or ('win' if score==1 else 'loss'),
            'illegal_actions':0,'turns':10,'board_sha256':pair,'simulator_seed':1}


class BenchmarkTests(unittest.TestCase):
    def test_smoke_continuation_reuses_only_exact_validated_prefix(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from contextlib import redirect_stdout
        from agent_apis.benchmark import run
        config=json.loads(CONFIG.read_text())
        config['candidate']['key']=config['opponent']['key']='RANDOM'
        calls=[]
        def fixture(pair,color,settings,stream):
            calls.append((pair['pair_id'],color.value))
            game=result(pair['pair_id'],color.value,float(color==Color.BLUE))
            event={'role':'candidate','elapsed_ms':1.,'cold':False,'forced':False,'overrun':False}
            stream.write(json.dumps(event)+'\n')
            return game,[event]
        with TemporaryDirectory() as temp, patch('agent_apis.benchmark.play_game',side_effect=fixture), redirect_stdout(io.StringIO()):
            smoke=run(config,Path(temp))
            self.assertEqual(len(calls),40)
            config['evaluation']['tier']='screening'
            screening=run(config,Path(temp),smoke)
            self.assertEqual(len(calls),200)
            record=json.loads(screening.read_text())
            self.assertEqual(record['metrics']['benchmark']['games'],200)
            self.assertEqual(record['metrics']['continuation']['reused_pairs'],20)
            changed=copy.deepcopy(config);changed['seed']+=1
            with self.assertRaisesRegex(ValueError,'changed more'):
                run(changed,Path(temp),smoke)
            with (smoke.parent/'games.jsonl').open('a') as stream:stream.write('{}\n')
            with self.assertRaisesRegex(ValueError,'logs changed'):
                run(config,Path(temp),smoke)

    def config(self):
        config=json.loads(CONFIG.read_text())
        config['candidate']['key']=config['opponent']['key']='RANDOM'
        config['evaluation']['max_actions']=20
        return config

    def test_frozen_seeds_are_disjoint_and_deterministic(self):
        config=self.config()
        excluded=corpus_seeds(config)
        data=json.loads(SEED_FILE.read_text())
        validate_seeds(data,excluded)
        self.assertEqual(data,frozen_seeds(excluded))
        invalid=copy.deepcopy(data)
        invalid['sets']['final'][0]['board_seed']=data['sets']['tuning'][0]['board_seed']
        with self.assertRaisesRegex(ValueError,'Seed overlap'):
            validate_seeds(invalid,excluded)

    def test_pair_bootstrap_does_not_treat_seats_or_moves_as_independent(self):
        games=[result(str(i),color,float(color=='BLUE')) for i in range(10) for color in ('BLUE','RED')]
        self.assertEqual(paired_bootstrap(games,repeats=100),[.5,.5])
        changed=[{**g,'match_score':g['match_score']-.1} for g in games]
        difference=paired_difference(games,changed,repeats=100)
        self.assertAlmostEqual(difference['difference'],.1)
        self.assertAlmostEqual(difference['ci95'][0],.1)
        with self.assertRaisesRegex(ValueError,'complete seat pairs'):
            paired_bootstrap(games[:-1],repeats=10)
        runs=[(seed,games) for seed in (11,23,47)]
        self.assertEqual(hierarchical_bootstrap(runs,repeats=100)['ci95'],[.5,.5])
        with self.assertRaisesRegex(ValueError,'training seeds'):
            hierarchical_bootstrap(runs[:1],repeats=10)

    def test_summary_separates_cutoffs_errors_and_all_timing_components(self):
        games=[result('a','BLUE',1),result('a','RED',.5,'cutoff'),
               result('b','BLUE',0),result('b','RED',0,'error')]
        timings=[{'role':'candidate','elapsed_ms':v,'cold':cold,'forced':forced,'overrun':v>100}
                 for v,cold,forced in [(500,True,False),(100,False,False),
                                       (200,False,False),(1,False,True)]]
        s=summarize(games,timings,100,bootstrap_repeats=100)
        self.assertEqual(s['completed_game_win_rate'],.5)
        self.assertEqual(s['overall_match_score'],.375)
        self.assertEqual(s['candidate_cold_first_decision']['count'],1)
        self.assertEqual(s['candidate_latency_warm_nonforced']['count'],2)
        self.assertEqual(s['candidate_overruns'],2)
        self.assertFalse(s['timing_gate_passed'])
        self.assertFalse(s['cutoff_gate_passed'])
        self.assertFalse(s['failure_gate_passed'])

    def test_repeated_random_games_have_identical_traces_and_seat_swapped_boards(self):
        config=self.config()
        pair=json.loads(SEED_FILE.read_text())['sets']['tuning'][0]
        first,_=play_game(pair,Color.BLUE,config)
        second,_=play_game(pair,Color.BLUE,config)
        swapped,_=play_game(pair,Color.RED,config)
        self.assertEqual(first['trajectory_sha256'],second['trajectory_sha256'])
        self.assertEqual(first['board_sha256'],swapped['board_sha256'])
        self.assertNotEqual(first['candidate_start_seat'],swapped['candidate_start_seat'])
        self.assertEqual(first['status'],'cutoff')
        self.assertEqual(first['match_score'],.5)
        self.assertEqual(first['actions'],20)

    def test_random_agent_does_not_consume_simulator_rng_and_forced_moves_bypass_provider(self):
        pair=json.loads(SEED_FILE.read_text())['sets']['tuning'][0]
        game=make_game(pair)
        before=game.state.random.getstate()
        agent=Agent('RANDOM',game.state.current_color(),11)
        context=DecisionContext(time.perf_counter(),500,'policy',None)
        selected=agent.select(game,game.playable_actions,context)
        self.assertIn(selected,game.playable_actions)
        self.assertEqual(game.state.random.getstate(),before)
        self.assertLess(context.admission_deadline,context.deadline)
        config=self.config()
        config['evaluation']['max_actions']=1
        game.playable_actions=game.playable_actions[:1]
        with patch('agent_apis.benchmark.make_game',return_value=game), patch.object(
                Agent,'select',side_effect=AssertionError('Forced move called provider')):
            completed,timings=play_game(pair,Color.BLUE,config)
        self.assertEqual(completed['status'],'cutoff')
        self.assertTrue(timings[0]['forced'])

    def test_final_seeds_and_protocol_overrides_are_rejected(self):
        from agent_apis.benchmark import run
        from pathlib import Path
        from tempfile import TemporaryDirectory
        config=self.config()
        config['evaluation']['seed_set']='final'
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'reserved for phase 10'):
                run(config,Path(temp))
            config['evaluation']['seed_set']='tuning'
            with self.assertRaisesRegex(ValueError,'frozen protocol'):
                run(config,Path(temp))  # custom action cap must not enter a published benchmark

    def test_errors_and_illegal_actions_are_logged_without_fallback(self):
        class BadAgent:
            def __init__(self,*args):
                self.init_ms=0;self.cold=True
            def select(self,*args):
                return None
        pair=json.loads(SEED_FILE.read_text())['sets']['tuning'][0]
        stream=io.StringIO()
        game,timings=play_game(pair,Color.BLUE,self.config(),stream,agent_factory=BadAgent)
        self.assertEqual(game['status'],'error')
        self.assertEqual(game['illegal_actions'],1)
        self.assertEqual(game['actions'],0)
        self.assertIn('illegal action',json.loads(stream.getvalue())['error'])
        self.assertEqual(len(timings),1)

    def test_frozen_checkpoint_hook_verifies_hash_and_receives_only_safe_observations(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from agent_apis.experiment import file_digest
        from catanatron.serialization import action_from_json
        class FrozenProvider:
            @staticmethod
            def availability():
                return True,''
            @staticmethod
            def from_checkpoint(path):
                return FrozenProvider()
            def benchmark_choose(self,observation,context,rng):
                assert 'random_state' not in observation
                assert 'seed' not in observation['game']
                assert type(observation['development_listdeck']) is int
                return observation['current_playable_actions'][0]
        with TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'fixture.bin'
            checkpoint.write_bytes(b'fake checkpoint for interface verification only')
            spec={'path':str(checkpoint),'sha256':file_digest(checkpoint)}
            with patch('agent_apis._entries',return_value=[{'key':'FROZEN'}]), patch(
                    'agent_apis._implementation',return_value=FrozenProvider):
                game=make_game(json.loads(SEED_FILE.read_text())['sets']['tuning'][0])
                agent=Agent('FROZEN',game.state.current_color(),11,spec)
                selected=agent.select(game,game.playable_actions,
                    DecisionContext(time.perf_counter(),500,'policy',None))
                self.assertIn(selected,game.playable_actions)
                with self.assertRaisesRegex(ValueError,'Checkpoint hash mismatch'):
                    Agent('FROZEN',game.state.current_color(),11,{**spec,'sha256':'0'*64})

    def test_heuristic_decision_is_invariant_to_authoritative_future_rng_and_deck(self):
        pair=json.loads(SEED_FILE.read_text())['sets']['tuning'][0]
        game=make_game(pair)
        for _ in range(8):
            game.execute(game.playable_actions[0])
        from catanatron.models.actions import generate_playable_actions
        from catanatron.state_functions import player_key
        key=player_key(game.state,game.state.current_color())
        game.state.player_state[key+'_KNIGHT_IN_HAND']=1
        game.state.player_state[key+'_KNIGHT_OWNED_AT_START']=True
        game.state.development_listdeck.remove('KNIGHT')
        game.playable_actions=generate_playable_actions(game.state)
        self.assertGreater(len(game.playable_actions),1)
        self.assertIn('ROLL',[a.action_type.value for a in game.playable_actions])
        first=Agent('VALUE',game.state.current_color(),11)
        second=Agent('VALUE',game.state.current_color(),11)
        context=DecisionContext(time.perf_counter(),500,'policy',None)
        selected=first.select(game,game.playable_actions,context)
        game.state.random.seed(998)
        game.state.development_listdeck.reverse()
        self.assertEqual(selected,second.select(game,game.playable_actions,context))


if __name__=='__main__':
    unittest.main()
