"""Re-rooting, hidden-world conditioning, session isolation and retained states."""
from copy import deepcopy
import random
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from catanatron.serialization import action_from_json,action_to_json
from agent_apis.neural.subtree_reuse import ReusableSearch,SearchSession,allocation_key,match_key,historical_action_id
from agent_apis.stochastic_search import ReferenceEvaluator,HeuristicOpponent
from agent_apis.stochastic import action_id,observation,world_from_observation,canonical_successor
from agent_apis.development_belief import DevelopmentBelief
from tests.test_stochastic_search import buy_position,root_view,hidden_card_view,set_hands,refresh


def descendants(tree):
    old=len(tree.view['action_records'])
    return [(p,g) for p,(_,g) in tree.positions.items() if p and p in tree.nodes and len(g.playable_actions)>1 and
            historical_action_id(observation(g,tree.root)['action_records'][old][0])==tree.selected_action]


class ReuseTests(unittest.TestCase):
    def tree(self):
        view=root_view(buy_position()[0])
        tree=ReusableSearch(view,seed=41,session_id='A');tree.run(180)
        choices=descendants(tree);self.assertTrue(choices)
        path,game=max(choices,key=lambda v:tree.nodes[v[0]].visits)
        return tree,path,game

    def test_reroot_retains_actual_states_and_prunes_unrelated_branches(self):
        tree,path,game=self.tree();view=observation(game,tree.root)
        advanced=tree.advance(view,77,tree.evaluator,tree.opponent,'A')
        self.assertGreater(advanced.metrics['reused_nodes'],0)
        self.assertIs(advanced.retained_root,tree.nodes[path])
        self.assertTrue(any(g is game for g in advanced.world_templates.values()))
        self.assertTrue(all(path+k in tree.nodes for k in advanced.nodes))
        with patch('agent_apis.stochastic_search.world_from_observation',side_effect=AssertionError('Decoded a retained root')):
            result=advanced.run(16)
        self.assertIn(result['action'],view['current_playable_actions'])
        self.assertEqual(result['metrics']['simulations'],16)
        self.assertEqual(result['metrics'].get('root_world_decodes',0),0)
        self.assertGreater(result['metrics'].get('root_world_cache_hits',0),0)

    def test_world_allocation_keys_match_fresh_belief_and_complete_state(self):
        view=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None),('other','END_TURN',None),
                               ('other','BUY_DEVELOPMENT_CARD',None)],other_total=2)
        belief=DevelopmentBelief(view);rng=random.Random(9)
        for _ in range(15):
            allocation=belief.sample(rng);game=world_from_observation(view,allocation)
            expected=(allocation['opponent_old'],allocation['opponent_new'],tuple(allocation['opponent_draws'].items()),allocation['deck'],allocation['opponent_start_flags'])
            self.assertEqual(allocation_key(game,view['perspective']),expected)

    def test_unknown_cards_reset_returns_and_refresh_belief(self):
        view=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None)],other_total=1)
        allocation=DevelopmentBelief(view).sample(random.Random(1));game=world_from_observation(view,allocation)
        set_hands(game,{game.state.current_color():[4]*5})
        game.state.player_state[f'P{game.state.current_player_index}_HAS_ROLLED']=True;refresh(game)
        tree=ReusableSearch(observation(game,view['perspective']),seed=11,session_id='A');tree.run(150)
        found=[(p,g) for p,g in descendants(tree) if len(observation(g,tree.root)['current_playable_actions'])>1]
        self.assertTrue(found)
        p,g=found[0];old=len(tree.view['action_records']);view=observation(g,tree.root)
        tree.selected_action=historical_action_id(view['action_records'][old][0])
        advanced=tree.advance(view,91,tree.evaluator,tree.opponent,'A')
        self.assertTrue(advanced.metrics['reuse_statistics_reset'])
        self.assertEqual(advanced.retained_root.visits,0)
        for n in advanced.nodes.values():
            self.assertEqual(n.visits,0)
            for e in n.edges.values():self.assertEqual((e.visits,e.total),(0,0.))
        result=advanced.run(10);self.assertIn(result['action'],view['current_playable_actions'])
        self.assertEqual(advanced.belief.view['state_index'],view['state_index'])

    def test_session_rewind_wrong_action_and_changed_private_information_reset(self):
        tree,path,g=self.tree();view=observation(g,tree.root)
        for session,request,reason in [('B',view,'different-or-missing-session'),
                                     ('A',tree.view,'history-rewind-or-divergence')]:
            advanced=tree.advance(request,1,tree.evaluator,tree.opponent,session)
            self.assertEqual(advanced.metrics['reuse_reset_reason'],reason)
            self.assertFalse(advanced.nodes)
        changed=deepcopy(view);seat=changed['colors'].index(tree.root)
        changed['player_state'][f'P{seat}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN']=not changed['player_state'][f'P{seat}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN']
        self.assertEqual(tree.advance(changed,1,tree.evaluator,tree.opponent,'A').metrics['reuse_reset_reason'],'unexplored-observation')
        tree.selected_action=-1
        self.assertEqual(tree.advance(view,1,tree.evaluator,tree.opponent,'A').metrics['reuse_reset_reason'],'different-played-action')

    def test_capped_statistics_reset_and_retained_node_limit(self):
        tree,path,g=self.tree();view=observation(g,tree.root);tree.metrics['capped_simulations']=1
        advanced=tree.advance(view,1,tree.evaluator,tree.opponent,'A')
        self.assertTrue(advanced.metrics['reuse_statistics_reset'])
        tree.max_retained_nodes=0
        self.assertEqual(tree.advance(view,1,tree.evaluator,tree.opponent,'A').metrics['reuse_reset_reason'],'retained-node-limit')
        tree.max_retained_nodes=4096;tree.max_retained_states=0
        reset=tree.advance(view,1,tree.evaluator,tree.opponent,'A')
        self.assertEqual(reset.metrics['reuse_reset_reason'],'retained-state-limit')
        self.assertFalse(reset.nodes);self.assertIsNone(reset.decoded_world)

    def test_parallel_worker_real_action_and_model_revision_isolation(self):
        view=root_view(buy_position()[0]);session=SearchSession();e=ReferenceEvaluator();o=HeuristicOpponent()
        context=SimpleNamespace(mode='simulations')
        first=session.run(view,11,e,o,2,100,context,session_id='A',model_revision=1)
        actual=action_id(action_from_json(first['action']))
        self.assertTrue(all(t.selected_action==actual for t in session.trees))
        t=session.trees[0];choices=descendants(t);self.assertTrue(choices)
        _,g=choices[0];next_view=observation(g,t.root)
        second=session.run(next_view,12,e,o,2,12,context,session_id='A',model_revision=1)
        self.assertEqual(second['metrics']['simulations'],12)
        self.assertGreater(second['metrics']['reused_nodes'],0)
        self.assertIn(second['action'],next_view['current_playable_actions'])
        fresh=session.run(next_view,13,e,o,2,12,context,session_id='A',model_revision=2)
        self.assertEqual(fresh['metrics']['reused_nodes'],0)
        session.clear();self.assertFalse(session.trees)

    def test_neural_runtime_opt_in_and_public_game_routing(self):
        from agent_apis.experiment import ROOT,EXPERIMENTS
        from agent_apis.neural.provider import NeuralRuntime
        from agent_apis.benchmark import DecisionContext,Agent
        import time
        state=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
        runtime=NeuralRuntime(ROOT/state['training']['XD-CNN:11:D2-W0']['checkpoint']['path'],'search')
        settings={'search':{'reuse_subtree':True,'workers':2,'c_puct':1.5,'max_actions':128,
            'opponent_checkpoint':json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']}}
        game,_=buy_position();set_hands(game,{game.state.current_color():[3]*5});view=root_view(game)
        context=DecisionContext(time.perf_counter(),500,'simulations',256,settings);context.session_id='real-checkpoint'
        action=runtime.benchmark_choose(view,context,random.Random(11))
        self.assertIn(action,view['current_playable_actions'])
        choices=[(t,p,g) for t in runtime.search_session.trees for p,g in descendants(t)]
        self.assertTrue(choices)
        t,_,g=choices[0];next_view=observation(g,t.root)
        context=DecisionContext(time.perf_counter(),500,'simulations',16,settings);context.session_id='real-checkpoint'
        action=runtime.benchmark_choose(next_view,context,random.Random(12))
        self.assertIn(action,next_view['current_playable_actions'])
        self.assertGreater(runtime.last_search_metrics['reused_nodes'],0)
        runtime.reset_search();self.assertFalse(runtime.search_session.trees)
        class Stub:
            def benchmark_choose(self,safe,ctx,rng):
                self.session=ctx.session_id
                assert 'id' not in safe['game']
                return safe['current_playable_actions'][0]
        agent=object.__new__(Agent);agent.key='NEURAL_SEARCH';agent.color=game.state.current_color()
        agent.rng=random.Random(1);agent.implementation=Stub()
        ctx=SimpleNamespace()
        agent.select(game,game.playable_actions,ctx)
        self.assertEqual(agent.implementation.session,game.id)

if __name__=='__main__':unittest.main()
