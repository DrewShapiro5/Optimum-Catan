import copy
from collections import Counter
import json
import math
from pathlib import Path
import random
import unittest

from catanatron import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import Action, ActionRecord, ActionType, ActionPrompt, RESOURCES
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import action_to_json, web_view

from agent_apis.development_belief import CARDS, SUPPLY, DevelopmentBelief
from agent_apis.observation import from_web_view
from agent_apis.stochastic import (NoFutureRandom, action_id, canonical_successor,
    elementary_outcomes, enumerate_classes, observation, observation_key, sample_class,
    successor, visible_successor_key, world_from_observation)
from agent_apis.stochastic_search import (ReferenceEvaluator, StochasticSearch, root_value)


def game_position(setup=False):
    game=Game([SimplePlayer(Color.BLUE),SimplePlayer(Color.RED)],seed=11,
              vps_to_win=15,discard_limit=9)
    for color in game.state.colors:
        for kind in ('ROAD','SETTLEMENT','CITY'):
            game.state.buildings_by_color[color][kind]
    if setup:
        for _ in range(8):
            game.execute(game.playable_actions[0])
    else:
        game.state.is_initial_build_phase=False
        game.state.current_prompt=ActionPrompt.PLAY_TURN
    game.random=game.state.random=NoFutureRandom(0)
    refresh(game)
    return game


def refresh(game):
    game.playable_actions=generate_playable_actions(game.state)


def set_hands(game,hands):
    for i,color in enumerate(game.state.colors):
        for resource,value in zip(RESOURCES,hands.get(color,[0]*5)):
            game.state.player_state[f'P{i}_{resource}_IN_HAND']=value
    game.state.resource_freqdeck=[19-sum(game.state.player_state[f'P{i}_{r}_IN_HAND']
        for i in range(2)) for r in RESOURCES]
    refresh(game)


def root_view(game):
    color=game.state.current_color()
    return from_web_view(web_view(game,color),color.value)


def buy_position():
    game=game_position(True)
    own=game.state.current_color()
    set_hands(game,{own:[0,0,1,1,1]})
    game.state.player_state[f'P{game.state.current_player_index}_HAS_ROLLED']=True
    refresh(game)
    return game,next(a for a in game.playable_actions if a.action_type==ActionType.BUY_DEVELOPMENT_CARD)


def theft_position():
    game=game_position(True)
    own=game.state.current_color()
    other=next(c for c in game.state.colors if c!=own)
    set_hands(game,{other:[2,0,0,0,1]})
    roll=next(a for a in game.playable_actions if a.action_type==ActionType.ROLL)
    game.execute(roll,action_record=ActionRecord(roll,(3,4)))
    return game,next(a for a in game.playable_actions if a.action_type==ActionType.MOVE_ROBBER and a.value[1]==other)


def hidden_card_view(events,own_cards=None,other_total=1):
    game=game_position()
    own=game.state.current_color().value
    other=next(c.value for c in game.state.colors if c.value!=own)
    view=root_view(game)
    view['action_records']=[]
    bought=0
    plays={own:Counter(),other:Counter()}
    flag={own:False,other:False}
    for actor,kind,result in events:
        color=own if actor=='own' else other
        value=result if kind=='BUY_DEVELOPMENT_CARD' and actor=='own' else None
        view['action_records'].append([[color,kind,value],result if actor=='own' and kind=='BUY_DEVELOPMENT_CARD' else None])
        if kind=='BUY_DEVELOPMENT_CARD': bought+=1
        if kind.startswith('PLAY_'):
            from agent_apis.observation import PLAY_TYPES
            plays[color][PLAY_TYPES[kind]]+=1;flag[color]=True
        if kind=='END_TURN':flag[color]=False
    view['state_index']=len(view['action_records'])
    seat=view['colors'].index(own)
    for card,n in (own_cards or {}).items():view['player_state'][f'P{seat}_{card}_IN_HAND']=n
    for i,color in enumerate(view['colors']):
        for card,n in plays[color].items():view['player_state'][f'P{i}_PLAYED_{card}']=n
        view['player_state'][f'P{i}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN']=flag[color]
    view['player_state'][f'P{1-seat}_NUM_DEVELOPMENT_CARDS_IN_HAND']=other_total
    view['development_listdeck']=25-bought
    return view


class ConstantEvaluator:
    def evaluate(self,view):
        assert all(a[2] is None and result is None for a,result in view['action_records']
                   if a[1]=='ROLL')
        from catanatron.serialization import action_from_json
        legal=[action_id(action_from_json(a)) for a in view['current_playable_actions']]
        return {i:1/len(legal) for i in legal},.8


class StochasticTests(unittest.TestCase):
    def test_registered_python_and_http_reference_provider(self):
        from agent_apis import choose_action, register_agents
        from play_catan import make_app
        register_agents()
        game=game_position()
        view=web_view(game,game.state.current_color())
        self.assertEqual(choose_action('STOCHASTIC_REFERENCE',view),action_to_json(game.playable_actions[0]))
        broken=copy.deepcopy(view);broken['game']['discard_limit']=7
        with self.assertRaisesRegex(ValueError,'Unsupported game rules'):
            choose_action('STOCHASTIC_REFERENCE',broken)
        initial=Game([SimplePlayer(Color.BLUE),SimplePlayer(Color.RED)],seed=11,vps_to_win=15,discard_limit=9)
        view=web_view(initial,initial.state.current_color())
        with make_app('sqlite:///:memory:').test_client() as client:
            response=client.post('/api/agents/STOCHASTIC_REFERENCE/decide',json={'observation':view})
        self.assertEqual(response.status_code,200)
        self.assertIn(response.json['action'],view['current_playable_actions'])

    def test_dice_merge_mass_and_phase_information(self):
        game=game_position()
        action=game.playable_actions[0]
        self.assertEqual(action.action_type,ActionType.ROLL)
        classes=enumerate_classes(game,action)
        self.assertEqual(len(elementary_outcomes(game,action)),36)
        self.assertEqual(len(classes),2)
        self.assertEqual(sorted(len(c.members) for c in classes),[6,30])
        self.assertEqual(sorted(round(c.probability,8) for c in classes),[round(6/36,8),round(30/36,8)])
        no_production=next(c for c in classes if len(c.members)==30)
        sums=Counter(sum(o.result) for o in no_production.members)
        self.assertEqual(sums[3]+sums[4],5)
        self.assertEqual(len(enumerate_classes(game,action,False)),36)
        first=successor(game,action,next(o for o in no_production.members if sum(o.result)==3))
        second=successor(game,action,next(o for o in no_production.members if sum(o.result)==4))
        self.assertEqual(canonical_successor(first),canonical_successor(second))
        self.assertEqual(observation_key(observation(first,action.color.value)),
                         observation_key(observation(second,action.color.value)))

    def test_steal_frequencies_and_resource_conservation(self):
        game,action=theft_position()
        outcomes=elementary_outcomes(game,action)
        self.assertEqual({o.result:o.probability for o in outcomes},{'WOOD':2/3,'ORE':1/3})
        for outcome in outcomes:
            child=successor(game,action,outcome)
            self.assertEqual(child.state.resource_freqdeck,game.state.resource_freqdeck)
            self.assertEqual(child.state.current_prompt,ActionPrompt.PLAY_TURN)
            for r in RESOURCES:
                self.assertEqual(sum(child.state.player_state[f'P{i}_{r}_IN_HAND'] for i in range(2)),
                                 sum(game.state.player_state[f'P{i}_{r}_IN_HAND'] for i in range(2)))
        with self.assertRaisesRegex(RuntimeError,'Implicit engine chance'):
            game.copy().execute(action)

    def test_draws_have_true_supply_and_age_restrictions_and_terminal_values(self):
        game,action=buy_position()
        outcomes=elementary_outcomes(game,action)
        self.assertEqual({o.result:o.probability for o in outcomes},dict(zip(CARDS,[n/25 for n in SUPPLY])))
        self.assertEqual(len(enumerate_classes(game,action)),5)
        knight=next(o for o in outcomes if o.result=='KNIGHT')
        child=successor(game,action,knight)
        self.assertFalse(any(a.action_type==ActionType.PLAY_KNIGHT_CARD for a in child.playable_actions))
        self.assertEqual(len(child.state.development_listdeck),24)
        seat=game.state.current_player_index
        game.state.player_state[f'P{seat}_VICTORY_POINTS']=14
        game.state.player_state[f'P{seat}_ACTUAL_VICTORY_POINTS']=14
        for outcome in outcomes:
            next_game=successor(game,action,outcome)
            self.assertEqual(next_game.winning_color()==action.color,outcome.result=='VICTORY_POINT')

    def test_bank_shortages_remain_explicit_public_outcome_effects(self):
        game=game_position(True)
        own=game.state.current_color()
        other=next(c for c in game.state.colors if c!=own)
        # Exhaust every resource. All non-seven rolls now leave hands unchanged,
        # but denied production is retained as an observable event effect.
        set_hands(game,{other:[19]*5})
        roll=next(a for a in game.playable_actions if a.action_type==ActionType.ROLL)
        classes=enumerate_classes(game,roll)
        self.assertTrue(any(c.bank_shortages for c in classes))
        self.assertGreater(len(classes),2)
        self.assertAlmostEqual(sum(c.probability for c in classes),1)

    def test_immediate_winning_build_receives_exact_root_utility(self):
        path=Path('data/jsettlers-corpora/full-250/game-0000.jsonl.gz')
        if not path.exists():self.skipTest('Local corpus unavailable')
        from agent_apis.observation_dataset import _frames
        last=None
        for _,_,view,row,_ in _frames(path):last=view
        class ZeroFrontier(ReferenceEvaluator):
            def value(self,view):return 0.
        result=StochasticSearch(last,seed=11,evaluator=ZeroFrontier()).run(64)
        belief=DevelopmentBelief(last)
        world=world_from_observation(last,belief.sample(random.Random(11)))
        from catanatron.serialization import action_from_json
        action=action_from_json(result['action'])
        self.assertEqual(len(elementary_outcomes(world,action)),1)
        final=successor(world,action,elementary_outcomes(world,action)[0])
        self.assertEqual(final.winning_color().value,last['perspective'])
        chosen=next(s for s in result['root_statistics'] if s['action_id']==action_id(action))
        self.assertEqual(chosen['mean_value'],1.)

    def test_discard_and_free_roads_preserve_actor_until_resolved(self):
        game=game_position(True)
        set_hands(game,{c:[10,0,0,0,0] for c in game.state.colors})
        roll=next(a for a in game.playable_actions if a.action_type==ActionType.ROLL)
        game.execute(roll,action_record=ActionRecord(roll,(3,4)))
        self.assertEqual(game.state.current_prompt,ActionPrompt.DISCARD)
        self.assertEqual(game.state.discard_counts,[5,5])
        actor=game.state.current_color()
        game=successor(game,game.playable_actions[0],elementary_outcomes(game,game.playable_actions[0])[0])
        self.assertEqual(game.state.current_color(),actor)
        self.assertEqual(game.state.discard_counts[game.state.current_player_index],4)
        game,action=buy_position()
        card=next(o for o in elementary_outcomes(game,action) if o.result=='ROAD_BUILDING')
        game=successor(game,action,card)
        # End both turns so the purchased card becomes eligible.
        for _ in range(2):
            game.state.player_state[f'P{game.state.current_player_index}_HAS_ROLLED']=True
            refresh(game)
            game.execute(next(a for a in game.playable_actions if a.action_type==ActionType.END_TURN))
        play=next(a for a in game.playable_actions if a.action_type==ActionType.PLAY_ROAD_BUILDING)
        game.execute(play)
        self.assertTrue(game.state.is_road_building)
        self.assertEqual(game.state.free_roads_available,2)
        actor=game.state.current_color()
        bank=list(game.state.resource_freqdeck)
        game.execute(game.playable_actions[0])
        self.assertEqual(game.state.current_color(),actor)
        self.assertEqual(game.state.resource_freqdeck,bank)
        self.assertEqual(game.state.free_roads_available,1)

    def test_long_card_history_does_not_depend_on_python_recursion_limit(self):
        events=[('other','BUY_DEVELOPMENT_CARD',None)]
        events += [('other','END_TURN',None),('own','END_TURN',None)]*1200
        belief=DevelopmentBelief(hidden_card_view(events,other_total=1))
        self.assertGreater(len(belief.events),1000)
        self.assertAlmostEqual(belief.evidence,1.)
        sample=belief.sample(random.Random(11))
        self.assertEqual(sum(sample['opponent_old']),1)
        self.assertEqual(sum(sample['opponent_new']),0)
        self.assertEqual(sum(sample['deck']),24)

    def test_iterative_belief_matches_recursive_probability_and_seeded_worlds(self):
        from functools import lru_cache
        from agent_apis.development_belief import ZERO,VP
        view=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None),('other','END_TURN',None),
            ('own','BUY_DEVELOPMENT_CARD','KNIGHT'),('other','BUY_DEVELOPMENT_CARD',None),
            ('other','PLAY_MONOPOLY',None)],own_cards={'KNIGHT':1},other_total=1)
        belief=DevelopmentBelief(view)
        iterative=belief.mass
        @lru_cache(None)
        def reference(position,old,new):
            if position==len(belief.events):
                deck=belief.deck_counts(belief.own_bought,belief.other_played,old,new)
                seat=view['colors'].index(view['perspective'])
                return float(sum(old)+sum(new)==belief.other_total and sum(deck)==belief.remaining_total
                    and min(deck)>=0 and old[VP]+new[VP]<=14-view['player_state'][f'P{1-seat}_VICTORY_POINTS'])
            return sum(p*reference(position+1,a,b) for _,p,a,b in belief.transitions(position,old,new))
        self.assertEqual(iterative(0,ZERO,ZERO),reference(0,ZERO,ZERO))
        first=random.Random(47);second=random.Random(47)
        samples=[belief.sample(first) for _ in range(100)]
        belief.mass=reference
        self.assertEqual(samples,[belief.sample(second) for _ in range(100)])

    def test_joint_hidden_draw_sampler_conditions_on_old_card_plays(self):
        view=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None),
            ('other','END_TURN',None),('other','BUY_DEVELOPMENT_CARD',None),
            ('other','PLAY_MONOPOLY',None)],other_total=1)
        belief=DevelopmentBelief(view)
        rng=random.Random(11)
        samples=[belief.sample(rng) for _ in range(10000)]
        self.assertTrue(all(s['opponent_draws'][0]=='MONOPOLY' for s in samples))
        self.assertTrue(all(sum(s['opponent_old'])==0 and sum(s['opponent_new'])==1 for s in samples))
        self.assertAlmostEqual(sum(s['opponent_draws'][2]=='KNIGHT' for s in samples)/len(samples),14/24,delta=.02)
        impossible=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None),
            ('other','PLAY_MONOPOLY',None)],other_total=0)
        with self.assertRaisesRegex(ValueError,'No development-card world'):
            DevelopmentBelief(impossible)

    def test_own_known_draw_likelihood_changes_opponent_posterior(self):
        view=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None),
            ('own','BUY_DEVELOPMENT_CARD','KNIGHT')],own_cards={'KNIGHT':1},other_total=1)
        belief=DevelopmentBelief(view)
        rng=random.Random(23)
        draws=[belief.sample(rng)['opponent_draws'][0] for _ in range(10000)]
        self.assertAlmostEqual(draws.count('KNIGHT')/len(draws),13/24,delta=.02)
        self.assertAlmostEqual(draws.count('MONOPOLY')/len(draws),2/24,delta=.015)

    def test_hidden_worlds_share_root_statistics_but_not_chance_world_keys(self):
        view=hidden_card_view([('other','BUY_DEVELOPMENT_CARD',None),('other','END_TURN',None)])
        seat=view['colors'].index(view['perspective'])
        view['player_state'][f'P{seat}_WOOD_IN_HAND']=4
        view['player_state'][f'P{seat}_HAS_ROLLED']=True
        view['resource_freqdeck'][0]=15
        belief=DevelopmentBelief(view)
        allocation=belief.sample(random.Random(1))
        worlds=[]
        for card in ('KNIGHT','MONOPOLY'):
            counts=tuple(int(c==card) for c in CARDS)
            current={**allocation,'opponent_old':counts,'opponent_new':(0,)*5,
                     'opponent_draws':{0:card},'deck':tuple(n-k for n,k in zip(SUPPLY,counts)),
                     'opponent_start_flags':tuple(bool(n) for n in counts)}
            # Obtain the correct legal root list after enabling bank trades.
            temp=game_position();set_hands(temp,{temp.state.current_color():[4,0,0,0,0]})
            temp.state.player_state[f'P{temp.state.current_player_index}_HAS_ROLLED']=True;refresh(temp)
            view['current_playable_actions']=[action_to_json(a) for a in temp.playable_actions]
            worlds.append(world_from_observation(view,current))
        self.assertNotEqual(canonical_successor(worlds[0]),canonical_successor(worlds[1]))
        self.assertEqual(visible_successor_key(worlds[0],view['perspective']),
                         visible_successor_key(worlds[1],view['perspective']))
        first_view,second_view=map(root_view,worlds)
        self.assertEqual(first_view,second_view)
        for seed in (11,23):
            first=StochasticSearch(first_view,seed=seed,evaluator=ConstantEvaluator()).run(8)
            second=StochasticSearch(second_view,seed=seed,evaluator=ConstantEvaluator()).run(8)
            self.assertEqual(first['root_statistics'],second['root_statistics'])
            self.assertEqual(first['action'],second['action'])
        search=StochasticSearch(view,evaluator=ConstantEvaluator())
        search._simulate(worlds[0],(),0,0)
        root=search.nodes[()]
        search._simulate(worlds[1],(),0,0)
        self.assertIs(search.nodes[()],root)
        self.assertEqual(root.visits,2)
        # Root-visible choices must not contain the opponent's private legal dev plays.
        for world in worlds:
            world.state.current_player_index=1-seat;refresh(world)
            self.assertEqual(observation(world,view['perspective'])['current_playable_actions'],[])

    def test_fixed_root_value_and_authoritative_future_invariance(self):
        game=game_position()
        own=game.state.current_color()
        set_hands(game,{own:[4,0,0,0,0]})
        game.state.player_state[f'P{game.state.current_player_index}_HAS_ROLLED']=True;refresh(game)
        before=root_view(game)
        game.random=game.state.random=random.Random(998)
        game.state.development_listdeck.reverse()
        after=root_view(game)
        for seed in (11,23,47):
            first=StochasticSearch(before,seed=seed,evaluator=ConstantEvaluator()).run(12)
            second=StochasticSearch(after,seed=seed,evaluator=ConstantEvaluator()).run(12)
            self.assertEqual(first['root_statistics'],second['root_statistics'])
            self.assertEqual(first['action'],second['action'])
            self.assertEqual(sum(e['visits'] for e in first['root_statistics']),12)
            self.assertTrue(all(abs(e['mean_value']-.8)<1e-12 for e in first['root_statistics'] if e['visits']))
        self.assertEqual(root_value(.8,'BLUE','RED'),1-.8)

    def test_private_opponent_draws_share_a_root_observation_but_own_draws_do_not(self):
        game,action=buy_position()
        root=next(c.value for c in game.state.colors if c!=action.color)
        outcomes=elementary_outcomes(game,action)
        knight=successor(game,action,next(o for o in outcomes if o.result=='KNIGHT'))
        monopoly=successor(game,action,next(o for o in outcomes if o.result=='MONOPOLY'))
        self.assertEqual(visible_successor_key(knight,root),visible_successor_key(monopoly,root))
        self.assertNotEqual(visible_successor_key(knight,action.color.value),
                            visible_successor_key(monopoly,action.color.value))
        self.assertNotEqual(canonical_successor(knight),canonical_successor(monopoly))

    def test_opponent_policy_receives_its_own_private_hand_only(self):
        from agent_apis.stochastic_search import HeuristicOpponent
        game,action=buy_position()
        game=successor(game,action,next(o for o in elementary_outcomes(game,action) if o.result=='KNIGHT'))
        root=action.color.value
        view=root_view(game)
        belief=DevelopmentBelief(view)
        world=world_from_observation(view,belief.sample(random.Random(11)))
        root_seat=world.state.color_to_index[Color[root]]
        class SpyOpponent(HeuristicOpponent):
            calls=0
            def probabilities(self,own_view):
                self.calls+=1
                assert f'P{root_seat}_KNIGHT_IN_HAND' not in own_view['player_state']
                assert type(own_view['development_listdeck']) is int
                assert all(a[2] is None and result is None for a,result in own_view['action_records']
                           if a[1]=='ROLL')
                assert all(result is None for a,result in own_view['action_records']
                           if a[0]==root and a[1]=='BUY_DEVELOPMENT_CARD')
                return super().probabilities(own_view)
        end=next(a for a in world.playable_actions if a.action_type==ActionType.END_TURN)
        world=successor(world,end,elementary_outcomes(world,end)[0])
        spy=SpyOpponent()
        search=StochasticSearch(view,evaluator=ConstantEvaluator(),opponent=spy)
        self.assertAlmostEqual(search._simulate(world,(),0,0),.8)
        self.assertGreater(spy.calls,0)

    def test_budget_stop_and_final_tie_break_do_not_play_exploration_bonus(self):
        import time
        from agent_apis.benchmark import DecisionContext
        game=game_position()
        own=game.state.current_color()
        set_hands(game,{own:[4,0,0,0,0]})
        game.state.player_state[f'P{game.state.current_player_index}_HAS_ROLLED']=True;refresh(game)
        view=root_view(game)
        result=StochasticSearch(view).run(128,DecisionContext(time.perf_counter()-10,100,'time',None))
        self.assertEqual(result['metrics']['simulations'],0)
        from catanatron.serialization import action_from_json
        self.assertEqual(action_id(action_from_json(result['action'])),min(action_id(a) for a in game.playable_actions))
        self.assertTrue(all(s['visits']==0 for s in result['root_statistics']))

    def test_merged_and_unmerged_weighted_one_step_values_match(self):
        cases=[(game_position(),None),buy_position(),theft_position()]
        evaluator=ReferenceEvaluator()
        for game,action in cases:
            action=action or game.playable_actions[0]
            def value(child):
                winner=child.winning_color()
                return float(winner==action.color) if winner is not None else evaluator.evaluate(observation(child,action.color.value))[1]
            expected=[]
            for merged in (False,True):
                classes=enumerate_classes(game,action,merged)
                expected.append(sum(c.probability*value(successor(game,action,c.members[0])) for c in classes))
            self.assertAlmostEqual(*expected,delta=1e-8)

    def test_100000_draw_frequency_gate_with_predeclared_simultaneous_bounds(self):
        # Familywise alpha=.001 over all classes in all three distributions.
        cases=[(game_position(),None),buy_position(),theft_position()]
        distributions=[enumerate_classes(g,a or g.playable_actions[0]) for g,a in cases]
        total_classes=sum(map(len,distributions))
        count=100000
        epsilon=math.sqrt(math.log(2*total_classes/.001)/(2*count))
        rng=random.Random(11)
        for classes in distributions:
            observed=Counter(sample_class(classes,rng).key for _ in range(count))
            for c in classes:
                self.assertLessEqual(abs(observed[c.key]/count-c.probability),epsilon)


if __name__=='__main__':
    unittest.main()
