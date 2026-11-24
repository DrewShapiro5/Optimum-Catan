import copy
import unittest
from unittest.mock import patch
from catanatron.models.player import Color
from catanatron.models.enums import ActionType,ActionPrompt
from catanatron.models.actions import generate_playable_actions
from catanatron.web.models import get_game_state,upsert_game_state
from play_catan import make_app
from agent_apis import benchmark

class PrivacyRuleTests(unittest.TestCase):
 def test_hidden_vp_does_not_remove_friendly_robber_protection(self):
  g=benchmark.make_game({'board_seed':1,'simulator_seed':2,'agent_seed':3},friendly_robber=True)
  for _ in range(8):g.execute(g.playable_actions[0])
  actor=g.state.current_color();enemy=next(c for c in g.state.colors if c!=actor)
  idx=g.state.colors.index(enemy);prefix=f'P{idx}_'
  g.state.current_prompt=ActionPrompt.MOVE_ROBBER
  g.state.player_state[prefix+'VICTORY_POINTS']=2
  sets=[]
  for hidden in (0,1,3):
   g.state.player_state[prefix+'VICTORY_POINT_IN_HAND']=hidden
   g.state.player_state[prefix+'ACTUAL_VICTORY_POINTS']=2+hidden
   legal=generate_playable_actions(g.state);sets.append(set(legal))
   enemy_nodes=set(g.state.buildings_by_color[enemy]['SETTLEMENT'])
   self.assertTrue(all(not enemy_nodes.intersection(g.state.board.map.land_tiles[a.value[0]].nodes.values()) for a in legal))
  self.assertEqual(sets[0],sets[1]);self.assertEqual(sets[1],sets[2])
  g.state.player_state[prefix+'VICTORY_POINTS']=3
  self.assertGreater(len(generate_playable_actions(g.state)),len(sets[0]))
 def test_browser_never_switches_to_bot_hand_during_search(self):
  app=make_app('sqlite:///:memory:');client=app.test_client()
  result=client.post('/api/games',json={'players':['HUMAN','CNN_INSPECT_SEARCH'],'vps_to_win':15,'discard_limit':9,'friendly_robber':True})
  self.assertEqual(result.status_code,200);gid=result.json['game_id'];endpoint=f'/api/games/{gid}'
  with app.app_context():
   g=get_game_state(gid);human=next(p.color for p in g.state.players if not p.is_bot);bot=next(p.color for p in g.state.players if p.is_bot)
   bot_seat=g.state.colors.index(bot);human_seat=g.state.colors.index(human)
   g.state.current_player_index=bot_seat;g.state.current_turn_index=bot_seat
   g.state.player_state[f'P{bot_seat}_KNIGHT_IN_HAND']=2
   g.state.player_state[f'P{bot_seat}_VICTORY_POINT_IN_HAND']=1
   g.state.player_state[f'P{human_seat}_WOOD_IN_HAND']=3
   g.playable_actions=generate_playable_actions(g.state);upsert_game_state(g)
  view=client.get(endpoint+'/states/latest').json
  self.assertEqual(view['viewer_color'],human.value)
  self.assertEqual(view['current_color'],bot.value)
  self.assertEqual(view['current_playable_actions'],[])
  self.assertNotIn(f'P{bot_seat}_KNIGHT_IN_HAND',view['player_state'])
  self.assertNotIn(f'P{bot_seat}_VICTORY_POINT_IN_HAND',view['player_state'])
  self.assertNotIn(f'P{bot_seat}_ACTUAL_VICTORY_POINTS',view['player_state'])
  self.assertEqual(view['player_state'][f'P{bot_seat}_NUM_DEVELOPMENT_CARDS_IN_HAND'],3)
  self.assertEqual(view['player_state'][f'P{human_seat}_WOOD_IN_HAND'],3)
  self.assertEqual(set(view['development_listdeck']),{'UNKNOWN'})
  self.assertEqual(client.get(endpoint+f'/states/latest?as={bot.value}').status_code,400)
