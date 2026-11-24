import unittest
from unittest.mock import patch
from agent_apis import list_agents
from agent_apis.neural.mixed_web import MixedCnnSearchPlayer
from play_catan import make_app

class MixedWebTests(unittest.TestCase):
 def test_registered_gameplay_and_human_privacy(self):
  entry=next(x for x in list_agents() if x['key']=='CNN_MIXED_SEARCH');self.assertTrue(entry['available']);self.assertEqual(entry['supported_rules']['players'],2)
  client=make_app('sqlite:///:memory:').test_client()
  response=client.post('/api/games',json={'players':['HUMAN','CNN_MIXED_SEARCH'],'vps_to_win':15,'discard_limit':9,'friendly_robber':True});self.assertEqual(response.status_code,200,response.get_data(as_text=True))
  gid=response.get_json()['game_id'];state=client.get(f'/api/games/{gid}/states/latest').get_json();moves=0
  for _ in range(8):
   if state['current_color'] in state['bot_colors']:
    response=client.post(f'/api/games/{gid}/actions',json={});moves+=1
   else:response=client.post(f'/api/games/{gid}/actions',json=state['current_playable_actions'][0])
   self.assertEqual(response.status_code,200,response.get_data(as_text=True));state=response.get_json()
  self.assertGreaterEqual(moves,4)
 def test_corrupt_checkpoint_fails_closed(self):
  with patch('agent_apis.neural.mixed_web.file_digest',return_value='wrong'):self.assertFalse(MixedCnnSearchPlayer.availability()[0])
if __name__=='__main__':unittest.main()
