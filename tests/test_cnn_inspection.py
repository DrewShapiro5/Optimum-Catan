import json
import unittest
from unittest.mock import patch
from agent_apis import list_agents
from agent_apis.experiment import ROOT
from agent_apis.neural.inspection import CONFIG,InspectionPolicyPlayer,InspectionSearchPlayer
from agent_apis.neural.provider import cached_runtime
from play_catan import make_app

class InspectionTests(unittest.TestCase):
 def test_pinned_CNN_and_live_HTTP_gameplay_both_modes(self):
  config=json.loads(CONFIG.read_text())
  self.assertEqual(config['training_seed'],23)
  self.assertEqual(config['selection']['wins'],20)
  for key,cls in [('CNN_INSPECT_POLICY',InspectionPolicyPlayer),('CNN_INSPECT_SEARCH',InspectionSearchPlayer)]:
   self.assertEqual(cls.availability(),(True,''))
   entry=next(a for a in list_agents() if a['key']==key)
   self.assertEqual(entry['supported_rules']['players'],2)
   app=make_app('sqlite:///:memory:');client=app.test_client()
   response=client.post('/api/games',json={'players':['HUMAN',key],'vps_to_win':15,'discard_limit':9,'friendly_robber':True})
   self.assertEqual(response.status_code,200,response.get_data(as_text=True))
   gid=response.get_json()['game_id'];endpoint=f'/api/games/{gid}'
   state=client.get(endpoint+'/states/latest').get_json();bot_moves=0
   for _ in range(12):
    before=len(state['action_records'])
    if state['current_color'] in state['bot_colors']:
     response=client.post(endpoint+'/actions',json={});bot_moves+=1
    else:response=client.post(endpoint+'/actions',json=state['current_playable_actions'][0])
    self.assertEqual(response.status_code,200,response.get_data(as_text=True))
    state=response.get_json();self.assertEqual(len(state['action_records']),before+1)
   self.assertGreaterEqual(bot_moves,4)
   self.assertTrue(state['game']['friendly_robber'])
  ref=config['checkpoint'];op=config['search']['opponent_checkpoint']
  runtime=cached_runtime(str(ROOT/ref['path']),ref['sha256'],'policy',op['sha256'],None)
  self.assertEqual(runtime.policy.metadata['model_id'],'XD-CNN')
  self.assertEqual(runtime.policy.metadata['training_seed'],23)
 def test_corrupt_checkpoint_is_unavailable_without_fallback(self):
  with patch('agent_apis.neural.inspection.file_digest',return_value='wrong'):
   self.assertFalse(InspectionPolicyPlayer.availability()[0])

if __name__=='__main__':unittest.main()
