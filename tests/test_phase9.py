import json
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from catanatron.serialization import web_view,action_to_json
from catanatron.models.player import Color
from agent_apis.neural import phase9
from tests.test_stochastic_search import game_position

class PilotTests(unittest.TestCase):
    def test_teacher_reconstruction_and_rejected_queries_have_no_fallback(self):
        game=game_position(True);actions=game.playable_actions;view=web_view(game,game.state.current_color())
        identity={'engine':'SOCRobotBrain','strategy':'SMART','state_index':len(view['action_records']),
                  'current_seat':view['colors'].index(actions[0].color.value),'translation_rejections':0}
        with patch.object(phase9.JSettlersPlayer,'synchronize',return_value=identity),patch.object(phase9.JSettlersPlayer,'choose',return_value=action_to_json(actions[0])):
            self.assertEqual(phase9.teacher_label(view,actions),phase9.action_id(actions[0]))
        broken={**identity,'state_index':999}
        with patch.object(phase9.JSettlersPlayer,'synchronize',return_value=broken):
            with self.assertRaises(ValueError):phase9.teacher_label(view,actions)
        with patch.object(phase9.JSettlersPlayer,'synchronize',side_effect=[identity,{**identity,'translation_rejections':1}]),patch.object(phase9.JSettlersPlayer,'choose',return_value=action_to_json(actions[0])):
            with self.assertRaises(ValueError):phase9.teacher_label(view,actions)

    def test_outcomes_attached_only_after_completion(self):
        rows=[{'features':[0.]*681,'legal':[2,5],'teacher':5}];original=deepcopy(rows)
        self.assertIsNone(phase9.finalize_rows(rows,None,Color.BLUE))
        win=phase9.finalize_rows(rows,Color.BLUE,Color.BLUE);loss=phase9.finalize_rows(rows,Color.RED,Color.BLUE)
        self.assertEqual(win['value'][0],1);self.assertEqual(loss['value'][0],0)
        np.testing.assert_array_equal(win['features'],loss['features']);self.assertEqual(rows,original)
        rows[0]['teacher']=9
        with self.assertRaises(ValueError):phase9.finalize_rows(rows,Color.BLUE,Color.BLUE)

    def test_collection_seeds_disjoint_and_pilot_bounded(self):
        p=json.loads(phase9.PROTOCOL.read_text());pairs=phase9.generation_pairs(p)
        self.assertEqual(len(pairs),10);self.assertEqual(p['collection_games'],20)
        self.assertEqual(p['updates'],1000);self.assertEqual(p['validation_games'],10)
        self.assertEqual(p['validation_opponent'],'JSETTLERS');self.assertFalse(p['automatic_scale_up'])
        seeds=[r[k] for r in pairs for k in ('board_seed','simulator_seed','agent_seed')]
        self.assertEqual(len(set(seeds)),30)

if __name__=='__main__':unittest.main()
