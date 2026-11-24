import copy
import gzip
import json
from pathlib import Path
import unittest
from agent_apis.dataset_bundle import exact_numbers, retained_steps
from agent_apis.dataset import replay_game

class BundleTests(unittest.TestCase):
    def test_exact_identity_distinguishes_equal_probability_numbers(self):
        board={'tiles':[{'type':'LAND','id':0,'number':6},{'type':'LAND','id':1,'number':8},{'type':'LAND','id':2,'number':None}]}
        other=copy.deepcopy(board)
        other['tiles'][1]['number']=6
        self.assertEqual(exact_numbers(board),[6,8,0])
        self.assertEqual(exact_numbers(other),[6,6,0])
        self.assertNotEqual(exact_numbers(board),exact_numbers(other))

    def test_sampling_is_per_actor_and_preserves_setup_and_tail(self):
        rows=[{'step':i,'actor':'BLUE' if i%2 else 'RED',
               'observation':{'is_initial_build_phase':i<8}} for i in range(100)]
        expected=set(range(8))|set(range(80,100))|{0,1,20,21,40,41,60,61}
        self.assertEqual(retained_steps(rows),expected)

    def test_replay_advances_across_omitted_actions(self):
        from tempfile import TemporaryDirectory
        from agent_apis.dataset import _generate_game, MAX_BYTES
        from agent_apis.dataset_bundle import convert
        result=_generate_game(0,100,1000,10000,MAX_BYTES)
        self.assertNotIn('error',result)
        packed,_,header,rows=convert(result['payload'],True)
        with TemporaryDirectory() as temp:
            p=Path(temp)/'game.gz'; p.write_bytes(packed)
            steps=[]
            for game,row in replay_game(p):
                self.assertEqual(len(game.state.action_records),row['step'])
                self.assertEqual(len(row['features']),633)
                steps.append(row['step'])
            self.assertEqual(steps,[r['step'] for r in rows])
            self.assertLess(len(steps),header['steps'])

if __name__=='__main__': unittest.main()
