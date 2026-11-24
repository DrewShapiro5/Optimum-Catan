import unittest
from copy import deepcopy
from agent_apis.neural.transformer_catchup import Catchup,validate_manifest
from agent_apis.neural.friendly_w4 import Workflow

class CatchupTests(unittest.TestCase):
    def manifest(self):
        return {'completed_games':2000,'rules':{'friendly_robber':True,'friendly_robber_basis':'visible-victory-points'},
            'splits':{k:[{'path':f'{k}/{i}'} for i in range(n)] for k,n in [('train',1800),('validation',100),('test',100)]}}
    def test_complete_whole_game_splits(self):
        m=self.manifest();self.assertEqual(len(validate_manifest(m)['train']),1800)
        m['splits']['test'][0]=deepcopy(m['splits']['train'][0])
        with self.assertRaisesRegex(ValueError,'cross-split'):validate_manifest(m)
    def test_incomplete_and_wrong_rules_rejected(self):
        m=self.manifest();m['completed_games']=1999
        with self.assertRaisesRegex(ValueError,'2000'):validate_manifest(m)
        m=self.manifest();m['rules']['friendly_robber_basis']='actual-victory-points'
        with self.assertRaisesRegex(ValueError,'visible'):validate_manifest(m)
    def test_pinned_trainer_inherited(self):
        self.assertIs(Catchup.train_w4,Workflow.train_w4)

if __name__=='__main__':unittest.main()
