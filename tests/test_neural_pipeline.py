"""Exercise the phase-5 experiment sequence with deterministic game fixtures."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from agent_apis.neural import pipeline


class PipelineTests(unittest.TestCase):
    def test_search_strength_can_keep_weak_policy_and_confirmation_uses_three_seeds(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary);calls=[];trained=set();written={}
            # CNN is strongest standalone; the GNN has the weakest policy but
            # strongest search. It must survive selection on playing strength.
            strengths={'MLP':(.25,.25),'XD-CNN':(.75,.75),'H-GNN':(0.,1.),'TRANSFORMER':(.25,.25)}
            class Fixture(pipeline.Pipeline):
                def __init__(self):
                    self.core={};self.state={'selection_rule':'fixture','benchmarks':{}}
                def persist(self):pass
                def trained(self,model,seed,wait=False):
                    trained.add((model,seed))
                    return folder/'training.json',{'checkpoint':{'path':'fixture','sha256':'0'*64},
                        'elapsed_seconds':1.,'configuration':{'content':{'search':{}}},
                        'metrics':{'training':{'parameters':{'MLP':10,'XD-CNN':20,'H-GNN':30,'TRANSFORMER':40}[model]}}}
                def evaluate(self,model,seed,mode,final=False):
                    calls.append((model,seed,mode,final))
                    path=folder/f'{model}-{seed}-{mode}-{final}'/'run.json';path.parent.mkdir()
                    score=strengths[model][mode=='search']
                    rows=[{'pair_id':str(i),'candidate_color':color,'board_sha256':str(i),
                           'simulator_seed':i,'match_score':score} for i in range(2) for color in ('BLUE','RED')]
                    (path.parent/'games.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                    return path,{'metrics':{'benchmark':{'overall_match_score':score,'failure_gate_passed':True,
                                  'timing_gate_passed':True,'cutoff_gate_passed':True}}}
            def capture(path,value):written[path.name]=value
            original_relative=Path.relative_to
            def fixture_relative(path,root,*args):
                if path.is_relative_to(folder):return Path(path.name)
                return original_relative(path,root,*args)
            with patch.object(pipeline,'write',side_effect=capture),patch.object(pipeline,'report'),patch.object(Path,'relative_to',fixture_relative):
                Fixture().execute()
            summary=written['PHASE5_RESULTS.json']
            self.assertEqual(summary['finalists'],['H-GNN','XD-CNN'])
            self.assertFalse(summary['final_evaluation_seeds_used'])
            self.assertEqual(summary['provisional_choice']['model'],'H-GNN')
            self.assertEqual(summary['provisional_choice']['mode'],'search')
            for model in ('H-GNN','XD-CNN'):
                self.assertTrue(all((model,seed) in trained for seed in (11,23,47)))
                self.assertTrue(all((model,11,mode,True) in calls for mode in ('policy','search')))
            self.assertFalse(any(seed!=11 for model,seed in trained if model in ('MLP','TRANSFORMER')))
            self.assertEqual(summary['architectures']['H-GNN']['hierarchical']['search_minus_policy']['ci95'],[1.,1.])


if __name__=='__main__':unittest.main()
