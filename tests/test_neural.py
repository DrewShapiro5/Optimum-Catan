import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from catanatron import Game
from catanatron.models.player import Color,SimplePlayer
from catanatron.serialization import web_view,action_to_json,action_from_json
from catanatron.gym.envs.action_space import to_action_space
from agent_apis.observation import encode_observation,from_web_view
from agent_apis.neural.models import MODEL_IDS,Inputs,build_model,losses
from agent_apis.neural.topology import topology
from agent_apis.neural.data import DecisionShuffle
from agent_apis.neural.train import lr_at


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def position(self):
        game=Game([SimplePlayer(Color.BLUE),SimplePlayer(Color.RED)],seed=23,vps_to_win=15,discard_limit=9)
        view=from_web_view(web_view(game,game.state.current_color()),game.state.current_color().value)
        return game,view
    def test_four_architectures_forward_backward_and_global_legal_normalization(self):
        game,view=self.position();raw=torch.tensor([encode_observation(view)]*2)
        legal=torch.zeros(2,332,dtype=torch.bool)
        ids=[to_action_space(a,(Color.BLUE,Color.RED),'BASE') for a in game.playable_actions];legal[:,ids]=True
        for name in MODEL_IDS:
            model=build_model(name);symmetry=torch.tensor([0,7]);target=torch.tensor([ids[0],ids[1]])
            augmented_target,augmented_legal=model.inputs.augment_targets(target,legal,symmetry)
            policy,value=model(raw,symmetry)
            self.assertEqual(policy.shape,(2,332));self.assertEqual(value.shape,(2,))
            pl,vl=losses(policy,value,augmented_legal,augmented_target,torch.tensor([0.,1.]))
            (pl+vl).backward()
            self.assertTrue(torch.isfinite(pl+vl));self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
            probabilities=policy.masked_fill(~augmented_legal,float('-inf')).softmax(-1)
            torch.testing.assert_close(probabilities.sum(1),torch.ones(2))
            self.assertTrue((probabilities[~augmented_legal]==0).all())
    def test_forced_rows_only_contribute_value_gradient(self):
        policy=torch.randn(3,332,requires_grad=True);value=torch.zeros(3,requires_grad=True)
        legal=torch.zeros(3,332,dtype=torch.bool);legal[:,7]=True
        pl,vl=losses(policy,value,legal,torch.tensor([7,7,7]),torch.tensor([0.,1.,0.]))
        self.assertEqual(pl.item(),0);(pl+vl).backward()
        self.assertEqual(policy.grad.abs().sum().item(),0);self.assertGreater(value.grad.abs().sum().item(),0)
        with self.assertRaisesRegex(ValueError,'Illegal demonstrated'):
            losses(policy,value,legal,torch.tensor([8,7,7]),torch.tensor([0.,1.,0.]))
    def test_all_geometric_symmetries_form_a_group_and_preserve_victim_colors(self):
        t=topology();permutations=[np.array(p) for p in t['action_forward']]
        for first in permutations:
            for second in permutations:self.assertTrue(any(np.array_equal(second[first],p) for p in permutations))
        for permutation in permutations:
            self.assertEqual(sorted(permutation.tolist()),list(range(332)))
            for i,(kind,value) in enumerate(t['action_vocabulary']):
                if kind=='MOVE_ROBBER':self.assertEqual(t['action_vocabulary'][permutation[i]][1][1],value[1])
    def test_full_physical_augmentation_audit_is_pinned(self):
        from agent_apis.experiment import EXPERIMENTS
        report=json.loads((EXPERIMENTS/'phase5-augmentation-audit.json').read_text())
        self.assertTrue(report['gate_passed']);self.assertEqual(report['checks'],144)
        self.assertEqual(report['topology_sha256'],topology()['sha256'])
    def test_bounded_shuffle_exact_epochs_and_restart(self):
        games=[{'offset':0,'steps':3,'split':'train'},{'offset':3,'steps':2,'split':'validation'},{'offset':5,'steps':5,'split':'train'}]
        sampler=DecisionShuffle(games,11,buffer_size=3)
        self.assertEqual(set(sampler.take(8)),{0,1,2,5,6,7,8,9})
        # Exercise the actual checkpoint boundary: state_dict holds live lists,
        # while serialization produces the independent restart snapshot.
        import io
        sampler.take(3);stream=io.BytesIO();torch.save(sampler.state_dict(),stream);stream.seek(0)
        restored=DecisionShuffle(games,999,buffer_size=3,state=torch.load(stream,weights_only=True))
        np.testing.assert_array_equal(sampler.take(20),restored.take(20))
    def test_learning_rate_boundaries(self):
        settings={'learning_rate':3e-4,'warmup_updates':500,'max_updates':20000,'minimum_learning_rate':3e-5}
        self.assertAlmostEqual(lr_at(1,settings),3e-4/500)
        self.assertAlmostEqual(lr_at(500,settings),3e-4)
        self.assertAlmostEqual(lr_at(20000,settings),3e-5)
        self.assertGreater(lr_at(1000,settings),lr_at(10000,settings))
    def test_trained_checkpoint_and_neural_search_interfaces(self):
        from agent_apis.experiment import EXPERIMENTS,ROOT
        from agent_apis.neural.provider import NeuralPolicyPlayer,NeuralSearchPlayer
        from agent_apis.benchmark import DecisionContext
        from agent_apis.stochastic import policy_observation
        import random,time
        frozen=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())
        game,view=self.position();view=policy_observation(view)
        settings={'search':{'opponent_checkpoint':frozen['checkpoint'],'c_puct':1.5,'max_root_depth':4,'max_actions':128}}
        policy=NeuralPolicyPlayer.from_checkpoint(ROOT/frozen['checkpoint']['path'])
        selected=policy.benchmark_choose(view,DecisionContext(time.perf_counter(),500,'policy',None,settings),random.Random(11))
        self.assertIn(selected,view['current_playable_actions'])
        priors,value=policy.policy.evaluate(view)
        self.assertAlmostEqual(sum(priors.values()),1,places=12);self.assertTrue(0<=value<=1)
        search=NeuralSearchPlayer.from_checkpoint(ROOT/frozen['checkpoint']['path'])
        selected=search.benchmark_choose(view,DecisionContext(time.perf_counter(),500,'simulations',2,settings),random.Random(11))
        self.assertIn(selected,view['current_playable_actions']);self.assertEqual(search.last_search_metrics['simulations'],2)


if __name__=='__main__':unittest.main()
