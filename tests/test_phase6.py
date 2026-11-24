"""Distribution, value-correction, resume and experiment scope regressions."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import numpy as np
import torch
from agent_apis.neural.recipes import RecipeSampler,RECIPES,balanced_mass,weighted_losses
from agent_apis.neural import phase6


class FixtureCache:
    def __init__(self):
        lengths=[6,12,6];n=sum(lengths)
        self.manifest={'rows':n,'feature_names':['x']*681,'families':['A','B','C'],
            'games':[{'offset':0,'steps':6,'split':'train','directory':'x/full-250'},
                     {'offset':6,'steps':12,'split':'train','directory':'x/downsampled'},
                     {'offset':18,'steps':6,'split':'validation','directory':'x/full-250'}]}
        self.manifest['feature_names'][638]='ACTOR_IS_BLUE'
        features=np.zeros((n,681),np.float32);features[:,638]=np.arange(n)%2
        mask=np.zeros((n,332),np.uint8);mask[:,:2]=1;mask[::4,1]=0
        self.arrays={'features':features,'legal':np.packbits(mask,axis=1,bitorder='little'),
                     'phase':np.array([0,1,1,1,1,2]*4),'family':np.array([0]*15+[1,1,2]+[0]*6),
                     'retained':np.array([1]*6+[1,0]*6+[1]*6)}


def data(recipe):
    composition,weighting,*variant=recipe.split('-')
    return {'composition':composition,'weighting':weighting,'value_weighting':variant[0] if variant else 'V0'}


class RecipeTests(unittest.TestCase):
    def test_coverage_and_split_isolation(self):
        cache=FixtureCache()
        for recipe,n in [('D0-W0',6),('D1-W0',12),('D2-W0',18)]:
            sampler=RecipeSampler(cache,data(recipe),11,buffer_size=4)
            self.assertEqual(len(sampler.indices),n)
            self.assertTrue(np.all(sampler.take(100)<18))
        # W0 keeps the baseline's exact once-per-epoch coverage.
        s=RecipeSampler(cache,data('D2-W0'),11,buffer_size=4)
        self.assertEqual(sorted(s.take(18)),list(range(18)))

    def test_uniform_games_and_phase_probabilities(self):
        c=FixtureCache();s=RecipeSampler(c,data('D2-W1'),11)
        self.assertAlmostEqual(s.probabilities[:6].sum(),.5)
        self.assertAlmostEqual(s.probabilities[6:].sum(),.5)
        s=RecipeSampler(c,data('D2-W2'),11)
        for phase in range(3):
            selected=c.arrays['phase'][:18]==phase
            self.assertAlmostEqual(s.probabilities[selected].sum(),.5*selected.mean()+1/6)
        np.testing.assert_allclose(s.probabilities*s.value_weights,np.ones(18)/18)

    def test_rare_families_cap_and_inverse_value_correction(self):
        for counts in ([1000,10,1],[1,1,1],[10000,1]):
            counts=np.array(counts);mass=balanced_mass(counts)
            q=.5/counts.sum()+.5*mass/counts
            self.assertLessEqual(float((counts.sum()*q).max()),5+1e-10)
            self.assertAlmostEqual(float((counts*q).sum()),1)
        s=RecipeSampler(FixtureCache(),data('D2-W4'),11)
        np.testing.assert_allclose(s.probabilities*s.value_weights,np.ones(18)/18)

    def test_curve_is_per_actor_and_training_only(self):
        c=FixtureCache();s=RecipeSampler(c,data('D2-W3-V1'),11)
        legal=np.unpackbits(c.arrays['legal'][:18],axis=1,bitorder='little')[:,:332]
        self.assertAlmostEqual(s.policy_weights[legal.sum(1)>1].mean(),1)
        self.assertAlmostEqual(s.value_weights.mean(),1)
        # Each actor ends independently in each trajectory.
        self.assertGreater(s.value_weights[4],s.value_weights[0])
        self.assertGreater(s.value_weights[5],s.value_weights[1])
        self.assertEqual(c.arrays['features'].shape[1],681)
        self.assertTrue(np.all(RecipeSampler(c,data('D2-W3-V0'),11).value_weights==1))

    def test_resume_reproduces_next_batch_for_every_recipe(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                c=FixtureCache();s=RecipeSampler(c,data(recipe),123,buffer_size=4)
                s.take(23);state=s.state_dict()
                restored=RecipeSampler(c,data(recipe),123,buffer_size=4,state=state)
                np.testing.assert_array_equal(s.take(37),restored.take(37))
                for x,y in zip(s.weights(s.indices),restored.weights(restored.indices)):
                    np.testing.assert_array_equal(x,y)

    def test_value_correction_matches_uniform_gradient(self):
        s=RecipeSampler(FixtureCache(),data('D2-W2'),11)
        logits=torch.linspace(-2,2,18,requires_grad=True)
        targets=torch.arange(18)%2
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets.float(),reduction='none')
        corrected=(bce*torch.tensor(s.probabilities*s.value_weights)).sum()
        uniform=bce.mean()
        np.testing.assert_allclose(torch.autograd.grad(corrected,logits,retain_graph=True)[0],
                                   torch.autograd.grad(uniform,logits)[0],rtol=1e-6,atol=1e-8)

    def test_forced_policy_exclusion_and_no_sample_weight_renormalization(self):
        legal=torch.tensor([[True,False],[True,True]])
        policy=torch.zeros((2,2),requires_grad=True);value=torch.zeros(2,requires_grad=True)
        p,v=weighted_losses(policy,value,legal,torch.tensor([0,1]),torch.tensor([0.,1.]),
                            torch.tensor([100.,2.]),torch.tensor([1.,3.]))
        self.assertAlmostEqual(p.item(),2*np.log(2),places=6)
        self.assertAlmostEqual(v.item(),2*np.log(2),places=6)
        (p+v).backward();self.assertEqual(policy.grad[0].abs().sum(),0)


class Phase6Tests(unittest.TestCase):
    def test_scope_and_equal_training_budget(self):
        for model in ('XD-CNN','TRANSFORMER'):
            for recipe in RECIPES:
                c=phase6.configuration(model,11,recipe)
                self.assertEqual(c['training']['max_updates'],20000)
                self.assertEqual(c['training']['batch_size'],256)
                self.assertEqual(c['data']['model_schema'],3)
        with self.assertRaises(ValueError):phase6.configuration('MLP',11,'D2-W0')
        with self.assertRaises(ValueError):phase6.configuration('H-GNN',11,'D2-W0')
        self.assertEqual(phase6.configuration('XD-CNN',11,'D2-W0'),phase6.config_for('XD-CNN',11))

    def test_smokes_precede_extensions_and_preserve_opponent_and_mode(self):
        calls=[]
        class Work(phase6.Phase6):
            def __init__(self):self.protocol={'validation_opponents':['RANDOM','JSETTLERS']}
            def measure(self,model,seed,recipe,training,mode,opponent,tier,prior=None):
                calls.append((mode,opponent,tier,prior))
                return Path(opponent+'-'+mode),{'metrics':{'benchmark':{'futility':{'stopped':False}}}}
        Work().validate('XD-CNN',11,'D2-W0',None)
        self.assertEqual([c[2] for c in calls],['smoke']*4+['screening']*4)
        self.assertEqual({c[1] for c in calls},{'RANDOM','JSETTLERS'})
        self.assertTrue(all(c[3]==Path(c[1]+'-'+c[0]) for c in calls[4:]))

    def test_match_configs_pin_two_opponents_and_frozen_search_policy(self):
        work=object.__new__(phase6.Phase6);work.core={}
        c=phase6.configuration('TRANSFORMER',11,'D2-W2')
        training=(phase6.OUTPUT/'fixture/run.json',{'configuration':{'content':c},'checkpoint':{'path':'best.pt','sha256':'a'*64}})
        for opponent in ('RANDOM','JSETTLERS'):
            for mode in ('policy','search'):
                resolved=work.match_config('TRANSFORMER',11,'D2-W2',training,mode,opponent,'smoke')
                self.assertEqual(resolved['opponent']['key'],opponent)
                self.assertEqual(resolved['evaluation']['seed_set'],'tuning')
                self.assertEqual(resolved['evaluation']['mode'],'time' if mode=='search' else 'policy')
                self.assertEqual(resolved['evaluation']['smoke_pairs'],5)
                self.assertTrue(resolved['search']['opponent_checkpoint']['sha256'])

if __name__=='__main__':unittest.main()
