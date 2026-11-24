import unittest
import torch
from agent_apis.neural.models import StructuralTransformer
from agent_apis.neural.value_ablation_models import build_variant,connector_graph,TypedExphormer,objective

class ValueAblationTests(unittest.TestCase):
    def test_value_variants_preserve_initial_predictions(self):
        torch.set_num_threads(1);torch.manual_seed(10);base=StructuralTransformer().eval()
        raw=torch.zeros(2,681)
        with torch.no_grad():expected=base(raw)
        for variant in ('control','logit-penalty','value-dropout','value-depth1','value-depth2'):
            model=build_variant(base.state_dict(),variant).eval()
            with torch.no_grad():result=model(raw)
            torch.testing.assert_close(result[0],expected[0],rtol=0,atol=0)
            torch.testing.assert_close(result[1],expected[1],rtol=0,atol=0)
    def test_sparse_types_connectors_and_blocked_attention(self):
        torch.set_num_threads(1);model=TypedExphormer();raw=torch.zeros(2,681)
        x=model.inputs(raw);types,bias=model.attention_graph(x)
        self.assertTrue(torch.isneginf(bias.masked_select((types[:,None]==0).expand_as(bias))).all())
        self.assertTrue((types[:,:148,148:]!=0).all());self.assertGreater(model.spectral_gap,.05)
        self.assertLess((types!=0).sum().item()/types.numel(),.2)
        self.assertEqual(types[0,0,19].item(),model.edge_types[0,19].item())
        a,b=model(raw);self.assertEqual(a.shape,(2,332));self.assertEqual(b.shape,(2,))
        (a.square().mean()+b.square().mean()).backward()
        self.assertGreater(model.connectors.grad.abs().sum().item(),0)
        grads=model.edge_bias.weight.grad.abs().sum(1)
        self.assertGreater((grads>0).sum().item(),5)
    def test_dynamic_edge_types_and_deterministic_graph(self):
        torch.set_num_threads(1)
        model=TypedExphormer();x=model.inputs(torch.zeros(1,681))
        x['owner_node'][0,0,0]=1;x['owner_road'][0,1,0]=1
        x['port_adj'][0,0,1]=x['port_adj'][0,1,0]=1
        types,bias=model.attention_graph(x)
        for a,b,kind in ((145,19,10),(19,145,11),(146,73,12),(73,146,13),(19,20,14)):
            self.assertEqual(types[0,a,b].item(),kind)
            self.assertTrue(torch.isfinite(bias[0,:,a,b]).all())
        a,gap=connector_graph();b,other=connector_graph()
        self.assertTrue(torch.equal(a,b));self.assertEqual(gap,other)

    def test_w4_correction_is_applied_once(self):
        from agent_apis.neural.friendly_w4 import losses
        values=torch.tensor([1.,-2.],requires_grad=True);weights=torch.tensor([.5,2.])
        logits=torch.zeros(2,3);legal=torch.ones(2,3,dtype=torch.bool);targets=torch.tensor([[1.,0,0],[0,1.,0]])
        outcomes=torch.tensor([1.,0.]);pl,vl,p=objective(logits,values,legal,targets,outcomes,weights,'logit-penalty')
        expected=losses(logits,values,legal,targets,outcomes,weights)
        torch.testing.assert_close(vl,expected[1]);torch.testing.assert_close(p,.01*(values.square()*weights).mean())

if __name__=='__main__':unittest.main()
