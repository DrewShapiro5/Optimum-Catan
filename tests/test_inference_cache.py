import unittest
from agent_apis.neural.inference_cache import InferenceCache
from tests.test_stochastic_search import hidden_card_view,ConstantEvaluator

class BatchEvaluator(ConstantEvaluator):
    def evaluate_batch(self,views):return [self.evaluate(v) for v in views]

class CacheTests(unittest.TestCase):
    def test_duplicates_are_coalesced_and_outputs_are_defensive(self):
        view=hidden_card_view([]);cache=InferenceCache(BatchEvaluator())
        a,b=cache.evaluate_batch([view,view]);self.assertEqual(a,b)
        self.assertEqual(cache.forward_rows,1);self.assertEqual(cache.coalesced,1)
        a[0].clear();self.assertTrue(cache.evaluate(view)[0]);self.assertEqual(cache.hits,1)
    def test_private_knowledge_isolates_entries(self):
        a=hidden_card_view([],own_cards={'KNIGHT':1})
        b=hidden_card_view([],own_cards={'YEAR_OF_PLENTY':1})
        cache=InferenceCache(BatchEvaluator());cache.evaluate_batch([a,b])
        self.assertEqual(cache.forward_rows,2);self.assertEqual(len(cache.cache),2)
    def test_capacity_and_model_scope(self):
        a=hidden_card_view([],own_cards={'KNIGHT':1});b=hidden_card_view([],own_cards={'YEAR_OF_PLENTY':1})
        cache=InferenceCache(BatchEvaluator(),1);cache.evaluate_batch([a,b])
        self.assertEqual(len(cache.cache),1);cache.evaluate(a);self.assertEqual(cache.forward_rows,3)
        other=InferenceCache(BatchEvaluator());other.evaluate(a);self.assertEqual(other.forward_rows,1)
    def test_rejects_missing_predictions(self):
        class Broken:
            def evaluate_batch(self,views):return []
        with self.assertRaises(ValueError):InferenceCache(Broken()).evaluate(hidden_card_view([]))

if __name__=='__main__':unittest.main()
