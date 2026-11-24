"""Bounded evaluator-local cache keyed by permitted features, knowledge and legality.

Use one instance per frozen checkpoint and per decision. Outputs are copied on
return; neither private knowledge nor predictions cross model/decision scopes.
"""
from collections import OrderedDict
from agent_apis.stochastic import observation_key,action_id
from catanatron.serialization import action_from_json
from agent_apis.observation import require


class InferenceCache:
    def __init__(self,evaluator,capacity=1024):
        require(isinstance(capacity,int) and capacity>0,'Positive cache capacity required')
        self.evaluator=evaluator;self.capacity=capacity;self.cache=OrderedDict()
        self.hits=self.misses=self.coalesced=self.forward_rows=0
    @staticmethod
    def copy(result):return dict(result[0]),float(result[1])
    def evaluate_batch(self,views):
        keys=[observation_key(v) for v in views];pending=OrderedDict();results={}
        for key,view in zip(keys,views):
            if key in self.cache:
                self.hits+=1;results[key]=self.cache[key];self.cache.move_to_end(key)
            elif key in pending:self.coalesced+=1
            else:self.misses+=1;pending[key]=view
        if pending:
            predictions=self.evaluator.evaluate_batch(list(pending.values()))
            require(len(predictions)==len(pending),'Inference lost pending rows')
            self.forward_rows+=len(pending)
            for key,result in zip(pending,predictions):
                stored=self.copy(result);results[key]=stored;self.cache[key]=stored
                self.cache.move_to_end(key)
                while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return [self.copy(results[key]) for key in keys]
    def evaluate(self,view):return self.evaluate_batch([view])[0]
    def probabilities(self,view):
        legal=view['current_playable_actions']
        if len(legal)==1:return {action_id(action_from_json(legal[0])):1.}
        return self.evaluate(view)[0]
    def metrics(self):return {'hits':self.hits,'misses':self.misses,'coalesced_requests':self.coalesced,'forward_rows':self.forward_rows,'entries':len(self.cache)}
