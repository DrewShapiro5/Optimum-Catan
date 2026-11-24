"""Phase-7 search instrumentation and neural-prior/heuristic-value control.

The search decision/backup implementation is inherited unchanged. Measured
subsystem times may overlap; threaded service times are not additive wall time.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import random
import time
from types import SimpleNamespace
from catanatron.serialization import action_from_json
from agent_apis.benchmark import named_seed
from agent_apis.neural.parallel_search import BatchedEvaluator
from agent_apis.neural.provider import NeuralRuntime,NeuralSearchPlayer,NeuralCheckpoint
from agent_apis.stochastic import action_id
from agent_apis.stochastic_search import StochasticSearch,ReferenceEvaluator
from agent_apis.observation import require,validate_rules
from agent_apis.experiment import ROOT,file_digest


class TelemetrySearch(StochasticSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metrics.update(observation_cache_hits=0,observation_cache_misses=0,
            observation_seconds=0.,simulator_transition_seconds=0.,leaf_evaluation_seconds=0.,
            chance_cache_hits=0,neural_output_cache_hits=0)
    def _view(self,game,color):
        key=(game,color)
        if key in self.view_cache:self.metrics['observation_cache_hits']+=1;return self.view_cache[key]
        start=time.perf_counter();result=super()._view(game,color)
        self.metrics['observation_cache_misses']+=1
        self.metrics['observation_seconds']+=time.perf_counter()-start
        return result
    def _step(self,game,edge):
        start=time.perf_counter();before=self.metrics['chance_distributions']
        result=super()._step(game,edge)
        self.metrics['simulator_transition_seconds']+=time.perf_counter()-start
        if before==self.metrics['chance_distributions']:self.metrics['chance_cache_hits']+=1
        return result
    def _evaluate(self,view):
        start=time.perf_counter();result=super()._evaluate(view)
        self.metrics['leaf_evaluation_seconds']+=time.perf_counter()-start
        return result


class HeuristicValue:
    """Same neural legal priors; replace only the leaf win estimate."""
    def __init__(self,neural):self.neural=neural;self.reference=ReferenceEvaluator()
    def evaluate_batch(self,views):
        predictions=self.neural.evaluate_batch(views)
        return [(priors,self.reference.value(view)) for view,(priors,_) in zip(views,predictions)]
    def evaluate(self,view):return self.evaluate_batch([view])[0]


def entropy(rows):
    visits=sum(row['visits'] for row in rows)
    if not visits:return 0.
    return -sum((row['visits']/visits)*math.log(row['visits']/visits) for row in rows if row['visits'])


def measured_search(view,seed,evaluator,opponent,workers,simulations,context,c_puct,max_actions=128):
    """Same independent-tree allocation/admission rules as optimized phase 6."""
    workers=min(workers,simulations);require(1<=workers<=8,'Invalid search worker count')
    start=time.perf_counter()
    policy=BatchedEvaluator(evaluator,workers);enemy=BatchedEvaluator(opponent,workers)
    try:
        def run(i):
            worker_context=context
            if context.mode=='time':
                worker_context=SimpleNamespace(mode='time',deadline=context.deadline,
                    admission_deadline=min(context.admission_deadline,context.deadline-.050))
            return TelemetrySearch(view,named_seed(seed,f'root-worker-{i}'),policy,enemy,
                c_puct=c_puct,max_actions=max_actions).run(simulations//workers+int(i<simulations%workers),worker_context)
        with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(run,range(workers)))
    finally:policy.close();enemy.close()
    combined={}
    for result in results:
        for row in result['root_statistics']:
            item=combined.setdefault(row['action_id'],{'action_id':row['action_id'],'visits':0,'total':0.,'prior':row['prior']})
            item['visits']+=row['visits'];item['total']+=row['mean_value']*row['visits']
    if not combined:return results[0]
    rows=[{**r,'mean_value':r['total']/r['visits'] if r['visits'] else .5} for r in combined.values()]
    require(rows,'No root statistics')
    selected=max(rows,key=lambda r:(r['visits'],r['mean_value'],-r['action_id']))
    action=next(a for a in view['current_playable_actions'] if action_id(action_from_json(a))==selected['action_id'])
    metrics={'workers':workers,'elapsed_seconds':time.perf_counter()-start,
             'inference_batches':policy.batches+enemy.batches,'inference_requests':policy.requests+enemy.requests,
             'inference_service_seconds':policy.forward_seconds+enemy.forward_seconds,'root_visit_entropy':entropy(rows)}
    for name in ('simulations','raw_outcomes','merged_classes','chance_distributions','retained_chance_children',
                 'worlds_sampled','evaluations','observation_cache_hits','observation_cache_misses','observation_seconds',
                 'simulator_transition_seconds','leaf_evaluation_seconds','chance_cache_hits','neural_output_cache_hits','capped_simulations'):
        metrics[name]=sum(r['metrics'][name] for r in results)
    for name in ('max_action_depth','max_root_depth','max_completed_turn_depth'):
        metrics[name]=max(r['metrics'][name] for r in results)
    metrics['simulations_per_second']=metrics['simulations']/metrics['elapsed_seconds']
    metrics['timing_semantics']='Subsystem times may overlap and sum across threads. Inference service includes packing/encoding/transfers. Observation seconds measures simulated observation construction.'
    return {'action':action,'root_statistics':rows,'metrics':metrics}


class Phase7Runtime(NeuralRuntime):
    def benchmark_choose(self,view,context,rng):
        validate_rules(view)
        if len(view['current_playable_actions'])==1:return view['current_playable_actions'][0]
        require(context.mode in ('time','simulations'),'Phase-7 runtime requires search mode')
        settings=context.settings['search'];frozen=settings['opponent_checkpoint']
        require(file_digest(ROOT/frozen['path'])==frozen['sha256'],'Frozen search opponent changed')
        if self.opponent is None:
            self.opponent=NeuralCheckpoint(ROOT/frozen['path'],device='cpu');self.opponent_sha=frozen['sha256']
            meta=self.opponent.metadata
            require(meta['model_id']=='MLP' and meta['training_seed']==11 and
                    meta['configuration']['data']['composition']=='D2' and meta['configuration']['data']['weighting']=='W0',
                    'Frozen opponent must be seed11 D2/W0 MLP')
        require(self.opponent_sha==frozen['sha256'],'Frozen opponent changed within match')
        leaf=settings.get('leaf_evaluator','neural');require(leaf in ('neural','heuristic'),'Unknown leaf evaluator')
        evaluator=self.policy if leaf=='neural' else HeuristicValue(self.policy)
        count=context.simulations if context.mode=='simulations' else 1000000
        result=measured_search(view,rng.getrandbits(64),evaluator,self.opponent,settings['workers'],count,
                               context,settings['c_puct'],settings.get('max_actions',128))
        self.last_search_metrics=result['metrics'];return result['action']


class Phase7SearchPlayer(NeuralSearchPlayer):
    LABEL='XD-CNN measured search (phase 7)'
    @classmethod
    def from_checkpoint(cls,path):return Phase7Runtime(path,'search')
