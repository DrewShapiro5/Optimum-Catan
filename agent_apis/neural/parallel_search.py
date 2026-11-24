"""Independent root trees in threads; shared, queued batched neural inference.

Python tree work still encounters the GIL. This is an opt-in measured variant,
not a guarantee that more threads are faster. Torch kernels release the GIL.
"""
from concurrent.futures import Future,ThreadPoolExecutor
from queue import Queue,Empty
import threading
import time
from types import SimpleNamespace
from catanatron.serialization import action_to_json
from agent_apis.benchmark import named_seed
from agent_apis.observation import require
from agent_apis.stochastic import action_id
from agent_apis.stochastic_search import StochasticSearch


class BatchedEvaluator:
    def __init__(self,evaluator,max_batch,wait_seconds=.0002):
        self.evaluator=evaluator;self.max_batch=max_batch;self.wait_seconds=wait_seconds
        self.queue=Queue();self.batches=0;self.requests=0;self.forward_seconds=0.
        self.thread=threading.Thread(target=self._serve,daemon=True);self.thread.start()
    def evaluate(self,view):
        future=Future();self.queue.put((view,future));return future.result()
    def probabilities(self,view):
        if len(view['current_playable_actions'])==1:
            from catanatron.serialization import action_from_json
            return {action_id(action_from_json(view['current_playable_actions'][0])):1.}
        return self.evaluate(view)[0]
    def _serve(self):
        while True:
            first=self.queue.get()
            if first is None:return
            batch=[first];deadline=time.perf_counter()+self.wait_seconds
            while len(batch)<self.max_batch:
                try:batch.append(self.queue.get(timeout=max(0,deadline-time.perf_counter())))
                except Empty:break
            start=time.perf_counter()
            try:
                if hasattr(self.evaluator,'evaluate_batch'):results=self.evaluator.evaluate_batch([v for v,f in batch])
                elif hasattr(self.evaluator,'evaluate'):results=[self.evaluator.evaluate(v) for v,f in batch]
                else:results=[(self.evaluator.probabilities(v),.5) for v,f in batch]
                require(len(results)==len(batch),'Inference batch lost requests')
                for (_,future),result in zip(batch,results):future.set_result(result)
            except BaseException as error:
                for _,future in batch:future.set_exception(error)
            finally:
                self.batches+=1;self.requests+=len(batch);self.forward_seconds+=time.perf_counter()-start
    def close(self):
        self.queue.put(None);self.thread.join()


def parallel_search(view,seed,evaluator,opponent,workers,simulations,context,c_puct=1.5,max_actions=128):
    require(type(workers)is int and 1<=workers<=8,'Search workers must be 1..8')
    require(simulations>0,'Positive simulations required')
    workers=min(workers,simulations)
    if workers==1:return StochasticSearch(view,seed,evaluator,opponent,max_actions=max_actions,c_puct=c_puct).run(simulations,context)
    start=time.perf_counter();policy=BatchedEvaluator(evaluator,workers);enemy=BatchedEvaluator(opponent,workers)
    try:
        def run(i):
            worker_context=context
            if context is not None and context.mode=='time':
                worker_context=SimpleNamespace(mode=context.mode,deadline=context.deadline,
                    admission_deadline=min(context.admission_deadline,context.deadline-.050))
            count=simulations//workers+int(i<simulations%workers)
            return StochasticSearch(view,named_seed(seed,f'root-worker-{i}'),policy,enemy,
                                    c_puct=c_puct,max_actions=max_actions).run(count,worker_context)
        with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(run,range(workers)))
    finally:policy.close();enemy.close()
    combined={}
    for result in results:
        for row in result['root_statistics']:
            item=combined.setdefault(row['action_id'],{'action_id':row['action_id'],'visits':0,'total':0.,'prior':row['prior']})
            item['visits']+=row['visits'];item['total']+=row['mean_value']*row['visits']
    if not combined:return results[0]
    rows=[{'action_id':i,'visits':r['visits'],'mean_value':r['total']/r['visits'] if r['visits'] else .5,'prior':r['prior']} for i,r in combined.items()]
    selected=max(rows,key=lambda r:(r['visits'],r['mean_value'],-r['action_id']))
    from catanatron.serialization import action_from_json
    action=next(a for a in view['current_playable_actions'] if action_id(action_from_json(a))==selected['action_id'])
    metrics={'workers':workers,'simulations':sum(r['metrics']['simulations'] for r in results),
             'elapsed_seconds':time.perf_counter()-start,'worker_metrics':[r['metrics'] for r in results],
             'inference_batches':policy.batches+enemy.batches,'inference_requests':policy.requests+enemy.requests,
             'inference_service_seconds':policy.forward_seconds+enemy.forward_seconds}
    for name in ('max_root_depth','max_action_depth','max_completed_turn_depth'):
        metrics[name]=max(r['metrics'][name] for r in results)
    return {'action':action,'root_statistics':rows,'metrics':metrics}
