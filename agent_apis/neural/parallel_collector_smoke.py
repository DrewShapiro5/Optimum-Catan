"""Disposable twelve-worker interruption/restart integration check; no training."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
import os
import time

from agent_apis import benchmark
from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.neural import parallel_cohorts as p
from agent_apis.neural.pipeline import write
from agent_apis.observation import require


def main():
    folder = ROOT/'runs/catan-ai/parallel-collector-smoke'
    folder.mkdir(exist_ok=True)
    state = json.loads(p.fresh.STATE.read_text())
    core, parent = state['source_sha256'], state['iterations']['1']['parent']
    require(all(file_digest(ROOT/name)==sha for name,sha in core.items()), 'Pinned engine changed')
    protocol = dict(state['protocol']); protocol['simulations'] = 4
    manifest = folder/'execution.json'
    write(manifest, {'purpose':'diagnostic only; excluded from training','workers':12,'simulations':4,
                     'collector_sha256':file_digest(p.__file__), 'original_source_sha256':core})
    execution = p.selfplay.reference(manifest)
    assignments=[]
    for i in range(12):
        pair={'pair_id':f'parallel-smoke-{i}'}
        for key in ('board','simulator','agent'):
            pair[key+'_seed']=benchmark.named_seed(20261118,f'parallel-smoke-{i}-{key}')
        assignments.append({'attempt':i,'slot':i,'pair':pair,'kind':'current','candidate_color':benchmark.COLORS[i%2].value,'enemy':parent})
    saved=[]
    with benchmark.exclusive_benchmark():
        for phase,seconds in ((1,10),(2,6)):
            ctx=mp.get_context('spawn');stop=ctx.Event();deadline=time.time()+seconds
            with ProcessPoolExecutor(max_workers=12,mp_context=ctx,initializer=p.initialize,
                    initargs=(protocol,core,deadline,stop,os.getpid(),execution,parent)) as pool:
                futures=[pool.submit(p.run_game,a,str((folder/'games').relative_to(ROOT)),parent,[parent]) for a in assignments]
                results=[f.result(timeout=90) for f in futures]
            require(len({r['pid'] for r in results})==12,'Smoke did not start twelve game workers')
            require(all(r['cpu_threads']==1 and r['inference_device']=='cpu' for r in results),'Wrong smoke runtime')
            for r in results:
                with gzip.open(ROOT/r['journal']['path'],'rt') as stream:record=json.load(stream)
                p.compact.replay_journal(record)
                require(record['events'],'Smoke did not exercise game actions')
                require(all(set(e)<= {'a','event','v'} for e in record['events']),'Unexpected snapshot persistence')
            saved.append({'phase':phase,'workers':12,'results':results})
            print(f'phase {phase}: twelve single-thread CPU workers; all prefixes replayed',flush=True)
        archives=list((folder/'games/partial-history').glob('*.gz'))
        require(len(archives)>=12,'Restart did not preserve interrupted prefixes')
        for path in archives:
            with gzip.open(path,'rt') as stream:p.compact.replay_journal(json.load(stream))
        write(folder/'result.json',{'status':'passed','phases':saved,'archived_prefixes':len(archives),'original_training_state_unchanged':json.loads(p.fresh.STATE.read_text())==state})
        require(json.loads(p.fresh.STATE.read_text())==state,'Smoke modified training progress')
        print('Pause/restart smoke passed; training progress unchanged',flush=True)


if __name__=='__main__':main()
