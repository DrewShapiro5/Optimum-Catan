"""Matched bounded multi-game process benchmark; never resumes training."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

import torch
from catanatron.serialization import web_view,action_record_to_json
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,file_digest,digest
from agent_apis.observation import from_web_view,require
from agent_apis.neural import selfplay,compact_iterations as compact
from agent_apis.neural.pipeline import write
from agent_apis.jsettlers.player import BRIDGE

OUTPUT=ROOT/'runs/catan-ai/game-worker-benchmark'
MODEL=None;OPPONENT=None;THREADS=None


def position(job):
    with gzip.open(ROOT/job['source']['path'],'rt') as stream:record=json.load(stream)
    require(file_digest(ROOT/job['source']['path'])==job['source']['sha256'],'Benchmark source changed')
    game=benchmark.make_game(record['pair'],friendly_robber=True)
    require(compact.snapshot_digest(game)==record['initial_sha256'],'Benchmark source initial state changed')
    prefix=record['events'][:job['prefix']]
    for event in prefix:
        actions={selfplay.action_id(a):a for a in game.playable_actions}
        require(event['a'] in actions,'Illegal source prefix')
        require(action_record_to_json(game.execute(actions[event['a']]))==event['event'],'Source prefix chance mismatch')
    return game,record,prefix


def initialize(ref,frozen,threads,barrier,pids,warmup_job):
    global MODEL,OPPONENT,THREADS
    torch.set_num_interop_threads(1)
    MODEL=selfplay.load_checkpoint(ref);OPPONENT=selfplay.load_checkpoint(frozen)
    torch.set_num_threads(threads);THREADS=threads
    game,_,_=position(warmup_job)
    safe=from_web_view(web_view(game,game.state.current_color()),game.state.current_color().value)
    MODEL.evaluate(safe);OPPONENT.evaluate(safe)
    pids.put(os.getpid());barrier.wait(timeout=90)


def ready():return os.getpid()


def run_job(job,protocol,checkpoint,output):
    require(torch.get_num_threads()==THREADS,'Worker CPU limit changed')
    start=time.perf_counter();game,source,prefix=position(job);BRIDGE.close()
    initial=source['initial_sha256'];pair=source['pair'];root=game.state.current_color()
    other=next(c for c in benchmark.COLORS if c!=root)
    seed=benchmark.named_seed(pair['agent_seed'],f"worker-benchmark-{job['prefix']}")
    seats={root:selfplay.SearchSeat(MODEL,OPPONENT,benchmark.named_seed(seed,root.value),protocol,game.id)}
    if job['kind']!='JSETTLERS':
        seats[other]=selfplay.SearchSeat(MODEL,OPPONENT,benchmark.named_seed(seed,other.value),protocol,game.id)
    native=benchmark.Agent('JSETTLERS',other,seed) if job['kind']=='JSETTLERS' else None
    events=list(prefix);decisions=simulations=reuse=0;timings=[];native_seconds=0.
    replay_start=0.
    try:
        while decisions<job['searched_decisions'] and game.winning_color() is None:
            require(len(events)-len(prefix)<400,'Benchmark prefix did not reach enough searches')
            actor=game.state.current_color();actions=list(game.playable_actions);visits=None
            if len(actions)==1:chosen=actions[0]
            elif actor in seats:
                safe=from_web_view(web_view(game,actor),actor.value);t=time.perf_counter()
                chosen,_,metrics=seats[actor].choose(safe);timings.append(time.perf_counter()-t)
                decisions+=1;simulations+=metrics['simulations'];reuse+=int(metrics['reused_nodes']>0)
                visits=[[int(i),int(e.visits)] for i,e in sorted(seats[actor].tree.nodes[()].edges.items()) if e.visits]
            else:
                t=time.perf_counter();chosen=native.select(game,actions,benchmark.DecisionContext(t,500,'policy',None));native.cold=False
                native_seconds+=time.perf_counter()-t
            require(chosen in actions,'Worker returned illegal action')
            event={'a':selfplay.action_id(chosen),'event':action_record_to_json(game.execute(chosen))}
            if visits:event['v']=visits
            events.append(event)
        generation_seconds=time.perf_counter()-start
        winner=game.winning_color()
        record={'pair':pair,'initial_sha256':initial,'events':events,'final_sha256':compact.snapshot_digest(game),
            'winner':winner.value if winner else None,'status':'completed' if winner else 'benchmark-prefix',
            'purpose':'bounded worker-throughput diagnostic; never train on this run',
            'source_prefix':job['source'],'prefix_actions':len(prefix),'candidate_checkpoint':checkpoint,
            'cpu_threads':THREADS,'job':job['id'],'opponent_kind':job['kind']}
        replay_start=time.perf_counter();compact.replay_journal(record);replay_seconds=time.perf_counter()-replay_start
        path=ROOT/output/(job['id']+'.json.gz');compact.save_journal(path,record)
        scale=1 if sys.platform=='darwin' else 1024
        return {'job':job['id'],'pid':os.getpid(),'searched_decisions':decisions,'simulations':simulations,
            'generation_seconds':generation_seconds,'replay_seconds':replay_seconds,'total_seconds':time.perf_counter()-start,
            'decision_seconds':timings,'reuse_matches':reuse,'native_seconds':native_seconds,
            'worker_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale,
            'journal':selfplay.reference(path),'final_sha256':record['final_sha256'],'replay_passed':True}
    finally:BRIDGE.close()


def process_memory(roots):
    """Sample summed RSS for worker processes and their Java children."""
    p=subprocess.run(['/bin/ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True)
    require(p.returncode==0,'Cannot sample process RSS')
    rows=[]
    for line in p.stdout.splitlines():
        values=line.split()
        if len(values)==3:rows.append(tuple(map(int,values)))
    family=set(roots)
    while True:
        additions={pid for pid,parent,rss in rows if parent in family}-family
        if not additions:break
        family.update(additions)
    return sum(rss*1024 for pid,parent,rss in rows if pid in family)


def one_case(workers,threads,jobs,protocol,ref,frozen,repeat):
    label=f'repeat-{repeat}-workers-{workers}-threads-{threads}'
    folder=OUTPUT/label;folder.mkdir(parents=True,exist_ok=True)
    ctx=mp.get_context('spawn');barrier=ctx.Barrier(workers+1);pids=ctx.Queue()
    started=time.perf_counter();results=[];peak=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize,
            initargs=(ref,frozen,threads,barrier,pids,jobs[0])) as pool:
        boots=[pool.submit(ready) for _ in range(workers)]
        barrier.wait(timeout=90)
        ids=[pids.get(timeout=10) for _ in range(workers)]
        for future in boots:future.result()
        startup=time.perf_counter()-started;measured=time.perf_counter()
        pending={pool.submit(run_job,job,protocol,ref,str(folder.relative_to(ROOT))) for job in jobs}
        while pending:
            done,pending=wait(pending,timeout=.25,return_when=FIRST_COMPLETED)
            peak=max(peak,process_memory(ids))
            for f in done:results.append(f.result())
        elapsed=time.perf_counter()-measured
    ordered=sorted(results,key=lambda r:r['job'])
    decisions=sum(r['searched_decisions'] for r in results);sims=sum(r['simulations'] for r in results)
    times=[x for r in results for x in r['decision_seconds']]
    result={'workers':workers,'cpu_threads_per_worker':threads,'repeat':repeat,'jobs':len(jobs),
        'startup_seconds':startup,'measured_seconds':elapsed,'search_decisions':decisions,'simulations':sims,
        'search_decisions_per_second':decisions/elapsed,'simulations_per_second':sims/elapsed,
        'peak_sampled_worker_and_child_rss_bytes':peak,'decision_p50_seconds':benchmark.quantile(times,.5),
        'decision_p95_seconds':benchmark.quantile(times,.95),'all_replay_passed':all(r['replay_passed'] for r in results),
        'trace_sha256':digest([(r['job'],r['final_sha256']) for r in ordered]),'results':ordered}
    write(folder/'result.json',result)
    return result


def make_jobs():
    sources=sorted((ROOT/'runs/catan-ai/fresh-xdcnn-visible/round-001/collection').glob('*.json.gz'))[:8]
    require(len(sources)==8,'Need eight completed source games')
    jobs=[]
    kinds=['current','current','historical','JSETTLERS']
    for phase,fraction in enumerate((0.,.25,.5)):
        for i,path in enumerate(sources):
            with gzip.open(path,'rt') as f:r=json.load(f)
            require(r['status']=='completed','Incomplete benchmark source')
            jobs.append({'id':f'phase-{phase}-board-{i}','source':selfplay.reference(path),
                'prefix':int(len(r['events'])*fraction),'searched_decisions':4,'kind':kinds[(i+phase)%4]})
    return jobs


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    ref=json.loads((ROOT/'runs/catan-ai/fresh-xdcnn-visible/pretraining/iteration-1/training.json').read_text())['checkpoint']
    frozen=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
    protocol=json.loads((EXPERIMENTS/'fresh-xdcnn-visible.json').read_text())
    jobs=make_jobs()
    if args.smoke:
        with benchmark.exclusive_benchmark():
            r=one_case(2,1,jobs[:2],protocol,ref,frozen,0)
        print(json.dumps({k:r[k] for k in ('workers','search_decisions','all_replay_passed','search_decisions_per_second')}),flush=True)
        return
    identity={'source_sha256':file_digest(Path(__file__)),'checkpoint':ref,'frozen':frozen,
        'jobs':jobs,'simulations':128,'purpose':'bounded concurrency benchmark; training stays paused'}
    path=OUTPUT/'protocol.json'
    if path.exists():require(json.loads(path.read_text())==identity,'Benchmark configuration changed')
    else:write(path,identity)
    configs=[(w,t) for w in (1,2,4,6,8) for t in (1,2)]+[(1,6)]
    random.Random(1802).shuffle(configs)
    cases=[(w,t,rep) for rep,order in [(1,configs),(2,list(reversed(configs)))] for w,t in order]
    with benchmark.exclusive_benchmark():
        all_results=[]
        for i,(workers,threads,repeat) in enumerate(cases):
            label=f'repeat-{repeat}-workers-{workers}-threads-{threads}';p=OUTPUT/label/'result.json'
            write(OUTPUT/'progress.json',{'status':'running','case':i+1,'total_cases':len(cases),'workers':workers,'threads':threads,'repeat':repeat,'completed_cases':len(all_results)})
            if p.exists():r=json.loads(p.read_text())
            else:r=one_case(workers,threads,jobs,protocol,ref,frozen,repeat)
            all_results.append(r)
            print(f'{i+1}/{len(cases)}: workers={workers} threads={threads} repeat={repeat}: {r["search_decisions_per_second"]:.2f} searches/s, RSS={r["peak_sampled_worker_and_child_rss_bytes"]/1e9:.2f}GB',flush=True)
        summary=[]
        for workers,threads in sorted(configs):
            selected=[r for r in all_results if r['workers']==workers and r['cpu_threads_per_worker']==threads]
            seconds=sum(r['measured_seconds'] for r in selected);decisions=sum(r['search_decisions'] for r in selected)
            summary.append({'workers':workers,'threads':threads,'search_decisions_per_second':decisions/seconds,
                'simulations_per_second':128*decisions/seconds,'peak_rss_gb':max(r['peak_sampled_worker_and_child_rss_bytes'] for r in selected)/1e9,
                'startup_seconds_mean':sum(r['startup_seconds'] for r in selected)/len(selected),
                'replays_passed':all(r['all_replay_passed'] for r in selected),
                'trace_identical_between_repeats':len({r['trace_sha256'] for r in selected})==1})
        write(OUTPUT/'summary.json',{'status':'completed','configurations':summary,'scope':'24 matched short game continuations, four searched decisions each, two reverse-order passes; CPU search with native opponent mixture; warm startup excluded from throughput but separately reported; includes replay and compact persistence; not a full-game strength or throughput certification'})
        write(OUTPUT/'progress.json',{'status':'completed','completed_cases':len(cases),'total_cases':len(cases)})
        print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
