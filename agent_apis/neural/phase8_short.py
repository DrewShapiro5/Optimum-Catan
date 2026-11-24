"""Bounded phase-8 throughput/diagnostic pass after user-shortened phase 7.

No match generation, chance allocation changes or automatic promotion.
"""
from collections import Counter
from datetime import datetime,timezone
import json
import math
import time
import statistics
from types import SimpleNamespace
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,file_digest,digest
from agent_apis.neural.pipeline import write
from agent_apis.neural.provider import NeuralCheckpoint
from agent_apis.neural.inference_cache import InferenceCache
from agent_apis.neural.phase7_runtime import measured_search
from agent_apis.search_validation import frames_for_positions
from agent_apis.stochastic_search import root_value

OUTPUT=ROOT/'runs/catan-ai'
STATE=OUTPUT/'phase8-progress.json'


def run():
    state={'status':'running','stage':'validation-positions','started_at':datetime.now(timezone.utc).isoformat(),
           'scope':'Bounded phase8 inference-cache correctness/throughput and value/opponent diagnostics; full chance allocation sweep deferred',
           'positions_planned':12,'final_seeds_used':False,'full_original_phase8_complete':False,
           'source_sha256':{p:file_digest(ROOT/p) for p in ('agent_apis/neural/phase8_short.py','agent_apis/neural/inference_cache.py','agent_apis/neural/phase7_runtime.py')}}
    write(STATE,state)
    phase6=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
    candidate=phase6['training']['XD-CNN:11:D2-W0']['checkpoint']
    enemy=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
    for reference in (candidate,enemy):assert file_digest(ROOT/reference['path'])==reference['sha256']
    model=NeuralCheckpoint(ROOT/candidate['path'],device='cpu');opponent=NeuralCheckpoint(ROOT/enemy['path'],device='cpu')
    state.update(checkpoint=candidate,frozen_opponent=enemy);write(STATE,state)
    suite=json.loads((EXPERIMENTS/'search-positions-v1.json').read_text());quota=Counter();chosen=[]
    for p in suite['positions']:
        if quota[p['stratum']]<2:chosen.append(p);quota[p['stratum']]+=1
    write(EXPERIMENTS/'phase8-short.json',{'phase':8,'scope':state['scope'],'positions':chosen,'budget_ms':500,
       'fixed_simulation_correctness_check':16,'timing_repeats':2,'checkpoint':candidate,'frozen_opponent':enemy,
       'no_new_games':True,'promotion':'No installation/promotion from position timing alone; retain C0/c=1.5',
       'final_seeds_used':False})
    samples=[]
    try:
        for position,_,view,row in frames_for_positions({'positions':chosen}):
            probabilities,value=model.evaluate(view);enemy_probs,_=opponent.evaluate(view)
            assert root_value(value,view['perspective'],view['perspective'])==value
            outcome=row['value_target']
            sample={'position':position,'value':value,'outcome':outcome,'brier':(value-outcome)**2,
                    'value_bce':-(outcome*math.log(max(value,1e-9))+(1-outcome)*math.log(max(1-value,1e-9))),
                    'frozen_opponent_top1_correct':max(enemy_probs,key=enemy_probs.get)==row['action_id'],
                    'frozen_opponent_teacher_action_probability':enemy_probs[row['action_id']],
                    'forced':len(probabilities)==1,'timings':{}}
            # Same 16 simulations, root seed and priors; cache has no sampling RNG.
            # One worker isolates numerical effects from thread scheduling.
            from agent_apis.neural.phase7_runtime import TelemetrySearch
            baseline=TelemetrySearch(view,seed=73,evaluator=model,opponent=opponent).run(16)
            cached_model=InferenceCache(model);cached_enemy=InferenceCache(opponent)
            cached=TelemetrySearch(view,seed=73,evaluator=cached_model,opponent=cached_enemy).run(16)
            assert baseline['action']==cached['action']
            assert baseline['root_statistics']==cached['root_statistics']
            sample['fixed_simulation_equivalent']=True
            if not sample['forced']:
                for repeat in range(2):
                    for use_cache in ((False,True) if repeat==0 else (True,False)):
                        evaluator=InferenceCache(model) if use_cache else model
                        enemy_eval=InferenceCache(opponent) if use_cache else opponent
                        started=time.perf_counter();context=benchmark.DecisionContext(started,500,'time',None)
                        result=measured_search(view,73+repeat,evaluator,enemy_eval,4,1000000,context,1.5)
                        elapsed=time.perf_counter()-started
                        name='cached' if use_cache else 'baseline'
                        sample['timings'].setdefault(name,[]).append({'elapsed_ms':elapsed*1000,'metrics':result['metrics'],
                            'cache':evaluator.metrics() if use_cache else None})
            samples.append(sample);state.update(stage=f'positions:{len(samples)}/12',positions_completed=len(samples));write(STATE,state)
            print(f"Phase8 bounded pass {len(samples)}/12: {position['stratum']}",flush=True)
        def median_throughput(name):
            values=[r['metrics']['simulations_per_second'] for s in samples for r in s['timings'].get(name,[])]
            return statistics.median(values) if values else None
        before=median_throughput('baseline');after=median_throughput('cached')
        state.update(status='completed',stage='completed-bounded-pass',samples=samples,
            summary={'positions':len(samples),'brier':statistics.mean(s['brier'] for s in samples),
              'mean_value_bce':statistics.mean(s['value_bce'] for s in samples),
              'frozen_opponent_teacher_top1':statistics.mean(s['frozen_opponent_top1_correct'] for s in samples),
              'baseline_median_simulations_per_second':before,'cached_median_simulations_per_second':after,
              'throughput_change_fraction':after/before-1 if before and after else None,
              'fixed_simulation_equivalence_passed':all(s['fixed_simulation_equivalent'] for s in samples),
              'perspective_check':'Actor-to-root conversion verified on every sample; full regression suite covers private information and chance invariance',
              'limitations':'12 stratified validation positions; value calibration and frozen-opponent imitation are diagnostic only, not on-policy strength or opponent-response calibration.',
              'decision':'Retain C0/c=1.5 and existing runtime; optional cache is not installed. No search strength gain established, so defer C1-C4 and progressive widening.',
              'full_original_phase8_complete':False,'next_work':'Learner improvement/data collection is the next substantive stage; no automatic large generation queue.'})
        write(STATE,state);write(EXPERIMENTS/'PHASE8_SHORT_RESULTS.json',state)
        lines=['# Phase 8 bounded diagnostic and throughput pass','',state['scope']+'.','',
            'No new games, training, Transformer or random-player runs. Twelve fixed validation positions; '
            'cache and baseline alternate measurement order at 500 ms, with a fixed-16-simulation correctness comparison.','',
            '```json',json.dumps(state['summary'],indent=2),'```','',
            'The cache key includes permitted features, legal actions and development-card knowledge. '
            'Caches are separate per checkpoint and per decision and are not installed as a playing default. '
            'The original phase-eight chance-allocation/progressive-widening comparison is deferred, not claimed complete.','']
        (EXPERIMENTS/'PHASE8_SHORT_RESULTS.md').write_text('\n'.join(lines))
        print(json.dumps(state['summary']),flush=True)
    except BaseException as error:
        state.update(status='needs-investigation',error=f'{type(error).__name__}: {error}');write(STATE,state);raise

if __name__=='__main__':
    with benchmark.exclusive_benchmark():run()
