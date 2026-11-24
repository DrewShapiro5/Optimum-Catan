"""Bounded real-move subtree-reuse audit; no training or model promotion."""
from collections import Counter
import json
import random
import statistics
import time
from catanatron.models.player import Color
from catanatron.serialization import web_view,action_from_json
from agent_apis import benchmark
from agent_apis.benchmark import DecisionContext,named_seed
from agent_apis.experiment import ROOT,EXPERIMENTS,file_digest
from agent_apis.neural.pipeline import write
from agent_apis.neural.provider import NeuralCheckpoint
from agent_apis.neural.subtree_reuse import SearchSession,allocation_key
from agent_apis.stochastic import canonical_successor,world_from_observation
from agent_apis.observation import from_web_view,require
from agent_apis.search_validation import frames_for_positions
from agent_apis.jsettlers.player import BRIDGE

OUTPUT=ROOT/'runs/catan-ai/subtree-reuse-audit.json'


def run():
    phase6=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
    reference=phase6['training']['XD-CNN:11:D2-W0']['checkpoint']
    enemy=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
    for r in (reference,enemy):require(file_digest(ROOT/r['path'])==r['sha256'],'Checkpoint changed')
    policy=NeuralCheckpoint(ROOT/reference['path'],device='cpu');opponent=NeuralCheckpoint(ROOT/enemy['path'],device='cpu')
    suite=json.loads((EXPERIMENTS/'search-positions-v1.json').read_text());quota=Counter();chosen=[]
    for p in suite['positions']:
        if quota[p['stratum']]<2:chosen.append(p);quota[p['stratum']]+=1
    report={'status':'running','scope':'12 frozen validation starts; real independently sampled transitions and native JSettlers opponent; paired cold/warm 500ms decisions',
        'checkpoint':reference,'frozen_opponent':enemy,'positions':[],'training_started':False,'final_seeds_used':False,
        'source_sha256':{p:file_digest(ROOT/p) for p in ('agent_apis/neural/subtree_reuse.py','agent_apis/stochastic_search.py','agent_apis/neural/provider.py','agent_apis/benchmark.py')}}
    write(OUTPUT,report)
    try:
        for index,(position,original,view,row) in enumerate(frames_for_positions({'positions':chosen})):
            if len(view['current_playable_actions'])==1:continue
            game=original.copy();game.random=game.state.random=random.Random(named_seed(index,'reuse-audit-real-chance'))
            root=Color[view['perspective']];session=SearchSession();session_id=f'audit-{index}'
            # Preparation and all timed calls get fresh admission deadlines.
            start=time.perf_counter();first=session.run(view,41,policy,opponent,4,1000000,
                DecisionContext(start,500,'time',None),session_id=session_id)
            game.execute(action_from_json(first['action']))
            BRIDGE.close();enemy_actor=benchmark.Agent('JSETTLERS',next(c for c in game.state.colors if c!=root),index)
            actions_between=1
            while game.winning_color() is None and actions_between<50:
                actor=game.state.current_color();actions=list(game.playable_actions)
                if actor==root and len(actions)>1:break
                if len(actions)==1:action=actions[0]
                elif actor!=root:
                    action=enemy_actor.select(game,actions,DecisionContext(time.perf_counter(),500,'policy',None));enemy_actor.cold=False
                else:break
                game.execute(action);actions_between+=1
            BRIDGE.close()
            if game.winning_color() is not None or game.state.current_color()!=root or len(game.playable_actions)==1:
                report['positions'].append({'position':position,'skipped':'terminal or no next nonforced root decision'});write(OUTPUT,report);continue
            next_view=from_web_view(web_view(game,root),root.value)
            trials={}
            for mode in (('warm','cold') if index%2 else ('cold','warm')):
                manager=session if mode=='warm' else SearchSession()
                start=time.perf_counter();result=manager.run(next_view,71,policy,opponent,4,1000000,
                    DecisionContext(start,500,'time',None),session_id=session_id)
                elapsed=(time.perf_counter()-start)*1000
                require(result['action'] in next_view['current_playable_actions'],'Illegal audit decision')
                trials[mode]={'elapsed_ms':elapsed,'metrics':result['metrics']}
                # Every retained root world must equal a fresh decode with its
                # own sampled private allocation; no authoritative secrets used.
                if mode=='warm':
                    checked=0
                    for tree in manager.trees:
                        if not tree.metrics['reused_nodes']:continue
                        for prototype in list(tree.world_templates.values())[:3]:
                            old,new,draws,deck,flags=allocation_key(prototype,tree.root)
                            allocation={'opponent_old':old,'opponent_new':new,'opponent_draws':dict(draws),'deck':deck,'opponent_start_flags':flags}
                            fresh=world_from_observation(tree.view,allocation)
                            require(canonical_successor(prototype)==canonical_successor(fresh),'Retained root world differs from fresh decode')
                            checked+=1
                    trials[mode]['root_world_equivalence_checks']=checked
            report['positions'].append({'position':position,'actions_between':actions_between,'trials':trials})
            write(OUTPUT,report);print(f"Audit {len(report['positions'])}/12: {position['stratum']}, reused nodes={trials['warm']['metrics']['reused_nodes']}",flush=True)
        measured=[p['trials'] for p in report['positions'] if 'trials' in p]
        hits=[t for t in measured if t['warm']['metrics']['reused_nodes']]
        def throughput(mode,group):return statistics.median(t[mode]['metrics']['simulations_per_second'] for t in group) if group else None
        before=throughput('cold',measured);after=throughput('warm',measured)
        report.update(status='completed',summary={'measured_transitions':len(measured),'transitions_with_reuse':len(hits),
            'cold_median_simulations_per_second':before,'warm_median_simulations_per_second':after,
            'median_throughput_change_fraction':after/before-1 if before else None,
            'warm_reuse_hit_median_simulations_per_second':throughput('warm',hits),'cold_same_hit_positions_median_simulations_per_second':throughput('cold',hits),
            'root_world_equivalence_checks':sum(t['warm']['root_world_equivalence_checks'] for t in measured),
            'warm_max_decision_ms':max(t['warm']['elapsed_ms'] for t in measured),
            'warm_p95_decision_ms':benchmark.quantile([t['warm']['elapsed_ms'] for t in measured],.95),
            'illegal_actions':0,'strength_improvement_claimed':False,
            'limitations':'Small position diagnostic; warm search starts with valid retained computation. Throughput counts new simulations only. No win-rate claim or self-play training.'})
        write(OUTPUT,report);write(EXPERIMENTS/'SUBTREE_REUSE_RESULTS.json',report)
        (EXPERIMENTS/'SUBTREE_REUSE_RESULTS.md').write_text('# Subtree reuse audit\n\n```json\n'+json.dumps(report['summary'],indent=2)+'\n```\n\nFull paired position results and source hashes are in SUBTREE_REUSE_RESULTS.json.\n')
        print(json.dumps(report['summary']),flush=True)
    except BaseException as error:
        report.update(status='needs-investigation',error=f'{type(error).__name__}: {error}');write(OUTPUT,report);raise
    finally:BRIDGE.close()

if __name__=='__main__':
    with benchmark.exclusive_benchmark():run()
