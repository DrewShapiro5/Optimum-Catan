"""Serial XD-CNN phase-7 search tuning; no training or reserved final seeds."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest
from agent_apis.neural.pipeline import write,games,eligible
from agent_apis.observation import require

OUTPUT=ROOT/'runs/catan-ai'
STATE=OUTPUT/'phase7-progress.json'
PROTOCOL=EXPERIMENTS/'phase7.json'
CORE=('agent_apis/neural/phase7.py','agent_apis/neural/phase7_runtime.py',
 'agent_apis/neural/parallel_search.py','agent_apis/neural/provider.py','agent_apis/neural/models.py',
 'agent_apis/neural/topology.py','agent_apis/stochastic_search.py','agent_apis/stochastic.py',
 'agent_apis/development_belief.py','agent_apis/observation.py','agent_apis/benchmark.py',
 'agent_apis/futility.py','agent_apis/agents.json','experiments/catan-ai/phase7.json',
 'experiments/catan-ai/evaluation-quick-v1.json','experiments/catan-ai/phase5-frozen-opponent.json',
 'experiments/catan-ai/PHASE6_RESULTS.json')


def variants():
    return {'policy':{'mode':'policy','budget_ms':500,'simulations':None,'c_puct':1.5,'leaf_evaluator':'neural'},
      'time500-c150':{'mode':'time','budget_ms':500,'simulations':None,'c_puct':1.5,'leaf_evaluator':'neural'},
      'time500-c075':{'mode':'time','budget_ms':500,'simulations':None,'c_puct':.75,'leaf_evaluator':'neural'},
      'time500-c300':{'mode':'time','budget_ms':500,'simulations':None,'c_puct':3.,'leaf_evaluator':'neural'},
      'time100':{'mode':'time','budget_ms':100,'simulations':None,'c_puct':1.5,'leaf_evaluator':'neural'},
      'heuristic500':{'mode':'time','budget_ms':500,'simulations':None,'c_puct':1.5,'leaf_evaluator':'heuristic'},
      'time2000':{'mode':'time','budget_ms':2000,'simulations':None,'c_puct':1.5,'leaf_evaluator':'neural'},
      **{f'sim{n}':{'mode':'simulations','budget_ms':500,'simulations':n,'c_puct':1.5,'leaf_evaluator':'neural'} for n in (128,512,2048)}}


class Phase7:
    def __init__(self):
        self.protocol=json.loads(PROTOCOL.read_text());self.core={p:file_digest(ROOT/p) for p in CORE}
        self.phase6=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
        require(self.phase6['selection']['model']=='XD-CNN' and self.phase6['selection']['chosen_recipe']=='D2-W0','Phase-7 training recipe differs')
        if STATE.exists():
            self.state=json.loads(STATE.read_text())
            if self.state['source_sha256']!=self.core:
                amendment=json.loads((EXPERIMENTS/'phase7-opponent-amendment.json').read_text())
                old=self.state['source_sha256']
                changed={p for p in set(old)|set(self.core) if old.get(p)!=self.core.get(p)}
                require(old==amendment['original_source_sha256'] and
                        changed==set(amendment['allowed_changed_sources']) and
                        self.protocol==amendment['new_protocol'] and self.protocol['opponents']==['JSETTLERS'],
                        'Phase-7 implementation changed beyond authorized opponent-only amendment')
                self.state['benchmark_configuration_core']=old
                self.state['scope_amendment']=amendment
                self.state['source_sha256']=self.core
                self.state['protocol']=self.protocol
        else:
            self.state={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'protocol':self.protocol,
                        'source_sha256':self.core,'runs':{},'final_seeds_used':False,'stage':'initializing'}
        self.state.update(pid=os.getpid(),status='running');self.state.pop('error',None);self.persist();self.report()

    def persist(self):
        self.state['updated_at']=datetime.now(timezone.utc).isoformat();write(STATE,self.state)

    def guard(self):
        require(all(file_digest(ROOT/p)==sha for p,sha in self.core.items()),'Phase-7 inputs changed')

    def reference(self,seed,mode,opponent):
        """Reuse exact phase-6 checkpoint/seed/opponent/mode evidence, never pool revisions."""
        key=f'XD-CNN:{seed}:D2-W0:{mode}:{opponent}:screening'
        saved=self.phase6['benchmarks'][key];path=ROOT/saved['run_record'];record=json.loads(path.read_text())
        config=record['configuration']['content']
        require(record['status'] in ('completed','stopped-early') and record['configuration']['sha256']==digest(config),'Bad historical baseline')
        require(config['model']['id']=='XD-CNN' and config['seed']==seed and
                config['search']['c_puct']==1.5 and config['opponent']['key']==opponent and
                config['evaluation']['mode']==('policy' if mode=='policy' else 'time') and
                config['evaluation']['budget_ms']==500 and config['evaluation']['seed_set']=='tuning','Wrong historical baseline')
        training=self.phase6['training'][f'XD-CNN:{seed}:D2-W0']
        require(config['candidate']['checkpoint']==training['checkpoint'],'Historical checkpoint differs')
        require(file_digest(ROOT/training['checkpoint']['path'])==training['checkpoint']['sha256'],'Historical checkpoint changed')
        for name in ('agent_apis/neural/provider.py','agent_apis/neural/models.py','agent_apis/neural/topology.py',
                     'agent_apis/neural/parallel_search.py','agent_apis/stochastic_search.py','agent_apis/stochastic.py',
                     'agent_apis/observation.py','agent_apis/benchmark.py','agent_apis/development_belief.py'):
            require(record['source']['files'][name]==file_digest(ROOT/name),f'Historical decision implementation changed: {name}')
        for name,sha in record['metrics']['artifacts'].items():require(file_digest(path.parent/name)==sha,'Historical logs changed')
        require(record['metrics']['providers']['opponent']==benchmark.provider_identity(opponent),'Historical validation opponent changed')
        return path,record

    def retain(self,seed,variant,opponent,path,record,reused=False):
        key=f'{seed}:{variant}:{opponent}'
        self.state['runs'][key]={'run_record':str(path.relative_to(ROOT)),'status':record['status'],
            'metrics':record['metrics']['benchmark'],'eligible':eligible(record),'reused_from_phase6':reused,
            'telemetry':'historical phase-6 subset' if reused else 'phase-7 subsystem instrumentation'}
        self.persist();self.report();return path,record

    def configuration(self,seed,variant,opponent,tier):
        training=self.phase6['training'][f'XD-CNN:{seed}:D2-W0']
        record=json.loads((ROOT/training['run_record']).read_text());config=record['configuration']['content']
        choice=variants()[variant]
        # Opponent-only amendments keep the exact original benchmark identity
        # for checksum-verified reuse; current executor hashes remain in state
        # and each newly written benchmark's source provenance.
        configuration_core=getattr(self,'state',{}).get('benchmark_configuration_core',self.core)
        config.update(stage='benchmark',experiment=f'phase7-XD-CNN-{variant}',implementation_sha256=configuration_core,
                      training_run_record=training['run_record'])
        config['candidate']={'key':'PHASE7_SEARCH','checkpoint':training['checkpoint']}
        config['opponent']={'key':opponent,'checkpoint':None}
        config['evaluation'].update(schedule='quick-v1',schedule_sha256=file_digest(EXPERIMENTS/'evaluation-quick-v1.json'),
            smoke_pairs=5,screening_pairs=25,tier=tier,seed_set='tuning',futility=dict(benchmark.DEFAULT_FUTILITY),
            mode=choice['mode'],budget_ms=choice['budget_ms'],simulations=choice['simulations'],opponent=opponent)
        config['search'].update(workers=4,max_root_depth=None,c_puct=choice['c_puct'],leaf_evaluator=choice['leaf_evaluator'],
            revision='phase7-measured-optimized-search-v1',
            opponent_checkpoint=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint'])
        return config

    def existing(self,config):
        for p in sorted(OUTPUT.glob('phase5q-*/run.json')):
            r=json.loads(p.read_text())
            if r['configuration']['sha256']!=digest(config):continue
            require(r['status'] in ('completed','stopped-early'),'Matching phase-7 run failed; investigate before resume')
            for name,sha in r['metrics']['artifacts'].items():require(file_digest(p.parent/name)==sha,'Saved phase-7 logs changed')
            require(r['metrics']['providers']=={role:benchmark.provider_identity(config[role]['key']) for role in ('candidate','opponent')},'Phase-7 provider changed')
            return p,r
        return None

    def measure(self,seed,variant,opponent,tier,prior=None):
        self.guard();config=self.configuration(seed,variant,opponent,tier)
        write(EXPERIMENTS/'phase7-configs'/f'seed{seed}-{variant}-{opponent}-{tier}.json',config)
        self.state['stage']=f'evaluate:{seed}:{variant}:{opponent}:{tier}';self.persist()
        found=self.existing(config)
        if not found:
            print(f'RUN seed{seed} {variant} {opponent} {tier}',flush=True)
            p=benchmark.run(config,OUTPUT,continuation=prior);found=p,json.loads(p.read_text())
        p,r=found;self.retain(seed,variant,opponent,p,r);return p,r

    def record_for(self,seed,variant,opponent):
        row=self.state['runs'][f'{seed}:{variant}:{opponent}'];p=ROOT/row['run_record']
        return p,json.loads(p.read_text())

    def finish(self,chosen):
        differences=[];search_runs=[];policy_runs=[];feasible=True;stopped=False
        for seed in (11,23,47):
            a,ar=self.record_for(seed,chosen,'JSETTLERS');b,br=self.record_for(seed,'policy','JSETTLERS')
            ga,gb=games(a),games(b);n=min(len(ga),len(gb))//2*2
            benchmark.paired_difference(ga[:n],gb[:n])
            lookup={(g['pair_id'],g['candidate_color']):g for g in gb[:n]}
            differences.append((seed,[{**g,'match_score':g['match_score']-lookup[g['pair_id'],g['candidate_color']]['match_score']} for g in ga[:n]]))
            search_runs.append((seed,ga));policy_runs.append((seed,gb))
            feasible=feasible and all(eligible(self.record_for(seed,chosen,o)[1]) for o in self.protocol['opponents'])
            stopped=stopped or ar['status']=='stopped-early' or br['status']=='stopped-early'
        # Align pairs across seeds when a run ended for futility. The resulting
        # interval is descriptive and cannot justify search promotion.
        common=set(g['pair_id'] for g in differences[0][1])
        for _,gs in differences:common&={g['pair_id'] for g in gs}
        diff=benchmark.hierarchical_bootstrap([(seed,[g for g in gs if g['pair_id'] in common]) for seed,gs in differences])
        if stopped:diff.update(descriptive_only=True,ci95_for_promotion=None)
        self.state['conclusion']={'chosen_500ms_variant':chosen,'search_minus_policy':diff,
            'all_search_gates_passed':feasible,'search_promoted':bool(feasible and not stopped and diff['ci95'][0]>0),
            'installed_default_changed':False,'final_seeds_used':False,
            'required_next_step':'Proceed only with demonstrated search gain; otherwise diagnose value calibration, perspective and frozen-opponent mismatch before adding complexity.'}
        self.state.update(status='completed',stage='completed',pid=None);self.persist();self.report(complete=True)
        print('Phase 7 complete. No automatic phase-8/self-play launch.',flush=True)

    def report(self,complete=False):
        destination='PHASE7_RESULTS' if complete else 'PHASE7_PROGRESS'
        write(EXPERIMENTS/(destination+'.json'),self.state)
        lines=['# Phase 7 XD-CNN search '+('results' if complete else 'progress'),'','D2/W0 checkpoints only; no new training. '
               'JSettlers SMART only under the user opponent-scope amendment; tuning seeds, 10 smoke / 50 total games. '
               'Verified phase-6 policy and 500 ms c=1.5 references are reused without replay. '
               'Fixed-simulation runs are diagnostic and have no wall-clock eligibility gate for promotion.','',
               '| Seed | Variant | Opponent | Games | Wins | Cutoffs | Score | Warm p95 ms | Source |',
               '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for key,row in self.state['runs'].items():
            seed,variant,opponent=key.split(':');m=row['metrics'];p95=m['candidate_latency_warm_nonforced']['p95_ms']
            if opponent not in self.protocol['opponents']:continue
            lines.append(f"| {seed} | {variant} | {opponent} | {m['games']} | {m['wins']} | {m['cutoffs']} | {m['overall_match_score']:.3f} | {p95:.1f} | {'phase6' if row['reused_from_phase6'] else 'phase7'} |")
        lines+=['','Variant definitions, source/checkpoint hashes, pair intervals, latency and provider gates are in the JSON. '
                'Per-decision timing logs record search telemetry; legacy references have only their original subset. '
                'Subsystem times overlap and sum across worker threads; they are not additive wall time. '
                'Heuristic control keeps neural priors and the same frozen MLP opponent, changing only the leaf estimate. '
                'Final evaluation seeds remain unused.','']
        if 'conclusion' in self.state:lines+=['Conclusion: '+json.dumps(self.state['conclusion']), '']
        (EXPERIMENTS/(destination+'.md')).write_text('\n'.join(lines))

    def execute(self):
        # Reuse all exact baseline checkpoints/results before scheduling new work.
        for seed in (11,23,47):
            for variant,mode in (('policy','policy'),('time500-c150','search')):
                for opponent in self.protocol['opponents']:
                    p,r=self.reference(seed,mode,opponent);self.retain(seed,variant,opponent,p,r,True)
        for variant in self.protocol['new_variants']:
            smokes={o:self.measure(11,variant,o,'smoke') for o in self.protocol['opponents']}
            for opponent,(p,r) in smokes.items():
                if not r['metrics']['benchmark'].get('futility',{}).get('stopped'):
                    self.measure(11,variant,opponent,'screening',p)
        candidates=[]
        for v in ('time500-c150','time500-c075','time500-c300'):
            _,r=self.record_for(11,v,'JSETTLERS')
            if all(eligible(self.record_for(11,v,o)[1]) for o in self.protocol['opponents']):
                candidates.append((r['metrics']['benchmark']['overall_match_score'],v))
        require(candidates,'No feasible 500 ms search; diagnose before confirmation')
        order=('time500-c150','time500-c075','time500-c300')
        chosen=max(candidates,key=lambda t:(t[0],-order.index(t[1])))[1]
        self.state['chosen_500ms_variant']=chosen;self.persist()
        if chosen!='time500-c150':
            for seed in (23,47):
                smokes={o:self.measure(seed,chosen,o,'smoke') for o in self.protocol['opponents']}
                for opponent,(p,r) in smokes.items():
                    if not r['metrics']['benchmark'].get('futility',{}).get('stopped'):
                        self.measure(seed,chosen,opponent,'screening',p)
        self.finish(chosen)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    if (EXPERIMENTS/'phase7-stop-request.json').exists():
        print('Phase 7 closed early by user. See PHASE7_EARLY_RESULTS.md; no games launched.')
        return
    with benchmark.exclusive_benchmark():
        work=Phase7()
        try:work.execute()
        except BaseException as error:
            work.state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'needs-investigation',error=f'{type(error).__name__}: {error}')
            work.persist();raise

if __name__=='__main__':main()
