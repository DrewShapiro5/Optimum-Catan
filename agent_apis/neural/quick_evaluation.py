"""User-requested 10-game smoke / 50-game total optimized-search evaluation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest
from agent_apis.neural.pipeline import config_for,locked,write,games,eligible
from agent_apis.neural.queue import completed
from agent_apis.neural.evaluation import common_prefix
from agent_apis.observation import require

ORDER=('H-GNN','TRANSFORMER','XD-CNN','MLP')
OUTPUT=ROOT/'runs/catan-ai'
STATE=OUTPUT/'phase5-quick-evaluation.json'
CORE=('agent_apis/benchmark.py','agent_apis/futility.py','agent_apis/stochastic.py',
      'agent_apis/stochastic_search.py','agent_apis/development_belief.py',
      'agent_apis/neural/provider.py','agent_apis/neural/parallel_search.py',
      'agent_apis/neural/models.py','agent_apis/neural/topology.py','agent_apis/observation.py',
      'agent_apis/neural/quick_evaluation.py','experiments/catan-ai/evaluation-quick-v1.json',
      'experiments/catan-ai/search-optimized-v1.json','experiments/catan-ai/phase5-frozen-opponent.json')


class QuickEvaluation:
    def __init__(self,report_only=False):
        if report_only:
            self.state=json.loads(STATE.read_text())
            self.core=self.state['source_sha256']
            return
        self.core={p:file_digest(ROOT/p) for p in CORE}
        self.state={'status':'running','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
            'schedule':{'smoke_games':10,'total_games':50,'includes_smoke':True},'budget_ms':500,
            'order':list(ORDER),'stage':'initializing','source_sha256':self.core,'runs':{},
            'scope':'Seed11 optimized search; reuse historical policy references; no new training or automatic confirmation',
            'final_seeds_used':False}
        self.persist()
    def persist(self):
        self.state['updated_at']=datetime.now(timezone.utc).isoformat();write(STATE,self.state)
    def guard(self):
        require(all(file_digest(ROOT/p)==sha for p,sha in self.core.items()),'Quick evaluation implementation changed')
    def configuration(self,model,tier):
        config=config_for(model,11);training=completed(config)
        require(training,'Required seed11 training run is missing; this runner never launches training')
        path,record=training
        config.update(stage='benchmark',experiment='phase5-quick-optimized-search-v1',implementation_sha256=self.core)
        config['candidate']={'key':'NEURAL_SEARCH','checkpoint':record['checkpoint']}
        config['opponent']={'key':'JSETTLERS','checkpoint':None}
        config['training_run_record']=str(path.relative_to(ROOT))
        config['evaluation'].update(schedule='quick-v1',schedule_sha256=file_digest(EXPERIMENTS/'evaluation-quick-v1.json'),
            tier=tier,seed_set='tuning',mode='time',budget_ms=500,simulations=None,futility=dict(benchmark.DEFAULT_FUTILITY),
            smoke_pairs=5,screening_pairs=25)
        optimization=json.loads((EXPERIMENTS/'search-optimized-v1.json').read_text())
        config['search'].update(max_root_depth=None,workers=optimization['workers_by_model'].get(model,1),
            revision=optimization['revision'],opponent_checkpoint=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint'])
        return config
    def existing(self,config):
        for p in sorted(OUTPUT.glob('phase5q-*/run.json')):
            record=json.loads(p.read_text())
            if record['configuration']['sha256']!=digest(config):continue
            require(record['status'] in ('completed','stopped-early'),'Matching quick run failed; investigate before resuming')
            for name,sha in record['metrics']['artifacts'].items():require(file_digest(p.parent/name)==sha,'Completed quick logs changed')
            return p,record
        return None
    def measure(self,model,tier,prior=None):
        config=self.configuration(model,tier)
        write(EXPERIMENTS/'phase5-quick-configs'/f'{model}-seed11-search-{tier}.json',config)
        self.state['stage']=f'{model}:search:{tier}';self.persist()
        with locked():
            self.guard();found=self.existing(config)
            if found:
                path,record=found;print(f'REUSE {model} {tier}: {record["metrics"]["benchmark"]["games"]} completed games',flush=True)
            else:
                print(f'RUN {model} seed11 {tier}: {10 if tier=="smoke" else 50} games TOTAL; workers={config["search"]["workers"]}; own_action_limit=None',flush=True)
                path=benchmark.run(config,OUTPUT,prior);record=json.loads(path.read_text())
        self.state['runs'][f'{model}:{tier}']={'run_record':str(path.relative_to(ROOT)),
            'metrics':record['metrics']['benchmark'],'status':record['status']}
        self.persist();return path,record
    def policy_reference(self,model):
        checkpoint=self.configuration(model,'smoke')['candidate']['checkpoint'];options=[]
        for p in OUTPUT.glob('phase3-NEURAL_POLICY-*/run.json'):
            r=json.loads(p.read_text());c=r['configuration']['content']
            if c['candidate']['checkpoint']!=checkpoint or r['status'] not in ('completed','stopped-early'):continue
            if r['status']=='stopped-early':
                require(r['metrics']['benchmark'].get('futility',{}).get('stopped'),
                        'Historical policy stopped without a recorded futility decision')
            for name in ('agent_apis/neural/models.py','agent_apis/neural/topology.py','agent_apis/observation.py'):
                require(r['source']['files'][name]==file_digest(ROOT/name),'Historical policy model/features changed')
            require(r['metrics']['providers']['opponent']==benchmark.provider_identity('JSETTLERS'),'Historical native opponent changed')
            for name,sha in r['metrics']['artifacts'].items():require(file_digest(p.parent/name)==sha,'Historical policy logs changed')
            rows=games(p)[:50];options.append((len(rows),p,rows))
        require(options,f'No preserved policy baseline for {model}')
        _,path,rows=max(options,key=lambda item:item[0])
        return path,rows
    def finish(self):
        measured={}
        # All models get smoke results before any long extension.
        for model in ORDER:measured[model]=self.measure(model,'smoke')
        for model in ORDER:
            path,record=measured[model]
            if record['metrics']['benchmark'].get('futility',{}).get('stopped'):continue
            measured[model]=self.measure(model,'screening',path)
        self.publish(measured)
    def report_saved(self):
        """Recover reporting without entering the game-playing or training paths."""
        with locked():
            require(all(file_digest(ROOT/p)==sha for p,sha in self.core.items()
                        if p!='agent_apis/neural/quick_evaluation.py'),
                    'Saved evaluation implementation changed beyond the reporting module')
            measured={}
            for model in ORDER:
                prior=None
                for tier in ('smoke','screening'):
                    if prior and prior[1]['metrics']['benchmark'].get('futility',{}).get('stopped'):break
                    saved=self.state['runs'].get(f'{model}:{tier}')
                    require(saved,f'Missing saved {model} {tier}; report-only never launches games')
                    path=ROOT/saved['run_record'];record=json.loads(path.read_text())
                    require(record['configuration']['sha256']==digest(self.configuration(model,tier))
                            and record['configuration']['sha256']==digest(record['configuration']['content']),
                            'Saved quick configuration changed')
                    require(record['status'] in ('completed','stopped-early')
                            and record['status']==saved['status']
                            and record['metrics']['benchmark']==saved['metrics'],
                            'Saved quick result changed or is incomplete')
                    for name,sha in record['metrics']['artifacts'].items():
                        require(file_digest(path.parent/name)==sha,'Saved quick logs changed')
                    if prior:
                        require(games(prior[0])==games(path)[:10],'Saved smoke prefix changed')
                    prior=(path,record)
                measured[model]=prior
            self.state['reporting_recovery']={'previous_status':self.state['status'],
                'previous_error':self.state.get('error'),'games_replayed':0,
                'reporter_sha256':file_digest(Path(__file__))}
            self.publish(measured)
    def publish(self,measured):
        results={'status':'completed','scope':self.state['scope'],'schedule':self.state['schedule'],'budget_ms':500,
                 'all_models_evaluated':True,'final_seeds_used':False,'training_seeds':[11],'architectures':{},'source_sha256':self.core}
        if 'reporting_recovery' in self.state:results['reporting_recovery']=self.state['reporting_recovery']
        lines=['# Phase 5 quick optimized-search evaluation','',
               '10-game smoke followed by at most 50 games total, including smoke. '
               'All four seed11 checkpoints use 500 ms optimized search. All smoke runs precede extensions. '
               'Existing policy references are reused, not replayed. No training seeds or final evaluation games are launched.','',
               '| Model | Search games | Wins | Losses | Cutoffs | Errors | Match score | Matched policy comparison |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for model,(path,record) in measured.items():
            metrics=record['metrics']['benchmark'];reference,policy=self.policy_reference(model)
            reference_status=json.loads(reference.read_text())['status']
            search=games(path);a,b=common_prefix([search,policy]);comparison=benchmark.paired_difference(a,b)
            comparison.update(games=len(a),descriptive_only=True)
            if metrics.get('futility',{}).get('stopped') or reference_status=='stopped-early':
                comparison['ci95']=[None,None]
                comparison['interval_status']='Fixed-sample interval suppressed because search or policy stopped early'
            results['architectures'][model]={'search_run':str(path.relative_to(ROOT)),'metrics':metrics,
                'status':record['status'],'promotion_eligible':eligible(record),
                'policy_reference':str(reference.relative_to(ROOT)),'policy_reference_sha256':file_digest(reference),
                'policy_reference_status':reference_status,
                'policy_reference_games_used':len(b),'search_minus_policy':comparison}
            lines.append(f"| {model} | {metrics['games']} | {metrics['wins']} | {metrics['losses']} | {metrics['cutoffs']} | {metrics['errors']} | {metrics['overall_match_score']:.3f} | {comparison['difference']:+.3f} over {len(a)} games |")
        lines+=['','These are exploratory single-seed tuning results, with broad uncertainty. '
                'Futility can end a run before 50 games. The existing statistical rule needs more than five pairs '
                'to establish futility; zero wins in a 10-game smoke alone is not proof of hopelessness. '
                'Full logs, paired bootstrap intervals and depth/simulation telemetry are linked from '
                '[PHASE5_QUICK_RESULTS.json](PHASE5_QUICK_RESULTS.json).','']
        write(EXPERIMENTS/'PHASE5_QUICK_RESULTS.json',results)
        (EXPERIMENTS/'PHASE5_QUICK_RESULTS.md').write_text('\n'.join(lines))
        self.state.pop('error',None)
        self.state.update(status='completed',stage='completed',summary='experiments/catan-ai/PHASE5_QUICK_RESULTS.json');self.persist()
        print('Quick evaluation completed; no automatic training or larger evaluation follows.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true',help='Verify saved completed games and regenerate the report without playing games')
    args=parser.parse_args();work=QuickEvaluation(report_only=args.report_only)
    try:
        if args.report_only:work.report_saved()
        else:work.finish()
    except BaseException as error:
        work.state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'needs-investigation',error=f'{type(error).__name__}: {error}');work.persist();raise

if __name__=='__main__':main()
