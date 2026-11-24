"""Serial, resumable phase-5 training and tuning-only architecture comparison.

Run after (or alongside) the initial training queue. The process waits for that
queue, then owns the same exclusive benchmark lock for each measurement. Every
resolved match configuration pins its checkpoint and relevant implementation.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest
from agent_apis.neural.models import MODEL_IDS
from agent_apis.neural.queue import completed
from agent_apis.neural.train import train
from agent_apis.observation import require

OUTPUT=ROOT/'runs/catan-ai'
STATE=OUTPUT/'phase5-pipeline.json'
CORE=('agent_apis/neural/parallel_search.py','experiments/catan-ai/search-optimized-v1.json','agent_apis/futility.py','agent_apis/benchmark.py','agent_apis/neural/provider.py',
      'agent_apis/neural/models.py','agent_apis/neural/topology.py',
      'agent_apis/neural/pipeline.py','agent_apis/stochastic.py',
      'agent_apis/stochastic_search.py','agent_apis/development_belief.py',
      'agent_apis/observation.py','agent_apis/neural/train.py',
      'agent_apis/neural/data.py','agent_apis/agents.json',
      'experiments/catan-ai/phase5-frozen-opponent.json',
      'experiments/catan-ai/feature-schema-v3.json')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def config_for(model,seed):
    return json.loads((EXPERIMENTS/f'phase5-configs/{model}-seed{seed}.json').read_text())


@contextmanager
def locked():
    while True:
        try:
            context=benchmark.exclusive_benchmark();context.__enter__();break
        except RuntimeError as error:
            if 'Another benchmark is running' not in str(error):raise
            time.sleep(5)
    try:yield
    finally:context.__exit__(None,None,None)


def games(path):
    return [json.loads(line) for line in (path.parent/'games.jsonl').read_text().splitlines()]


def eligible(record):
    metrics=record['metrics']['benchmark']
    return not metrics.get('futility',{}).get('stopped',False) and all(metrics[key] for key in ('failure_gate_passed','timing_gate_passed','cutoff_gate_passed'))


class Pipeline:
    def __init__(self,cache):
        self.cache=cache
        self.core={name:file_digest(ROOT/name) for name in CORE}
        self.state={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
                    'source':self.core,'stage':'waiting-for-seed11-training','benchmarks':{},
                    'selection_rule':'Rank each architecture by its highest feasible seed-11 match score across policy and 500ms search; select top two, ties by fewer parameters. Report pair-clustered uncertainty; selection is tuning, not a confirmed claim.',
                    'checkpoint_diagnostic':'For each finalist seed11 compare validation-selected best and final20k on identical 100 tuning pairs, in policy and 500ms search; retain the predeclared best checkpoint for the baseline comparison.',
                    'final_seeds_used':False}
        self.persist()

    def persist(self):
        self.state['updated_at']=datetime.now(timezone.utc).isoformat();write(STATE,self.state)

    def guard(self):
        require(all(file_digest(ROOT/name)==sha for name,sha in self.core.items()),
                'Pipeline implementation changed during execution; stop and resume with reviewed sources')

    def trained(self,model,seed,wait=False):
        config=config_for(model,seed)
        while True:
            self.guard();found=completed(config)
            if found:return found
            if wait:time.sleep(5);continue
            with locked():
                self.guard();found=completed(config)
                if found:return found
                resumable=[p for p in OUTPUT.glob('phase5-*')
                           if (p/'configuration.json').exists() and (p/'resume.pt').exists()
                           and not (p/'run.json').exists()
                           and digest(json.loads((p/'configuration.json').read_text()))==digest(config)]
                train(config,self.cache,OUTPUT,max(resumable) if resumable else None)
                if torch.backends.mps.is_available():torch.mps.empty_cache()
            found=completed(config);require(found,'Training did not produce a completed run');return found

    def benchmark_record(self,config):
        for path in sorted(OUTPUT.glob('phase3-*/run.json')):
            record=json.loads(path.read_text())
            if record['configuration']['sha256']!=digest(config):continue
            for name,sha in record['metrics']['artifacts'].items():
                require(file_digest(path.parent/name)==sha,'Saved benchmark logs changed')
            require(record['status'] in ('completed','stopped-early'),'A matching benchmark failed; investigate its run record before resuming')
            return path,record
        return None

    def evaluate(self,model,seed,mode,final=False):
        training_path,training=self.trained(model,seed)
        config=json.loads(json.dumps(training['configuration']['content']))
        checkpoint=training['checkpoint']
        if final:
            path=training_path.parent/'final.pt'
            checkpoint={'path':str(path.relative_to(ROOT)),'sha256':file_digest(path)}
        config['stage']='benchmark';config['experiment']='phase5-D2-W0-architecture-tuning'
        config['candidate']={'key':'NEURAL_POLICY' if mode=='policy' else 'NEURAL_SEARCH','checkpoint':checkpoint}
        config['opponent']={'key':'JSETTLERS','checkpoint':None}
        config['evaluation']['futility']=dict(benchmark.DEFAULT_FUTILITY)
        config['evaluation'].update(tier='smoke',seed_set='tuning',mode='policy' if mode=='policy' else 'time',budget_ms=500,simulations=None)
        if mode=='search':
            config['search']['max_root_depth']=None
            optimization=EXPERIMENTS/'search-optimized-v1.json'
            options=json.loads(optimization.read_text()) if optimization.exists() else {}
            config['search']['workers']=options.get('workers_by_model',{}).get(model,1)
            config['search']['revision']='optimized-unbounded-own-actions-v1'
            if optimization.exists():config['search']['optimization_config_sha256']=file_digest(optimization)
        config['search']['opponent_checkpoint']=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
        config['implementation_sha256']=self.core
        config['training_run_record']=str(training_path.relative_to(ROOT))
        label=f'{model}:{seed}:{mode}:{"final" if final else "best"}'
        prior=None
        for tier in ('smoke','screening'):
            config['evaluation']['tier']=tier
            destination=EXPERIMENTS/'phase5-match-configs'/f'{model}-seed{seed}-{mode}-{"final" if final else "best"}-{tier}.json'
            write(destination,config)
            self.state['stage']=f'evaluate:{label}:{tier}';self.persist()
            with locked():
                self.guard();found=self.benchmark_record(config)
                if found:path,record=found
                else:
                    path=benchmark.run(config,OUTPUT,continuation=prior)
                    record=json.loads(path.read_text())
            self.state['benchmarks'][f'{label}:{tier}']={'run_record':str(path.relative_to(ROOT)),
                                                       'metrics':record['metrics']['benchmark'],'eligible':eligible(record)}
            self.persist();prior=path
            if record['metrics']['benchmark'].get('futility',{}).get('stopped') and self.state.get('continue_after_futility'):
                return path,record
            require(not record['metrics']['benchmark'].get('futility',{}).get('stopped'),
                    f'Futility stop ({label}); no screening extension or confirmation runs will launch. Diagnose {path}')
            # Errors already raise in run; timing/cutoffs require investigation.
            if not record['metrics']['benchmark']['cutoff_gate_passed'] and self.state.get('continue_after_investigated_cutoffs'):
                return path,record
            require(record['metrics']['benchmark']['cutoff_gate_passed'],f'Phase5 cutoff gate failed ({label}, {tier}); investigate {path}')

        return path,record

    def execute(self):
        for model in MODEL_IDS:self.trained(model,11)
        screened={}
        for model in MODEL_IDS:
            screened[model]={mode:self.evaluate(model,11,mode) for mode in ('policy','search')}
        feasible=[m for m in MODEL_IDS if any(eligible(r) for _,r in screened[m].values())]
        require(len(feasible)>=2,'Fewer than two feasible architectures; investigate timing before confirmation')
        ranks=sorted(feasible,key=lambda m:(-max(r['metrics']['benchmark']['overall_match_score'] for _,r in screened[m].values() if eligible(r)),
                                             self.trained(m,11)[1]['metrics']['training']['parameters']))
        finalists=ranks[:2];self.state['finalists']=finalists
        self.state['seed11_ranking']=ranks;self.state['stage']='confirmation-training';self.persist()
        for model in finalists:
            for seed in (23,47):self.trained(model,seed)
        results={}
        for model in finalists:
            results[model]={11:screened[model]}
            for seed in (23,47):
                results[model][seed]={mode:self.evaluate(model,seed,mode) for mode in ('policy','search')}
        diagnostics={}
        for model in finalists:
            diagnostics[model]={}
            for mode in ('policy','search'):
                final_path,_=self.evaluate(model,11,mode,final=True)
                diagnostics[model][mode]=benchmark.paired_difference(games(final_path),games(screened[model][mode][0]))
        summary={'status':'completed','finalists':finalists,'seed11_ranking':ranks,
                 'final_evaluation_seeds_used':False,'selection_rule':self.state['selection_rule'],
                 'checkpoint_diagnostic_final_minus_best':diagnostics,'architectures':{},
                 'source_sha256':self.core,'device_preflight':json.loads((EXPERIMENTS/'phase5-devices.json').read_text())}
        for model in MODEL_IDS:
            seeds=(11,23,47) if model in finalists else (11,)
            row={'training':{},'matches':{},'confirmed_across_training_seeds':model in finalists}
            for seed in seeds:
                path,record=self.trained(model,seed)
                row['training'][str(seed)]={'run_record':str(path.relative_to(ROOT)),
                    'elapsed_seconds':record['elapsed_seconds'],'metrics':record['metrics']['training'],
                    'checkpoint':record['checkpoint']}
                modes=results[model][seed] if model in finalists else screened[model]
                row['matches'][str(seed)]={mode:{'run_record':str(p.relative_to(ROOT)),'metrics':r['metrics']['benchmark']}
                                           for mode,(p,r) in modes.items()}
                row['matches'][str(seed)]['search_minus_policy']=benchmark.paired_difference(games(modes['search'][0]),games(modes['policy'][0]))
            if model in finalists:
                row['hierarchical']={mode:benchmark.hierarchical_bootstrap([(seed,games(results[model][seed][mode][0])) for seed in seeds]) for mode in ('policy','search')}
                # Resample training seeds and board pairs for search-minus-policy too.
                differences=[]
                for seed in seeds:
                    a=games(results[model][seed]['search'][0]);b=games(results[model][seed]['policy'][0])
                    benchmark.paired_difference(a,b)
                    lookup={(g['pair_id'],g['candidate_color']):g for g in b}
                    differences.append((seed,[{**g,'match_score':g['match_score']-lookup[g['pair_id'],g['candidate_color']]['match_score']} for g in a]))
                row['hierarchical']['search_minus_policy']=benchmark.hierarchical_bootstrap(differences)
            summary['architectures'][model]=row
        # Lock a provisional tuning choice. Overlapping intervals are reported as a
        # tie and resolved by parameter count among statistically tied finalists.
        feasible_modes={m:[mode for mode in ('policy','search') if all(eligible(results[m][seed][mode][1]) for seed in (11,23,47))] for m in finalists}
        require(all(feasible_modes.values()),'A finalist has no feasible mode across all three training seeds')
        def best_mode(m):return max(feasible_modes[m],key=lambda mode:summary['architectures'][m]['hierarchical'][mode]['mean'])
        choice=max(finalists,key=lambda m:summary['architectures'][m]['hierarchical'][best_mode(m)]['mean'])
        intervals={m:summary['architectures'][m]['hierarchical'][best_mode(m)]['ci95'] for m in finalists}
        tied=[m for m in finalists if intervals[m][1]>=intervals[choice][0] and intervals[m][0]<=intervals[choice][1]]
        if len(tied)>1:choice=min(tied,key=lambda m:self.trained(m,11)[1]['metrics']['training']['parameters'])
        mode=best_mode(choice)
        if mode=='search' and summary['architectures'][choice]['hierarchical']['search_minus_policy']['ci95'][0]<=0:mode='policy'
        require(all(eligible(results[choice][seed][mode][1]) for seed in (11,23,47)),
                'Provisional winner failed a confirmation feasibility gate')
        selected=self.trained(choice,11)[1]
        frozen=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
        settings=selected['configuration']['content']['search'];settings['opponent_checkpoint']=frozen
        summary['provisional_choice']={'model':choice,'mode':mode,'seed':11,'statistically_tied_finalists':tied,
                                      'checkpoint':selected['checkpoint'],'reserved_final_confirmation_pending':True}
        write(EXPERIMENTS/'phase5-selected.json',{'checkpoint':selected['checkpoint'],'search':settings,
               'provisional':True,'phase5_complete':True,'mode':mode,'final_seeds_used':False})
        write(EXPERIMENTS/'PHASE5_RESULTS.json',summary)
        report(summary)
        self.state['stage']='completed';self.state['status']='completed';self.state['summary']='experiments/catan-ai/PHASE5_RESULTS.json';self.persist()
        print('Phase 5 completed; reserved final evaluation seeds remain unused.',flush=True)


def report(summary):
    lines=['# Phase 5 results','',
           'All four seed-11 architectures completed 20,000 updates on D2/W0. '
           'The two finalists completed seeds 23 and 47 as well. All games use '
           'the frozen tuning set; final evaluation seeds remain reserved for phase 10.','',
           'Selection uses paired-game policy and 500 ms search performance, with '
           'timing, cutoff and failure gates. Checkpoints use the predeclared '
           'validation policy-NLL plus value-BCE criterion. The final checkpoint '
           'diagnostics do not change that baseline selection rule.','',
           '| Architecture | Seed | Parameters | Best update | Train seconds | Test policy NLL | Test top-1 | Value BCE | Brier |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model,row in summary['architectures'].items():
        for seed,training in row['training'].items():
            m=training['metrics'];t=m['test']
            lines.append(f"| {model} | {seed} | {m['parameters']} | {m['best_step']} | {training['elapsed_seconds']:.1f} | {t['policy_nll']:.4f} | {t['top1_accuracy']:.4f} | {t['value_bce']:.4f} | {t['brier']:.4f} |")
    lines+=['','| Architecture | Seed | Mode | Match score | Pair-clustered 95% CI | Warm p95 ms | Cutoffs | Errors |',
            '| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |']
    for model,row in summary['architectures'].items():
        for seed,modes in row['matches'].items():
            for mode in ('policy','search'):
                m=modes[mode]['metrics'];ci=m['match_score_ci95']
                lines.append(f"| {model} | {seed} | {mode} | {m['overall_match_score']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {m['candidate_latency_warm_nonforced']['p95_ms']:.2f} | {m['cutoffs']} | {m['errors']} |")
    lines+=['','Three-seed hierarchical results, paired search improvements, final-versus-best '
            'checkpoint diagnostics, calibration bins, metrics by phase/action family, '
            'consumption counts and all run/checkpoint hashes are in [PHASE5_RESULTS.json](PHASE5_RESULTS.json). '
            'Batch-1 and batch-32 measurements for both devices are in '
            '[phase5-devices.json](phase5-devices.json).','',
            'Finalists: '+', '.join(summary['finalists'])+'. '
            'The provisional choice is '+summary['provisional_choice']['model']+' in '
            +summary['provisional_choice']['mode']+' mode. '
            'Overlapping intervals are reported as a tie and resolved by parameter count; '
            'search is selected only when its paired three-seed improvement excludes zero. '
            'No claim of superiority over JSettlers follows without a match-score '
            'lower bound above 50%.','']
    (EXPERIMENTS/'PHASE5_RESULTS.md').write_text('\n'.join(lines))
    plan=ROOT/'AI_EXPERIMENT_PLAN.md'
    text=plan.read_text()
    text=text.replace('phases 1–4 are complete.','phases 1–5 are complete.')
    start=text.index('Phase 5 is in progress.')
    end=text.index('\n\nThe objective',start)
    text=text[:start]+('Phase 5 completed the four architecture runs, paired tuning matches and '
        'three-seed finalist comparisons; see [the phase-five results]'
        '(experiments/catan-ai/PHASE5_RESULTS.md). Phases 6–10 remain pending. '
        'Final evaluation seeds remain unused.')+text[end:]
    plan.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args();pipeline=Pipeline(args.cache)
    try:pipeline.execute()
    except BaseException as error:
        pipeline.state['status']='needs-investigation';pipeline.state['error']=f'{type(error).__name__}: {error}';pipeline.persist();raise


if __name__=='__main__':main()
