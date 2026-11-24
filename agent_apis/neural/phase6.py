"""Resumable corpus/weighting screening and three-seed finalist validation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest
from agent_apis.neural.data import Cache
from agent_apis.neural.pipeline import config_for,write,games,eligible
from agent_apis.neural.queue import completed
from agent_apis.neural.recipes import RECIPES,RecipeSampler
from agent_apis.neural.train import train
from agent_apis.observation import require

OUTPUT=ROOT/'runs/catan-ai'
STATE=OUTPUT/'phase6-progress.json'
PROTOCOL=EXPERIMENTS/'phase6.json'
CORE=('agent_apis/neural/phase6.py','agent_apis/neural/recipes.py','agent_apis/neural/train.py',
      'agent_apis/neural/data.py','agent_apis/neural/models.py','agent_apis/neural/provider.py',
      'agent_apis/neural/topology.py','agent_apis/neural/parallel_search.py','agent_apis/observation.py',
      'agent_apis/benchmark.py','agent_apis/futility.py','agent_apis/stochastic.py',
      'agent_apis/stochastic_search.py','agent_apis/development_belief.py',
      'experiments/catan-ai/phase6.json','experiments/catan-ai/search-optimized-v1.json',
      'experiments/catan-ai/evaluation-quick-v1.json','experiments/catan-ai/phase5-frozen-opponent.json')


def configuration(model,seed,recipe):
    require(model in ('XD-CNN','TRANSFORMER') and seed in (11,23,47),'Outside authorized finalists/seeds')
    require(recipe in RECIPES,'Unknown recipe')
    config=config_for(model,seed)
    # Exact baseline training configurations permit verified reuse of phase 5.
    if recipe=='D2-W0':return config
    composition,weighting,*variant=recipe.split('-')
    config.update(phase=6,experiment=f'phase6-{recipe}')
    config['data'].update(composition=composition,weighting=weighting,value_weighting=variant[0] if variant else 'V0')
    if composition=='D0':config['data']['corpora']=['data/jsettlers-corpora/full-250']
    if composition=='D1':config['data']['view']='retained-rows'
    config['training']['validation_distribution']='common full unweighted D2 validation; no test-based selection'
    return config


class Phase6:
    def __init__(self,cache_path):
        self.cache_path=Path(cache_path);self.protocol=json.loads(PROTOCOL.read_text())
        self.core={p:file_digest(ROOT/p) for p in CORE}
        self.core['cache_manifest']=file_digest(self.cache_path/'manifest.json')
        for model in self.protocol['architectures']:
            for seed in (11,23,47):
                p=f'experiments/catan-ai/phase5-configs/{model}-seed{seed}.json';self.core[p]=file_digest(ROOT/p)
        if STATE.exists():
            self.state=json.loads(STATE.read_text())
            require(self.state['source_sha256']==self.core,'Phase-6 sources changed; investigate before resuming')
        else:
            self.state={'status':'running','stage':'initializing','started_at':datetime.now(timezone.utc).isoformat(),
                        'source_sha256':self.core,'protocol':self.protocol,'training':{},'benchmarks':{},'final_seeds_used':False}
        self.state.update(pid=os.getpid(),status='running');self.state.pop('error',None);self.persist();self.report()

    def persist(self):
        self.state['updated_at']=datetime.now(timezone.utc).isoformat();write(STATE,self.state)

    def guard(self):
        for p,sha in self.core.items():
            target=self.cache_path/'manifest.json' if p=='cache_manifest' else ROOT/p
            require(file_digest(target)==sha,f'Phase-6 input changed: {p}')

    def trained(self,model,seed,recipe):
        self.guard();config=configuration(model,seed,recipe);key=f'{model}:{seed}:{recipe}'
        found=completed(config) if recipe=='D2-W0' else None
        if not found:
            for p in sorted(OUTPUT.glob('phase6-*/run.json')):
                record=json.loads(p.read_text())
                if record['configuration']['sha256']==digest(config):
                    require(record['status']=='completed','Matching training failed')
                    require(file_digest(ROOT/record['checkpoint']['path'])==record['checkpoint']['sha256'],'Checkpoint changed')
                    found=p,record;break
        if not found:
            self.state['stage']='train:'+key;self.persist()
            write(EXPERIMENTS/'phase6-configs'/f'{model}-seed{seed}-{recipe}.json',config)
            resumable=[p for p in OUTPUT.glob('phase*-*') if (p/'resume.pt').exists() and
                       not (p/'run.json').exists() and (p/'configuration.json').exists() and
                       digest(json.loads((p/'configuration.json').read_text()))==digest(config)]
            print('TRAIN '+key,flush=True)
            folder=train(config,self.cache_path,OUTPUT,max(resumable) if resumable else None)
            found=folder/'run.json',json.loads((folder/'run.json').read_text())
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        p,record=found
        self.state['training'][key]={'run_record':str(p.relative_to(ROOT)),
            'checkpoint':record['checkpoint'],'metrics':record['metrics']['training'],'elapsed_seconds':record['elapsed_seconds']}
        self.persist();return p,record

    def match_config(self,model,seed,recipe,training,mode,opponent,tier):
        path,record=training;config=json.loads(json.dumps(record['configuration']['content']))
        config.update(stage='benchmark',experiment=f'phase6-{recipe}-validation',implementation_sha256=self.core,
                      training_run_record=str(path.relative_to(ROOT)))
        config['candidate']={'key':'NEURAL_POLICY' if mode=='policy' else 'NEURAL_SEARCH','checkpoint':record['checkpoint']}
        config['opponent']={'key':opponent,'checkpoint':None}
        config['evaluation'].update(opponent=opponent,schedule='quick-v1',
            schedule_sha256=file_digest(EXPERIMENTS/'evaluation-quick-v1.json'),
            tier=tier,seed_set='tuning',mode='policy' if mode=='policy' else 'time',budget_ms=500,simulations=None,
            smoke_pairs=5,screening_pairs=25,futility=dict(benchmark.DEFAULT_FUTILITY))
        optimization=json.loads((EXPERIMENTS/'search-optimized-v1.json').read_text())
        config['search'].update(max_root_depth=None,workers=optimization['workers_by_model'].get(model,1),
            revision=optimization['revision'],opponent_checkpoint=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint'])
        return config

    def measure(self,model,seed,recipe,training,mode,opponent,tier,prior=None):
        self.guard();config=self.match_config(model,seed,recipe,training,mode,opponent,tier)
        key=f'{model}:{seed}:{recipe}:{mode}:{opponent}:{tier}'
        self.state['stage']='validate:'+key;self.persist()
        write(EXPERIMENTS/'phase6-match-configs'/f'{key.replace(":","-")}.json',config)
        found=None
        for p in sorted(OUTPUT.glob('phase5q-*/run.json')):
            r=json.loads(p.read_text())
            if r['configuration']['sha256']!=digest(config):continue
            require(r['status'] in ('completed','stopped-early'),'Matching validation failed; investigate, do not reroll')
            require(r['metrics']['providers']=={role:benchmark.provider_identity(config[role]['key']) for role in ('candidate','opponent')},'Validation provider changed')
            for name,sha in r['metrics']['artifacts'].items():require(file_digest(p.parent/name)==sha,'Validation logs changed')
            found=p,r;break
        if not found:
            print('VALIDATE '+key,flush=True)
            p=benchmark.run(config,OUTPUT,continuation=prior);found=p,json.loads(p.read_text())
        p,r=found
        self.state['benchmarks'][key]={'run_record':str(p.relative_to(ROOT)),'status':r['status'],
                                      'metrics':r['metrics']['benchmark'],'eligible':eligible(r)}
        self.persist();self.report();return p,r

    def validate(self,model,seed,recipe,training):
        measured={}
        # Both opponents and both modes receive smoke before any extension.
        for opponent in self.protocol['validation_opponents']:
            for mode in ('policy','search'):
                measured[opponent,mode]=self.measure(model,seed,recipe,training,mode,opponent,'smoke')
        for opponent,mode in list(measured):
            prior,r=measured[opponent,mode]
            if r['metrics']['benchmark'].get('futility',{}).get('stopped'):continue
            measured[opponent,mode]=self.measure(model,seed,recipe,training,mode,opponent,'screening',prior)
        return measured

    def latest(self,model,seed,recipe,opponent='JSETTLERS',mode='search'):
        for tier in ('screening','smoke'):
            row=self.state['benchmarks'].get(f'{model}:{seed}:{recipe}:{mode}:{opponent}:{tier}')
            if row:return ROOT/row['run_record'],json.loads((ROOT/row['run_record']).read_text())
        raise ValueError('Missing validation result')

    def select(self):
        # Predeclared conservative control: confirm uniform plus one challenger.
        model=self.protocol['screening_architecture'];base_path,base=self.latest(model,11,'D2-W0')
        candidates=[];comparisons={}
        for recipe in RECIPES:
            if recipe=='D2-W0':continue
            path,record=self.latest(model,11,recipe)
            if not eligible(record) or not eligible(self.latest(model,11,recipe,'RANDOM')[1]):continue
            a,b=games(path),games(base_path);length=min(len(a),len(b))//2*2
            comparison=benchmark.paired_difference(a[:length],b[:length])
            if base['status']=='stopped-early' or record['status']=='stopped-early':comparison['ci95']=[None,None]
            comparisons[recipe]=comparison
            candidates.append((record['metrics']['benchmark']['overall_match_score'],recipe))
        require(candidates,'No feasible challenger; retain uniform and investigate instead of automatic confirmation')
        challenger=max(candidates,key=lambda row:(row[0],-RECIPES.index(row[1])))[1]
        self.state['screening_comparisons_search_minus_uniform']=comparisons
        self.state['selected_recipes']=['D2-W0',challenger];self.persist()
        return self.state['selected_recipes']

    def report(self,complete=False):
        summary={'status':'completed' if complete else 'in-progress','final_seeds_used':False,
                 'protocol':self.protocol,'training':self.state['training'],'benchmarks':self.state['benchmarks'],
                 'source_sha256':self.core,'selected_recipes':self.state.get('selected_recipes'),
                 'selection':self.state.get('selection'),'updated_at':self.state['updated_at']}
        destination='PHASE6_RESULTS' if complete else 'PHASE6_PROGRESS'
        write(EXPERIMENTS/(destination+'.json'),summary)
        lines=['# Phase 6 corpus and weighting '+('results' if complete else 'progress'),'',
               'Transformer and XD-CNN only. Random player and native JSettlers SMART validation; '
               '10-game smoke / 50-game total per model, seed, recipe, mode and opponent. '
               'Search uses 500 ms; final seeds remain unused.','',
               '| Model | Seed | Recipe | Mode | Opponent | Games | Wins | Score | Pair 95% CI | Eligible |',
               '| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |']
        latest={}
        for key,row in self.state['benchmarks'].items():
            identity,tier=key.rsplit(':',1)
            if tier=='screening' or identity not in latest:latest[identity]=row
        for key,row in latest.items():
            model,seed,recipe,mode,opponent=key.split(':');m=row['metrics']
            ci=m['match_score_ci95'];interval='suppressed after futility' if ci[0] is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
            lines.append(f"| {model} | {seed} | {recipe} | {mode} | {opponent} | {m['games']} | {m['wins']} | {m['overall_match_score']:.3f} | {interval} | {row['eligible']} |")
        lines+=['','Run records contain imitation/calibration metrics, consumed phase/family counts, unique games, '
                'sampling probabilities, latency, cutoffs, failures, checksums and pair intervals. '
                'Random-opponent results diagnose competence; JSettlers search matches determine recipe selection. '
                'The original phase-5 checkpoints are reused only as exact D2/W0 training controls.','']
        if summary['selection']:lines.append('Selection: '+json.dumps(summary['selection'])+'\n')
        (EXPERIMENTS/(destination+'.md')).write_text('\n'.join(lines))

    def finish(self):
        selected=self.state['selected_recipes'];analysis={};promotion=[]
        challenger=selected[1]
        for model in self.protocol['architectures']:
            comparisons=[];ready=True
            for seed in (11,23,47):
                a,ar=self.latest(model,seed,challenger);b,br=self.latest(model,seed,'D2-W0')
                ready=ready and eligible(ar) and eligible(br)
                ready=ready and all(eligible(self.latest(model,seed,recipe,'RANDOM')[1]) for recipe in selected)
                ga,gb=games(a),games(b);length=min(len(ga),len(gb))//2*2
                difference=benchmark.paired_difference(ga[:length],gb[:length])
                if ar['status']=='stopped-early' or br['status']=='stopped-early':ready=False
                lookup={(g['pair_id'],g['candidate_color']):g for g in gb[:length]}
                comparisons.append((seed,[{**g,'match_score':g['match_score']-lookup[g['pair_id'],g['candidate_color']]['match_score']} for g in ga[:length]]))
            interval=benchmark.hierarchical_bootstrap(comparisons)
            if not ready:
                interval['descriptive_only']=True
            analysis[model]={'search_challenger_minus_uniform':interval,'all_runs_eligible':ready}
            if ready and interval['ci95'][0]>0:promotion.append(model)
        self.state['selection']={'uniform_control':'D2-W0','challenger':challenger,'by_architecture':analysis,
            'improved_architectures':promotion,'chosen_recipe':challenger if len(promotion)==2 else 'D2-W0',
            'reason':'Promote only if paired three-seed search improvement is positive on both finalists; otherwise retain uniform.',
            'installed_default_changed':False,'reserved_final_confirmation_pending':True}
        self.state.update(status='completed',stage='completed');self.persist();self.report(complete=True)
        print('Phase 6 completed; final seeds unused. Results: experiments/catan-ai/PHASE6_RESULTS.md',flush=True)

    def execute(self):
        require(torch.backends.mps.is_available(),'Phase-6 pinned MPS training/inference is unavailable; no fallback')
        cache=Cache(self.cache_path)
        distributions={recipe:RecipeSampler(cache,configuration('XD-CNN',11,recipe)['data'],11).summary for recipe in RECIPES}
        write(EXPERIMENTS/'phase6-sampling-audit.json',{'cache_manifest_sha256':self.core['cache_manifest'],
              'recipes':distributions,'metadata_only':['actor-relative final decision ordinal'],
              'validation_distribution':'common complete D2, unweighted','gate_passed':True})
        # Begin the new phase with its first coverage training run. Reuse that
        # checkpoint below after validating the two existing baseline controls.
        self.trained(self.protocol['screening_architecture'],11,'D0-W0')
        # Validate both requested finalists against both opponents immediately.
        for model in self.protocol['architectures']:
            training=self.trained(model,11,'D2-W0');self.validate(model,11,'D2-W0',training)
        for recipe in RECIPES:
            if recipe=='D2-W0':continue
            model=self.protocol['screening_architecture']
            training=self.trained(model,11,recipe);self.validate(model,11,recipe,training)
        selected=self.state.get('selected_recipes') or self.select()
        for model in self.protocol['architectures']:
            for recipe in selected:
                for seed in (11,23,47):
                    training=self.trained(model,seed,recipe);self.validate(model,seed,recipe,training)
        self.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=OUTPUT/'d2-cache-89b777ce01b9ebd7')
    args=parser.parse_args()
    scope=EXPERIMENTS/'phase6-active-scope.json'
    if scope.exists():
        require(json.loads(scope.read_text())['active_architectures']==['XD-CNN'],'Unknown scope amendment')
        print('Transformer retired by user; phase-6 XD-CNN results are complete. See PHASE6_RESULTS.md. No games launched.')
        return
    with benchmark.exclusive_benchmark():
        work=Phase6(args.cache)
        try:work.execute()
        except BaseException as error:
            work.state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'needs-investigation',
                              error=f'{type(error).__name__}: {error}');work.persist();raise

if __name__=='__main__':main()
