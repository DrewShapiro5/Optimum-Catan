"""Finalize verified XD-CNN results under the user's reduced phase-6 scope."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,file_digest,digest
from agent_apis.neural.pipeline import write,games,eligible
from agent_apis.observation import require

OUTPUT=ROOT/'runs/catan-ai'
STATE=OUTPUT/'phase6-progress.json'
SCOPE=EXPERIMENTS/'phase6-active-scope.json'


def finalize():
    with benchmark.exclusive_benchmark():
        state=json.loads(STATE.read_text())
        if state.get('scope_amendment') and state['status']=='completed':
            print('XD-CNN-only phase 6 already finalized; no games launched.');return
        original=deepcopy(state)
        write(OUTPUT/'phase6-before-xd-only.json',original)
        # The queue entry point alone changes to prevent relaunching retired work.
        for name,sha in state['source_sha256'].items():
            if name=='agent_apis/neural/phase6.py':continue
            target=OUTPUT/'d2-cache-89b777ce01b9ebd7/manifest.json' if name=='cache_manifest' else ROOT/name
            require(file_digest(target)==sha,f'Historical input changed: {name}')
        for row in state['training'].values():
            require(file_digest(ROOT/row['checkpoint']['path'])==row['checkpoint']['sha256'],'Training checkpoint changed')
        for row in state['benchmarks'].values():
            p=ROOT/row['run_record'];record=json.loads(p.read_text())
            require(record['configuration']['sha256']==digest(record['configuration']['content']),'Configuration changed')
            for name,sha in record['metrics']['artifacts'].items():require(file_digest(p.parent/name)==sha,'Game logs changed')
        scope={'date':'2026-11-11','authorized_by':'User: move forward with XD-CNN; stop remaining Transformer tests',
               'active_architectures':['XD-CNN'],'retired_architectures':['TRANSFORMER'],
               'original_two_architecture_protocol_complete':False,'historical_results_preserved':True,
               'stopped_stage':state['stage'],'stopped_pid':state.get('pid'),
               'original_protocol':state['protocol'],
               'historical_entrypoint_sha256':state['source_sha256']['agent_apis/neural/phase6.py'],
               'current_entrypoint_sha256':file_digest(ROOT/'agent_apis/neural/phase6.py'),
               'finalizer_sha256':file_digest(__file__)}
        def latest(seed,recipe,mode,opponent):
            key=f'XD-CNN:{seed}:{recipe}:{mode}:{opponent}:screening'
            require(key in state['benchmarks'],f'Missing XD-CNN confirmation: {key}')
            row=state['benchmarks'][key];p=ROOT/row['run_record'];r=json.loads(p.read_text())
            require(r['status']=='completed' and r['metrics']['benchmark']['games']==50,'Incomplete XD-CNN confirmation')
            return p,r
        recipes={};differences=[];all_search_eligible=True;table=[]
        for recipe in ('D2-W0','D2-W4'):
            modes={}
            for mode in ('policy','search'):
                runs=[];wins=losses=cutoffs=0;feasible=True
                for seed in (11,23,47):
                    p,r=latest(seed,recipe,mode,'JSETTLERS');m=r['metrics']['benchmark']
                    runs.append((seed,games(p)));wins+=m['wins'];losses+=m['losses'];cutoffs+=m['cutoffs']
                    feasible=feasible and eligible(r) and eligible(latest(seed,recipe,mode,'RANDOM')[1])
                    table.append((recipe,seed,mode,m['wins'],m['cutoffs'],eligible(r)))
                modes[mode]={'wins':wins,'losses':losses,'cutoffs':cutoffs,'games':150,
                             'hierarchical':benchmark.hierarchical_bootstrap(runs),'all_runs_eligible':feasible}
            recipes[recipe]=modes
            all_search_eligible=all_search_eligible and modes['search']['all_runs_eligible']
        for seed in (11,23,47):
            a=games(latest(seed,'D2-W4','search','JSETTLERS')[0]);b=games(latest(seed,'D2-W0','search','JSETTLERS')[0])
            benchmark.paired_difference(a,b)
            lookup={(g['pair_id'],g['candidate_color']):g for g in b}
            differences.append((seed,[{**g,'match_score':g['match_score']-lookup[g['pair_id'],g['candidate_color']]['match_score']} for g in a]))
        improvement=benchmark.hierarchical_bootstrap(differences)
        chosen='D2-W4' if all_search_eligible and improvement['ci95'][0]>0 else 'D2-W0'
        selection={'model':'XD-CNN','chosen_recipe':chosen,'challenger':'D2-W4',
                   'search_challenger_minus_uniform':improvement,'all_search_runs_eligible':all_search_eligible,
                   'reason':'Three-seed paired search comparison on XD-CNN only, under user-authorized reduced scope.',
                   'installed_default_changed':False,'reserved_final_confirmation_pending':True}
        state.update(status='completed',stage='completed-xd-cnn-only',pid=None,scope_amendment=scope,
                     selection=selection,updated_at=datetime.now(timezone.utc).isoformat())
        state.pop('error',None)
        summary={'status':'completed','scope':'XD-CNN-only user amendment; original Transformer comparison incomplete',
                 'scope_amendment':scope,'final_seeds_used':False,'selection':selection,'architectures':{'XD-CNN':recipes},
                 'training':state['training'],'benchmarks':state['benchmarks'],
                 'historical_source_sha256':state['source_sha256'],'updated_at':state['updated_at']}
        lines=['# Phase 6 results — XD-CNN-only scope','',
               'On November 11, 2026, the user retired Transformer and requested early stopping. '
               'All six XD-CNN model/recipe/seed combinations were already complete, including both modes and both opponents. '
               'The active Transformer evaluation was terminated; completed games/checkpoints and its partial logs remain preserved. '
               'The original two-architecture protocol is incomplete. Final evaluation seeds remain unused.','',
               '| Recipe | Seed | Mode | JSettlers wins / 50 | Cutoffs | JSettlers eligible |',
               '| --- | ---: | --- | ---: | ---: | --- |']
        for recipe,seed,mode,wins,cutoffs,feasible in table:
            lines.append(f'| {recipe} | {seed} | {mode} | {wins} | {cutoffs} | {feasible} |')
        lines+=['','| Recipe | Mode | Wins / 150 | Match score | Hierarchical 95% CI | All opponent gates |',
                '| --- | --- | ---: | ---: | --- | --- |']
        for recipe,modes in recipes.items():
            for mode,m in modes.items():
                h=m['hierarchical'];lo,hi=h['ci95']
                lines.append(f"| {recipe} | {mode} | {m['wins']} | {h['mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {m['all_runs_eligible']} |")
        lo,hi=improvement['ci95']
        lines+=['',f"W4 minus W0 search match score: {improvement['mean']:+.1%}, paired hierarchical 95% interval [{lo:+.1%}, {hi:+.1%}]. Selected recipe for subsequent XD-CNN work: **{chosen}**.",'',
                'Intervals resample training seeds and shared board/seat pairs. Results are tuning comparisons after recipe screening; '
                'they do not establish superiority over JSettlers or supply reserved final-set confirmation. '
                'Search must separately demonstrate improvement over policy before selecting a deployed search default. '
                'No installed checkpoint default was changed by this report.','']
        write(SCOPE,scope);write(STATE,state);write(EXPERIMENTS/'PHASE6_RESULTS.json',summary)
        (EXPERIMENTS/'PHASE6_RESULTS.md').write_text('\n'.join(lines))
        print(json.dumps({'selection':selection,'recipes':recipes},indent=2))

if __name__=='__main__':finalize()
