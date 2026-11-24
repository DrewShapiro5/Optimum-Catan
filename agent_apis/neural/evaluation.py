"""Finish phase-5 comparisons with per-run futility and explicit negative results.

Historical MLP logs are imported as historical observations, not relabeled as
prospectively early-stopped runs. A weak policy never suppresses its search test.
Only architectures with a feasible, non-futile mode receive additional seeds.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
from collections import Counter
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,file_digest,digest,validate_record
from agent_apis.neural.models import MODEL_IDS
from agent_apis.neural.pipeline import Pipeline,eligible,games,write
from agent_apis.neural.diagnose import replay,summarize as score_summary
from agent_apis.observation import require

STATE=ROOT/'runs/catan-ai/phase5-evaluation.json'
ORCHESTRATION=('agent_apis/neural/evaluation.py','agent_apis/neural/pipeline.py')

def equivalent_decision_configuration(first,second):
    a=json.loads(json.dumps(first));b=json.loads(json.dumps(second))
    for config in (a,b):
        for name in ORCHESTRATION:config.get('implementation_sha256',{}).pop(name,None)
    if all(c.get('candidate',{}).get('key')=='NEURAL_POLICY' and c.get('evaluation',{}).get('mode')=='policy' for c in (a,b)):
        # Pure policies import the action vocabulary but never evaluate beliefs.
        for c in (a,b):c.get('implementation_sha256',{}).pop('agent_apis/development_belief.py',None)
    return a==b


def common_prefix(datasets):
    keys=[{(r['pair_id'],r['candidate_color']) for r in rows} for rows in datasets]
    common=set.intersection(*keys)
    counts=Counter(pair for pair,color in common)
    common={key for key in common if counts[key[0]]==2}
    require(common,'No shared complete board/seat pairs')
    return [[r for r in rows if (r['pair_id'],r['candidate_color']) in common] for rows in datasets]


def descriptor(path,record):
    m=record['metrics']['benchmark']
    return {'run_record':str(path.relative_to(ROOT)) if path.exists() else None,
            'partial_run_directory':str(path.parent.relative_to(ROOT)) if not path.exists() else None,
            'run_record_sha256':file_digest(path) if path.exists() else None,
            'status':record['status'],'metrics':m,'promotion_eligible':eligible(record),
            'historical_import':record.get('historical_import')}


class Evaluation(Pipeline):
    def __init__(self,cache):
        super().__init__(cache)
        self.core['agent_apis/neural/evaluation.py']=file_digest(Path(__file__))
        self.state.update(continue_after_futility=True,continue_after_investigated_cutoffs=True,stage='architecture-evaluation',
                          selection_rule='Rank by highest feasible, non-futile policy or 500ms search score. Confirm at most two eligible architectures on seeds23/47; do not force a shortlist when every architecture is futile.')
        self.persist()

    def persist(self):
        self.state['updated_at']=datetime.now(timezone.utc).isoformat();write(STATE,self.state)

    def benchmark_record(self,config):
        exact=super().benchmark_record(config)
        if exact:return exact
        candidates=[]
        for path in (ROOT/'runs/catan-ai').glob('phase3-*/run.json'):
            record=json.loads(path.read_text())
            if equivalent_decision_configuration(record['configuration']['content'],config):candidates.append((path,record))
        for path,record in candidates:
            # A changed orchestrator can reuse an already completed full tier;
            # it cannot silently change the strict smoke continuation contract.
            if (config['evaluation']['tier']=='smoke' and record['status']=='completed'
                and record['metrics']['benchmark'].get('cutoff_gate_passed',True)
                and not record['metrics']['benchmark'].get('futility',{}).get('stopped')):
                full=json.loads(json.dumps(config));full['evaluation']['tier']='screening'
                if not any(equivalent_decision_configuration(json.loads(p.read_text())['configuration']['content'],full)
                           for p in (ROOT/'runs/catan-ai').glob('phase3-*/run.json')):continue
            validate_record(record)
            require(record['status'] in ('completed','stopped-early'),'A compatible historical benchmark failed')
            for name,sha in record['metrics']['artifacts'].items():
                require(file_digest(path.parent/name)==sha,'Compatible benchmark artifact changed')
            for role in ('candidate','opponent'):
                require(record['metrics']['providers'][role]==benchmark.provider_identity(config[role]['key']),
                        'Compatible provider identity changed')
            self.state.setdefault('compatible_artifact_reuse',{})[str(path.relative_to(ROOT))]={
                'original_configuration_sha256':record['configuration']['sha256'],
                'requested_configuration_sha256':digest(config),
                'reason':'Only orchestration or unused pure-policy belief code differs; checkpoint, executed decision implementations, native identity, rules, seeds, budget and logs verified unchanged'}
            self.persist();return path,record
        return None

    def legacy_mlp(self,mode):
        diagnosis=json.loads((ROOT/'runs/catan-ai/phase5-loss-diagnosis.json').read_text())
        key='NEURAL_POLICY' if mode=='policy' else 'NEURAL_SEARCH';entry=diagnosis['runs'][key]
        folder=ROOT/entry['folder'];config=entry['configuration']
        trained=self.trained('MLP',11)[1]
        require(config['candidate']['checkpoint']==trained['checkpoint'],'Historical MLP checkpoint differs')
        for name in ('games','timings'):
            require(file_digest(folder/f'{name}.jsonl')==entry[f'{name}_sha256'],'Historical MLP log changed')
        # Verify that model/search/observation and native decision implementations
        # match the historical completed smoke record. Harness logging and stop
        # policy changed; original results keep their original provenance.
        candidates=[]
        for path in (ROOT/'runs/catan-ai').glob(f'phase3-{key}-*/run.json'):
            record=json.loads(path.read_text())
            if record['configuration']['content']['candidate']['checkpoint']==trained['checkpoint']:
                candidates.append((path,record))
        require(candidates,'No historical provenance for MLP import')
        _,reference=candidates[0]
        critical=['agent_apis/neural/provider.py','agent_apis/neural/models.py','agent_apis/neural/topology.py',
                  'agent_apis/observation.py','agent_apis/jsettlers/player.py','agent_apis/jsettlers/Bridge.java']
        if mode=='search':critical+=['agent_apis/stochastic.py','agent_apis/stochastic_search.py','agent_apis/development_belief.py']
        belief_exception=None
        for name in critical:
            if file_digest(ROOT/name)==reference['source']['files'][name]:continue
            if name=='agent_apis/development_belief.py':
                amendment=json.loads((EXPERIMENTS/'phase5-belief-fix.json').read_text())
                require(reference['source']['files'][name]==amendment['old_sha256'] and file_digest(ROOT/name)==amendment['new_sha256'],
                        'Historical belief implementation differs from documented fix')
                belief_exception='Historical pre-fix baseline; sampled-world equivalence verified; never eligible for promotion'
                continue
            require(False,f'Historical decision implementation changed: {name}')
        require(reference['metrics']['providers']['opponent']==benchmark.provider_identity('JSETTLERS'),'Historical SMART identity changed')
        rows=entry['games'];counts=Counter(r['pair_id'] for r in rows)
        complete=[r for r in rows if counts[r['pair_id']]==2]
        keep={(r['pair_id'],r['candidate_color']) for r in complete}
        timings=[json.loads(line) for line in (folder/'timings.jsonl').read_text().splitlines()]
        timings=[t for t in timings if (t['pair_id'],t['candidate_color']) in keep]
        metrics=benchmark.summarize(complete,timings,500)
        require(all(r['status']=='loss' for r in rows),'Historical MLP no longer satisfies the zero-win import condition')
        metrics.update(futility={'stopped':True,'retrospective':True,
            'reason':'Existing zero-win baseline; no new matches or prospective stopping claim'},
            all_completed_games=len(rows),unpaired_completed_games=len(rows)-len(complete),
            terminal_score_summary=entry['summary'])
        metrics['match_score_ci95']=[None,None];metrics['interval_status']='Historical zero-win sample; no superiority or optional-stopping confidence claim'
        return folder/'run.json',{'status':'historical-completed' if mode=='policy' else 'historical-interrupted',
            'metrics':{'benchmark':metrics},'historical_import':{'diagnosis':str((ROOT/'runs/catan-ai/phase5-loss-diagnosis.json').relative_to(ROOT)),
                'diagnosis_sha256':file_digest(ROOT/'runs/catan-ai/phase5-loss-diagnosis.json'),
                'games_sha256':entry['games_sha256'],'timings_sha256':entry['timings_sha256'],
                'original_provenance':str(candidates[0][0].relative_to(ROOT)),
                'belief_fix_exception':belief_exception}}

    def measure(self,model,seed,mode,final=False):
        print(f'EVALUATING {model} seed {seed}: {mode}; checkpoint={"final" if final else "validation-selected best"}',flush=True)
        if model=='MLP' and seed==11 and not final:return self.legacy_mlp(mode)
        return self.evaluate(model,seed,mode,final)

    def finish(self):
        result={'status':'running','evaluation_complete':False,'final_evaluation_seeds_used':False,
                'futility_rule':'experiments/catan-ai/FUTILITY.md','source_sha256':self.core,
                'selection_rule':self.state['selection_rule'],'architectures':{},'finalists':[]}
        measured={};training={}
        # Complete every seed11 policy first, then every search, so inexpensive
        # information is available before long search runs. Always test search
        # even if the same architecture's policy was futile.
        for model in MODEL_IDS:
            path,record=self.trained(model,11)
            training[model]={11:(path,record)};measured[model]={11:{}}
            result['architectures'][model]={'training':{},'matches':{},'confirmed_across_training_seeds':False}
        def capture(model,seed,mode,final=False):
            pair=self.measure(model,seed,mode,final)
            if not final:
                measured[model][seed][mode]=pair
                result['architectures'][model]['matches'].setdefault(str(seed),{})[mode]=descriptor(*pair)
            write(EXPERIMENTS/'phase5-evaluation-progress.json',result)
            return pair
        for mode in ('policy','search'):
            for model in MODEL_IDS:capture(model,11,mode)
        ranked=sorted([m for m in MODEL_IDS if any(eligible(r) for _,r in measured[m][11].values())],
            key=lambda m:(-max(r['metrics']['benchmark']['overall_match_score'] for _,r in measured[m][11].values() if eligible(r)),
                          training[m][11][1]['metrics']['training']['parameters']))
        finalists=ranked[:2];result['seed11_ranking']=ranked;result['finalists']=finalists
        self.state['finalists']=finalists;self.persist()
        for model in finalists:
            for seed in (23,47):
                training[model][seed]=self.trained(model,seed);measured[model][seed]={}
                for mode in ('policy','search'):capture(model,seed,mode)
        for model in MODEL_IDS:
            row=result['architectures'][model]
            for seed,(path,record) in training[model].items():
                row['training'][str(seed)]={'run_record':str(path.relative_to(ROOT)),
                    'checkpoint':record['checkpoint'],'elapsed_seconds':record['elapsed_seconds'],'metrics':record['metrics']['training']}
                # score margins are recovered via exact replay for new runs;
                # original logs/checkpoints are never changed.
                for mode,(p,r) in measured[model][seed].items():
                    if not r.get('historical_import'):
                        scored=replay(p.parent,neural_sample_games=0)
                        row['matches'][str(seed)][mode]['terminal_score_summary']=score_summary(scored)
                a,b=[games(measured[model][seed][mode][0]) for mode in ('search','policy')]
                a,b=common_prefix([a,b])
                stopped=any(not eligible(r) for _,r in measured[model][seed].values())
                comparison=benchmark.paired_difference(a,b)
                comparison.update(complete_pairs=len(a)//2,descriptive_only=stopped)
                if stopped:comparison['ci95']=[None,None]
                row['matches'][str(seed)]['search_minus_policy']=comparison
            if model in finalists:
                row['confirmed_across_training_seeds']=True;row['hierarchical']={}
                for mode in ('policy','search'):
                    sets=common_prefix([games(measured[model][seed][mode][0]) for seed in (11,23,47)])
                    aggregate=benchmark.hierarchical_bootstrap(list(zip((11,23,47),sets)))
                    aggregate['complete_pairs_per_seed']=len(sets[0])//2
                    aggregate['all_runs_eligible']=all(eligible(measured[model][seed][mode][1]) for seed in (11,23,47))
                    if not aggregate['all_runs_eligible']:aggregate['ci95']=[None,None]
                    row['hierarchical'][mode]=aggregate
                row['checkpoint_diagnostics']={}
                for mode in ('policy','search'):
                    final=capture(model,11,mode,True)
                    a,b=common_prefix([games(final[0]),games(measured[model][11][mode][0])])
                    d=benchmark.paired_difference(a,b)
                    if not eligible(final[1]) or not eligible(measured[model][11][mode][1]):d['ci95']=[None,None]
                    row['checkpoint_diagnostics'][mode]={'final_run':descriptor(*final),'final_minus_best':d,'complete_pairs':len(a)//2}
        viable=[(model,mode) for model in finalists for mode in ('policy','search')
                if result['architectures'][model]['hierarchical'][mode]['all_runs_eligible']]
        result['qualified_for_provisional_selection']=viable
        result['provisional_choice']=None
        if viable:
            raw=max(viable,key=lambda pair:result['architectures'][pair[0]]['hierarchical'][pair[1]]['mean'])
            interval=result['architectures'][raw[0]]['hierarchical'][raw[1]]['ci95']
            tied=[pair for pair in viable if (result['architectures'][pair[0]]['hierarchical'][pair[1]]['ci95'][1]>=interval[0] and
                  result['architectures'][pair[0]]['hierarchical'][pair[1]]['ci95'][0]<=interval[1])]
            simpler=min({p[0] for p in tied},key=lambda m:training[m][11][1]['metrics']['training']['parameters'])
            chosen=max([p for p in viable if p[0]==simpler],key=lambda p:result['architectures'][p[0]]['hierarchical'][p[1]]['mean'])
            result['statistically_tied_architectures']=sorted({p[0] for p in tied})
            # Do not promote search without a positive paired three-seed interval.
            if chosen[1]=='search':
                # First 20 pairs are collected before any possible futility stop.
                # Fix this comparison prefix prospectively so its sample size is
                # unaffected by which modes stop or continue.
                sets=common_prefix([games(measured[chosen[0]][seed][mode][0])[:40] for seed in (11,23,47) for mode in ('search','policy')])
                differences=[]
                for i,seed in enumerate((11,23,47)):
                    a,b=sets[2*i:2*i+2];lookup={(g['pair_id'],g['candidate_color']):g for g in b}
                    differences.append((seed,[{**g,'match_score':g['match_score']-lookup[g['pair_id'],g['candidate_color']]['match_score']} for g in a]))
                diff=benchmark.hierarchical_bootstrap(differences)
                diff['comparison_prefix']='predeclared first20 pairs per seed, before any futility decision'
                result['search_promotion_comparison']=diff
                if diff['ci95'][0]<=0:chosen=(chosen[0],'policy')
            if chosen in viable:
                record=training[chosen[0]][11][1]
                result['provisional_choice']={'model':chosen[0],'mode':chosen[1],'seed':11,'checkpoint':record['checkpoint'],'final_confirmation_pending':True}
                settings=json.loads(json.dumps(record['configuration']['content']['search']))
                settings['opponent_checkpoint']=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
                write(EXPERIMENTS/'phase5-selected.json',{'checkpoint':record['checkpoint'],'search':settings,'provisional':True,
                    'evaluation_complete':True,'mode':chosen[1],'final_seeds_used':False})
        if not result['provisional_choice']:
            result['conclusion']='No competitive architecture qualified under the predeclared futility and feasibility gates; no winner is selected and no hopeless configuration receives additional training seeds.'
            default=json.loads((EXPERIMENTS/'phase5-selected.json').read_text())
            default.update(evaluation_complete=True,competitive=False,selection_reason='Existing experimental default only; no model qualified in phase5 evaluation')
            write(EXPERIMENTS/'phase5-selected.json',default)
        result.update(status='completed',evaluation_complete=True,
            confirmation_performed=bool(finalists),all_four_seed11_training_complete=True,
            completed_at=datetime.now(timezone.utc).isoformat())
        write(EXPERIMENTS/'PHASE5_RESULTS.json',result);self.report(result)
        self.state.update(status='completed',stage='evaluation-completed',summary='experiments/catan-ai/PHASE5_RESULTS.json');self.persist()
        print('Phase 5 evaluation completed.',flush=True)

    def report(self,result):
        lines=['# Phase 5 evaluation results','',result.get('conclusion','Architecture screening and eligible finalist confirmation completed.'),'',
            'All four seed-11 models completed 20,000 updates on the same D2/W0 corpus. '
            'Policy and 500 ms search use the same frozen tuning pairs and frozen MLP opponent. '
            'Final evaluation seeds remain unused. Historical MLP observations retain their original provenance; '
            'new hopeless runs stop under the prospectively fixed futility rule.','',
            '| Architecture | Seed | Mode | Games | Wins | Match score | Mean VP margin | Status |',
            '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
        for model,row in result['architectures'].items():
            for seed,modes in row['matches'].items():
                for mode in ('policy','search'):
                    e=modes[mode];m=e['metrics'];score=e.get('terminal_score_summary') or m.get('terminal_score_summary')
                    count=m.get('all_completed_games',m['games'])
                    margin=f"{score['vp_margin_mean']:.2f}" if score else 'n/a'
                    lines.append(f"| {model} | {seed} | {mode} | {count} | {m['wins']} | {m['overall_match_score']:.3f} | {margin} | {e['status']} |")
        lines+=['','Futility uses independent board/seat pairs and corrected one-sided upper bounds. '
                'Stopped runs are not full 200-game comparisons; ordinary bootstrap intervals are suppressed. '
                'Complete-pair intersections support descriptive comparisons where schedules differ. '
                'A negative result is an evaluation outcome, not evidence of superiority or three-seed confirmation.','',
                'Training/validation/test metrics by phase/action family, value calibration, checkpoint and source hashes, '
                'matched comparisons and run references are in [PHASE5_RESULTS.json](PHASE5_RESULTS.json). '
                'Device and batch inference measurements remain in [phase5-devices.json](phase5-devices.json).','']
        (EXPERIMENTS/'PHASE5_RESULTS.md').write_text('\n'.join(lines))
        plan=ROOT/'AI_EXPERIMENT_PLAN.md';text=plan.read_text();start=text.index('Phase 5 is ');end=text.index('\n\nThe objective',start)
        text=text[:start]+('Phase 5 evaluation is complete under the user-authorized futility amendment; '
            'see [the results](experiments/catan-ai/PHASE5_RESULTS.md). All four seed-11 '
            'models trained and were evaluated in policy and 500 ms search modes. '
            'Hopeless runs stop early; additional seeds are limited to qualifying finalists. '
            'No superiority claim is made. Phases 6–10 remain pending; final seeds remain unused.')+text[end:]
        plan.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args();work=Evaluation(args.cache)
    try:work.finish()
    except BaseException as error:
        work.state.update(status='needs-investigation',error=f'{type(error).__name__}: {error}');work.persist();raise


if __name__=='__main__':main()
