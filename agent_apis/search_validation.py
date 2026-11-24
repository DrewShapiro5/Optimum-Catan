"""Freeze and audit the phase-4 stratified corpus validation suite.

python -m agent_apis.search_validation freeze
python -m agent_apis.search_validation audit [--limit N]
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time
import tracemalloc
import unittest

from catanatron.gym.envs.action_space import from_action_space
from catanatron.serialization import state_to_json
from agent_apis.benchmark import named_seed, quantile, exclusive_benchmark
from agent_apis.experiment import (ROOT, EXPERIMENTS, corpus_snapshot, digest,
                                   file_digest, source_snapshot, stream_seeds,
                                   validate_record)
from agent_apis.observation import require
from agent_apis.observation_dataset import _frames, permitted_view
from agent_apis.stochastic import (COLORS, action_id, enumerate_classes, observation,
                                   sample_class, successor, world_from_observation)
from agent_apis.development_belief import DevelopmentBelief
from agent_apis.stochastic_search import ReferenceEvaluator, StochasticSearch

CONFIG = EXPERIMENTS/'phase4.json'
SUITE = EXPERIMENTS/'search-positions-v1.json'


def stratum(view):
    if view['is_initial_build_phase']:
        return 'setup'
    if view['is_discarding']:
        return 'discard'
    if view['is_road_building']:
        return 'free-road'
    if view['current_prompt']=='MOVE_ROBBER':
        return 'robber'
    if max(view['player_state'][f'P{i}_VICTORY_POINTS'] for i in range(2))>=12:
        return 'endgame'
    return 'middle'


def freeze(config):
    pools = {name:[] for name in config['validation']['strata']}
    for directory in config['data']['corpora']:
        manifest = json.loads((ROOT/directory/'manifest.json').read_text())
        for entry in manifest['games']:
            if entry['split'] != 'validation':
                continue
            path = ROOT/directory/entry['file']
            require(file_digest(path)==entry['sha256'],'Corpus changed during suite generation')
            for _,_,view,row,_ in _frames(path):
                group = stratum(view)
                rank = digest(['phase4-suite-v1',entry['sha256'],row['step']])
                pools[group].append({'corpus':directory,'file':entry['file'],
                    'source_sha256':entry['sha256'],'game_id':entry['game_id'],
                    'step':row['step'],'stratum':group,'rank':rank})
        print(f'Scanned validation games in {directory}',flush=True)
    positions = []
    for name,count in config['validation']['strata'].items():
        require(len(pools[name])>=count,f'Not enough {name} positions')
        positions.extend(sorted(pools[name],key=lambda p:p['rank'])[:count])
    document = {'version':1,'split':'validation','selection':'lowest SHA-256 rank per predeclared stratum',
                'available_counts':{k:len(v) for k,v in pools.items()},'positions':positions}
    encoded = json.dumps(document,indent=2)+'\n'
    if SUITE.exists():
        require(SUITE.read_text()==encoded,'Refusing to replace a different frozen position suite')
    else:
        SUITE.write_text(encoded)
    print(f'{SUITE}: {len(positions)} positions, sha256={file_digest(SUITE)}',flush=True)


def frames_for_positions(document,limit=None):
    selected = document['positions'][:limit] if limit else document['positions']
    by_file = {}
    for position in selected:
        key = (position['corpus'],position['file'])
        by_file.setdefault(key,{})[position['step']] = position
    for (directory,name),steps in sorted(by_file.items()):
        path = ROOT/directory/name
        require(all(file_digest(path)==p['source_sha256'] for p in steps.values()),'Suite source hash mismatch')
        found = set()
        for game,doc,view,row,_ in _frames(path):
            if row['step'] in steps:
                position = steps[row['step']]
                require(row['game_id']==position['game_id'] and stratum(view)==position['stratum'],
                        'Frozen position identity/phase changed')
                found.add(row['step'])
                yield position,game,view,row
        require(found==set(steps),'Frozen position is missing')


def one_step_value(game,action,merged,evaluator):
    classes = enumerate_classes(game,action,merged)
    total = 0.
    for outcome in classes:
        child = successor(game,action,outcome.members[0])
        winner = child.winning_color()
        value = float(winner==action.color) if winner is not None else evaluator.value(observation(child,action.color.value))
        total += outcome.probability*value
    return total,classes


def audit(config,output,limit=None):
    started = time.perf_counter()
    source = source_snapshot()
    datasets = corpus_snapshot(config)
    require(file_digest(SUITE)==config['validation']['suite_sha256'],'Frozen validation suite changed')
    document = json.loads(SUITE.read_text())
    validation = config['validation']
    require(document['version']==1 and document['split']=='validation','Wrong validation suite')
    require(len(document['positions'])==validation['positions']==1000,'Expected 1000 frozen positions')
    require(dict(Counter(p['stratum'] for p in document['positions']))==validation['strata'],
            'Frozen suite strata differ from configuration')
    # Match each suite reference against the manifest's validated game-level split.
    entries = {(c['path'],g['file']):g for c in datasets for g in c['games']}
    require(len({(p['corpus'],p['file'],p['step']) for p in document['positions']})==1000,'Duplicate position')
    for position in document['positions']:
        entry = entries[(position['corpus'],position['file'])]
        require(entry['split']=='validation' and entry['sha256']==position['source_sha256'],
                'Suite contains a different split or changed shard')
    test_stream = io.StringIO()
    tests = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_stochastic_search.py', top_level_dir=str(ROOT))
    tested = unittest.TextTestRunner(stream=test_stream,verbosity=2).run(tests)
    require(tested.wasSuccessful(),test_stream.getvalue())
    print(f'{tested.testsRun} search correctness tests passed',flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f'phase4-validation-{digest(config)[:12]}-{stamp}'
    folder = output/run_id
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'tests.txt').write_text(test_stream.getvalue())
    (folder/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    failures,rows,frequency,probed_strata = [],[],{},set()
    max_error = 0.
    evaluator = ReferenceEvaluator()
    with (folder/'positions.jsonl').open('w') as stream:
        for index,(position,authoritative,view,row) in enumerate(frames_for_positions(document,limit)):
            try:
                seed = named_seed(validation['search_seed'],position['rank'])
                belief = DevelopmentBelief(view)
                world = world_from_observation(view,belief.sample(random.Random(seed)))
                action = from_action_space(row['action_id'],world.state.current_color(),COLORS,'BASE')
                values,distributions = [],[]
                for merged in (False,True):
                    value,classes = one_step_value(world,action,merged,evaluator)
                    values.append(value);distributions.append(classes)
                error = abs(values[0]-values[1])
                max_error = max(max_error,error)
                require(error<=validation['weighted_value_tolerance'],'Merged one-step value mismatch')
                # Perturb the actual future, not the simulated one; model input is
                # identical. The engine has no path back to this authoritative game.
                altered = authoritative.copy()
                altered.random = altered.state.random = random.Random(seed+1)
                altered.state.development_listdeck.reverse()
                altered_view = permitted_view(state_to_json(altered),view['perspective'],view['current_playable_actions'])
                require(altered_view==view,'Authoritative future changed permitted observation')
                measured = {}
                traced = position['stratum'] not in probed_strata and len(view['current_playable_actions'])>1
                for merged in (False,True):
                    gc.collect()
                    if traced:
                        tracemalloc.start()
                    try:
                        result = StochasticSearch(view,seed=seed,merged=merged,
                            c_puct=config['search']['c_puct'],max_root_depth=validation['max_root_depth'],
                            max_actions=validation['max_actions_per_simulation']).run(validation['simulations'])
                        peak = tracemalloc.get_traced_memory()[1] if traced else None
                    finally:
                        if traced:
                            tracemalloc.stop()
                    require(result['action'] in view['current_playable_actions'],'Illegal search result')
                    measured['merged' if merged else 'unmerged'] = {
                        **result,'peak_traced_bytes':peak,'memory_traced':traced,
                        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
                if traced:
                    probed_strata.add(position['stratum'])
                record = {**position,'one_step_value_error':error,
                          'raw_one_step_outcomes':sum(len(c.members) for c in distributions[0]),
                          'merged_one_step_classes':len(distributions[1]),**measured}
                rows.append(record)
                stream.write(json.dumps(record)+'\n');stream.flush()
                family = action.action_type.value
                if family in ('ROLL','MOVE_ROBBER','BUY_DEVELOPMENT_CARD') and len(distributions[1])>1 and family not in frequency:
                    frequency[family] = distributions[1]
            except Exception as error:
                failure = {**position,'error':f'{type(error).__name__}: {error}'}
                failures.append(failure)
                stream.write(json.dumps({'failure':failure})+'\n');stream.flush()
                print(f'FAILED {position["file"]}:{position["step"]}: {error}',flush=True)
            if (index+1)%25==0:
                print(f'{index+1}/{limit or 1000} positions, failures={len(failures)}',flush=True)
    # Predeclared simultaneous Hoeffding bounds, across all classes in all cases.
    confidence = []
    classes_total = sum(len(v) for v in frequency.values())
    n = validation['frequency_draws_per_distribution']
    alpha = validation['familywise_alpha']
    epsilon = math.sqrt(math.log(2*max(1,classes_total)/alpha)/(2*n))
    frequency_ok = True
    for name,classes in sorted(frequency.items()):
        rng = random.Random(named_seed(validation['search_seed'],'frequency:'+name))
        observed = Counter(sample_class(classes,rng).key for _ in range(n))
        outcomes = [{'probability':c.probability,'observed':observed[c.key],
                     'frequency':observed[c.key]/n,'members':[str(o.result) for o in c.members],
                     'within_bound':abs(observed[c.key]/n-c.probability)<=epsilon}
                    for c in classes]
        frequency_ok &= all(o['within_bound'] for o in outcomes)
        confidence.append({'family':name,'draws':n,'absolute_bound':epsilon,'classes':outcomes})
    variants = {}
    for mode in ('unmerged','merged'):
        values = [r[mode] for r in rows]
        elapsed = sum(v['metrics']['elapsed_seconds'] for v in values)
        simulations = sum(v['metrics']['simulations'] for v in values)
        untraced = [v for v in values if not v['memory_traced']]
        untraced_elapsed = sum(v['metrics']['elapsed_seconds'] for v in untraced)
        untraced_simulations = sum(v['metrics']['simulations'] for v in untraced)
        peaks = [v['peak_traced_bytes'] for v in values if v['memory_traced']]
        variants[mode] = {'positions':len(values),'forced_positions':sum(v['metrics']['forced'] for v in values),
            'simulations':simulations,'elapsed_search_seconds':elapsed,
            'uninstrumented_seconds':untraced_elapsed,'uninstrumented_simulations':untraced_simulations,
            'simulations_per_second':untraced_simulations/untraced_elapsed if untraced_elapsed else None,
            'raw_outcomes':sum(v['metrics']['raw_outcomes'] for v in values),
            'outcome_classes':sum(v['metrics']['merged_classes'] for v in values),
            'retained_chance_children':sum(v['metrics']['retained_chance_children'] for v in values),
            'traced_memory_probes':len(peaks),'peak_traced_bytes_p50':quantile(peaks,.5),
            'peak_traced_bytes_max':max(peaks,default=0),
            'process_peak_rss_bytes_max':max((v['process_peak_rss_bytes'] for v in values),default=0),
            'max_action_depth':max((v['metrics']['max_action_depth'] for v in values),default=0),
            'max_root_depth':max((v['metrics']['max_root_depth'] for v in values),default=0),
            'max_completed_turn_depth':max((v['metrics']['max_completed_turn_depth'] for v in values),default=0),
            'capped_simulations':sum(v['metrics']['capped_simulations'] for v in values)}
    gate = (not failures and limit is None and len(rows)==1000 and frequency_ok
            and len(frequency)==3 and tested.wasSuccessful())
    record = {'schema_version':1,'run_id':run_id,'status':'completed' if not failures else 'failed',
              'created_at':datetime.now(timezone.utc).isoformat(),
              'configuration':{'content':config,'sha256':digest(config)},
              'seed':validation['search_seed'],'rng_seeds':stream_seeds(validation['search_seed']),
              'source':source,'datasets':datasets,'checkpoint':None,
              'software':{'python':sys.version,'executable':sys.executable,
                  'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))},
              'hardware':{'platform':platform.platform(),'machine':platform.machine(),'available_devices':['cpu']},
              'metrics':{'search_validation':{'gate_passed':gate,'scope':'partial' if limit else 'all',
                  'suite_sha256':file_digest(SUITE),'positions':len(rows),'failures':failures,
                  'strata':dict(Counter(r['stratum'] for r in rows)),
                  'unit_tests_passed':tested.testsRun,'maximum_weighted_one_step_value_error':max_error,
                  'frequency_gate_passed':frequency_ok,'frequency':confidence,
                  'variants':variants,'different_selected_actions':sum(r['merged']['action']!=r['unmerged']['action'] for r in rows),
                  'artifacts':{'positions.jsonl':file_digest(folder/'positions.jsonl'),
                               'tests.txt':file_digest(folder/'tests.txt')}}},
              'elapsed_seconds':time.perf_counter()-started}
    validate_record(record)
    (folder/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(folder/'run.json',flush=True)
    require(not failures and frequency_ok,'Search validation failed; see run record')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','audit'))
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/catan-ai')
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    require(args.limit is None or args.limit>0,'Positive limit required')
    config = json.loads(args.config.read_text())
    with exclusive_benchmark():
        if args.command=='freeze':
            freeze(config)
        else:
            audit(config,args.output,args.limit)


if __name__=='__main__':
    main()
