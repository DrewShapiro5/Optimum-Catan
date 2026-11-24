"""Paired, serial Catan benchmark. Final seeds are reserved for phase 10.

python -m agent_apis.benchmark run --candidate RANDOM --tier smoke
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from catanatron import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import action_from_json, action_to_json, map_to_json, web_view
from catanatron.gym.envs.action_space import to_action_space

from agent_apis.experiment import (ROOT, EXPERIMENTS, corpus_snapshot, digest,
                                   file_digest, source_snapshot, stream_seeds,
                                   validate_record)
from agent_apis.observation import from_web_view, require
from agent_apis.futility import DEFAULT as DEFAULT_FUTILITY,check as check_futility

SEED_FILE = EXPERIMENTS / 'evaluation-seeds-v1.json'
PROTOCOL = EXPERIMENTS / 'benchmark-protocol-v1.json'
CONFIG = EXPERIMENTS / 'phase3.json'
TIERS = {'smoke': 20, 'screening': 100, 'expanded-tuning': 500}
COLORS = (Color.BLUE, Color.RED)


def named_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'catan-benchmark-v1:{seed}:{name}'.encode()).digest()[:8], 'big')


def corpus_seeds(config):
    paths = [ROOT / p / 'manifest.json' for p in config['data']['corpora']]
    legacy = ROOT / 'data/jsettlers-1v1/manifest.json'
    if legacy.exists():
        paths.append(legacy)
    return {g['seed'] for p in paths for g in json.loads(p.read_text())['games']}


def frozen_seeds(excluded, master=2026110403, count=500):
    used = set(excluded)
    result = {'version': 1, 'master_seed': master, 'sets': {}}
    for name in ('tuning', 'final'):
        pairs = []
        for index in range(count):
            values = {}
            for stream in ('board', 'simulator', 'agent'):
                value = named_seed(master, f'{name}:{index}:{stream}')
                while value in used:
                    value = named_seed(value, 'collision')
                used.add(value)
                values[stream + '_seed'] = value
            pairs.append({'pair_id': f'{name}-{index:04d}', **values})
        result['sets'][name] = pairs
    return result


def validate_seeds(document, excluded):
    require(document['version'] == 1, 'Unsupported seed list')
    used = set(excluded)
    ids = set()
    for name in ('tuning', 'final'):
        pairs = document['sets'][name]
        require(len(pairs) == 500, 'Expected 500 tuning and 500 final pairs')
        for pair in pairs:
            require(pair['pair_id'] not in ids, 'Duplicate pair ID')
            ids.add(pair['pair_id'])
            for key in ('board_seed', 'simulator_seed', 'agent_seed'):
                value = pair[key]
                require(type(value) is int and value >= 0 and value not in used,
                        'Seed overlap with corpus, another stream or another evaluation set')
                used.add(value)


@contextmanager
def exclusive_benchmark():
    """Do not permit overlapping harness runs, even across different outputs."""
    from agent_apis.neural.device_utils import file_lock
    folder = ROOT / 'runs/catan-ai'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.benchmark.lock').open('a') as lock:
        try:
            with file_lock(lock, nonblocking=True):
                yield
        except BlockingIOError as error:
            raise RuntimeError('Another benchmark is running; timing jobs must be serial') from error


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower]) * (position-lower)


def interval(values):
    return [quantile(values, .025), quantile(values, .975)]


def paired_bootstrap(games, seed=11, repeats=5000):
    groups = {}
    for game in games:
        groups.setdefault(game['pair_id'], []).append(game['match_score'])
    require(all(len(v) == 2 for v in groups.values()), 'Bootstrap requires complete seat pairs')
    means = [sum(v)/2 for v in groups.values()]
    if not means:
        return [None, None]
    rng = random.Random(named_seed(seed, 'bootstrap'))
    return interval([sum(rng.choices(means, k=len(means)))/len(means) for _ in range(repeats)])


def paired_difference(first, second, seed=11, repeats=5000):
    def indexed(games):
        result = {}
        for game in games:
            key = (game['pair_id'], game['candidate_color'])
            require(key not in result, 'Duplicate paired game')
            result[key] = game
        return result
    a, b = indexed(first), indexed(second)
    require(a.keys() == b.keys() and a, 'Comparisons require identical board/seat pairs')
    groups = {}
    for key in a:
        require(a[key]['board_sha256'] == b[key]['board_sha256'] and
                a[key]['simulator_seed'] == b[key]['simulator_seed'], 'Different evaluation boards/seeds')
        groups.setdefault(key[0], []).append(a[key]['match_score'] - b[key]['match_score'])
    require(all(len(v) == 2 for v in groups.values()), 'Incomplete seat pairs')
    means = [sum(v)/2 for v in groups.values()]
    rng = random.Random(named_seed(seed, 'difference-bootstrap'))
    samples = [sum(rng.choices(means, k=len(means)))/len(means) for _ in range(repeats)]
    return {'difference': sum(means)/len(means), 'ci95': interval(samples)}


def hierarchical_bootstrap(runs, seed=11, repeats=5000):
    """Resample training seed, then shared board pairs; never resample moves."""
    require(len(runs) >= 3, 'Confirmation requires at least three training seeds')
    by_seed = {}
    for training_seed, games in runs:
        require(training_seed not in by_seed, 'Duplicate training seed')
        pairs = {}
        for game in games:
            pairs.setdefault(game['pair_id'], []).append(game['match_score'])
        require(all(len(v) == 2 for v in pairs.values()), 'Incomplete seat pairs')
        by_seed[training_seed] = {p: sum(v)/2 for p, v in pairs.items()}
    names = list(by_seed)
    pairs = list(by_seed[names[0]])
    require(pairs and all(set(v) == set(pairs) for v in by_seed.values()), 'Unshared evaluation pairs')
    rng = random.Random(named_seed(seed, 'hierarchical-bootstrap'))
    samples = []
    for _ in range(repeats):
        sampled_seeds = rng.choices(names, k=len(names))
        # Share pair resampling across seeds to retain common-board dependence.
        sampled_pairs = rng.choices(pairs, k=len(pairs))
        samples.append(sum(by_seed[s][p] for s in sampled_seeds for p in sampled_pairs)
                       / (len(names)*len(pairs)))
    return {'mean': sum(sum(v.values())/len(pairs) for v in by_seed.values())/len(names),
            'ci95': interval(samples),
            'per_training_seed': {str(s): sum(v.values())/len(v) for s, v in by_seed.items()}}


class DecisionContext:
    def __init__(self, started, budget_ms, mode, simulations, settings=None):
        self.deadline = started + budget_ms/1000
        self.admission_deadline = self.deadline - max(.020, budget_ms*.00005)
        self.mode, self.simulations = mode, simulations
        self.settings = settings or {}

    def remaining_ms(self):
        return max(0., (self.deadline-time.perf_counter())*1000)


class Agent:
    def __init__(self, key, color, seed, checkpoint=None):
        started = time.perf_counter()
        self.key, self.color = key, color
        self.rng = random.Random(seed)
        self.cold = True
        self.implementation = None
        if key not in ('RANDOM', 'VALUE'):
            from agent_apis import _entries, _implementation
            entry = next((e for e in _entries() if e['key'] == key), None)
            require(entry is not None, f'Unregistered provider: {key}')
            self.implementation = _implementation(entry)
            ready, reason = self.implementation.availability()
            require(ready, reason)
        if checkpoint:
            require(file_digest(ROOT / checkpoint['path']) == checkpoint['sha256'], 'Checkpoint hash mismatch')
            require(hasattr(self.implementation, 'from_checkpoint'), 'Provider cannot load frozen checkpoints')
            self.implementation = self.implementation.from_checkpoint(ROOT / checkpoint['path'])
        self.init_ms = (time.perf_counter()-started)*1000

    def select(self, game, actions, context):
        if self.key == 'RANDOM':
            # Include the same observation boundary in candidate decision timing.
            from_web_view(web_view(game, self.color), self.color.value)
            return self.rng.choice(actions)
        if self.key == 'VALUE':
            # Reuse framework weights on deterministic/sampled afterstates. Never
            # evaluate a clone of the live future RNG or actual hidden dev deck.
            from catanatron.players.value import base_fn
            from_web_view(web_view(game, self.color), self.color.value)
            value_fn = base_fn()
            best, best_value = None, -math.inf
            for action in actions:
                clone = game.copy()
                clone.random = clone.state.random = random.Random(self.rng.getrandbits(64))
                # The framework scorer depends only on hand total for draws.
                # A neutral knight draw avoids querying hidden deck composition.
                clone.state.development_listdeck = ['KNIGHT'] * len(clone.state.development_listdeck)
                clone.execute(action)
                value = value_fn(clone, self.color)
                if value > best_value:
                    best, best_value = action, value
            return best
        observation = web_view(game, self.color)
        if self.key == 'JSETTLERS':
            if self.cold:
                identity = self.implementation.synchronize(observation, self.color.value)
                require(identity['engine'] == 'SOCRobotBrain' and identity['strategy'] == 'SMART',
                        'Native provider identity/strategy mismatch')
            return action_from_json(self.implementation.choose(observation))
        # New learned providers receive ONLY the audited safe boundary and an
        # explicit budget context, never the authoritative Game.
        safe = from_web_view(observation, self.color.value)
        # Public routing metadata, deliberately outside learned observations.
        context.session_id=game.id
        require(hasattr(self.implementation, 'benchmark_choose'), 'Provider lacks budget-aware observation API')
        return action_from_json(self.implementation.benchmark_choose(safe, context, self.rng))


def make_game(pair, friendly_robber=True):
    game = Game([SimplePlayer(c) for c in COLORS], seed=pair['board_seed'],
                vps_to_win=15, discard_limit=9, friendly_robber=friendly_robber)
    # Board and starting-seat generation are isolated from simulator futures.
    game.random = game.state.random = random.Random(pair['simulator_seed'])
    game.state.development_listdeck = starting_devcard_bank()
    game.random.shuffle(game.state.development_listdeck)
    for color in COLORS:
        for kind in ('ROAD', 'SETTLEMENT', 'CITY'):
            game.state.buildings_by_color[color][kind]
    return game


def play_game(pair, candidate_color, config, timing_stream=None, agent_factory=Agent):
    from agent_apis.jsettlers.player import BRIDGE
    BRIDGE.close()
    game = make_game(pair, friendly_robber=config.get('rules', {}).get('friendly_robber', True))
    board_hash = digest(map_to_json(game.state.board.map))
    opponent_color = next(c for c in COLORS if c != candidate_color)
    agents = {}
    result = {**pair, 'candidate_color': candidate_color.value,
              'candidate_start_seat': game.state.colors.index(candidate_color),
              'board_sha256': board_hash, 'status': 'error', 'winner': None,
              'match_score': 0., 'error': None, 'illegal_actions': 0,
              'initialization_ms': {}, 'turns': 0, 'actions': 0}
    timings = []
    started = time.perf_counter()
    try:
        for color, spec, role in [(candidate_color, config['candidate'], 'candidate'),
                                  (opponent_color, config['opponent'], 'opponent')]:
            agents[color] = agent_factory(spec['key'], color,
                named_seed(pair['agent_seed'], role), spec.get('checkpoint'))
            result['initialization_ms'][role] = agents[color].init_ms
        while game.winning_color() is None:
            if game.state.num_turns >= config['evaluation']['max_turns'] or len(game.state.action_records) >= config['evaluation']['max_actions']:
                result.update(status='cutoff', match_score=.5)
                break
            color = game.state.current_color()
            agent = agents[color]
            actions = list(game.playable_actions)
            role = 'candidate' if color == candidate_color else 'opponent'
            decision_start = time.perf_counter()
            forced = len(actions) == 1
            cold = agent.cold and not forced
            event = {'pair_id': pair['pair_id'], 'candidate_color': candidate_color.value,
                     'step': len(game.state.action_records), 'role': role,
                     'actor': color.value, 'legal_actions': len(actions), 'forced': forced,
                     'cold': cold, 'budget_ms': config['evaluation']['budget_ms'],
                     'action_id': None, 'error': None}
            try:
                if forced:
                    action = actions[0]
                else:
                    context = DecisionContext(decision_start, config['evaluation']['budget_ms'],
                        config['evaluation']['mode'], config['evaluation']['simulations'],config)
                    action = agent.select(game, actions, context)
                    agent.cold = False
                    search_metrics=getattr(getattr(agent,'implementation',None),'last_search_metrics',None)
                    if search_metrics is not None:
                        event['search']={k:v for k,v in search_metrics.items() if k!='worker_metrics'}
                if action not in actions:
                    result['illegal_actions'] += 1
                    raise ValueError('Agent returned an illegal action')
                event['action_id'] = to_action_space(action, COLORS, 'BASE')
            except Exception as error:
                event['error'] = f'{type(error).__name__}: {error}'
                raise
            finally:
                event['elapsed_ms'] = (time.perf_counter()-decision_start)*1000
                event['overrun'] = event['elapsed_ms'] > event['budget_ms']
                timings.append(event)
                if timing_stream:
                    timing_stream.write(json.dumps(event) + '\n')
            game.execute(action)
        winner = game.winning_color()
        if winner is not None:
            result.update(status='win' if winner == candidate_color else 'loss',
                          winner=winner.value, match_score=float(winner == candidate_color))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result.update(turns=game.state.num_turns, actions=len(game.state.action_records),
                      elapsed_seconds=time.perf_counter()-started)
        from catanatron.state_functions import get_actual_victory_points,get_visible_victory_points
        result['terminal_scores']={c.value:{'actual_vp':get_actual_victory_points(game.state,c),
            'public_vp':get_visible_victory_points(game.state,c)} for c in COLORS}
        result['candidate_vp_margin']=(get_actual_victory_points(game.state,candidate_color)-
                                       get_actual_victory_points(game.state,opponent_color))
        result['native_translation_rejections'] = {}
        if BRIDGE.process is not None and BRIDGE.process.poll() is None:
            for color, agent in agents.items():
                if agent.key == 'JSETTLERS':
                    try:
                        status = BRIDGE.request(op='status', session=f'{game.id}:{color.value}')
                        result['native_translation_rejections'][color.value] = status['translation_rejections']
                    except Exception as error:
                        result.update(status='error', match_score=0.,
                                      error=f'Native diagnostics failed: {type(error).__name__}: {error}')
        from catanatron.serialization import action_record_to_json
        result['trajectory_sha256'] = digest([action_record_to_json(r) for r in game.state.action_records])
        BRIDGE.close()
    return result, timings


def summarize(games, timings, budget_ms, bootstrap_seed=11, bootstrap_repeats=5000):
    counts = Counter(g['status'] for g in games)
    completed = counts['win'] + counts['loss']
    valid_without_cutoffs = len(games) - counts['cutoff']
    candidate = [t for t in timings if t['role'] == 'candidate']
    warm = [t['elapsed_ms'] for t in candidate if not t['cold'] and not t['forced']]
    all_times = [t['elapsed_ms'] for t in candidate]
    cold = [t['elapsed_ms'] for t in candidate if t['cold']]
    limit = budget_ms + max(20, .05 * budget_ms)
    def latencies(values):
        return {'count': len(values), 'p50_ms': quantile(values, .5),
                'p95_ms': quantile(values, .95), 'p99_ms': quantile(values, .99)}
    mean_score = sum(g['match_score'] for g in games)/len(games) if games else None
    ci = paired_bootstrap(games, bootstrap_seed, bootstrap_repeats)
    seats = {}
    for seat in (0, 1):
        subset = [g for g in games if g['candidate_start_seat'] == seat]
        seats[str(seat)] = {'games': len(subset), 'wins': sum(g['status']=='win' for g in subset),
                           'losses': sum(g['status']=='loss' for g in subset),
                           'cutoffs': sum(g['status']=='cutoff' for g in subset),
                           'errors': sum(g['status']=='error' for g in subset),
                           'match_score': sum(g['match_score'] for g in subset)/len(subset) if subset else None}
    return {'games': len(games), 'wins': counts['win'], 'losses': counts['loss'],
            'cutoffs': counts['cutoff'], 'errors': counts['error'],
            'illegal_actions': sum(g['illegal_actions'] for g in games),
            'completed_game_win_rate': counts['win']/completed if completed else None,
            'overall_match_score': mean_score, 'match_score_ci95': ci,
            'score_without_cutoffs': sum(g['match_score'] for g in games if g['status']!='cutoff')/valid_without_cutoffs if valid_without_cutoffs else None,
            'seat_specific': seats, 'turns': {'p50': quantile([g['turns'] for g in games], .5), 'max': max((g['turns'] for g in games), default=0)},
            'candidate_latency_all': latencies(all_times), 'candidate_latency_warm_nonforced': latencies(warm),
            'candidate_cold_first_decision': latencies(cold),
            'candidate_decision_seconds': sum(all_times)/1000,
            'candidate_decisions_per_second': len(all_times)/(sum(all_times)/1000) if sum(all_times) else None,
            'candidate_overruns': sum(t['overrun'] for t in candidate),
            'native_translation_rejections': sum(sum(g.get('native_translation_rejections',{}).values()) for g in games),
            'candidate_warm_overruns': sum(t['overrun'] for t in candidate if not t['cold']),
            'p95_limit_ms': limit,
            'timing_gate_passed': bool(warm) and quantile(warm, .95) <= limit,
            'cutoff_gate_passed': bool(games) and counts['cutoff']/len(games) <= .01,
            'failure_gate_passed': counts['error'] == 0 and sum(g['illegal_actions'] for g in games)==0,
            'lower_bound_above_50_percent': ci[0] is not None and ci[0] > .5,
            'failure_score_convention': 'Errors are invalid games with zero candidate credit; never wins. Any error blocks promotion.'}


def provider_identity(key):
    if key in ('RANDOM', 'VALUE'):
        result = {'key': key, 'implementation': 'agent_apis.benchmark:Agent',
                  'sha256': file_digest(Path(__file__))}
        if key == 'VALUE':
            import catanatron.players.value as value
            result['framework_value_sha256'] = file_digest(value.__file__)
            result['weights'] = value.DEFAULT_WEIGHTS
            result['variant'] = 'Independent-future afterstates; neutral hidden development draw; framework base_fn weights'
        return result
    from agent_apis import _entries, _implementation
    entry = next((e for e in _entries() if e['key'] == key), None)
    require(entry is not None, f'Unknown provider {key}')
    cls = _implementation(entry)
    ready, reason = cls.availability()
    require(ready, reason)
    import inspect
    result = {**entry, 'source_sha256': file_digest(inspect.getfile(cls))}
    if key == 'JSETTLERS':
        upstream = ROOT / 'integrations/jsettlers/upstream'
        revision = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(['git', '-C', str(upstream), 'status', '--porcelain'],
                                capture_output=True, text=True, check=True).stdout.strip()
        require(revision == '7a756baac6ad675f0930605bfb3a8e1bd7a796a5' and not status,
                'JSettlers upstream differs from pinned clean revision')
        result.update(revision=revision, strategy='SMART', engine='SOCRobotBrain',
                      jars={p.name: file_digest(p) for p in sorted((ROOT/'integrations/jsettlers/lib').glob('*.jar'))})
        result['java_version'] = subprocess.run(['java', '-version'], capture_output=True,
                                               text=True, check=True).stderr.strip()
        require(result['jars'].get('gson-2.8.6.jar') == 'c8fb4839054d280b3033f800d1f5a97de2f028eb8ba2eb458ad287e536f3f25f', 'Gson checksum mismatch')
    return result


def schedule_sizes(spec):
    if spec.get('schedule','original-v1')=='original-v1':return dict(TIERS)
    require(spec.get('schedule')=='quick-v1','Unknown evaluation schedule')
    path=EXPERIMENTS/'evaluation-quick-v1.json'
    require(file_digest(path)==spec.get('schedule_sha256'),'Quick schedule checksum changed')
    schedule=json.loads(path.read_text())
    require(schedule['id']=='quick-v1' and schedule['seed_set']=='tuning' and
            schedule['smoke_pairs']==5 and schedule['screening_pairs']==25 and
            schedule['base_protocol_sha256']==file_digest(PROTOCOL),'Quick schedule differs from amendment')
    return {'smoke':5,'screening':25}


def run(config, output, continuation=None):
    started = time.perf_counter()
    config=json.loads(json.dumps(config))
    config['evaluation'].setdefault('futility',dict(DEFAULT_FUTILITY))
    require(config['stage'] == 'benchmark', 'Expected benchmark configuration')
    spec = config['evaluation']
    model_label=(config.get('model') or {}).get('id',config['candidate']['key'])
    sizes=schedule_sizes(spec)
    require(spec['seed_set'] == 'tuning' and spec['tier'] in sizes, 'Final seeds are reserved for phase 10')
    require(spec['budget_ms'] in (100, 500, 2000), 'Unsupported time budget')
    require(spec['mode'] in ('policy', 'time', 'simulations'), 'Unknown decision mode')
    require(spec['mode'] != 'simulations' or spec['simulations'] in (128, 512, 2048), 'Unsupported simulation budget')
    require(file_digest(SEED_FILE) == spec['seed_file_sha256'] and file_digest(PROTOCOL) == spec['protocol_sha256'], 'Frozen protocol or seed list changed')
    protocol = json.loads(PROTOCOL.read_text())
    require(config['rules'] == protocol['rules'], 'Rules differ from frozen protocol')
    require(spec['max_turns'] == protocol['max_turns'] and
            spec['max_actions'] == protocol['max_actions'] and
            spec['cutoff_score'] == protocol['cutoff_score'] and
            spec['bootstrap_repeats'] == protocol['bootstrap']['repeats'],
            'Limits/scoring/interval settings differ from frozen protocol')
    require(spec.get('schedule')=='quick-v1' or sizes[spec['tier']] == protocol['schedule'][
        {'smoke':'smoke_pairs','screening':'screening_pairs','expanded-tuning':'expanded_tuning_pairs'}[spec['tier']]],
        'Evaluation schedule differs from frozen protocol')
    seeds = json.loads(SEED_FILE.read_text())
    validate_seeds(seeds, corpus_seeds(config))
    require(spec['mode'] == 'policy' or config['candidate']['key'] not in ('RANDOM', 'VALUE', 'JSETTLERS'), 'Baseline agents support policy mode only')
    providers = {role: provider_identity(config[role]['key']) for role in ('candidate', 'opponent')}
    pairs = seeds['sets']['tuning'][:sizes[spec['tier']]]
    datasets = corpus_snapshot(config)
    source = source_snapshot()
    partial_folder=None;resume_info=None;elapsed_before_resume=0.
    if spec.get('schedule')=='quick-v1':
        for candidate in sorted(output.glob('phase5q-*'),reverse=True):
            if (candidate/'run.json').exists() or not (candidate/'progress.json').exists():continue
            if digest(json.loads((candidate/'configuration.json').read_text()))!=digest(config):continue
            provenance=json.loads((candidate/'provenance.json').read_text())
            require(provenance['providers']==providers,'Partial run provider identities changed')
            engine=lambda snap:{k:v for k,v in snap['files'].items() if k.startswith('framework/catanatron/catanatron/') and k.endswith('.py')}
            require(engine(provenance['source'])==engine(source),'Partial run simulator implementation changed')
            resume_info=json.loads((candidate/'progress.json').read_text())
            require(resume_info['configuration_sha256']==digest(config),'Partial checkpoint configuration changed')
            for name in ('games','timings'):
                path=candidate/(name+'.jsonl');raw=path.read_bytes();kept=raw[:resume_info[name+'_bytes']]
                require(hashlib.sha256(kept).hexdigest()==resume_info[name+'_sha256'],'Partial checkpoint logs changed')
                if len(raw)>len(kept):(candidate/(name+'.uncommitted-tail')).write_bytes(raw[len(kept):])
                path.write_bytes(kept)
            source=provenance['source'];partial_folder=candidate
            elapsed_before_resume=resume_info['elapsed_seconds'];break
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    label=(f'phase5q-{model_label}-seed{config["seed"]}-{config["evaluation"]["mode"]}' if spec.get('schedule')=='quick-v1' else f'phase3-{config["candidate"]["key"]}')
    run_id = f'{label}-{digest(config)[:12]}-{stamp}'
    folder = partial_folder or output / run_id
    if partial_folder:run_id=folder.name
    else:folder.mkdir(parents=True, exist_ok=False)
    # Preserve resolved choices before launching any game.
    (folder/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    (folder/'provenance.json').write_text(json.dumps({'source':source,'providers':providers,'configuration_sha256':digest(config)},indent=2)+'\n')
    games, timings = [], []
    reused_pairs=0
    prior=json.loads(Path(continuation).read_text()) if continuation else None
    if partial_folder:
        games=[json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
        timings=[json.loads(line) for line in (folder/'timings.jsonl').read_text().splitlines()]
        require(len(games)==resume_info['completed_games'],'Partial game count differs')
        require(all(g['status']!='error' for g in games),'Partial run contains an error; investigate before resuming')
        require([(g['pair_id'],g['candidate_color']) for g in games]==[(p['pair_id'],c.value) for p in pairs for c in COLORS][:len(games)],'Partial games are not the exact schedule prefix')
        print(f'RESUME {model_label}: preserving {len(games)} completed games',flush=True)
        reused_pairs=sizes['smoke'] if continuation else 0
    if continuation is not None and not partial_folder:
        prior_path=Path(continuation);prior=json.loads(prior_path.read_text());validate_record(prior)
        before=json.loads(json.dumps(prior['configuration']['content']))
        after=json.loads(json.dumps(config))
        require(before['evaluation']['tier']=='smoke' and spec['tier']=='screening','Only smoke-to-screening continuation is supported')
        before['evaluation']['tier']=after['evaluation']['tier']
        require(before==after,'Continuation changed more than the schedule tier')
        require(prior['metrics']['providers']==providers,'Continuation provider identities changed')
        require(prior['metrics']['benchmark']['failure_gate_passed'],'Cannot continue a failed smoke run')
        require(prior['status']=='completed' and not prior['metrics']['benchmark'].get('futility',{}).get('stopped'),
                'Cannot extend an early-stopped smoke run')
        for name,checksum in prior['metrics']['artifacts'].items():
            require(file_digest(prior_path.parent/name)==checksum,'Continuation logs changed')
        games=[json.loads(line) for line in (prior_path.parent/'games.jsonl').read_text().splitlines()]
        timings=[json.loads(line) for line in (prior_path.parent/'timings.jsonl').read_text().splitlines()]
        reused_pairs=sizes['smoke']
        require(len(games)==2*reused_pairs and
                [(g['pair_id'],g['candidate_color']) for g in games]==
                [(p['pair_id'],c.value) for p in pairs[:reused_pairs] for c in COLORS],
                'Continuation is not the exact complete smoke prefix')
    futility=check_futility(games,len(pairs),spec['futility']) if len(games)%2==0 else {'stopped':False}
    with (folder/'games.jsonl').open('w') as game_stream, (folder/'timings.jsonl').open('w') as timing_stream:
        for game in games:game_stream.write(json.dumps(game)+'\n')
        for event in timings:timing_stream.write(json.dumps(event)+'\n')
        completed_at_start=len(games)
        for index, pair in enumerate(pairs[completed_at_start//2:],start=completed_at_start//2):
            if futility['stopped']:break
            hashes = [g['board_sha256'] for g in games if g['pair_id']==pair['pair_id']]
            for color in COLORS:
                if any(g['pair_id']==pair['pair_id'] and g['candidate_color']==color.value for g in games):continue
                game, measured = play_game(pair, color, config, timing_stream)
                hashes.append(game['board_sha256'])
                games.append(game)
                timings.extend(measured)
                game_stream.write(json.dumps(game)+'\n')
                game_stream.flush()
                timing_stream.flush()
                if spec.get('schedule')=='quick-v1':
                    checkpoint={'completed_games':len(games),'games_bytes':game_stream.tell(),'timings_bytes':timing_stream.tell(),
                        'games_sha256':file_digest(folder/'games.jsonl'),'timings_sha256':file_digest(folder/'timings.jsonl'),
                        'elapsed_seconds':elapsed_before_resume+time.perf_counter()-started,'configuration_sha256':digest(config)}
                    temporary=folder/'progress.tmp';temporary.write_text(json.dumps(checkpoint)+'\n');temporary.replace(folder/'progress.json')
                if game['status']=='error':
                    print(f'ERROR {pair["pair_id"]} {color.value}: {game["error"]}', flush=True)
            require(hashes[0] == hashes[1], 'Seat-swap pair has different boards')
            futility=check_futility(games,len(pairs),spec['futility'])
            if spec.get('schedule')!='quick-v1':
                (folder/'progress.json').write_text(json.dumps({'completed_pairs':index+1,'completed_games':len(games),'scheduled_pairs':len(pairs),'configuration_sha256':digest(config)})+'\n')
            print(f'{index+1}/{len(pairs)} pairs; wins={sum(g["status"]=="win" for g in games)} errors={sum(g["status"]=="error" for g in games)}', flush=True)
            if spec.get('schedule')=='quick-v1' and any(g['status']=='error' for g in games):break
    summary = summarize(games, timings, spec['budget_ms'], config['seed'], spec['bootstrap_repeats'])
    summary['futility']=futility
    if futility['stopped']:
        summary['descriptive_pair_bootstrap_ci95']=summary['match_score_ci95']
        summary['match_score_ci95']=[None,None]
        summary['interval_status']='Fixed-sample bootstrap suppressed after optional stopping; use the corrected futility bound'
        summary['lower_bound_above_50_percent']=False
        print('FUTILITY STOP: '+futility['reason'],flush=True)
    record = {'schema_version': 1, 'run_id': run_id,
              'status': ('stopped-early' if futility['stopped'] else 'completed') if summary['failure_gate_passed'] else 'failed',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'configuration': {'content': config, 'sha256': digest(config)},
              'seed': config['seed'], 'rng_seeds': stream_seeds(config['seed']),
              'source': source, 'datasets': datasets,
              'software': {'python': sys.version, 'executable': sys.executable,
                           'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))},
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                           'available_devices': ['cpu']+(['mps'] if config.get('runtime',{}).get('evaluation_device')=='mps' else []),
                           'evaluation_device':config.get('runtime',{}).get('evaluation_device') or 'cpu'},
              'checkpoint': config['candidate'].get('checkpoint'),
              'metrics': {'benchmark': summary, 'providers': providers,
                          'evaluation_pairs': pairs[:len(games)//2],
                          'artifacts': {name: file_digest(folder/name) for name in ('games.jsonl','timings.jsonl')},
                          'continuation':{'run_record':str(continuation),'reused_pairs':reused_pairs} if continuation else None},
              'elapsed_seconds': elapsed_before_resume+time.perf_counter()-started+(prior['elapsed_seconds'] if continuation else 0)}
    validate_record(record)
    (folder/'run.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(folder/'run.json', flush=True)
    require(summary['failure_gate_passed'], 'Benchmark recorded failures; see run record. No fallback was used.')
    return folder/'run.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--candidate')
    p.add_argument('--opponent')
    p.add_argument('--tier', choices=TIERS)
    p.add_argument('--budget-ms', type=int, choices=(100, 500, 2000))
    p.add_argument('--seed', type=int)
    p.add_argument('--output', type=Path, default=ROOT/'runs/catan-ai')
    p.add_argument('--continue-from',type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for role in ('candidate','opponent'):
        if getattr(args,role):
            config[role] = {'key': getattr(args,role).upper(), 'checkpoint': None}
    if args.tier:
        config['evaluation']['tier'] = args.tier
    if args.budget_ms:
        config['evaluation']['budget_ms'] = args.budget_ms
    if args.seed is not None:
        config['seed'] = args.seed
    with exclusive_benchmark():
        run(config,args.output,args.continue_from)


if __name__ == '__main__':
    main()
