"""Resumable phase-nine search self-play, training, and tuning evaluation.

Run with the project Python: python -m agent_apis.neural.selfplay
The installed provider selection is never modified.
"""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from catanatron.models.player import Color
from catanatron.serialization import action_from_json, action_record_to_json, web_view

from agent_apis import benchmark
from agent_apis.benchmark import named_seed
from agent_apis.experiment import ROOT, EXPERIMENTS, digest, file_digest, source_snapshot
from agent_apis.observation import encode_observation, from_web_view, require
from agent_apis.stochastic import action_id
from agent_apis.neural.data import Cache
from agent_apis.neural.pipeline import write
from agent_apis.neural.provider import NeuralCheckpoint
from agent_apis.neural.subtree_reuse import ReusableSearch, SearchSession
from agent_apis.neural.train import evaluate, lr_at, save_checkpoint
from agent_apis.jsettlers.player import BRIDGE

PROTOCOL = EXPERIMENTS / 'phase9-selfplay.json'
OUTPUT = ROOT / 'runs/catan-ai/phase9-selfplay'
STATE = ROOT / 'runs/catan-ai/selfplay-progress.json'
CORE = (
    'agent_apis/neural/selfplay.py', 'agent_apis/neural/subtree_reuse.py',
    'agent_apis/neural/parallel_search.py', 'agent_apis/neural/inference_cache.py',
    'agent_apis/neural/provider.py', 'agent_apis/neural/models.py',
    'agent_apis/neural/topology.py', 'agent_apis/neural/train.py',
    'agent_apis/neural/data.py', 'agent_apis/stochastic.py',
    'agent_apis/stochastic_search.py', 'agent_apis/development_belief.py',
    'agent_apis/observation.py', 'agent_apis/benchmark.py',
    'agent_apis/jsettlers/player.py', 'agent_apis/jsettlers/Bridge.java',
    'experiments/catan-ai/phase9-selfplay.json',
    'experiments/catan-ai/phase5-frozen-opponent.json',
    'experiments/catan-ai/evaluation-seeds-v1.json',
    'experiments/catan-ai/phase5-augmentation-audit.json',
)


def reference(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': file_digest(path)}


def load_checkpoint(ref):
    require(file_digest(ROOT / ref['path']) == ref['sha256'], 'Checkpoint changed')
    return NeuralCheckpoint(ROOT / ref['path'], device='cpu')


def visit_target(rows, temperature):
    """Use untempered visits for learning, tempered visits only for acting."""
    require(temperature > 0 and rows, 'Missing search visits')
    ids = np.asarray([r['action_id'] for r in rows], np.int64)
    visits = np.asarray([r['visits'] for r in rows], np.float64)
    require(len(set(ids)) == len(ids) and ((ids >= 0) & (ids < 332)).all(), 'Invalid root action IDs')
    require(np.isfinite(visits).all() and (visits >= 0).all() and visits.sum() > 0, 'Invalid root visits')
    target = np.zeros(332, np.float32)
    target[ids] = visits / visits.sum()
    weights = np.zeros_like(visits)
    positive = visits > 0
    logs = np.log(visits[positive]) / temperature
    weights[positive] = np.exp(logs - logs.max())
    weights /= weights.sum()
    return target, ids, weights


class ExploratorySearch(ReusableSearch):
    """Noise affects current root decision priors only, never chance weights."""
    def __init__(self, view, seed=11, *args, **kwargs):
        super().__init__(view, seed, *args, **kwargs)
        self.noise_rng = np.random.default_rng(named_seed(seed, 'root-noise'))

    def run(self, simulations=512, context=None, noise_fraction=.25, concentration=10.):
        require(0 <= noise_fraction <= 1 and concentration > 0, 'Invalid root noise')
        root = self.retained_root
        if root is None:
            root, _ = self._node(self.view)
        # Refresh base priors even after re-rooting: never compound old noise.
        priors, _ = self._evaluate(self.view)
        ids = sorted(root.edges)
        noise = self.noise_rng.dirichlet(np.full(len(ids), concentration / len(ids)))
        for i, n in zip(ids, noise):
            root.edges[i].prior = (1 - noise_fraction) * priors[i] + noise_fraction * float(n)
        self.retained_root = root
        result = super().run(simulations, context)
        result['metrics']['root_noise_fraction'] = noise_fraction
        return result


class SearchSeat:
    def __init__(self, model, opponent, seed, protocol, session):
        self.model, self.opponent = model, opponent
        self.rng = random.Random(named_seed(seed, 'search'))
        self.action_rng = np.random.default_rng(named_seed(seed, 'action-sampling'))
        self.protocol, self.session, self.tree = protocol, session, None

    def choose(self, safe):
        seed = self.rng.getrandbits(64)
        if self.tree is None:
            self.tree = ExploratorySearch(safe, seed, self.model, self.opponent,
                c_puct=self.protocol['c_puct'], max_actions=self.protocol['max_search_actions'],
                session_id=self.session)
        else:
            self.tree = self.tree.advance(safe, seed, self.model, self.opponent, self.session)
        result = self.tree.run(self.protocol['simulations'],
            noise_fraction=self.protocol['root_noise_fraction'],
            concentration=self.protocol['dirichlet_total_concentration'])
        require(result['metrics']['simulations'] == self.protocol['simulations'], 'Short self-play search')
        setup = safe['is_initial_build_phase']
        temperature = self.protocol['early_temperature'] if setup or safe['num_turns'] < self.protocol['early_turns'] else self.protocol['late_temperature']
        target, ids, weights = visit_target(result['root_statistics'], temperature)
        selected = int(self.action_rng.choice(ids, p=weights))
        # Reuse must follow the exploration action actually played.
        self.tree.selected_action = selected
        action = next(action_from_json(a) for a in safe['current_playable_actions']
                      if action_id(action_from_json(a)) == selected)
        return action, target, result['metrics']


def completed_rows(rows, winner):
    if winner is None:
        return None
    require(rows, 'Completed game has no searched learner moves')
    arrays = {
        'features': np.asarray([r['features'] for r in rows], np.float32),
        'legal': np.asarray([r['legal'] for r in rows], bool),
        'policy': np.asarray([r['policy'] for r in rows], np.float32),
        'value': np.asarray([float(r['actor'] == winner.value) for r in rows], np.float32),
    }
    require(np.isfinite(arrays['features']).all() and np.isfinite(arrays['policy']).all(), 'Nonfinite training data')
    require((arrays['policy'] >= 0).all() and np.allclose(arrays['policy'].sum(1), 1), 'Invalid policy target')
    require((arrays['policy'][~arrays['legal']] == 0).all(), 'Search target includes illegal actions')
    return arrays


def soft_losses(logits, values, legal, targets, outcomes):
    logp = F.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)
    eligible = legal.sum(1) > 1
    policy = -(targets * logp).sum(1)
    policy = policy[eligible].mean() if eligible.any() else logits.sum() * 0
    return policy, F.binary_cross_entropy_with_logits(values, outcomes)


class ShardSampler:
    """Uniform decisions across completed generations, with bounded RAM."""
    def __init__(self, records):
        self.records = [r for r in records if r.get('dataset')]
        require(self.records, 'No completed self-play data')
        self.ends = np.cumsum([r['dataset']['rows'] for r in self.records])
        self.cache = OrderedDict()
        for r in self.records:
            require(file_digest(ROOT / r['dataset']['path']) == r['dataset']['sha256'], 'Self-play data changed')

    def batch(self, rng, count):
        indices = rng.integers(int(self.ends[-1]), size=count)
        shards = np.searchsorted(self.ends, indices, side='right')
        starts = np.r_[0, self.ends[:-1]]
        result = None
        for shard in np.unique(shards):
            if shard not in self.cache:
                with np.load(ROOT / self.records[shard]['dataset']['path']) as loaded:
                    self.cache[shard] = {k: loaded[k].copy() for k in loaded.files}
            self.cache.move_to_end(shard)
            data = self.cache[shard]
            if result is None:
                result = {k: np.empty((count, *a.shape[1:]), a.dtype) for k, a in data.items()}
            chosen = np.flatnonzero(shards == shard)
            for k in result:
                result[k][chosen] = data[k][indices[chosen] - starts[shard]]
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
        return result


class PolicyMixture:
    def __init__(self, models):
        self.models = models

    def probabilities(self, view):
        if len(view['current_playable_actions']) == 1:
            return {action_id(action_from_json(view['current_playable_actions'][0])): 1.}
        predictions = [m.probabilities(view) for m in self.models]
        return {i: sum(p[i] for p in predictions) / len(predictions) for i in predictions[0]}


class EvaluationRuntime:
    """Noise-free search, including a separate simulated-opponent ablation."""
    def __init__(self, model, opponent, protocol):
        self.model, self.opponent, self.protocol = model, opponent, protocol
        self.session = SearchSession()
        self.last_search_metrics = None

    def benchmark_choose(self, safe, context, rng):
        result = self.session.run(safe, rng.getrandbits(64), self.model, self.opponent,
            1, 1000000, context, self.protocol['c_puct'], self.protocol['max_search_actions'],
            session_id=context.session_id)
        self.last_search_metrics = result['metrics']
        return result['action']


class SelfPlay:
    def __init__(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.protocol = json.loads(PROTOCOL.read_text())
        self.core = {p: file_digest(ROOT / p) for p in CORE}
        self.core.update({str(p.relative_to(ROOT)): file_digest(p)
            for p in sorted((ROOT / 'framework/catanatron/catanatron').rglob('*.py'))})
        phase6 = json.loads((EXPERIMENTS / 'PHASE6_RESULTS.json').read_text())
        self.parent = phase6['training']['XD-CNN:11:D2-W0']['checkpoint']
        self.frozen_ref = json.loads((EXPERIMENTS / 'phase5-frozen-opponent.json').read_text())['checkpoint']
        if STATE.exists():
            self.state = json.loads(STATE.read_text())
            require(self.state['source_sha256'] == self.core and self.state['protocol'] == self.protocol,
                    'Self-play implementation/protocol changed; preserve this run and investigate')
            require(self.state['parent'] == self.parent, 'Initial checkpoint reference changed')
        else:
            self.state = {'status': 'running', 'stage': 'preflight', 'started_at': self.now(),
                'source_sha256': self.core, 'protocol': self.protocol, 'parent': self.parent,
                'iterations': {}, 'installed_default_changed': False, 'final_seeds_used': False}
            write(OUTPUT / 'source.json', source_snapshot())
        self.state.update(status='running', pid=os.getpid())
        self.persist()
        require(torch.__version__ == '2.15.0', 'PyTorch version differs')
        require(self.protocol['training_device'] != 'mps' or torch.backends.mps.is_available(),
                'Pinned MPS device unavailable; launch outside the restricted process sandbox')
        self.cache = Cache(ROOT / self.protocol['cache'])
        self.frozen = load_checkpoint(self.frozen_ref)
        require(self.frozen.metadata['model_id'] == 'MLP', 'Wrong frozen search opponent')
        learner = load_checkpoint(self.parent)
        require(learner.metadata['model_id'] == 'XD-CNN', 'Only XD-CNN is configured')
        audit = json.loads((EXPERIMENTS / 'phase5-augmentation-audit.json').read_text())
        require(audit['gate_passed'], 'D6 augmentation audit failed')
        self.state['hardware'] = {'platform': platform.platform(), 'torch': torch.__version__,
            'training_device': self.protocol['training_device'], 'inference_device': 'cpu',
            'cpu_threads': torch.get_num_threads()}
        self.state['native_opponent'] = benchmark.provider_identity('JSETTLERS')
        self.persist()
        self.known_seeds = set()
        for folder in ('full-250', 'downsampled'):
            manifest = json.loads((ROOT / f'data/jsettlers-corpora/{folder}/manifest.json').read_text())
            self.known_seeds.update(g['seed'] for g in manifest['games'])
        for pairs in json.loads((EXPERIMENTS / 'evaluation-seeds-v1.json').read_text())['sets'].values():
            for pair in pairs:
                self.known_seeds.update(pair[k] for k in ('board_seed', 'simulator_seed', 'agent_seed'))
        # Previously generated learner-state data is a separate training route.
        for path in (ROOT / 'runs/catan-ai/phase9-pilot').glob('phase9-train-*.json'):
            pair = json.loads(path.read_text())['pair']
            self.known_seeds.update(pair[k] for k in ('board_seed', 'simulator_seed', 'agent_seed'))

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def persist(self):
        self.state['updated_at'] = self.now()
        write(STATE, self.state)

    def guard(self):
        require(all(file_digest(ROOT / p) == sha for p, sha in self.core.items()), 'Self-play source changed during execution')

    def game_spec(self, iteration, attempt, completed_slot=None):
        slot = attempt if completed_slot is None else completed_slot
        block = slot // 4
        order = list(self.protocol['opponent_blocks'])
        random.Random(named_seed(self.protocol['seed'], f'selfplay-{iteration}-mix-{block}')).shuffle(order)
        pair = {'pair_id': f'selfplay-{iteration}-{attempt:05d}'}
        for kind in ('board', 'simulator', 'agent'):
            pair[kind + '_seed'] = named_seed(self.protocol['seed'], f'selfplay-{iteration}-{attempt}-{kind}')
            require(pair[kind + '_seed'] not in self.known_seeds, 'Generation seed overlaps reserved data')
        learner = benchmark.COLORS[(block + slot % 4) % 2]
        return pair, order[slot % 4], learner

    def collect(self, iteration, parent, history, record):
        folder = OUTPUT / f'iteration-{iteration}'
        folder.mkdir(exist_ok=True)
        model = load_checkpoint(parent)
        records = []
        completed = 0
        for attempt in range(self.protocol['max_attempts_per_iteration']):
            if completed >= self.protocol['completed_games_per_iteration']:
                break
            self.guard()
            # Retry a cut-off mixture slot with fresh seeds, so the completed
            # cohort retains exactly 50/25/25 rather than completion bias.
            pair, kind, learner = self.game_spec(iteration, attempt, completed)
            path = folder / f'game-{attempt:05d}.json'
            if path.exists():
                game_record = json.loads(path.read_text())
                require(game_record['source_sha256'] == digest(self.core) and game_record['parent'] == parent
                    and game_record['pair'] == pair and game_record['opponent_kind'] == kind,
                    'Committed generation configuration changed')
                if game_record['dataset']:
                    require(file_digest(ROOT / game_record['dataset']['path']) == game_record['dataset']['sha256'], 'Generated shard changed')
            else:
                historical = history[named_seed(pair['agent_seed'], 'historical-selection') % len(history)]
                enemy_model = load_checkpoint(historical) if kind == 'historical' else model
                game_record = self.collect_game(pair, kind, learner, model, enemy_model, iteration, attempt)
                game_record.update(parent=parent, historical_checkpoint=historical if kind == 'historical' else None,
                                   source_sha256=digest(self.core))
                arrays = game_record.pop('arrays')
                if arrays is not None:
                    destination = path.with_suffix('.npz')
                    temporary = path.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, **arrays)
                    temporary.replace(destination)
                    game_record['dataset'] = {**reference(destination), 'rows': len(arrays['value'])}
                else:
                    game_record['dataset'] = None
                write(path, game_record)
            records.append(game_record)
            completed += int(game_record['status'] == 'completed')
            record.update(completed_games=completed, attempted_games=len(records),
                cutoffs=len(records) - completed, generation_seconds=sum(r['elapsed_seconds'] for r in records),
                searched_training_rows=sum(r['dataset']['rows'] for r in records if r['dataset']))
            self.persist()
            print(f'GENERATION {iteration}: {completed}/{self.protocol["completed_games_per_iteration"]} completed; attempts={len(records)}', flush=True)
        require(completed == self.protocol['completed_games_per_iteration'], 'Generation attempt limit reached; investigate cutoffs')
        write(folder / 'collection.json', records)
        return records

    def collect_game(self, pair, kind, learner, model, enemy_model, iteration, attempt):
        BRIDGE.close()
        game = benchmark.make_game(pair, friendly_robber=self.protocol['friendly_robber'])
        other = next(c for c in benchmark.COLORS if c != learner)
        seats = {learner: SearchSeat(model, self.frozen, named_seed(pair['agent_seed'], learner.value), self.protocol, game.id)}
        if kind != 'JSETTLERS':
            seats[other] = SearchSeat(enemy_model, self.frozen, named_seed(pair['agent_seed'], other.value), self.protocol, game.id)
        native = benchmark.Agent('JSETTLERS', other, pair['agent_seed']) if kind == 'JSETTLERS' else None
        rows = []
        started = time.perf_counter()
        simulations = reused = 0
        timings = []
        telemetry = OUTPUT / f'iteration-{iteration}' / f'game-{attempt:05d}-decisions.jsonl'
        self.state.update(stage=f'collect:iteration-{iteration}:game-{attempt:05d}', active_game={
            'iteration': iteration, 'attempt': attempt, 'opponent': kind, 'actions': 0, 'turns': 0})
        self.persist()
        try:
            with telemetry.open('w') as stream:
                while game.winning_color() is None and game.state.num_turns < self.protocol['max_turns'] and len(game.state.action_records) < self.protocol['max_game_actions']:
                    actor = game.state.current_color()
                    actions = list(game.playable_actions)
                    start = time.perf_counter()
                    metrics = None
                    if len(actions) == 1:
                        action = actions[0]
                    elif actor in seats:
                        safe = from_web_view(web_view(game, actor), actor.value)
                        action, target, metrics = seats[actor].choose(safe)
                        # Historical opponents supply trajectories, not new current-agent labels.
                        if actor == learner or kind == 'current':
                            legal = np.zeros(332, bool)
                            legal[[action_id(a) for a in actions]] = True
                            rows.append({'features': encode_observation(safe), 'legal': legal,
                                         'policy': target, 'actor': actor.value})
                        simulations += metrics['simulations']
                        reused += metrics['reused_nodes']
                    else:
                        context = benchmark.DecisionContext(start, 500, 'policy', None)
                        action = native.select(game, actions, context)
                        native.cold = False
                    require(action in actions, 'Illegal collection action')
                    elapsed = time.perf_counter() - start
                    if metrics is not None:
                        timings.append(elapsed)
                    event = {'step': len(game.state.action_records), 'actor': actor.value,
                        'action_id': action_id(action), 'forced': len(actions) == 1,
                        'elapsed_seconds': elapsed, 'search': metrics}
                    stream.write(json.dumps(event, allow_nan=False) + '\n')
                    stream.flush()
                    game.execute(action)
                    self.state['active_game'].update(actions=len(game.state.action_records), turns=game.state.num_turns,
                        elapsed_seconds=time.perf_counter()-started, simulations=simulations,
                        searched_rows=len(rows), reused_nodes=reused)
                    if len(game.state.action_records) % 5 == 0:
                        self.persist()
            winner = game.winning_color()
            return {'pair': pair, 'learner_color': learner.value, 'opponent_kind': kind,
                'status': 'completed' if winner else 'cutoff', 'winner': winner.value if winner else None,
                'steps': len(game.state.action_records), 'turns': game.state.num_turns,
                'elapsed_seconds': time.perf_counter()-started, 'simulations': simulations,
                'reused_nodes': reused, 'search_latency_p50_seconds': benchmark.quantile(timings, .5),
                'search_latency_p95_seconds': benchmark.quantile(timings, .95),
                'trajectory_sha256': digest([action_record_to_json(r) for r in game.state.action_records]),
                'telemetry': reference(telemetry), 'arrays': completed_rows(rows, winner)}
        finally:
            BRIDGE.close()

    def train(self, iteration, parent, records):
        folder = OUTPUT / f'iteration-{iteration}'
        destination = folder / 'training.json'
        if destination.exists():
            result = json.loads(destination.read_text())
            load_checkpoint(result['checkpoint'])
            return result
        self.state['stage'] = f'train:iteration-{iteration}'
        self.persist()
        sampler = ShardSampler(records)
        data_sha = digest([r['dataset'] for r in records if r['dataset']])
        checkpoint = load_checkpoint(parent)
        model = checkpoint.model
        device = self.protocol['training_device']
        model.to(device)
        config = deepcopy(checkpoint.metadata['configuration'])
        config.update(phase=9, stage='search-selfplay-training', selfplay_iteration=iteration)
        config['data']['selfplay'] = {'manifest_sha256': data_sha, 'expert_fraction': self.protocol['expert_fractions'][iteration-1]}
        settings = config['training']
        settings.update(max_updates=self.protocol['updates_per_iteration'], warmup_updates=self.protocol['warmup_updates'])
        meta = deepcopy(checkpoint.metadata)
        meta.update(configuration=config, configuration_sha256=digest(config), parent_checkpoint=parent,
                    source_sha256=digest(self.core), selfplay_iteration=iteration)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        rng = np.random.default_rng(named_seed(self.protocol['seed'], f'selfplay-train-{iteration}'))
        augment = torch.Generator().manual_seed(named_seed(self.protocol['seed'], f'selfplay-augment-{iteration}'))
        torch.manual_seed(named_seed(self.protocol['seed'], f'selfplay-torch-{iteration}'))
        train_indices = self.cache.indices('train')
        resume = folder / 'resume.pt'
        step = 0
        metrics = []
        previous_seconds = 0.
        if resume.exists():
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            require(saved['data_sha256'] == data_sha and saved['source_sha256'] == self.core and saved['parent'] == parent, 'Training resume changed')
            model.load_state_dict(saved['state_dict'])
            optimizer.load_state_dict(saved['optimizer'])
            rng.bit_generator.state = saved['sampling_rng']
            augment.set_state(saved['augmentation_rng'])
            torch.random.set_rng_state(saved['torch_rng'])
            if device == 'mps':
                torch.mps.set_rng_state(saved['mps_rng'])
            step, metrics = saved['step'], saved['validation']
            previous_seconds = saved['elapsed_seconds']
        else:
            metrics.append({'step': 0, **evaluate(model, self.cache, 'validation', device)})
        started = time.perf_counter()

        def save():
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                'sampling_rng': rng.bit_generator.state, 'augmentation_rng': augment.get_state(),
                'torch_rng': torch.random.get_rng_state(), 'mps_rng': torch.mps.get_rng_state() if device == 'mps' else None,
                'parent': parent, 'source_sha256': self.core, 'data_sha256': data_sha, 'validation': metrics,
                'elapsed_seconds': previous_seconds + time.perf_counter()-started}, folder / 'resume.tmp')
            (folder / 'resume.tmp').replace(resume)

        try:
            while step < settings['max_updates']:
                if step % self.protocol['save_interval'] == 0:
                    self.guard()
                old_count = round(self.protocol['batch_size'] * self.protocol['expert_fractions'][iteration-1])
                old = self.cache.batch(rng.choice(train_indices, size=old_count))
                new = sampler.batch(rng, self.protocol['batch_size'] - old_count)
                old_policy = np.eye(332, dtype=np.float32)[old['action']]
                batch = {k: np.concatenate((old[k], new[k])) for k in ('features', 'legal', 'value')}
                target = torch.from_numpy(np.concatenate((old_policy, new['policy']))).to(device)
                raw = torch.from_numpy(batch['features']).to(device)
                legal = torch.from_numpy(batch['legal']).to(device)
                outcome = torch.from_numpy(batch['value']).to(device)
                sym = torch.randint(12, (len(raw),), generator=augment).to(device)
                inverse = model.inputs.action_inverse[sym]
                target, legal = target.gather(1, inverse), legal.gather(1, inverse)
                model.train()
                for group in optimizer.param_groups:
                    group['lr'] = lr_at(step+1, settings)
                optimizer.zero_grad(set_to_none=True)
                logits, value = model(raw, sym)
                pl, vl = soft_losses(logits, value, legal, target, outcome)
                (pl + vl).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % self.protocol['validation_interval'] == 0 or step == settings['max_updates']:
                    metrics.append({'step': step, **evaluate(model, self.cache, 'validation', device)})
                if step % self.protocol['save_interval'] == 0 or step == settings['max_updates']:
                    save()
                    self.state['training_updates'] = step
                    self.persist()
                    print(f'TRAIN {iteration}: {step}/{settings["max_updates"]}', flush=True)
        except BaseException:
            save()
            raise
        meta['step'] = step
        save_checkpoint(folder / 'final.pt', model, meta)
        model.to('cpu')
        result = {'checkpoint': reference(folder / 'final.pt'), 'updates': step, 'parent': parent,
            'validation': metrics, 'expert_fraction': self.protocol['expert_fractions'][iteration-1],
            'elapsed_seconds': previous_seconds + time.perf_counter()-started,
            'selection': 'predeclared final update; supervised validation is diagnostic, paired strength gates deployment'}
        write(destination, result)
        return result

    def validation_match(self, iteration, name, candidate, opponent, simulated_opponent):
        folder = OUTPUT / f'iteration-{iteration}' / 'evaluation' / name
        folder.mkdir(parents=True, exist_ok=True)
        self.state['stage'] = f'evaluate:iteration-{iteration}:{name}'
        self.persist()
        models = {ref['sha256']: load_checkpoint(ref) for ref in [candidate] + ([opponent] if opponent else [])}
        spec = {'rules': {'friendly_robber': self.protocol['friendly_robber']},
            'candidate': {'key': 'NEURAL_SEARCH', 'checkpoint': candidate},
            'opponent': {'key': 'NEURAL_SEARCH', 'checkpoint': opponent} if opponent else {'key': 'JSETTLERS'},
            'evaluation': {'max_turns': self.protocol['max_turns'], 'max_actions': self.protocol['max_game_actions'],
                'budget_ms': self.protocol['evaluation_budget_ms'], 'mode': 'time', 'simulations': None}}

        def factory(key, color, seed, checkpoint=None):
            if key == 'JSETTLERS':
                return benchmark.Agent(key, color, seed)
            agent = benchmark.Agent(key, color, seed)
            # Apply the same simulated opponent to candidate and frozen prior.
            agent.implementation = EvaluationRuntime(models[checkpoint['sha256']], simulated_opponent, self.protocol)
            return agent

        pairs = json.loads((EXPERIMENTS / 'evaluation-seeds-v1.json').read_text())['sets']['tuning'][:self.protocol['evaluation_pairs']]
        results, timings = [], []
        identity = digest({'spec': spec, 'source': self.core, 'name': name})
        for pair in pairs:
            for color in benchmark.COLORS:
                self.guard()
                path = folder / f'{pair["pair_id"]}-{color.value}.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    require(saved['identity'] == identity, 'Evaluation resume configuration changed')
                    result, events = saved['game'], saved['timings']
                else:
                    result, events = benchmark.play_game(pair, color, spec, agent_factory=factory)
                    require(result['status'] != 'error', f'Evaluation failed: {result["error"]}')
                    write(path, {'identity': identity, 'game': result, 'timings': events})
                results.append(result)
                timings.extend(events)
                self.state['evaluation_games'] = len(results)
                self.persist()
        summary = benchmark.summarize(results, timings, self.protocol['evaluation_budget_ms'])
        write(folder / 'summary.json', summary)
        return results, summary

    def validate(self, iteration, candidate, parent, history, record):
        old, old_summary = self.validation_match(iteration, 'parent-vs-JSETTLERS', parent, None, self.frozen)
        new, summary = self.validation_match(iteration, 'candidate-vs-JSETTLERS', candidate, None, self.frozen)
        _, prior_summary = self.validation_match(iteration, 'candidate-vs-prior', candidate, parent, self.frozen)
        mixture = PolicyMixture([load_checkpoint(ref) for ref in history])
        ablation, ablation_summary = self.validation_match(iteration, 'historical-search-opponent-vs-JSETTLERS', candidate, None, mixture)
        change = benchmark.paired_difference(new, old)
        training = record['training']
        calibration_ok = training['validation'][-1]['brier'] <= training['validation'][0]['brier']
        gates = all(summary[k] and prior_summary[k] for k in ('failure_gate_passed', 'cutoff_gate_passed', 'timing_gate_passed'))
        eligible = gates and calibration_ok and change['ci95'][0] > 0 and prior_summary['match_score_ci95'][0] > .5
        gain = change['difference']
        result = {'parent': old_summary, 'candidate': summary, 'candidate_vs_prior': prior_summary,
            'candidate_minus_parent': change, 'historical_search_opponent': ablation_summary,
            'historical_minus_frozen_search_opponent': benchmark.paired_difference(ablation, new),
            'strength_gain_per_generation_hour': gain / max(record['generation_seconds']/3600, 1e-9),
            'strength_gain_per_training_hour': gain / max(training['elapsed_seconds']/3600, 1e-9),
            'strength_gain_per_generated_game': gain / record['completed_games'],
            'common_validation_brier_nonregression': calibration_ok, 'promotion_eligible': eligible,
            'installed_default_changed': False, 'final_seeds_used': False,
            'scope': 'Single-seed tuning results; deployment requires review of all diagnostics'}
        write(OUTPUT / f'iteration-{iteration}' / 'results.json', result)
        return result

    def execute(self):
        parent = self.parent
        history = [self.parent]
        all_records = []
        for iteration in range(1, self.protocol['iterations']+1):
            record = self.state['iterations'].setdefault(str(iteration), {})
            records = self.collect(iteration, parent, history, record)
            all_records.extend(records)
            training = self.train(iteration, parent, all_records)
            record['training'] = training
            self.persist()
            record['evaluation'] = self.validate(iteration, training['checkpoint'], parent, history, record)
            record['status'] = 'completed'
            parent = training['checkpoint']
            history.append(parent)
            self.persist()
        self.state.update(status='completed', stage='completed-three-iterations', pid=None)
        self.persist()
        write(EXPERIMENTS / 'PHASE9_SELFPLAY_RESULTS.json', self.state)
        print('Three self-play iterations complete. No automatic deployment or scale-up.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    with benchmark.exclusive_benchmark():
        work = SelfPlay()
        try:
            if args.preflight:
                print(json.dumps({'parent': work.parent, 'protocol': work.protocol, 'hardware': work.state['hardware']}), flush=True)
                work.state.update(status='ready', stage='preflight-passed', pid=None)
                work.persist()
            else:
                work.execute()
        except BaseException as error:
            work.state.update(status='needs-investigation', error=f'{type(error).__name__}: {error}', pid=None)
            work.persist()
            raise


if __name__ == '__main__':
    main()
