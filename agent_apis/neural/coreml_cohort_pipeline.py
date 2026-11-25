"""Multi-cohort CoreML GPU self-play (2,000 games/cohort) + W4 training for CNN & Transformer.

Per cohort (2,000 completed, replay-verified games, 128 simulations, w128-b64-s4-l1):
  - 1,000 CNN vs CNN         (native Rust CoreML --pool; both seats supply soft-visit targets)
  - 1,000 JSettlers vs CNN   (native Rust CoreML --bridge-pool; learner CNN visits only)
  (Transformer is kept out of self-play for speed, and trained on the same 2,000-game cohort.)

After saving and replay-verifying all 2,000 games:
  - Stratified whole-game train/validation split (default 95/5 via --validation-fraction 0.05).
  - Loads all eligible training positions into RAM once per cohort and reslices
    --positions-per-game (default 64, initial placements pinned) every --resample-every updates.
  - Validation games are derived once per cohort and never resampled.
  - Continues W4 training for both XD-CNN and TYPED-EXPHORMER for 2,000 updates on MPS
    with value loss downweighted to 0.25x (configurable via --value-loss-weight),
    saving the lowest validation-loss checkpoint (`best.pt`) and `final.pt`.
  - Advances sequentially for `--n-cohorts N` cohorts without restarting, feeding each
    cohort's lowest-validation `best.pt` into the next cohort. Never auto-deploys.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import shutil
import subprocess
import threading
import time
import zlib

import numpy as np
import torch
from catanatron.models.player import Color
from catanatron.serialization import action_record_to_json, map_to_json

from agent_apis import benchmark
from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.jsettlers import player as js_player
from agent_apis.neural import (
    compact_iterations as compact,
    fresh_cohorts as fresh,
    friendly_w4 as w4,
    selfplay,
    sparse_catchup as sparse_module,
)
from agent_apis.neural.device_utils import (
    configure_torch_device,
    default_gpu_backend,
    empty_device_cache,
    resolve_torch_device,
)
from agent_apis.neural.matchup_benchmark import NativeBridgePool
from agent_apis.neural.native_assets import (
    NATIVE,
    ensure_binary,
    export_checkpoint,
    resolve_frozen_opponent,
    runtime_environment,
)
from agent_apis.neural.native_codec import Codec, job_meta
from agent_apis.neural.native_gpu_server import (
    export_checkpoint_coreml_static,
    export_checkpoint_dynamic,
)
from agent_apis.neural.native_selfplay import (
    JournalRecorder,
    PROTOCOL_VERSION,
    execution_identity,
    validate_protocol,
)
from agent_apis.neural.pipeline import write
from agent_apis.neural.selfplay_only_comparison import corrected_evaluate
from agent_apis.neural.sparse_catchup import SparseCatchup
from agent_apis.observation import require
from agent_apis.stochastic import action_id

_LATEST_CNN = ROOT / 'runs/catan-ai/coreml-cohort-1000/cnn/iteration-1/best.pt'
_FALLBACK_CNN = ROOT / 'runs/catan-ai/selfplay-only-comparison/cnn/iteration-1/best.pt'
DEFAULT_CNN = _LATEST_CNN if _LATEST_CNN.exists() else _FALLBACK_CNN

_LATEST_TR = ROOT / 'runs/catan-ai/coreml-cohort-1000/transformer/iteration-1/best.pt'
_FALLBACK_TR = ROOT / 'runs/catan-ai/selfplay-only-comparison/sparse/iteration-1/best.pt'
DEFAULT_TRANSFORMER = _LATEST_TR if _LATEST_TR.exists() else _FALLBACK_TR

DEFAULT_OUTPUT = ROOT / 'runs/catan-ai/coreml-cohorts-2000'


def make_pair(seed, block_name, attempt):
    return {
        'pair_id': f'cohort-{seed}-{block_name}-{attempt:05d}',
        'board_seed': benchmark.named_seed(seed, f'{block_name}-board-{attempt}'),
        'simulator_seed': benchmark.named_seed(seed, f'{block_name}-sim-{attempt}'),
        'agent_seed': benchmark.named_seed(seed, f'{block_name}-agent-{attempt}'),
    }


class _JvmBridge:
    """Single JSettlers Java Bridge subprocess with stderr silenced (suppresses RESOURCE < 0 noise)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.reader = None
        self.replies = None

    def close(self):
        with self.lock:
            self._close_locked()

    def _close_locked(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            if self.reader is not None:
                self.reader.join(timeout=2)
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.stdout.close()
            except Exception:
                pass
            self.process = None

    def request(self, **payload):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.process = subprocess.Popen(
                    [
                        shutil.which('java'),
                        '-Xmx768m',
                        '-Djava.awt.headless=true',
                        '-cp',
                        str(js_player.LIB / '*'),
                        'Bridge',
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                replies = queue.Queue()

                def read(stream):
                    for line in stream:
                        replies.put(line)
                    replies.put(None)

                self.reader = threading.Thread(target=read, args=(self.process.stdout,), daemon=True)
                self.reader.start()
                self.replies = replies
            try:
                self.process.stdin.write(json.dumps(payload) + '\n')
                self.process.stdin.flush()
                line = self.replies.get(timeout=45)
                if line is None:
                    raise RuntimeError('JSettlers adapter exited')
                reply = json.loads(line)
                if 'error' in reply:
                    raise RuntimeError(reply['error'])
                return reply
            except queue.Empty as error:
                self._close_locked()
                raise TimeoutError('JSettlers decision exceeded 45 seconds') from error
            except Exception:
                self._close_locked()
                raise


class PooledJavaBridge:
    """Pool of independent JSettlers JVMs, consistently sharded by game session."""

    def __init__(self, n_jvms=12):
        self.bridges = [_JvmBridge() for _ in range(max(1, n_jvms))]

    def request(self, **payload):
        session = payload.get('session')
        if session:
            idx = zlib.crc32(session.encode('utf-8')) % len(self.bridges)
        else:
            idx = 0
        return self.bridges[idx].request(**payload)

    def close(self):
        for b in self.bridges:
            b.close()


def load_existing_completed(block_dir, target_games):
    """Load and replay-verify any previously saved completed games for resumability."""
    completed = []
    if not block_dir.exists():
        return completed, 0
    existing_files = sorted(block_dir.glob('game-*.json.gz'))
    max_attempt = 0
    for path in existing_files:
        stem = path.name.replace('.json.gz', '')
        try:
            idx = int(stem.split('-')[-1])
            max_attempt = max(max_attempt, idx + 1)
        except ValueError:
            pass
        try:
            with gzip.open(path, 'rt') as stream:
                rec = json.load(stream)
            if rec.get('status') == 'completed' and rec.get('winner') is not None:
                compact.replay_journal(rec)
                completed.append(selfplay.reference(path))
                if len(completed) >= target_games:
                    break
        except (OSError, EOFError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            print(f'  [resume] Skipping partial/corrupted file {path.name}: {exc}', flush=True)
    return completed, max_attempt


def collect_neural_block(
    block_name,
    block_dir,
    target_games,
    seed,
    binary,
    model_a_ref,
    model_b_ref,
    onnx_a,
    onnx_b,
    frozen_ref,
    frozen_onnx,
    protocol,
    workers,
    coreml_batch,
    gpu_streams,
    mcts_lanes,
    virtual_loss,
    wait_us,
    gpu_backend='coreml',
    mcts_opponent='self-policy',
    vp_margin_alpha=0.03,
    setup_simulations=None,
    setup_exploration_games=0,
    setup_temperature=1.5,
    print_games=False,
    fixed_batch=None,
    dynamic_onnx_a=None,
    dynamic_onnx_b=None,
    tail_over_dispatch=32,
):
    """Collect `target_games` completed games using native Rust GPU --pool."""
    require(tail_over_dispatch >= 0, 'tail_over_dispatch must be >= 0')
    block_dir.mkdir(parents=True, exist_ok=True)
    completed, next_attempt = load_existing_completed(block_dir, target_games)
    if len(completed) >= target_games:
        print(f'[{block_name}] Reusing {len(completed)}/{target_games} verified completed games.', flush=True)
        return completed[:target_games], {'reused': True}

    eff_setup_sims = setup_simulations if setup_simulations is not None else protocol['simulations']
    remaining_needed = target_games - len(completed)
    n_workers = min(workers, remaining_needed + tail_over_dispatch)
    if fixed_batch is None:
        fixed_batch = coreml_batch if gpu_backend == 'coreml' else 0
    print(
        f'\n[{block_name}] Collecting {remaining_needed} completed games '
        f'(backend={gpu_backend}, w={n_workers}, over_dispatch={tail_over_dispatch}, b={coreml_batch}, fixed_b={fixed_batch}, s={gpu_streams}, l={mcts_lanes}, sims={protocol["simulations"]}, '
        f'setup_sims={eff_setup_sims}, setup_explore={setup_exploration_games}@T={setup_temperature})...',
        flush=True,
    )
    log = (NATIVE / 'target/worker-stderr.log').open('a')
    cmd = [
        str(binary), '--pool', '--workers', str(n_workers), '--gpu-backend', gpu_backend,
        '--max-batch', str(coreml_batch), '--fixed-batch', str(fixed_batch),
        '--wait-us', str(wait_us), '--gpu-streams', str(gpu_streams),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        text=True, bufsize=1, env=runtime_environment(),
    )
    identity = {**execution_identity(binary), 'backend': f'rust-onnx-{gpu_backend}-batch'}
    recorders = {}
    game_paths = {}
    pool_metrics = {}
    in_flight = 0
    started = time.perf_counter()
    stop_flag = block_dir / '.stop'
    stop_flag.unlink(missing_ok=True)

    def dispatch_one():
        nonlocal next_attempt, in_flight
        attempt = next_attempt
        next_attempt += 1
        gid = f'{block_name}-{attempt:05d}'
        pair = make_pair(seed, block_name, attempt)
        seat_a = 'BLUE' if attempt % 2 == 0 else 'RED'
        game = benchmark.make_game(pair, friendly_robber=True)
        meta = job_meta(game)
        state = list(game.state.random.getstate()[1])
        colors = [c.value for c in game.state.colors]
        paths = [str(onnx_a if c == seat_a else onnx_b) for c in colors]
        out_path = block_dir / f'game-{attempt:05d}.json.gz'
        record = {
            'format': 'catan-compact-native-v1',
            'pair': pair,
            'candidate_color': seat_a,
            'opponent_kind': 'current',
            'matchup_block': block_name,
            'mode': 'search',
            'candidate_checkpoint': model_a_ref,
            'opponent_checkpoint': model_b_ref,
            'frozen_search_opponent': frozen_ref,
            'source_sha256': digest({'block': block_name, 'protocol': protocol}),
            'collection_execution': None,
            'native': {
                'execution': identity,
                'candidate_export_sha256': file_digest(onnx_a),
                'enemy_export_sha256': file_digest(onnx_b),
                'frozen_export_sha256': file_digest(frozen_onnx),
                'search_protocol': protocol,
                'budget_seconds': None,
                'limits': {'max_turns': 1000, 'max_actions': 10000},
                'lanes': mcts_lanes,
                'virtual_loss': virtual_loss,
                'mcts_opponent': mcts_opponent,
                'vp_margin_alpha': vp_margin_alpha,
                'setup_simulations': eff_setup_sims,
                'setup_temperature': setup_temperature if (setup_exploration_games > 0 and attempt < setup_exploration_games) else None,
            },
        }
        game_setup_temp = setup_temperature if (setup_exploration_games > 0 and attempt < setup_exploration_games) else None
        recorders[gid] = JournalRecorder(game, record, protocol['simulations'], setup_simulations=eff_setup_sims)
        game_paths[gid] = out_path
        req = {
            'version': PROTOCOL_VERSION,
            'game_id': gid,
            'task': {
                'job': meta,
                'start': meta['initial'],
                'start_rng': state,
                'logs': [],
                'id': out_path.name,
                'prefix': 0,
                'seat_seeds': [benchmark.named_seed(pair['agent_seed'], c) for c in colors],
                'rng_state': state,
            },
            'models': paths,
            'frozen': str(frozen_onnx),
            'simulations': protocol['simulations'],
            'setup_simulations': eff_setup_sims,
            'setup_temperature': game_setup_temp,
            'seconds': None,
            'deadline': time.time() + 12 * 3600,
            'stop_file': str(stop_flag.resolve()),
            'max_turns': 1000,
            'max_actions': 10000,
            'lanes': mcts_lanes,
            'virtual_loss': virtual_loss,
            'eval_mode': False,
            'mcts_opponent': mcts_opponent,
            'vp_margin_alpha': vp_margin_alpha,
        }
        if dynamic_onnx_a is not None and dynamic_onnx_b is not None:
            req['dynamic_models'] = [
                str(dynamic_onnx_a if c == seat_a else dynamic_onnx_b) for c in colors
            ]
        proc.stdin.write(json.dumps(req, separators=(',', ':'), allow_nan=False) + '\n')
        proc.stdin.flush()
        in_flight += 1

    try:
        ready = json.loads(proc.stdout.readline())
        require(ready.get('type') == 'ready', f'Unexpected pool ready: {ready}')
        for _ in range(n_workers):
            dispatch_one()

        stdin_closed = False
        for line in proc.stdout:
            msg = json.loads(line)
            mtype = msg.get('type')
            if mtype == 'error':
                raise RuntimeError(f'CoreML pool error ({msg.get("game_id")}): {msg.get("error")}')
            elif mtype == 'events':
                if len(completed) < target_games:
                    recorders[msg['game_id']].append(msg['events'])
            elif mtype == 'done':
                gid = msg['game_id']
                in_flight -= 1
                if len(completed) >= target_games:
                    # Target already reached; skip Python replay for cancelled tail games.
                    recorders.pop(gid, None)
                    game_paths.pop(gid, None)
                else:
                    rec = recorders.pop(gid).finish(msg)
                    out_path = game_paths.pop(gid)
                    if rec['status'] == 'completed':
                        compact.save_journal(out_path, rec)
                        completed.append(selfplay.reference(out_path))
                        if print_games or len(completed) % 50 == 0 or len(completed) == target_games:
                            elapsed = time.perf_counter() - started
                            print(
                                f'  [{block_name}] {len(completed)}/{target_games} completed '
                                f'(last: {rec["winner"]} in {rec["turns"]} turns | {elapsed/60.0:.1f} min)',
                                flush=True,
                            )
                if len(completed) >= target_games:
                    stop_flag.touch()
                    if in_flight == 0 and not stdin_closed:
                        proc.stdin.close()
                        stdin_closed = True
                elif len(completed) + in_flight < target_games + tail_over_dispatch:
                    dispatch_one()
            elif mtype == 'pool_metrics':
                pool_metrics = msg.get('metrics', {})
                break
        proc.wait(timeout=30)
    finally:
        stop_flag.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()

    elapsed = time.perf_counter() - started
    mean_b = (pool_metrics['rows'] / pool_metrics['batches']) if pool_metrics.get('batches') else 0.0
    print(
        f'[{block_name}] Done: {len(completed)}/{target_games} verified games in {elapsed/60.0:.1f} min '
        f'(mean_batch={mean_b:.1f}, batches={pool_metrics.get("batches", 0)})',
        flush=True,
    )
    return completed[:target_games], {'elapsed_seconds': elapsed, 'pool_metrics': pool_metrics}


def collect_jsettlers_block(
    block_name,
    block_dir,
    target_games,
    seed,
    binary,
    learner_ref,
    learner_onnx,
    frozen_ref,
    frozen_onnx,
    protocol,
    workers,
    jsettlers_jvms,
    coreml_batch,
    gpu_streams,
    mcts_lanes,
    virtual_loss,
    wait_us,
    gpu_backend='coreml',
    mcts_opponent='self-policy',
    vp_margin_alpha=0.03,
    setup_simulations=None,
    setup_exploration_games=0,
    setup_temperature=1.5,
    print_games=False,
    fixed_batch=None,
    dynamic_learner_onnx=None,
    tail_over_dispatch=32,
):
    """Collect `target_games` completed JSettlers vs Model games via native GPU --bridge-pool + pooled JVMs."""
    require(tail_over_dispatch >= 0, 'tail_over_dispatch must be >= 0')
    block_dir.mkdir(parents=True, exist_ok=True)
    completed, next_attempt = load_existing_completed(block_dir, target_games)
    if len(completed) >= target_games:
        print(f'[{block_name}] Reusing {len(completed)}/{target_games} verified completed games.', flush=True)
        return completed[:target_games], {'reused': True}

    eff_setup_sims = setup_simulations if setup_simulations is not None else protocol['simulations']
    remaining_needed = target_games - len(completed)
    n_workers = min(workers, remaining_needed + tail_over_dispatch)
    n_jvms = min(jsettlers_jvms, n_workers)
    if fixed_batch is None:
        fixed_batch = coreml_batch if gpu_backend == 'coreml' else 0
    print(
        f'\n[{block_name}] Collecting {remaining_needed} completed games vs JSettlers '
        f'(reused {len(completed)}/{target_games} | backend={gpu_backend}, w={n_workers}, jvms={n_jvms}, b={coreml_batch}, fixed_b={fixed_batch}, s={gpu_streams}, l={mcts_lanes}, '
        f'sims={protocol["simulations"]}, setup_sims={eff_setup_sims}, setup_explore={setup_exploration_games}@T={setup_temperature})...',
        flush=True,
    )
    bridge = NativeBridgePool(binary, n_workers, gpu_backend, coreml_batch, wait_us, gpu_streams, fixed_batch=fixed_batch)
    orig_js_bridge = js_player.BRIDGE
    pooled_js = PooledJavaBridge(n_jvms=n_jvms)
    js_player.BRIDGE = pooled_js
    lock = threading.Lock()
    state_ctl = {'next_attempt': next_attempt}
    started = time.perf_counter()

    def next_job():
        with lock:
            if len(completed) >= target_games:
                return None
            a = state_ctl['next_attempt']
            state_ctl['next_attempt'] += 1
            return a

    def worker_loop():
        while True:
            attempt = next_job()
            if attempt is None:
                return
            pair = make_pair(seed, block_name, attempt)
            learner_color = Color.BLUE if attempt % 2 == 0 else Color.RED
            opp_color = Color.RED if learner_color == Color.BLUE else Color.BLUE
            game = benchmark.make_game(pair, friendly_robber=True)
            codec = Codec(game)
            meta = job_meta(game)
            rng_state = list(game.state.random.getstate()[1])
            colors = [c.value for c in game.state.colors]
            learner_seat = colors.index(learner_color.value)
            gid = f'{block_name}-{attempt:05d}'
            task = {
                'job': meta,
                'start': meta['initial'],
                'start_rng': rng_state,
                'logs': [],
                'id': gid,
                'prefix': 0,
                'seat_seeds': [benchmark.named_seed(pair['agent_seed'], c) for c in colors],
                'rng_state': rng_state,
            }
            game_setup_temp = setup_temperature if (setup_exploration_games > 0 and attempt < setup_exploration_games) else None
            bridge.call(
                op='init', game_id=gid, task=task, learner_seat=learner_seat,
                model=str(learner_onnx),
                dynamic_model=(str(dynamic_learner_onnx) if dynamic_learner_onnx is not None else None),
                frozen=str(frozen_onnx),
                simulations=protocol['simulations'], setup_simulations=eff_setup_sims,
                setup_temperature=game_setup_temp, lanes=mcts_lanes,
                virtual_loss=virtual_loss, eval_mode=False,
                mcts_opponent=mcts_opponent,
                vp_margin_alpha=vp_margin_alpha,
            )
            jsettlers = benchmark.Agent('JSETTLERS', opp_color, benchmark.named_seed(pair['agent_seed'], 'JSETTLERS'))
            initial_sha = compact.snapshot_digest(game)
            events = []
            t0 = time.perf_counter()
            try:
                while game.winning_color() is None and game.state.num_turns < 1000 and len(events) < 10000:
                    if len(completed) >= target_games:
                        break
                    actor = game.state.current_color()
                    legal = list(game.playable_actions)
                    visits = None
                    if len(legal) == 1:
                        action = legal[0]
                    elif actor == learner_color:
                        reply = bridge.call(op='choose', game_id=gid)
                        aid = reply['action_id']
                        visits = reply.get('visits') or None
                        action = next((a for a in legal if action_id(a) == aid), None)
                        require(action is not None, f'Illegal native action {aid}')
                    else:
                        action = jsettlers.select(
                            game, legal, benchmark.DecisionContext(time.perf_counter(), 500, 'policy', None)
                        )
                        jsettlers.cold = False
                    result = game.execute(action)
                    aid = action_id(action)
                    ev = {'a': aid, 'event': action_record_to_json(result)}
                    if visits and actor == learner_color:
                        ev['v'] = visits
                    events.append(ev)
                    bridge.call(op='step', game_id=gid, action_id=aid, chance=codec.chance(action, result))
            finally:
                try:
                    bridge.call(op='close', game_id=gid)
                except Exception:
                    pass

            winner = game.winning_color()
            if winner is None:
                continue
            record = {
                'format': 'catan-compact-native-v1',
                'pair': pair,
                'candidate_color': learner_color.value,
                'opponent_kind': 'JSETTLERS',
                'matchup_block': block_name,
                'mode': 'search',
                'candidate_checkpoint': learner_ref,
                'opponent_checkpoint': None,
                'frozen_search_opponent': frozen_ref,
                'board_sha256': digest(map_to_json(game.state.board.map)),
                'simulator_seed': pair['simulator_seed'],
                'initial_sha256': initial_sha,
                'final_sha256': compact.snapshot_digest(game),
                'events': events,
                'status': 'completed',
                'winner': winner.value,
                'turns': game.state.num_turns,
                'elapsed_seconds': time.perf_counter() - t0,
                'searched_rows': sum(bool(e.get('v')) for e in events),
                'rules': {'friendly_robber': True, 'friendly_robber_basis': 'visible-victory-points'},
                'native': {
                    'events_sha256': digest(events),
                    'lanes': mcts_lanes,
                    'virtual_loss': virtual_loss,
                    'mcts_opponent': mcts_opponent,
                    'vp_margin_alpha': vp_margin_alpha,
                    'setup_simulations': eff_setup_sims,
                    'setup_temperature': game_setup_temp,
                    'search_protocol': protocol,
                },
            }
            compact.replay_journal(record)
            out_path = block_dir / f'game-{attempt:05d}.json.gz'
            with lock:
                if len(completed) >= target_games:
                    return
                compact.save_journal(out_path, record)
                completed.append(selfplay.reference(out_path))
                done_n = len(completed)
            if print_games or done_n % 50 == 0 or done_n == target_games:
                elapsed = time.perf_counter() - started
                print(
                    f'  [{block_name}] {done_n}/{target_games} completed '
                    f'(last: {winner.value} in {record["turns"]} turns | {elapsed/60.0:.1f} min)',
                    flush=True,
                )

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(worker_loop) for _ in range(n_workers)]
            for f in futs:
                f.result()
    finally:
        bridge.close()
        pooled_js.close()
        js_player.BRIDGE = orig_js_bridge

    elapsed = time.perf_counter() - started
    pm = bridge.pool_metrics
    mean_b = (pm['rows'] / pm['batches']) if pm.get('batches') else 0.0
    print(
        f'[{block_name}] Done: {len(completed)}/{target_games} verified games in {elapsed/60.0:.1f} min '
        f'(mean_batch={mean_b:.1f}, batches={pm.get("batches", 0)})',
        flush=True,
    )
    return completed[:target_games], {'elapsed_seconds': elapsed, 'pool_metrics': pm}


def _init_derive_worker():
    torch.set_num_threads(1)


def _derive_workers(n_items, load_workers=None):
    if n_items <= 4:
        return 1
    if load_workers is not None and load_workers > 0:
        return max(1, min(load_workers, n_items))
    return max(1, min(os.cpu_count() or 1, 16, n_items))


def _derive_val_game_task(item):
    ref, seed, limit = item
    path = ROOT / ref['path']
    require(file_digest(path) == ref['sha256'], f'Journal modified: {path}')
    with gzip.open(path, 'rt') as stream:
        rec = json.load(stream)
    return fresh.derive(rec, limit, benchmark.named_seed(seed, rec['pair']['pair_id']))


def _derive_train_game_task(ref):
    path = ROOT / ref['path']
    sha = ref['sha256']
    require(file_digest(path) == sha, f'Journal modified: {path}')
    with gzip.open(path, 'rt') as stream:
        rec = json.load(stream)
    d, n_init = _derive_all_game_positions(rec)
    if d is not None:
        w4.check_arrays(d)
    return sha, rec['pair']['pair_id'], d, n_init


def derive_memory_data(refs, seed, limit=64, load_workers=None):
    tasks = [(ref, seed, limit) for ref in refs]
    n_workers = _derive_workers(len(tasks), load_workers)
    if n_workers == 1:
        derived = [_derive_val_game_task(t) for t in tasks]
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_derive_worker) as pool:
            derived = list(pool.map(_derive_val_game_task, tasks, chunksize=4))
    chunks = [d for d in derived if d is not None]
    return compact.MemoryData(chunks, refs)


def _derive_all_game_positions(rec):
    """Replay a game once and return all eligible positions plus initial build-phase count."""
    events = rec['events']
    eligible = [i for i, e in enumerate(events) if e.get('v') or e.get('demonstration')]
    if not eligible:
        return None, 0
    first_roll = next((i for i, e in enumerate(events) if e['event'][0][1] == 'ROLL'), len(events))
    n_init = sum(1 for i in eligible if i < first_roll)
    data = fresh.derive(rec, len(eligible), seed=0, always_include_initial=True)
    return data, n_init


class ResamplingMemoryData:
    """Keeps all eligible training positions in RAM once and reslices per-game subsets in ms."""

    def __init__(
        self,
        refs,
        base_seed,
        limit=64,
        resample_every=500,
        calls_per_step=2,
        derived_cache=None,
        load_workers=None,
    ):
        self.refs = list(refs)
        self.base_seed = base_seed
        self.limit = limit
        self.resample_every = resample_every
        self.calls_per_step = calls_per_step
        self.manifest = {'families': w4.FAMILIES}
        self._round = 0
        self._sample_calls = 0

        t0 = time.perf_counter()
        cache = derived_cache if derived_cache is not None else {}
        uncached = []
        seen_uncached = set()
        cached_hits = 0
        for ref in self.refs:
            sha = ref['sha256']
            if sha in cache:
                require(file_digest(ROOT / ref['path']) == sha, f'Journal modified: {ROOT / ref["path"]}')
                cached_hits += 1
            elif sha not in seen_uncached:
                seen_uncached.add(sha)
                uncached.append(ref)

        n_workers = _derive_workers(len(uncached), load_workers)
        if uncached:
            if n_workers == 1:
                results = [_derive_train_game_task(ref) for ref in uncached]
            else:
                ctx = mp.get_context('spawn')
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_derive_worker) as pool:
                    results = list(pool.map(_derive_train_game_task, uncached, chunksize=8))
            for sha, pair_id, d, n_init in results:
                cache[sha] = (pair_id, d, n_init)

        chunks = []
        self._game_spans = []
        offset = 0
        for ref in self.refs:
            pair_id, d, n_init = cache[ref['sha256']]
            if d is None:
                continue
            n_rows = len(d['value'])
            chunks.append(d)
            self._game_spans.append((pair_id, offset, min(n_init, n_rows), n_rows))
            offset += n_rows
        require(chunks, 'No completed searched examples')

        self._pool = {k: np.concatenate([d[k] for d in chunks]) for k in ('features', 'legal', 'policy', 'value')}
        self._pool['action'] = self._pool['policy'].argmax(1).astype(np.int64)
        self._pool['family'] = w4.ACTION_FAMILY[self._pool['action']]
        self._pool['phase'] = np.ones(len(self._pool['value']), np.uint8)
        n_pool = len(self._pool['value'])
        self._pool_family_mass = np.zeros((n_pool, len(w4.FAMILIES)), np.float64)
        for action, family in enumerate(w4.ACTION_FAMILY):
            self._pool_family_mass[:, family] += self._pool['policy'][:, action]
        self._refs_sha256 = digest(self.refs)

        pool_bytes = sum(arr.nbytes for arr in self._pool.values()) + self._pool_family_mass.nbytes
        print(
            f'  [pool] Loaded {n_pool:,} eligible positions from {len(self._game_spans)} games '
            f'({cached_hits} cached, {len(uncached)} derived via {n_workers}w) '
            f'into RAM ({pool_bytes / (1024**3):.2f} GB) in {time.perf_counter() - t0:.1f}s',
            flush=True,
        )
        self._build_round(0)

    def _round_seed(self, round_idx):
        return self.base_seed if round_idx == 0 else benchmark.named_seed(self.base_seed, f'resample-{round_idx}')

    def _build_round(self, round_idx):
        round_seed = self._round_seed(round_idx)
        selected = []
        for pair_id, offset, n_init, n_rows in self._game_spans:
            if n_init > 0:
                selected.extend(range(offset, offset + n_init))
            rem = max(0, self.limit - n_init)
            n_later = n_rows - n_init
            if rem > 0 and n_later > 0:
                if rem >= n_later:
                    selected.extend(range(offset + n_init, offset + n_rows))
                else:
                    picked = random.Random(benchmark.named_seed(round_seed, pair_id)).sample(
                        range(offset + n_init, offset + n_rows), rem
                    )
                    picked.sort()
                    selected.extend(picked)
        self._sel = np.asarray(selected, dtype=np.int64)
        n = len(self._sel)
        require(n > 0, 'No selected positions for training round')
        fm = self._pool_family_mass[self._sel]
        counts = fm.sum(0)
        active = counts > 0
        mass = np.zeros_like(counts)
        mass[active] = w4.balanced_mass(counts[active])
        self.factors = 0.5 / n + 0.5 * np.divide(mass, counts, out=np.zeros_like(mass), where=active)
        self.q = fm @ self.factors
        require(
            np.isclose(self.q.sum(), 1) and (self.q > 0).all() and np.max(n * self.q) <= 5 + 1e-6,
            'Invalid W4 replay distribution',
        )
        self.summary = {
            'rows': n,
            'pool_rows': len(self._pool['value']),
            'dataset_references_sha256': self._refs_sha256,
            'max_relative_sampling_probability': float(np.max(n * self.q)),
            'value_weight_expectation': float(np.sum(self.q / (n * self.q))),
            'probabilities_sha256': digest(self.q.tolist()),
        }

    def reset(self):
        self._sample_calls = 0
        if self._round != 0:
            self._round = 0
            self._build_round(0)

    @property
    def arrays(self):
        return {k: v[self._sel] for k, v in self._pool.items()}

    def indices(self, split):
        return np.arange(len(self._sel))

    def batch(self, indices):
        pool_ids = self._sel[indices]
        return {k: v[pool_ids].copy() for k, v in self._pool.items()}

    def sample(self, rng, count):
        if self.resample_every > 0:
            period = self.resample_every * self.calls_per_step
            if self._sample_calls > 0 and self._sample_calls % period == 0:
                self._round += 1
                t0 = time.perf_counter()
                self._build_round(self._round)
                step_num = self._sample_calls // self.calls_per_step
                print(
                    f'  [resample] step {step_num} (round {self._round}): '
                    f'sliced {len(self._sel):,} positions from in-RAM pool '
                    f'({self.limit}/game, initial placements pinned) in {(time.perf_counter() - t0) * 1000:.1f}ms',
                    flush=True,
                )
        self._sample_calls += 1
        ids = rng.choice(len(self.q), count, p=self.q)
        batch = self.batch(ids)
        batch['policy'] *= (self.factors[w4.ACTION_FAMILY][None, :] / self.q[ids, None]).astype(np.float32)
        require(np.allclose(batch['policy'].sum(1), 1, atol=2e-6), 'W4 target normalization failed')
        batch['value_weight'] = (1 / (len(self.q) * self.q[ids])).astype(np.float32)
        return batch


def run_one_cohort(
    cohort_idx,
    cohort_dir,
    cohort_seed,
    cnn_ref,
    tr_ref,
    frozen_ref,
    binary,
    frozen_onnx,
    protocol,
    args,
    derived_cache=None,
):
    results_path = cohort_dir / 'results.json'
    if results_path.exists():
        saved = json.loads(results_path.read_text())
        if saved.get('status') == 'completed' and 'results' in saved:
            print(f'\n=== Cohort {cohort_idx:03d}: already completed, reusing saved checkpoints ===', flush=True)
            return saved['results']['cnn']['best_checkpoint'], saved['results']['transformer']['best_checkpoint'], saved

    cohort_dir.mkdir(parents=True, exist_ok=True)
    games_root = cohort_dir / 'games'
    games_root.mkdir(exist_ok=True)

    print(f'\n================ COHORT {cohort_idx:03d}/{args.n_cohorts:03d} ================', flush=True)
    print(f'CNN parent        : {cnn_ref["path"]}', flush=True)
    print(f'Transformer parent: {tr_ref["path"]}', flush=True)
    print(f'Directory         : {cohort_dir}', flush=True)
    total_games = args.cnn_selfplay_games + args.jsettlers_cnn_games
    setup_exp_total = min(args.setup_exploration_games, total_games)
    cnn_setup_exp = round(setup_exp_total * args.cnn_selfplay_games / total_games) if total_games > 0 else 0
    js_setup_exp = setup_exp_total - cnn_setup_exp
    eff_setup_sims = args.setup_simulations if args.setup_simulations is not None else args.simulations
    use_static_batch = (args.gpu_backend == 'coreml') or (args.gpu_backend == 'cuda' and args.cuda_fixed_batch)
    eff_fixed_batch = args.coreml_batch if use_static_batch else 0
    onnx_dtype = 'float32' if (args.fp32 or args.gpu_backend != 'cuda') else 'float16'
    print(
        f'Collection        : {args.cnn_selfplay_games} CNN vs CNN + {args.jsettlers_cnn_games} JSettlers vs CNN '
        f'(w{args.workers}/js-w{args.jsettlers_workers}-jvms{args.jsettlers_jvms}, over_dispatch={args.tail_over_dispatch}, b{args.coreml_batch}-fixed{eff_fixed_batch}-s{args.gpu_streams}-l{args.mcts_lanes}, '
        f'backend={args.gpu_backend}, dtype={onnx_dtype}, {args.simulations} sims, setup_sims={eff_setup_sims}, setup_explore={setup_exp_total}@T={args.setup_temperature}, '
        f'mcts_opponent={args.mcts_opponent}, vp_margin_alpha={args.vp_margin_alpha})',
        flush=True,
    )

    if use_static_batch:
        cnn_onnx = export_checkpoint_coreml_static(cnn_ref, batch_size=args.coreml_batch, dtype=onnx_dtype)
        cnn_dynamic_onnx = (
            export_checkpoint_dynamic(cnn_ref, dtype=onnx_dtype)
            if args.gpu_backend != 'coreml'
            else None
        )
    else:
        cnn_onnx = export_checkpoint_dynamic(cnn_ref, dtype=onnx_dtype)
        cnn_dynamic_onnx = None

    block_refs = {}
    block_stats = {}
    t_collect = time.perf_counter()

    if args.cnn_selfplay_games > 0:
        block_refs['cnn_vs_cnn'], block_stats['cnn_vs_cnn'] = collect_neural_block(
            'cnn_vs_cnn', games_root / 'cnn_vs_cnn', args.cnn_selfplay_games, cohort_seed,
            binary, cnn_ref, cnn_ref, cnn_onnx, cnn_onnx, frozen_ref, frozen_onnx,
            protocol, args.workers, args.coreml_batch, args.gpu_streams,
            args.mcts_lanes, args.virtual_loss, args.wait_us,
            gpu_backend=args.gpu_backend,
            mcts_opponent=args.mcts_opponent,
            vp_margin_alpha=args.vp_margin_alpha,
            setup_simulations=eff_setup_sims,
            setup_exploration_games=cnn_setup_exp,
            setup_temperature=args.setup_temperature,
            print_games=args.print_games,
            fixed_batch=eff_fixed_batch,
            dynamic_onnx_a=cnn_dynamic_onnx,
            dynamic_onnx_b=cnn_dynamic_onnx,
            tail_over_dispatch=args.tail_over_dispatch,
        )
    if args.jsettlers_cnn_games > 0:
        block_refs['jsettlers_vs_cnn'], block_stats['jsettlers_vs_cnn'] = collect_jsettlers_block(
            'jsettlers_vs_cnn', games_root / 'jsettlers_vs_cnn', args.jsettlers_cnn_games, cohort_seed,
            binary, cnn_ref, cnn_onnx, frozen_ref, frozen_onnx,
            protocol, args.jsettlers_workers, args.jsettlers_jvms, args.coreml_batch, args.gpu_streams,
            args.mcts_lanes, args.virtual_loss, args.wait_us,
            gpu_backend=args.gpu_backend,
            mcts_opponent=args.mcts_opponent,
            vp_margin_alpha=args.vp_margin_alpha,
            setup_simulations=eff_setup_sims,
            setup_exploration_games=js_setup_exp,
            setup_temperature=args.setup_temperature,
            print_games=args.print_games,
            fixed_batch=eff_fixed_batch,
            dynamic_learner_onnx=cnn_dynamic_onnx,
            tail_over_dispatch=args.tail_over_dispatch,
        )
    collect_seconds = time.perf_counter() - t_collect

    # Stratified whole-game train / validation split across blocks (no separate test split).
    splits = {'train': [], 'validation': []}
    for name, refs in block_refs.items():
        n = len(refs)
        n_val = max(1, min(n - 1, round(n * args.validation_fraction))) if n > 1 else 0
        n_train = n - n_val
        splits['train'].extend(refs[:n_train])
        splits['validation'].extend(refs[n_train:])

    manifest = {
        'cohort': cohort_idx,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'completed_games': sum(len(v) for v in splits.values()),
        'per_block_counts': {k: len(v) for k, v in block_refs.items()},
        'split_counts': {k: len(v) for k, v in splits.items()},
        'collection_seconds': collect_seconds,
        'collection_config': {
            'workers': args.workers,
            'jsettlers_workers': args.jsettlers_workers,
            'jsettlers_jvms': args.jsettlers_jvms,
            'tail_over_dispatch': args.tail_over_dispatch,
            'gpu_backend': args.gpu_backend,
            'gpu_dtype': onnx_dtype,
            'coreml_batch': args.coreml_batch,
            'fixed_batch': eff_fixed_batch,
            'gpu_streams': args.gpu_streams,
            'mcts_lanes': args.mcts_lanes,
            'virtual_loss': args.virtual_loss,
            'mcts_opponent': args.mcts_opponent,
            'vp_margin_alpha': args.vp_margin_alpha,
            'simulations': args.simulations,
            'setup_simulations': eff_setup_sims,
            'setup_exploration_games': setup_exp_total,
            'setup_temperature': args.setup_temperature,
            'validation_fraction': args.validation_fraction,
            'positions_per_game': args.positions_per_game,
            'resample_every': args.resample_every,
            'value_loss_weight': args.value_loss_weight,
            'updates': args.updates,
            'replay_cohorts': args.replay_cohorts,
        },
        'rules': {'friendly_robber': True, 'friendly_robber_basis': 'visible-victory-points'},
        'splits': splits,
        'block_stats': block_stats,
    }

    # Sliding replay buffer: combine train splits from the last `replay_cohorts` cohorts
    # while keeping validation strictly from the current cohort (no leakage).
    replay_train_refs = []
    next_cache_refs = []
    included_cohorts = []
    first_replay = max(1, cohort_idx - args.replay_cohorts + 1)
    first_next_replay = max(1, cohort_idx + 1 - args.replay_cohorts + 1)
    for past_idx in range(first_replay, cohort_idx):
        past_manifest_path = args.output / f'cohort-{past_idx:03d}' / 'manifest.json'
        if past_manifest_path.exists():
            past_manifest = json.loads(past_manifest_path.read_text())
            past_train = past_manifest.get('splits', {}).get('train', [])
            if past_train:
                replay_train_refs.extend(past_train)
                included_cohorts.append(past_idx)
                if past_idx >= first_next_replay:
                    next_cache_refs.extend(past_train)
    replay_train_refs.extend(splits['train'])
    if cohort_idx >= first_next_replay:
        next_cache_refs.extend(splits['train'])
    included_cohorts.append(cohort_idx)
    manifest['replay_window'] = {
        'replay_cohorts': args.replay_cohorts,
        'included_cohorts': included_cohorts,
        'train_games': len(replay_train_refs),
    }
    write(cohort_dir / 'manifest.json', manifest)
    print(
        f'\n=== Cohort {cohort_idx:03d} Verified: {manifest["completed_games"]} completed games '
        f'({manifest["split_counts"]}) in {collect_seconds/60.0:.1f} min ===',
        flush=True,
    )

    if args.skip_training:
        return cnn_ref, tr_ref, {'cohort': cohort_idx, 'status': 'collected-only', 'manifest': str(cohort_dir / 'manifest.json')}

    train_device = resolve_torch_device(args.train_device)
    configure_torch_device(train_device)
    torch.set_num_threads(1)

    print(
        f'Replay buffer: cohorts {included_cohorts} ({len(replay_train_refs)} train games) | '
        f'val={len(splits["validation"])} from cohort {cohort_idx:03d} '
        f'(validation_fraction={args.validation_fraction}, train_device={train_device})',
        flush=True,
    )
    print(
        f'Loading all eligible training positions into RAM once '
        f'(slicing {args.positions_per_game}/game with initial placements pinned, '
        f'resample_every={args.resample_every} updates; validation fixed)...',
        flush=True,
    )
    load_workers = getattr(args, 'load_workers', 0) or None
    train_data = ResamplingMemoryData(
        replay_train_refs,
        cohort_seed,
        limit=args.positions_per_game,
        resample_every=args.resample_every,
        calls_per_step=2,
        derived_cache=derived_cache,
        load_workers=load_workers,
    )
    if derived_cache is not None:
        keep_shas = {r['sha256'] for r in next_cache_refs}
        for sha in list(derived_cache):
            if sha not in keep_shas:
                del derived_cache[sha]
    val_data = derive_memory_data(
        splits['validation'],
        cohort_seed,
        limit=args.positions_per_game,
        load_workers=load_workers,
    )
    print(
        f'Active positions: train={train_data.summary["rows"]:,} '
        f'(in-RAM pool={train_data.summary["pool_rows"]:,}), '
        f'val={len(val_data.arrays["value"]):,}',
        flush=True,
    )

    sources = {str(Path(__file__).relative_to(ROOT)).replace('\\', '/'): file_digest(Path(__file__))}
    results = {}
    for name, klass, parent_ref in [
        ('cnn', fresh.Fresh, cnn_ref),
        ('transformer', SparseCatchup, tr_ref),
    ]:
        print(
            f'\n=== Cohort {cohort_idx:03d}: Training {name.upper()} '
            f'({args.updates} W4 updates on {train_device}, value_loss_weight={args.value_loss_weight}x) ===',
            flush=True,
        )
        model_dir = cohort_dir / name
        model_dir.mkdir(exist_ok=True)
        write(model_dir / 'manifest.json', manifest)
        work = klass.__new__(klass)
        work.parent = parent_ref
        work.protocol = dict(protocol)
        work.core = sources
        work.cache = val_data
        work.state = {}
        work.device = train_device
        work.deadline = time.time() + 6 * 3600
        work.stop_requested = False
        work.persist = lambda: None
        w4.OUTPUT = model_dir
        w4.CORPUS = model_dir

        train_data.reset()
        train_res = work.train_w4(train_data, train_data)

        best_meta = torch.load(ROOT / train_res['checkpoint']['path'], map_location='cpu', weights_only=True)['metadata']
        best_step = best_meta.get('step')

        metrics = {}
        for label, ref in [
            ('incoming', parent_ref),
            ('best_val', train_res['checkpoint']),
            ('final', train_res['final_checkpoint']),
        ]:
            cp = selfplay.load_checkpoint(ref) if name == 'cnn' else work.load_sparse(ref)
            model = cp.model.to(train_device)
            torch.set_num_threads(1)
            metrics[label] = {}
            r = corrected_evaluate(model, val_data, 'validation', train_device)
            metrics[label]['validation'] = {
                k: r[k] for k in ('policy_nll', 'hard_policy_nll', 'value_bce', 'selection_loss', 'brier', 'top1_accuracy')
            }
            model.to('cpu')
            del model, cp
            empty_device_cache(train_device)

        results[name] = {
            'parent': parent_ref,
            'best_step': best_step,
            'best_checkpoint': train_res['checkpoint'],
            'final_checkpoint': train_res['final_checkpoint'],
            'training': train_res,
            'metrics': metrics,
        }

    saved = {
        'cohort': cohort_idx,
        'status': 'completed',
        'validation_fraction': args.validation_fraction,
        'value_loss_weight': args.value_loss_weight,
        'manifest': str(cohort_dir / 'manifest.json'),
        'results': results,
        'deployed': False,
    }
    write(results_path, saved)

    print(f'\n=== Cohort {cohort_idx:03d} Results (value_loss_weight={args.value_loss_weight}x) ===')
    print(f'{"Model":<12} {"Stage":<10} {"Step":>5} {"Val Loss":>10} {"Val Pol":>9} {"Val Val":>9} {"Val Brier":>10} {"Val Top1":>10}')
    for name in ('cnn', 'transformer'):
        entry = results[name]
        steps = {'incoming': 0, 'best_val': entry['best_step'], 'final': args.updates}
        for stage in ('incoming', 'best_val', 'final'):
            vm = entry['metrics'][stage]['validation']
            print(
                f'{name:<12} {stage:<10} {str(steps[stage]):>5} {vm["selection_loss"]:>10.4f} '
                f'{vm["policy_nll"]:>9.4f} {vm["value_bce"]:>9.4f} '
                f'{vm["brier"]:>10.4f} {vm["top1_accuracy"]*100:>9.2f}%'
            )

    return results['cnn']['best_checkpoint'], results['transformer']['best_checkpoint'], saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-cohorts', type=int, default=1, help='Number of sequential 2,000-game cohorts to run')
    parser.add_argument('--cnn-checkpoint', type=Path, default=DEFAULT_CNN)
    parser.add_argument('--transformer-checkpoint', type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cnn-selfplay-games', type=int, default=1000, help='CNN vs CNN games per cohort (default: 1000)')
    parser.add_argument('--jsettlers-cnn-games', type=int, default=1000, help='JSettlers vs CNN games per cohort (default: 1000)')
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--setup-simulations', type=int, default=None, help='MCTS simulations during initial build phase (default: matches --simulations)')
    parser.add_argument('--setup-exploration-games', type=int, default=0, help='Number of games per cohort played with --setup-temperature during initial placements (default: 0)')
    parser.add_argument('--setup-temperature', type=float, default=1.5, help='Initial build phase sampling temperature for --setup-exploration-games (default: 1.5)')
    parser.add_argument('--validation-fraction', type=float, default=0.05, help='Fraction of games per block reserved for validation (default: 0.05; remaining games go to train)')
    parser.add_argument('--positions-per-game', type=int, default=64, help='Positions sliced per game per resampling round from the in-RAM pool, always pinning initial placements (default: 64)')
    parser.add_argument('--resample-every', type=int, default=500, help='Reslice --positions-per-game per game from RAM every N training updates (default: 500; 0 = once per cohort)')
    parser.add_argument('--workers', type=int, default=128, help='Workers for CNN vs CNN CoreML pool')
    parser.add_argument('--jsettlers-workers', type=int, default=None, help='Workers for JSettlers vs CNN CoreML bridge-pool (default: matches --workers)')
    parser.add_argument('--jsettlers-jvms', type=int, default=12, help='Number of parallel Java JSettlers Bridge JVMs (default: 12)')
    parser.add_argument('--tail-over-dispatch', type=int, default=32, help='Extra games kept in flight near block completion to avoid batch=1 tail stalls (default: 32)')
    parser.add_argument('--load-workers', type=int, default=0, help='Parallel Python worker processes for loading/deriving games into RAM (0 = auto)')
    parser.add_argument('--gpu-backend', choices=('coreml', 'cuda', 'cpu-batch'), default=default_gpu_backend(), help='Native Rust ONNX execution provider for game collection (default: cuda if available, else coreml on macOS, else cpu-batch)')
    parser.add_argument('--train-device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto', help='PyTorch device for W4 training and evaluation (default: auto)')
    parser.add_argument('--coreml-batch', '--batch-size', dest='coreml_batch', type=int, default=64, help='Max/static GPU inference batch size (default: 64)')
    parser.add_argument('--cuda-fixed-batch', '--fixed-batch', dest='cuda_fixed_batch', action='store_true', help='Use static padded batches for CUDA EP (pairs with --batch-size / --coreml-batch)')
    parser.add_argument('--gpu-streams', type=int, default=4)
    parser.add_argument('--mcts-lanes', type=int, default=1)
    parser.add_argument('--virtual-loss', type=float, default=1.0)
    parser.add_argument('--mcts-opponent', choices=('self-policy', 'adversarial-puct', 'frozen-mlp'), default='self-policy', help='Opponent model inside MCTS simulations (default: self-policy)')
    parser.add_argument('--adversarial-puct', action='store_true', help='Shorthand for --mcts-opponent adversarial-puct')
    parser.add_argument('--vp-margin-alpha', type=float, default=0.03, help='Weight on (2p-1)^2-gated VP-margin utility in MCTS (default: 0.03; 0.0 disables)')
    parser.add_argument('--wait-us', type=int, default=50)
    parser.add_argument('--updates', type=int, default=2000, help='W4 training updates per model per cohort')
    parser.add_argument('--value-loss-weight', type=float, default=0.25, help='Value loss multiplier during training (default: 0.25)')
    parser.add_argument('--replay-cohorts', type=int, default=2, help='Sliding replay buffer window size in cohorts, including current cohort (default: 2; 1 = current only)')
    parser.add_argument('--seed', type=int, default=2026112001)
    parser.add_argument('--print-games', action='store_true')
    parser.add_argument('--fp32', action='store_true', help='Use float32 instead of default float16 for CUDA ONNX export')
    parser.add_argument('--skip-training', action='store_true', help='Only collect and verify games')
    args = parser.parse_args()

    if args.adversarial_puct:
        args.mcts_opponent = 'adversarial-puct'
    if args.jsettlers_workers is None:
        args.jsettlers_workers = args.workers
    require(args.n_cohorts > 0, '--n-cohorts must be >= 1')
    require(args.tail_over_dispatch >= 0, '--tail-over-dispatch must be >= 0')
    require(args.load_workers >= 0, '--load-workers must be >= 0')
    require(0 < args.validation_fraction < 1, '--validation-fraction must be between 0 and 1 (exclusive)')
    require(0.0 <= args.vp_margin_alpha <= 0.5, '--vp-margin-alpha must be between 0.0 and 0.5')
    require(args.value_loss_weight > 0, '--value-loss-weight must be > 0')
    require(args.replay_cohorts > 0, '--replay-cohorts must be >= 1')
    require(args.setup_simulations is None or args.setup_simulations > 0, '--setup-simulations must be >= 1')
    require(args.setup_exploration_games >= 0, '--setup-exploration-games must be >= 0')
    require(args.setup_temperature > 0, '--setup-temperature must be > 0')
    require(args.positions_per_game > 0, '--positions-per-game must be >= 1')
    require(args.resample_every >= 0, '--resample-every must be >= 0')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    protocol = json.loads((ROOT / 'experiments/catan-ai/fresh-xdcnn-visible.json').read_text())
    protocol['simulations'] = args.simulations
    protocol['updates_per_iteration'] = args.updates
    validate_protocol(protocol)

    # Isolated in-process hooks: apply 0.25x value loss weight and full soft-visit validation
    # without modifying historical/pinned trainer source files.
    _orig_losses = w4.losses

    def _weighted_losses(logits, values, legal, targets, outcomes, value_weights):
        pl, vl = _orig_losses(logits, values, legal, targets, outcomes, value_weights)
        return pl, vl * args.value_loss_weight

    w4.losses = _weighted_losses
    sparse_module.losses = _weighted_losses
    w4.evaluate = corrected_evaluate
    sparse_module.evaluate = corrected_evaluate

    cnn_ref = selfplay.reference(args.cnn_checkpoint.resolve())
    tr_ref = selfplay.reference(args.transformer_checkpoint.resolve())
    frozen_ref = resolve_frozen_opponent(cnn_ref, args.mcts_opponent)

    args.output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = ensure_binary()
    frozen_onnx = export_checkpoint(frozen_ref)

    cohort_summaries = []
    derived_cache = {}
    with benchmark.exclusive_benchmark():
        for cohort_idx in range(1, args.n_cohorts + 1):
            cohort_dir = args.output / f'cohort-{cohort_idx:03d}'
            cohort_seed = args.seed + cohort_idx * 100003
            cnn_ref, tr_ref, summary = run_one_cohort(
                cohort_idx,
                cohort_dir,
                cohort_seed,
                cnn_ref,
                tr_ref,
                frozen_ref,
                binary,
                frozen_onnx,
                protocol,
                args,
                derived_cache=derived_cache,
            )
            cohort_summaries.append(summary)

    write(
        args.output / 'multi_cohort_summary.json',
        {
            'n_cohorts': args.n_cohorts,
            'mcts_opponent': args.mcts_opponent,
            'vp_margin_alpha': args.vp_margin_alpha,
            'validation_fraction': args.validation_fraction,
            'replay_cohorts': args.replay_cohorts,
            'value_loss_weight': args.value_loss_weight,
            'updates_per_cohort': args.updates,
            'games_per_cohort': args.cnn_selfplay_games + args.jsettlers_cnn_games,
            'latest_cnn_best': cnn_ref,
            'latest_transformer_best': tr_ref,
            'cohorts': cohort_summaries,
            'deployed': False,
        },
    )

    if not args.skip_training:
        print(f'\n================ MULTI-COHORT SUMMARY ({args.n_cohorts} cohorts) ================')
        print(f'{"Cohort":<8} {"Model":<12} {"BestStep":>8} {"Val Loss":>10} {"Val Pol":>9} {"Val Val":>9} {"Val Brier":>10} {"Val Top1":>10} {"Best Checkpoint"}')
        lines = [
            f'# CoreML Multi-Cohort Training Summary ({args.n_cohorts} cohorts)',
            '',
            f'- **Games/cohort**: {args.cnn_selfplay_games} CNN vs CNN + {args.jsettlers_cnn_games} JSettlers vs CNN ({args.simulations} sims, `mcts_opponent={args.mcts_opponent}`, `vp_margin_alpha={args.vp_margin_alpha}`, `validation_fraction={args.validation_fraction}`, `w{args.workers}-b{args.coreml_batch}-s{args.gpu_streams}-l{args.mcts_lanes}`)',
            f'- **Training**: {args.updates} W4 updates/cohort on MPS, `value_loss_weight={args.value_loss_weight}x`, `replay_cohorts={args.replay_cohorts}`, lowest validation selection, no auto-deployment',
            '',
            '| Cohort | Model | Best Step | Val Loss | Val Policy NLL | Val Value BCE | Val Brier | Val Top-1 | Best Checkpoint |',
            '|---|---|---:|---:|---:|---:|---:|---:|---|',
        ]
        for s in cohort_summaries:
            if s.get('status') != 'completed':
                continue
            cnum = s['cohort']
            for name in ('cnn', 'transformer'):
                entry = s['results'][name]
                vm = entry['metrics']['best_val']['validation']
                cp = entry['best_checkpoint']['path']
                print(
                    f'{cnum:03d}      {name:<12} {str(entry["best_step"]):>8} {vm["selection_loss"]:>10.4f} '
                    f'{vm["policy_nll"]:>9.4f} {vm["value_bce"]:>9.4f} {vm["brier"]:>10.4f} '
                    f'{vm["top1_accuracy"]*100:>9.2f}% {cp}'
                )
                lines.append(
                    f'| {cnum:03d} | {name} | {entry["best_step"]} | {vm["selection_loss"]:.4f} | '
                    f'{vm["policy_nll"]:.4f} | {vm["value_bce"]:.4f} | {vm["brier"]:.4f} | '
                    f'{vm["top1_accuracy"]*100:.2f}% | `{cp}` |'
                )
        (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        print(f'\nSaved summary to {args.output / "multi_cohort_summary.json"} and {args.output / "REPORT.md"}')


if __name__ == '__main__':
    main()
