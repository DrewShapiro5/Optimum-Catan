"""Benchmark Rust CPU-only vs Rust multi-worker batched GPU self-play (MPS socket & CoreML pool).

Uses the most recent XD-CNN checkpoint by default:
    runs/catan-ai/selfplay-only-comparison/cnn/iteration-1/best.pt

Every game is independently replay-verified through the unchanged Python compact replayer.
Outputs diagnostic results only; no training or checkpoint deployment is triggered.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

from agent_apis import benchmark
from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.observation import require
from agent_apis.neural import compact_iterations as compact, selfplay
from agent_apis.neural.device_utils import resolve_torch_device
from agent_apis.neural.native_assets import (
    NATIVE,
    ensure_binary,
    export_checkpoint,
    resolve_frozen_opponent,
    runtime_environment,
)
from agent_apis.neural.native_codec import job_meta
from agent_apis.neural.native_gpu_server import (
    BinaryGpuBatchServer,
    FEATURES_DIM,
    export_checkpoint_coreml_static,
    export_checkpoint_dynamic,
    load_inference_model,
    prepare_gpu_model,
)
from agent_apis.neural.native_selfplay import (
    JournalRecorder,
    NativeClient,
    PROTOCOL_VERSION,
    execution_identity,
    validate_protocol,
)
from agent_apis.neural.pipeline import write

DEFAULT_CHECKPOINTS = {
    'cnn': ROOT / 'runs/catan-ai/selfplay-only-comparison/cnn/iteration-1/best.pt',
    'transformer': ROOT / 'runs/catan-ai/selfplay-only-comparison/sparse/iteration-1/best.pt',
}
DEFAULT_OUTPUTS = {
    'cnn': ROOT / 'runs/catan-ai/native-batched-gpu-benchmark',
    'transformer': ROOT / 'runs/catan-ai/native-batched-gpu-benchmark-transformer',
}
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINTS['cnn']
DEFAULT_OUTPUT = DEFAULT_OUTPUTS['cnn']


def check_gpu_parity(checkpoint_ref, device='auto', dtype='float16'):
    """Verify numerical parity between PyTorch CPU (FP32) and PyTorch GPU (FP16 or FP32) on real observations."""
    dev = resolve_torch_device(device)
    if dev == 'cpu':
        return {'available': False, 'device': 'cpu', 'dtype': 'float32', 'reason': 'No CUDA or MPS GPU available'}
    from catanatron.serialization import web_view
    from agent_apis.observation import encode_observation, from_web_view

    cpu_model = load_inference_model(ROOT / checkpoint_ref['path'], device='cpu')
    gpu_model = prepare_gpu_model(
        load_inference_model(ROOT / checkpoint_ref['path'], device='cpu'),
        dev,
        dtype=dtype,
    )
    samples = []
    for idx in range(32):
        game = benchmark.make_game(make_pair(2026111999, idx), friendly_robber=True)
        actor = game.state.current_color()
        view = from_web_view(web_view(game, actor), actor.value)
        samples.append(encode_observation(view))
    all_x = np.asarray(samples, dtype=np.float32)

    max_logit_diff = 0.0
    max_val_diff = 0.0
    with torch.inference_mode():
        for b in (1, 16, 32):
            x = torch.from_numpy(all_x[:b])
            c_logits, c_val = cpu_model(x)
            g_logits, g_val = gpu_model(x.to(dev))
            logit_diff = float((c_logits - g_logits.cpu()).abs().max().item())
            val_diff = float((c_val.sigmoid() - g_val.cpu().sigmoid()).abs().max().item())
            max_logit_diff = max(max_logit_diff, logit_diff)
            max_val_diff = max(max_val_diff, val_diff)
    logit_tol = 2e-2 if dtype == 'float16' else 1e-3
    val_tol = 5e-3 if dtype == 'float16' else 1e-4
    return {
        'available': True,
        'device': dev,
        'dtype': dtype,
        'max_logit_diff': max_logit_diff,
        'max_value_prob_diff': max_val_diff,
        'passed': max_logit_diff < logit_tol and max_val_diff < val_tol,
    }


def make_pair(seed, index):
    return {
        'pair_id': f'gpu-bench-{seed}-{index}',
        'board_seed': benchmark.named_seed(seed, f'board-{index}'),
        'simulator_seed': benchmark.named_seed(seed, f'simulator-{index}'),
        'agent_seed': benchmark.named_seed(seed, f'agent-{index}'),
    }


def _new_tree_stats():
    return {
        'decisions': 0,
        'reuse_hits': 0,
        'tree_nodes': 0,
        'max_tree_nodes': 0,
        'tree_edges': 0,
        'reused_nodes': 0,
        'reused_opponent_nodes': 0,
        'reused_world_states': 0,
        'reused_root_visits': 0,
        'max_action_depth_sum': 0,
        'max_action_depth': 0,
        'retained_chance_children': 0,
        'feature_seconds': 0.0,
        'inference_seconds': 0.0,
        'chance_enumeration_seconds': 0.0,
        'posterior_seconds': 0.0,
        'reset_reasons': Counter(),
    }


def _record_tree_events(agg, events):
    for ev in events:
        m = ev.get('metrics')
        if not isinstance(m, dict):
            continue
        agg['decisions'] += 1
        reused = m.get('reused_nodes', 0)
        if reused > 0:
            agg['reuse_hits'] += 1
        nodes = m.get('tree_nodes', 0)
        agg['tree_nodes'] += nodes
        agg['max_tree_nodes'] = max(agg['max_tree_nodes'], nodes)
        agg['tree_edges'] += m.get('tree_edges', 0)
        agg['reused_nodes'] += reused
        agg['reused_opponent_nodes'] += m.get('reused_opponent_nodes', 0)
        agg['reused_world_states'] += m.get('reused_world_states', 0)
        agg['reused_root_visits'] += m.get('reused_root_visits', 0)
        depth = m.get('max_action_depth', 0)
        agg['max_action_depth_sum'] += depth
        agg['max_action_depth'] = max(agg['max_action_depth'], depth)
        agg['retained_chance_children'] += m.get('retained_chance_children', 0)
        agg['feature_seconds'] += m.get('feature_seconds', 0.0)
        agg['inference_seconds'] += m.get('inference_seconds', 0.0)
        agg['chance_enumeration_seconds'] += m.get('chance_enumeration_seconds', 0.0)
        agg['posterior_seconds'] += m.get('posterior_seconds', 0.0)
        agg['reset_reasons'][m.get('reset_reason') or 'fresh'] += 1


def _finalize_tree_stats(agg):
    n = agg['decisions']
    if n == 0:
        return None
    hits = agg['reuse_hits']
    return {
        'decisions': n,
        'reuse_hits': hits,
        'reuse_rate': hits / n,
        'mean_tree_nodes': agg['tree_nodes'] / n,
        'max_tree_nodes': agg['max_tree_nodes'],
        'mean_tree_edges': agg['tree_edges'] / n,
        'mean_reused_nodes': agg['reused_nodes'] / n,
        'mean_reused_opp_nodes': agg['reused_opponent_nodes'] / n,
        'mean_reused_total': (agg['reused_nodes'] + agg['reused_opponent_nodes']) / n,
        'mean_reused_worlds': agg['reused_world_states'] / n,
        'mean_reused_visits': agg['reused_root_visits'] / n,
        'mean_reused_total_on_hit': ((agg['reused_nodes'] + agg['reused_opponent_nodes']) / hits) if hits else 0.0,
        'mean_reused_worlds_on_hit': (agg['reused_world_states'] / hits) if hits else 0.0,
        'mean_max_depth': agg['max_action_depth_sum'] / n,
        'max_action_depth': agg['max_action_depth'],
        'mean_retained_chance_children': agg['retained_chance_children'] / n,
        'mean_feature_ms': 1000.0 * agg['feature_seconds'] / n,
        'mean_inference_ms': 1000.0 * agg['inference_seconds'] / n,
        'mean_chance_ms': 1000.0 * agg['chance_enumeration_seconds'] / n,
        'mean_posterior_ms': 1000.0 * agg['posterior_seconds'] / n,
        'reset_reasons': dict(agg['reset_reasons']),
    }


class SocketNativeClient:
    """Long-lived Rust worker process connected to a shared binary Unix-socket GPU batch server."""

    def __init__(self, binary, socket_path, worker_id):
        self.binary = binary
        self.identity = {**execution_identity(binary), 'backend': 'rust-gpu-socket'}
        self.log = (NATIVE / 'target/worker-stderr.log').open('a')
        self.process = subprocess.Popen(
            [str(binary), '--gpu-socket', str(socket_path), '--worker-id', str(worker_id)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=runtime_environment(),
        )
        self.messages = queue.Queue()

        def read():
            try:
                for line in self.process.stdout:
                    self.messages.put(json.loads(line))
            except BaseException as exc:
                self.messages.put(exc)
            finally:
                self.messages.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        self.closed = False
        ready = self._receive(30)
        require(
            ready.get('type') == 'ready' and ready['version'] == PROTOCOL_VERSION,
            f'Unexpected socket worker ready message: {ready}',
        )

    def _receive(self, timeout=0.25):
        msg = self.messages.get(timeout=timeout)
        if isinstance(msg, BaseException):
            raise RuntimeError('Invalid native worker output') from msg
        require(msg is not None, 'Native socket worker exited unexpectedly; check target/worker-stderr.log')
        require(msg.get('type') != 'error', f'Native socket worker error: {msg.get("error")}')
        return msg

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except BaseException:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdout.close()
        if not self.process.stdin.closed:
            self.process.stdin.close()
        self.log.close()

    def play(self, path, pair, candidate, frozen, candidate_onnx, frozen_onnx, protocol, deadline, lanes=1, virtual_loss=1.0, mcts_opponent='self-policy', setup_simulations=None, tree_agg=None, tree_lock=None):
        validate_protocol(protocol)
        eff_setup_sims = setup_simulations if setup_simulations is not None else protocol['simulations']
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        game = benchmark.make_game(pair, friendly_robber=True)
        meta = job_meta(game)
        initial = meta['initial']
        state = list(game.state.random.getstate()[1])
        colors = [c.value for c in game.state.colors]
        stop = path.with_suffix('.stop')
        stop.unlink(missing_ok=True)
        task = {
            'job': meta,
            'start': initial,
            'start_rng': state,
            'logs': [],
            'id': path.name,
            'prefix': 0,
            'seat_seeds': [benchmark.named_seed(pair['agent_seed'], c) for c in colors],
            'rng_state': state,
        }
        record = {
            'format': 'catan-compact-native-v1',
            'pair': pair,
            'candidate_color': 'BLUE',
            'opponent_kind': 'current',
            'mode': 'search',
            'candidate_checkpoint': candidate,
            'opponent_checkpoint': candidate,
            'frozen_search_opponent': frozen,
            'source_sha256': digest({}),
            'collection_execution': None,
            'native': {
                'execution': self.identity,
                'candidate_export_sha256': file_digest(candidate_onnx),
                'enemy_export_sha256': file_digest(candidate_onnx),
                'frozen_export_sha256': file_digest(frozen_onnx),
                'search_protocol': protocol,
                'setup_simulations': eff_setup_sims,
                'budget_seconds': None,
                'limits': {'max_turns': 1000, 'max_actions': 10000},
            },
        }
        recorder = JournalRecorder(game, record, protocol['simulations'], setup_simulations=eff_setup_sims)
        request = {
            'version': PROTOCOL_VERSION,
            'task': task,
            'models': [str(candidate_onnx), str(candidate_onnx)],
            'model_ids': [0, 0],
            'frozen': str(frozen_onnx),
            'simulations': protocol['simulations'],
            'setup_simulations': eff_setup_sims,
            'seconds': None,
            'deadline': deadline,
            'stop_file': str(stop.resolve()),
            'max_turns': 1000,
            'max_actions': 10000,
            'lanes': lanes,
            'virtual_loss': virtual_loss,
            'mcts_opponent': mcts_opponent,
        }
        self.process.stdin.write(json.dumps(request, separators=(',', ':'), allow_nan=False) + '\n')
        self.process.stdin.flush()
        try:
            while True:
                try:
                    msg = self._receive(timeout=0.5)
                except queue.Empty:
                    continue
                if msg['type'] == 'events':
                    if tree_agg is not None:
                        if tree_lock is not None:
                            with tree_lock:
                                _record_tree_events(tree_agg, msg['events'])
                        else:
                            _record_tree_events(tree_agg, msg['events'])
                    recorder.append(msg['events'])
                elif msg['type'] == 'done':
                    return recorder.finish(msg)
                else:
                    raise ValueError(f'Unexpected message: {msg}')
        finally:
            stop.unlink(missing_ok=True)


def run_timed_cpu(binary, workers, seconds, seed, candidate, frozen, protocol, out_dir, print_games=False, setup_simulations=None):
    """Run the existing Rust CPU-only workflow for a timed window."""
    require(
        setup_simulations is None or setup_simulations == protocol['simulations'],
        'CPU baseline NativeClient does not support --setup-simulations != --simulations; pass --skip-cpu',
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    clients = [NativeClient() for _ in range(workers)]
    counter = {'i': 0}
    lock = threading.Lock()
    records = []

    def next_index():
        with lock:
            i = counter['i']
            counter['i'] += 1
            return i

    started = time.perf_counter()
    deadline = time.time() + seconds

    def worker_fn(client):
        local_records = []
        while time.time() < deadline:
            idx = next_index()
            pair = make_pair(seed, idx)
            rec = client.play(
                out_dir / f'game-{idx:05d}.json.gz',
                pair,
                'BLUE',
                candidate,
                candidate,
                frozen,
                'current',
                protocol,
                {},
                deadline,
            )
            if print_games and rec['status'] == 'completed':
                print(
                    f'  [done] {rec["pair"]["pair_id"]}: winner={rec["winner"]} '
                    f'in {rec["turns"]} turns ({rec["elapsed_seconds"]:.1f}s)',
                    flush=True,
                )
            local_records.append(rec)
        return local_records

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(worker_fn, clients):
                records.extend(chunk)
        elapsed = time.perf_counter() - started
    finally:
        for c in clients:
            c.close()

    return summarize_case(f'rust-cpu-w{workers}', 'rust-onnx-cpu', workers, elapsed, records, None)


def run_timed_mps_socket(
    binary,
    workers,
    seconds,
    seed,
    candidate,
    frozen,
    candidate_onnx,
    frozen_onnx,
    protocol,
    out_dir,
    max_batch,
    wait_us,
    device='mps',
    dtype='float16',
    lanes=1,
    virtual_loss=1.0,
    mcts_opponent='self-policy',
    print_games=False,
    setup_simulations=None,
):
    """Run multi-worker Rust search submitting candidate CNN requests over Unix socket to MPS batch server."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sock_dir = Path(tempfile.mkdtemp(prefix='catan-gpu-'))
    socket_path = sock_dir / 'infer.sock'
    server = BinaryGpuBatchServer(
        socket_path=socket_path,
        model_refs={0: candidate},
        device=device,
        max_batch_rows=max_batch,
        wait_seconds=wait_us / 1_000_000.0,
        dtype=dtype,
    ).start()
    clients = []
    counter = {'i': 0}
    lock = threading.Lock()
    tree_lock = threading.Lock()
    tree_agg = _new_tree_stats()
    records = []

    def next_index():
        with lock:
            i = counter['i']
            counter['i'] += 1
            return i

    try:
        clients = [SocketNativeClient(binary, server.endpoint, wid) for wid in range(workers)]
        started = time.perf_counter()
        deadline = time.time() + seconds

        def worker_fn(client):
            local_records = []
            while time.time() < deadline:
                idx = next_index()
                pair = make_pair(seed, idx)
                rec = client.play(
                    out_dir / f'game-{idx:05d}.json.gz',
                    pair,
                    candidate,
                    frozen,
                    candidate_onnx,
                    frozen_onnx,
                    protocol,
                    deadline,
                    lanes=lanes,
                    virtual_loss=virtual_loss,
                    mcts_opponent=mcts_opponent,
                    setup_simulations=setup_simulations,
                    tree_agg=tree_agg,
                    tree_lock=tree_lock,
                )
                if print_games and rec['status'] == 'completed':
                    print(
                        f'  [done] {rec["pair"]["pair_id"]}: winner={rec["winner"]} '
                        f'in {rec["turns"]} turns ({rec["elapsed_seconds"]:.1f}s)',
                        flush=True,
                    )
                local_records.append(rec)
            return local_records

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(worker_fn, clients):
                records.extend(chunk)
        elapsed = time.perf_counter() - started
    finally:
        for c in clients:
            c.close()
        server_metrics = server.stop()
        shutil.rmtree(sock_dir, ignore_errors=True)

    m0 = server_metrics.get(0, {})
    batch_stats = {
        'requests': m0.get('requests', 0),
        'batches': m0.get('batches', 0),
        'mean_batch_size': (m0['rows'] / m0['batches']) if m0.get('batches') else 0.0,
        'forward_seconds': m0.get('forward_seconds', 0.0),
        'queue_window_seconds': m0.get('queue_window_seconds', 0.0),
    }
    suffix = f'-l{lanes}' if lanes > 1 else ''
    return summarize_case(
        f'rust-{device}-socket-w{workers}-b{max_batch}{suffix}',
        f'rust-{device}-socket',
        workers,
        elapsed,
        records,
        batch_stats,
        tree_stats=_finalize_tree_stats(tree_agg),
    )


def run_timed_inprocess_pool(
    binary,
    workers,
    seconds,
    seed,
    candidate,
    frozen,
    dynamic_onnx,
    frozen_onnx,
    protocol,
    out_dir,
    backend,
    max_batch,
    wait_us,
    fixed_batch=0,
    streams=1,
    lanes=1,
    virtual_loss=1.0,
    mcts_opponent='self-policy',
    print_games=False,
    setup_simulations=None,
):
    """Run pure-Rust multi-worker pool with in-process CoreML/ONNX batch queue."""
    eff_setup_sims = setup_simulations if setup_simulations is not None else protocol['simulations']
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (NATIVE / 'target/worker-stderr.log').open('a')
    cmd = [
        str(binary),
        '--pool',
        '--workers',
        str(workers),
        '--gpu-backend',
        backend,
        '--max-batch',
        str(max_batch),
        '--wait-us',
        str(wait_us),
        '--gpu-streams',
        str(streams),
    ]
    if fixed_batch > 0:
        cmd.extend(['--fixed-batch', str(fixed_batch)])
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
        env=runtime_environment(),
    )
    identity = {**execution_identity(binary), 'backend': f'rust-onnx-{backend}-batch'}
    recorders = {}
    records = []
    pool_metrics = {}
    tree_agg = _new_tree_stats()
    in_flight = 0
    next_idx = 0
    started = None
    deadline = None

    def dispatch_one():
        nonlocal next_idx, in_flight
        idx = next_idx
        next_idx += 1
        gid = f'game-{idx:05d}'
        pair = make_pair(seed, idx)
        game = benchmark.make_game(pair, friendly_robber=True)
        meta = job_meta(game)
        state = list(game.state.random.getstate()[1])
        colors = [c.value for c in game.state.colors]
        stop = out_dir / f'{gid}.stop'
        stop.unlink(missing_ok=True)
        task = {
            'job': meta,
            'start': meta['initial'],
            'start_rng': state,
            'logs': [],
            'id': f'{gid}.json.gz',
            'prefix': 0,
            'seat_seeds': [benchmark.named_seed(pair['agent_seed'], c) for c in colors],
            'rng_state': state,
        }
        record = {
            'format': 'catan-compact-native-v1',
            'pair': pair,
            'candidate_color': 'BLUE',
            'opponent_kind': 'current',
            'mode': 'search',
            'candidate_checkpoint': candidate,
            'opponent_checkpoint': candidate,
            'frozen_search_opponent': frozen,
            'source_sha256': digest({}),
            'collection_execution': None,
            'native': {
                'execution': identity,
                'candidate_export_sha256': file_digest(dynamic_onnx),
                'enemy_export_sha256': file_digest(dynamic_onnx),
                'frozen_export_sha256': file_digest(frozen_onnx),
                'search_protocol': protocol,
                'setup_simulations': eff_setup_sims,
                'budget_seconds': None,
                'limits': {'max_turns': 1000, 'max_actions': 10000},
            },
        }
        recorders[gid] = JournalRecorder(game, record, protocol['simulations'], setup_simulations=eff_setup_sims)
        req = {
            'version': PROTOCOL_VERSION,
            'game_id': gid,
            'task': task,
            'models': [str(dynamic_onnx), str(dynamic_onnx)],
            'frozen': str(frozen_onnx),
            'simulations': protocol['simulations'],
            'setup_simulations': eff_setup_sims,
            'seconds': None,
            'deadline': deadline,
            'stop_file': str(stop.resolve()),
            'max_turns': 1000,
            'max_actions': 10000,
            'lanes': lanes,
            'virtual_loss': virtual_loss,
            'mcts_opponent': mcts_opponent,
        }
        try:
            proc.stdin.write(json.dumps(req, separators=(',', ':'), allow_nan=False) + '\n')
            proc.stdin.flush()
        except BrokenPipeError as exc:
            for line in proc.stdout:
                msg = json.loads(line)
                if msg.get('type') == 'error':
                    raise RuntimeError(f'Pool worker error ({msg.get("game_id")}): {msg.get("error")}') from exc
            raise RuntimeError('Pool process exited unexpectedly while dispatching; check target/worker-stderr.log') from exc
        in_flight += 1

    try:
        ready_line = proc.stdout.readline()
        require(bool(ready_line), 'Pool process exited before ready; check target/worker-stderr.log')
        ready = json.loads(ready_line)
        require(ready.get('type') == 'ready', f'Unexpected pool ready message: {ready}')

        started = time.perf_counter()
        deadline = time.time() + seconds
        for _ in range(workers):
            dispatch_one()

        stdin_closed = False
        for line in proc.stdout:
            msg = json.loads(line)
            mtype = msg.get('type')
            if mtype == 'error':
                raise RuntimeError(f'Pool worker error ({msg.get("game_id")}): {msg.get("error")}')
            elif mtype == 'events':
                _record_tree_events(tree_agg, msg['events'])
                recorders[msg['game_id']].append(msg['events'])
            elif mtype == 'done':
                gid = msg['game_id']
                rec = recorders.pop(gid).finish(msg)
                if print_games and rec['status'] == 'completed':
                    print(
                        f'  [done] {gid}: winner={rec["winner"]} '
                        f'in {rec["turns"]} turns ({rec["elapsed_seconds"]:.1f}s)',
                        flush=True,
                    )
                records.append(rec)
                in_flight -= 1
                if time.time() < deadline:
                    dispatch_one()
                elif in_flight == 0 and not stdin_closed:
                    proc.stdin.close()
                    stdin_closed = True
            elif mtype == 'pool_metrics':
                pool_metrics = msg.get('metrics', {})
                break
        elapsed = time.perf_counter() - started
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()

    batch_stats = {
        'requests': pool_metrics.get('requests', 0),
        'batches': pool_metrics.get('batches', 0),
        'mean_batch_size': (pool_metrics['rows'] / pool_metrics['batches']) if pool_metrics.get('batches') else 0.0,
        'forward_seconds': pool_metrics.get('forward_seconds', 0.0),
        'queue_window_seconds': pool_metrics.get('queue_window_seconds', 0.0),
    }
    s_suf = f'-s{streams}' if streams > 1 else ''
    l_suf = f'-l{lanes}' if lanes > 1 else ''
    return summarize_case(
        f'rust-{backend}-pool-w{workers}-b{max_batch}{s_suf}{l_suf}',
        f'rust-onnx-{backend}-batch',
        workers,
        elapsed,
        records,
        batch_stats,
        tree_stats=_finalize_tree_stats(tree_agg),
    )


def summarize_case(name, backend, workers, elapsed, records, batch_stats, tree_stats=None):
    turns = sum(r['turns'] for r in records)
    decisions = sum(r['searched_rows'] for r in records)
    simulations = sum(r['simulations'] for r in records)
    completed = sum(1 for r in records if r['status'] == 'completed')
    return {
        'name': name,
        'backend': backend,
        'workers': workers,
        'elapsed_seconds': elapsed,
        'games_started': len(records),
        'completed_games': completed,
        'turns': turns,
        'searched_decisions': decisions,
        'simulations': simulations,
        'turns_per_second': turns / elapsed if elapsed > 0 else 0.0,
        'decisions_per_second': decisions / elapsed if elapsed > 0 else 0.0,
        'simulations_per_second': simulations / elapsed if elapsed > 0 else 0.0,
        'all_replay_verified': True,
        'batch_stats': batch_stats,
        'tree_stats': tree_stats,
    }


def parse_worker_list(raw):
    items = [raw] if isinstance(raw, str) else list(raw)
    out = []
    for item in items:
        for part in str(item).split(','):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('cnn', 'transformer'), default='cnn', help='Preset latest checkpoint (cnn or graph transformer)')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--seconds', type=int, default=120, help='Timed window per configuration (seconds)')
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--setup-simulations', type=int, default=None, help='MCTS simulations during initial build phase (default: matches --simulations)')
    parser.add_argument('--seed', type=int, default=2026111901)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    is_macos = sys.platform == 'darwin'
    parser.add_argument('--cpu-workers', nargs='+', default=['12'])
    parser.add_argument('--mps-workers', nargs='+', default=['12,16,24,48,64'])
    parser.add_argument('--coreml-workers', nargs='+', default=['16,32'])
    parser.add_argument('--coreml-batch', type=int, default=32, help='Static padded batch size for CoreML EP')
    parser.add_argument('--cuda-workers', nargs='+', default=['32,64,128'], help='Worker counts (comma- or space-separated) for in-process Rust CUDA EP pool')
    parser.add_argument('--cuda-batch', type=int, default=64, help='Max (or fixed when --cuda-fixed-batch) batch size for CUDA EP pool')
    parser.add_argument('--cuda-fixed-batch', '--fixed-batch', dest='cuda_fixed_batch', action='store_true', help='Use static padded batches for CUDA EP pool')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of default FP16 for GPU benchmarks')
    parser.add_argument('--cuda-socket-workers', nargs='+', default=['32,64'], help='Worker counts (comma- or space-separated) for PyTorch CUDA socket server')
    parser.add_argument('--gpu-streams', type=int, default=1, help='Concurrent GPU inference streams for CoreML/CUDA pool')
    parser.add_argument('--mcts-lanes', type=int, default=1, help='Concurrent virtual-loss MCTS lanes per game')
    parser.add_argument('--virtual-loss', type=float, default=1.0, help='Virtual loss penalty when --mcts-lanes > 1')
    parser.add_argument('--mcts-opponent', choices=('self-policy', 'adversarial-puct', 'frozen-mlp'), default='self-policy', help='Opponent model inside MCTS simulations (default: self-policy)')
    parser.add_argument('--adversarial-puct', action='store_true', help='Shorthand for --mcts-opponent adversarial-puct')
    parser.add_argument('--max-batch', type=int, default=64)
    parser.add_argument('--wait-us', type=int, default=500)
    parser.add_argument('--skip-cpu', action='store_true')
    parser.add_argument('--skip-coreml', action='store_true', default=not is_macos)
    parser.add_argument('--skip-mps', action='store_true', default=not has_mps)
    parser.add_argument('--skip-cuda', action='store_true', default=not has_cuda)
    parser.add_argument('--skip-cuda-socket', action='store_true', default=not has_cuda)
    parser.add_argument('--merge-existing', action='store_true', help='Merge new runs into existing summary.json')
    parser.add_argument('--print-games', action='store_true', help='Print a line to console whenever a game completes')
    args = parser.parse_args()
    if args.adversarial_puct:
        args.mcts_opponent = 'adversarial-puct'
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINTS[args.model]
    if args.output is None:
        args.output = DEFAULT_OUTPUTS[args.model]
    require(args.simulations > 0, '--simulations must be >= 1')
    require(args.setup_simulations is None or args.setup_simulations > 0, '--setup-simulations must be >= 1')

    protocol = json.loads((ROOT / 'experiments/catan-ai/fresh-xdcnn-visible.json').read_text())
    protocol['simulations'] = args.simulations
    validate_protocol(protocol)

    candidate = selfplay.reference(args.checkpoint.resolve())
    frozen = resolve_frozen_opponent(candidate, args.mcts_opponent)

    print(f'Checkpoint: {candidate["path"]} (sha256={candidate["sha256"][:12]})', flush=True)
    if frozen == candidate:
        print(f'Frozen MLP: unused (--mcts-opponent={args.mcts_opponent}); placeholder={frozen["path"]}', flush=True)
    else:
        print(f'Frozen MLP: {frozen["path"]} (sha256={frozen["sha256"][:12]})', flush=True)
    print('Building native Rust worker (release)...', flush=True)
    binary = ensure_binary()
    print(f'Built binary: {binary}', flush=True)

    gpu_dtype = 'float32' if args.fp32 else 'float16'
    candidate_onnx = export_checkpoint(candidate)
    frozen_onnx = candidate_onnx if frozen == candidate else export_checkpoint(frozen)
    coreml_onnx = None if args.skip_coreml else export_checkpoint_coreml_static(candidate, batch_size=args.coreml_batch)
    cuda_onnx = None if args.skip_cuda else (
        export_checkpoint_coreml_static(candidate, batch_size=args.cuda_batch, dtype=gpu_dtype)
        if args.cuda_fixed_batch
        else export_checkpoint_dynamic(candidate, dtype=gpu_dtype)
    )

    parity = check_gpu_parity(candidate, device='auto', dtype=gpu_dtype)
    print(f'GPU numerical parity check: {json.dumps(parity)}', flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    summary_path = args.output / 'summary.json'
    if args.merge_existing and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        results = list(existing.get('results', []))
        print(f'Loaded {len(results)} existing results from {summary_path}', flush=True)

    def record_result(res):
        for idx, prev in enumerate(results):
            if prev.get('name') == res.get('name'):
                results[idx] = res
                return
        results.append(res)

    with benchmark.exclusive_benchmark():
        if not args.skip_cpu:
            for w in parse_worker_list(args.cpu_workers):
                print(f'\n[RUN] Rust CPU-only baseline ({w} workers, {args.seconds}s)...', flush=True)
                res = run_timed_cpu(
                    binary, w, args.seconds, args.seed, candidate, frozen, protocol, args.output / f'cpu-w{w}',
                    print_games=args.print_games,
                    setup_simulations=args.setup_simulations,
                )
                record_result(res)
                print(
                    f'  -> {res["turns_per_second"]:.2f} turns/s | {res["decisions_per_second"]:.2f} dec/s | '
                    f'{res["simulations_per_second"]:.0f} sims/s | {res["completed_games"]} completed',
                    flush=True,
                )

        if not args.skip_mps:
            require(has_mps, 'PyTorch MPS is required for --mps-workers')
            for w in parse_worker_list(args.mps_workers):
                print(
                    f'\n[RUN] Rust + Batched MPS GPU queue ({w} workers, max_batch={args.max_batch}, '
                    f'wait={args.wait_us}us, dtype={gpu_dtype}, {args.seconds}s)...',
                    flush=True,
                )
                res = run_timed_mps_socket(
                    binary,
                    w,
                    args.seconds,
                    args.seed,
                    candidate,
                    frozen,
                    candidate_onnx,
                    frozen_onnx,
                    protocol,
                    args.output / f'mps-w{w}-l{args.mcts_lanes}',
                    args.max_batch,
                    args.wait_us,
                    device='mps',
                    dtype=gpu_dtype,
                    lanes=args.mcts_lanes,
                    virtual_loss=args.virtual_loss,
                    mcts_opponent=args.mcts_opponent,
                    print_games=args.print_games,
                    setup_simulations=args.setup_simulations,
                )
                record_result(res)
                bs = res['batch_stats']
                print(
                    f'  -> {res["turns_per_second"]:.2f} turns/s | {res["decisions_per_second"]:.2f} dec/s | '
                    f'{res["simulations_per_second"]:.0f} sims/s | {res["completed_games"]} completed | '
                    f'mean_batch={bs["mean_batch_size"]:.1f}',
                    flush=True,
                )

        if not args.skip_coreml:
            for w in parse_worker_list(args.coreml_workers):
                print(
                    f'\n[RUN] Rust In-Process Pool + Static CoreML EP ({w} workers, static_batch={args.coreml_batch}, '
                    f'streams={args.gpu_streams}, lanes={args.mcts_lanes}, wait={args.wait_us}us, {args.seconds}s)...',
                    flush=True,
                )
                try:
                    res = run_timed_inprocess_pool(
                        binary,
                        w,
                        args.seconds,
                        args.seed,
                        candidate,
                        frozen,
                        coreml_onnx,
                        frozen_onnx,
                        protocol,
                        args.output / f'coreml-w{w}-b{args.coreml_batch}-s{args.gpu_streams}-l{args.mcts_lanes}',
                        'coreml',
                        args.coreml_batch,
                        args.wait_us,
                        fixed_batch=args.coreml_batch,
                        streams=args.gpu_streams,
                        lanes=args.mcts_lanes,
                        virtual_loss=args.virtual_loss,
                        mcts_opponent=args.mcts_opponent,
                        print_games=args.print_games,
                        setup_simulations=args.setup_simulations,
                    )
                    record_result(res)
                    bs = res['batch_stats']
                    print(
                        f'  -> {res["turns_per_second"]:.2f} turns/s | {res["decisions_per_second"]:.2f} dec/s | '
                        f'{res["simulations_per_second"]:.0f} sims/s | {res["completed_games"]} completed | '
                        f'mean_batch={bs["mean_batch_size"]:.1f}',
                        flush=True,
                    )
                except Exception as exc:
                    print(f'  -> CoreML EP pool failed: {exc}', flush=True)
                    record_result({'name': f'rust-coreml-pool-w{w}-b{args.coreml_batch}', 'error': str(exc)})

        if not args.skip_cuda_socket:
            require(has_cuda, 'PyTorch CUDA is required for --cuda-socket-workers')
            for w in parse_worker_list(args.cuda_socket_workers):
                print(
                    f'\n[RUN] Rust + Batched CUDA GPU socket ({w} workers, max_batch={args.max_batch}, '
                    f'wait={args.wait_us}us, dtype={gpu_dtype}, {args.seconds}s)...',
                    flush=True,
                )
                res = run_timed_mps_socket(
                    binary,
                    w,
                    args.seconds,
                    args.seed,
                    candidate,
                    frozen,
                    candidate_onnx,
                    frozen_onnx,
                    protocol,
                    args.output / f'cuda-socket-w{w}-l{args.mcts_lanes}',
                    args.max_batch,
                    args.wait_us,
                    device='cuda',
                    dtype=gpu_dtype,
                    lanes=args.mcts_lanes,
                    virtual_loss=args.virtual_loss,
                    mcts_opponent=args.mcts_opponent,
                    print_games=args.print_games,
                    setup_simulations=args.setup_simulations,
                )
                record_result(res)
                bs = res['batch_stats']
                print(
                    f'  -> {res["turns_per_second"]:.2f} turns/s | {res["decisions_per_second"]:.2f} dec/s | '
                    f'{res["simulations_per_second"]:.0f} sims/s | {res["completed_games"]} completed | '
                    f'mean_batch={bs["mean_batch_size"]:.1f}',
                    flush=True,
                )

        if not args.skip_cuda:
            cuda_fixed = args.cuda_batch if args.cuda_fixed_batch else 0
            cuda_mode = 'Static' if cuda_fixed > 0 else 'Dynamic'
            for w in parse_worker_list(args.cuda_workers):
                print(
                    f'\n[RUN] Rust In-Process Pool + {cuda_mode} CUDA EP ({w} workers, '
                    f'{"fixed_batch" if cuda_fixed > 0 else "max_batch"}={args.cuda_batch}, '
                    f'streams={args.gpu_streams}, lanes={args.mcts_lanes}, wait={args.wait_us}us, '
                    f'dtype={gpu_dtype}, {args.seconds}s)...',
                    flush=True,
                )
                try:
                    res = run_timed_inprocess_pool(
                        binary,
                        w,
                        args.seconds,
                        args.seed,
                        candidate,
                        frozen,
                        cuda_onnx,
                        frozen_onnx,
                        protocol,
                        args.output / f'cuda-w{w}-b{args.cuda_batch}-s{args.gpu_streams}-l{args.mcts_lanes}',
                        'cuda',
                        args.cuda_batch,
                        args.wait_us,
                        fixed_batch=cuda_fixed,
                        streams=args.gpu_streams,
                        lanes=args.mcts_lanes,
                        virtual_loss=args.virtual_loss,
                        mcts_opponent=args.mcts_opponent,
                        print_games=args.print_games,
                        setup_simulations=args.setup_simulations,
                    )
                    record_result(res)
                    bs = res['batch_stats']
                    print(
                        f'  -> {res["turns_per_second"]:.2f} turns/s | {res["decisions_per_second"]:.2f} dec/s | '
                        f'{res["simulations_per_second"]:.0f} sims/s | {res["completed_games"]} completed | '
                        f'mean_batch={bs["mean_batch_size"]:.1f}',
                        flush=True,
                    )
                except Exception as exc:
                    print(f'  -> CUDA EP pool failed: {exc}', flush=True)
                    record_result({'name': f'rust-cuda-pool-w{w}-b{args.cuda_batch}', 'error': str(exc)})

    baseline = next((r for r in results if r.get('backend') == 'rust-onnx-cpu'), None)
    base_dec = baseline['decisions_per_second'] if baseline else 0.0
    base_turns = baseline['turns_per_second'] if baseline else 0.0

    print('\n=== Benchmark Summary (vs Rust CPU-only baseline) ===')
    print(
        f'{"Config":<28} {"Turns/s":>7} {"Dec/s":>7} {"Sims/s":>8} {"Done":>5} {"Batch":>6} '
        f'{"Reuse%":>7} {"Nodes":>7} {"Reused":>7} {"Worlds":>7} {"Depth":>6} {"Spd(Dec)":>9} {"Spd(Turn)":>10}'
    )
    for r in results:
        if 'error' in r:
            print(f'{r["name"]:<28} ERROR: {r["error"]}')
            continue
        bs = r['batch_stats']
        ts = r.get('tree_stats')
        mean_b = f'{bs["mean_batch_size"]:.1f}' if bs else '-'
        reuse_s = f'{ts["reuse_rate"] * 100:.1f}%' if ts else '-'
        nodes_s = f'{ts["mean_tree_nodes"]:.0f}' if ts else '-'
        reused_s = f'{ts["mean_reused_total"]:.0f}' if ts else '-'
        worlds_s = f'{ts["mean_reused_worlds"]:.0f}' if ts else '-'
        depth_s = f'{ts["mean_max_depth"]:.1f}' if ts else '-'
        sp_dec = f'{r["decisions_per_second"] / base_dec:.2f}x' if base_dec > 0 else '-'
        sp_turn = f'{r["turns_per_second"] / base_turns:.2f}x' if base_turns > 0 else '-'
        print(
            f'{r["name"]:<28} {r["turns_per_second"]:>7.2f} {r["decisions_per_second"]:>7.2f} '
            f'{r["simulations_per_second"]:>8.0f} {r["completed_games"]:>5d} {mean_b:>6} '
            f'{reuse_s:>7} {nodes_s:>7} {reused_s:>7} {worlds_s:>7} {depth_s:>6} {sp_dec:>9} {sp_turn:>10}'
        )

    for r in results:
        ts = r.get('tree_stats')
        if not ts:
            continue
        reasons = ', '.join(f'{k}={v}' for k, v in sorted(ts['reset_reasons'].items()))
        print(
            f'  [{r["name"]}] max_nodes={ts["max_tree_nodes"]} max_depth={ts["max_action_depth"]} '
            f'hit_reused={ts["mean_reused_total_on_hit"]:.0f} hit_worlds={ts["mean_reused_worlds_on_hit"]:.0f} '
            f'ms/dec(feat={ts["mean_feature_ms"]:.2f}, inf={ts["mean_inference_ms"]:.2f}, '
            f'chance={ts["mean_chance_ms"]:.2f}, post={ts["mean_posterior_ms"]:.2f}) | {reasons}'
        )

    summary = {
        'checkpoint': candidate,
        'frozen_opponent': frozen,
        'protocol': protocol,
        'setup_simulations': args.setup_simulations if args.setup_simulations is not None else args.simulations,
        'gpu_dtype': gpu_dtype,
        'parity': parity,
        'results': results,
        'training_started': False,
    }
    write(args.output / 'summary.json', summary)
    print(f'\nSaved full results to {args.output / "summary.json"}')


if __name__ == '__main__':
    main()
