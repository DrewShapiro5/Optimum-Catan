"""Head-to-head matchup benchmark: Model checkpoint vs JSettlers or Model vs Model.

Supports:
- Presets (`cnn`, `transformer`) or any `.pt` checkpoint path (XD-CNN, TYPED-EXPHORMER, MLP, H-GNN, Transformer).
- `--opponent jsettlers` (official SMART JSettlers adapter) OR `--opponent <checkpoint|cnn|transformer>`.
- `--policy-only` (0 simulations, greedy argmax policy) OR `--simulations N` (MCTS search).
- Multi-worker execution (`--workers N`), optional GPU batching (`--backend cpu|mps|coreml`) for model-vs-model search,
  paired seats (BLUE/RED on matched seeds), Wilson 95% CI, and published win-rate summary (stdout + JSON + Markdown).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import random
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
import torch
from catanatron.gym.envs.action_space import from_action_space
from catanatron.models.player import Color
from catanatron.serialization import action_from_json, web_view

from agent_apis import benchmark
from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.jsettlers.player import BRIDGE
from agent_apis.neural import selfplay
from agent_apis.neural.native_assets import (
    NATIVE,
    ensure_binary,
    export_checkpoint,
    resolve_frozen_opponent,
    runtime_environment,
)
from agent_apis.neural.device_utils import (
    nonexistent_stop_file,
    resolve_torch_device,
)
from agent_apis.neural.native_codec import Codec, job_meta
from agent_apis.neural.native_gpu_server import (
    BinaryGpuBatchServer,
    export_checkpoint_coreml_static,
    export_checkpoint_dynamic,
    load_inference_model,
)
from agent_apis.neural.native_selfplay import PROTOCOL_VERSION, validate_protocol
from agent_apis.neural.pipeline import write
from agent_apis.observation import encode_observation, from_web_view, require, validate_rules
from agent_apis.stochastic import SearchObservation, action_id

PRESETS = {
    'cnn': ROOT / 'runs/catan-ai/selfplay-only-comparison/cnn/iteration-1/best.pt',
    'transformer': ROOT / 'runs/catan-ai/selfplay-only-comparison/sparse/iteration-1/best.pt',
}
COHORTS_ROOT = ROOT / 'runs/catan-ai/coreml-cohorts-2000'
COLORS = (Color.BLUE, Color.RED)


def _latest_cohort_checkpoint(model_kind):
    if COHORTS_ROOT.is_dir():
        candidates = sorted(COHORTS_ROOT.glob(f'cohort-*/{model_kind}/iteration-1/best.pt'))
        if candidates:
            return candidates[-1]
    return PRESETS[model_kind]


def resolve_checkpoint(spec):
    low = spec.lower()
    if low == 'latest-cnn':
        path = _latest_cohort_checkpoint('cnn')
    elif low == 'latest-transformer':
        path = _latest_cohort_checkpoint('transformer')
    elif low in PRESETS:
        path = PRESETS[low]
    else:
        path = Path(spec)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
    require(path.exists(), f'Checkpoint not found: {path}')
    return selfplay.reference(path)


def wilson_ci(wins, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = wins / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def build_schedule(games, seed, paired=True):
    """Build deterministic game schedule with paired BLUE/RED seats by default."""
    schedule = []
    if paired:
        for i in range(games):
            pair_idx = i // 2
            seat = 'BLUE' if i % 2 == 0 else 'RED'
            pair = {
                'pair_id': f'matchup-{seed}-p{pair_idx:04d}-{seat}',
                'board_seed': benchmark.named_seed(seed, f'board-{pair_idx}'),
                'simulator_seed': benchmark.named_seed(seed, f'simulator-{pair_idx}'),
                'agent_seed': benchmark.named_seed(seed, f'agent-{pair_idx}'),
            }
            schedule.append({'index': i, 'pair_idx': pair_idx, 'candidate_color': seat, 'pair': pair})
    else:
        for i in range(games):
            seat = 'BLUE' if i % 2 == 0 else 'RED'
            pair = {
                'pair_id': f'matchup-{seed}-g{i:04d}-{seat}',
                'board_seed': benchmark.named_seed(seed, f'board-{i}'),
                'simulator_seed': benchmark.named_seed(seed, f'simulator-{i}'),
                'agent_seed': benchmark.named_seed(seed, f'agent-{i}'),
            }
            schedule.append({'index': i, 'pair_idx': i, 'candidate_color': seat, 'pair': pair})
    return schedule


# ---------------------------------------------------------------------------
# Python evaluator (used for Policy-Only and for Model vs JSettlers Search)
# ---------------------------------------------------------------------------

class EvalModel:
    """Thin wrapper supporting both NeuralCheckpoint (XD-CNN/MLP/etc.) and TypedExphormer."""

    def __init__(self, checkpoint_path, device='cpu'):
        self.path = Path(checkpoint_path)
        self.device = device
        self.model = load_inference_model(self.path, device=device)

    def evaluate_batch(self, views):
        legal, features = [], []
        for view in views:
            validate_rules(view)
            ids = [action_id(action_from_json(a)) for a in view['current_playable_actions']]
            require(ids, 'No legal neural actions')
            legal.append(ids)
            features.append(view.features if isinstance(view, SearchObservation) else encode_observation(view))
        raw = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits, value = self.model(raw)
            scores = logits.cpu().numpy().astype(np.float64)
            values = value.sigmoid().cpu().numpy()
        out = []
        for row, ids, prob in zip(scores, legal, values):
            selected = row[ids]
            prob = float(prob)
            require(np.isfinite(selected).all() and np.isfinite(prob) and 0.0 <= prob <= 1.0, 'Nonfinite prediction')
            weights = np.exp(selected - selected.max())
            weights /= weights.sum()
            out.append(({int(i): float(p) for i, p in zip(ids, weights)}, prob))
        return out

    def evaluate(self, view):
        return self.evaluate_batch([view])[0]

    def probabilities(self, view):
        if len(view['current_playable_actions']) == 1:
            return {action_id(action_from_json(view['current_playable_actions'][0])): 1.0}
        return self.evaluate(view)[0]

    def greedy_action(self, game, color):
        legal = list(game.playable_actions)
        if len(legal) == 1:
            return legal[0]
        view = from_web_view(web_view(game, color), color.value)
        probs = self.probabilities(view)
        best_id = max(probs, key=lambda i: (probs[i], -i))
        return from_action_space(best_id, color, COLORS, 'BASE')


_WORKER_STATE = {}


def _py_worker_init(candidate_path, opponent_path, frozen_path, device):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_STATE['candidate'] = EvalModel(candidate_path, device=device)
    _WORKER_STATE['opponent'] = (
        None if opponent_path is None else (
            _WORKER_STATE['candidate']
            if str(opponent_path) == str(candidate_path)
            else EvalModel(opponent_path, device=device)
        )
    )
    _WORKER_STATE['frozen'] = None if frozen_path is None else EvalModel(frozen_path, device='cpu')


def _choose_search(model, frozen, game, color, simulations, search_session, rng, mcts_opponent='self-policy', vp_margin_alpha=0.03):
    from agent_apis.neural.subtree_reuse import SearchSession
    legal = list(game.playable_actions)
    if len(legal) == 1:
        return legal[0]
    view = from_web_view(web_view(game, color), color.value)
    context = benchmark.DecisionContext(time.perf_counter(), 60000, 'simulations', simulations, {})
    result = search_session.run(
        view,
        rng.getrandbits(64),
        model,
        frozen,
        1,
        simulations,
        context,
        1.5,
        128,
        session_id=f'{game.id}:{color.value}',
        model_revision=(str(model.path), 0, str(frozen.path)),
        mcts_opponent=mcts_opponent,
        vp_margin_alpha=vp_margin_alpha,
    )
    return action_from_json(result['action'])


def _py_play_one(item, opponent_kind, simulations, opp_simulations, max_turns, max_actions,
                 mcts_opponent='self-policy', opp_mcts_opponent='self-policy',
                 vp_margin_alpha=0.03, opp_vp_margin_alpha=0.03):
    from agent_apis.neural.subtree_reuse import SearchSession

    pair = item['pair']
    cand_color = Color[item['candidate_color']]
    opp_color = Color.RED if cand_color == Color.BLUE else Color.BLUE
    game = benchmark.make_game(pair, friendly_robber=True)
    started = time.perf_counter()

    cand_model = _WORKER_STATE['candidate']
    opp_model = _WORKER_STATE['opponent']
    frozen = _WORKER_STATE['frozen']
    cand_session = SearchSession() if simulations > 0 else None
    opp_session = SearchSession() if (opponent_kind == 'model' and opp_simulations > 0) else None
    cand_rng = random.Random(benchmark.named_seed(pair['agent_seed'], cand_color.value))
    opp_rng = random.Random(benchmark.named_seed(pair['agent_seed'], opp_color.value))

    jsettlers = None
    if opponent_kind == 'jsettlers':
        BRIDGE.close()
        jsettlers = benchmark.Agent('JSETTLERS', opp_color, benchmark.named_seed(pair['agent_seed'], 'JSETTLERS'))

    actions = 0
    try:
        while game.winning_color() is None and game.state.num_turns < max_turns and actions < max_actions:
            actor = game.state.current_color()
            legal = list(game.playable_actions)
            if len(legal) == 1:
                action = legal[0]
            elif actor == cand_color:
                if simulations == 0:
                    action = cand_model.greedy_action(game, cand_color)
                else:
                    action = _choose_search(
                        cand_model, frozen, game, cand_color, simulations, cand_session, cand_rng,
                        mcts_opponent=mcts_opponent, vp_margin_alpha=vp_margin_alpha,
                    )
            else:
                if opponent_kind == 'jsettlers':
                    action = jsettlers.select(
                        game, legal, benchmark.DecisionContext(time.perf_counter(), 500, 'policy', None)
                    )
                    jsettlers.cold = False
                elif opp_simulations == 0:
                    action = opp_model.greedy_action(game, opp_color)
                else:
                    action = _choose_search(
                        opp_model, frozen, game, opp_color, opp_simulations, opp_session, opp_rng,
                        mcts_opponent=opp_mcts_opponent, vp_margin_alpha=opp_vp_margin_alpha,
                    )
            game.execute(action)
            actions += 1
    finally:
        if opponent_kind == 'jsettlers':
            BRIDGE.close()

    winner = game.winning_color()
    status = 'completed' if winner is not None else 'cutoff'
    return {
        'index': item['index'],
        'pair_id': pair['pair_id'],
        'candidate_color': cand_color.value,
        'winner': winner.value if winner is not None else None,
        'candidate_won': (winner == cand_color) if winner is not None else None,
        'status': status,
        'turns': game.state.num_turns,
        'actions': actions,
        'elapsed_seconds': time.perf_counter() - started,
    }


def run_python_matchup(schedule, candidate_ref, opponent_ref, opponent_kind, frozen_ref,
                       simulations, opp_simulations, workers, device, max_turns, max_actions,
                       mcts_opponent='self-policy', opp_mcts_opponent='self-policy',
                       vp_margin_alpha=0.03, opp_vp_margin_alpha=0.03):
    cand_path = str(ROOT / candidate_ref['path'])
    opp_path = None if opponent_kind == 'jsettlers' else str(ROOT / opponent_ref['path'])
    froz_path = None if (simulations == 0 and (opponent_kind == 'jsettlers' or opp_simulations == 0)) else str(ROOT / frozen_ref['path'])
    results = []

    def _record(res):
        results.append(res)
        outcome = 'WIN' if res['candidate_won'] is True else ('LOSS' if res['candidate_won'] is False else 'CUTOFF')
        print(
            f"  [{len(results):>{len(str(len(schedule)))}}/{len(schedule)}] "
            f"{res['pair_id']} ({res['candidate_color']}): {outcome} in {res['turns']} turns ({res['elapsed_seconds']:.1f}s)",
            flush=True,
        )

    n_workers = min(workers, len(schedule))
    if n_workers <= 1:
        _py_worker_init(cand_path, opp_path, froz_path, device)
        for item in schedule:
            _record(_py_play_one(
                item, opponent_kind, simulations, opp_simulations, max_turns, max_actions,
                mcts_opponent, opp_mcts_opponent, vp_margin_alpha, opp_vp_margin_alpha,
            ))
    else:
        try:
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_py_worker_init,
                initargs=(cand_path, opp_path, froz_path, device),
            ) as pool:
                futs = [
                    pool.submit(
                        _py_play_one, item, opponent_kind, simulations, opp_simulations, max_turns, max_actions,
                        mcts_opponent, opp_mcts_opponent, vp_margin_alpha, opp_vp_margin_alpha,
                    )
                    for item in schedule
                ]
                for fut in as_completed(futs):
                    _record(fut.result())
        except PermissionError:
            _py_worker_init(cand_path, opp_path, froz_path, device)
            for item in schedule:
                _record(_py_play_one(
                    item, opponent_kind, simulations, opp_simulations, max_turns, max_actions,
                    mcts_opponent, opp_mcts_opponent, vp_margin_alpha, opp_vp_margin_alpha,
                ))
    results.sort(key=lambda r: r['index'])
    return results


# ---------------------------------------------------------------------------
# Fast Native Rust evaluator (Model vs Model with MCTS search: CPU / MPS / CoreML)
# ---------------------------------------------------------------------------

def _native_eval_request(item, cand_onnx, opp_onnx, frozen_onnx, simulations, opp_simulations=None,
                         model_ids=None, lanes=1, virtual_loss=1.0, max_turns=1000, max_actions=10000,
                         mcts_opponent='self-policy', opp_mcts_opponent='self-policy',
                         cand_dyn_onnx=None, opp_dyn_onnx=None,
                         vp_margin_alpha=0.03, opp_vp_margin_alpha=0.03):
    if opp_simulations is None:
        opp_simulations = simulations
    pair = item['pair']
    game = benchmark.make_game(pair, friendly_robber=True)
    meta = job_meta(game)
    state = list(game.state.random.getstate()[1])
    colors = [c.value for c in game.state.colors]
    cand_color = item['candidate_color']
    paths = [str(cand_onnx if c == cand_color else opp_onnx) for c in colors]
    mids = None if model_ids is None else [model_ids[0] if c == cand_color else model_ids[1] for c in colors]
    seat_sims = [simulations if c == cand_color else opp_simulations for c in colors]
    seat_opp = [mcts_opponent if c == cand_color else opp_mcts_opponent for c in colors]
    seat_alpha = [vp_margin_alpha if c == cand_color else opp_vp_margin_alpha for c in colors]
    task = {
        'job': meta,
        'start': meta['initial'],
        'start_rng': state,
        'logs': [],
        'id': f'matchup-{item["index"]:05d}',
        'prefix': 0,
        'seat_seeds': [benchmark.named_seed(pair['agent_seed'], c) for c in colors],
        'rng_state': state,
    }
    req = {
        'version': PROTOCOL_VERSION,
        'game_id': f'game-{item["index"]:05d}',
        'task': task,
        'models': paths,
        'frozen': str(frozen_onnx),
        'simulations': simulations,
        'seconds': None,
        'deadline': time.time() + 12 * 3600,
        'stop_file': nonexistent_stop_file(),
        'max_turns': max_turns,
        'max_actions': max_actions,
        'lanes': lanes,
        'virtual_loss': virtual_loss,
        'eval_mode': True,
        'mcts_opponent': mcts_opponent,
        'vp_margin_alpha': vp_margin_alpha,
    }
    if simulations != opp_simulations:
        req['seat_simulations'] = seat_sims
    if mcts_opponent != opp_mcts_opponent:
        req['seat_mcts_opponent'] = seat_opp
    if vp_margin_alpha != opp_vp_margin_alpha:
        req['seat_vp_margin_alpha'] = seat_alpha
    if mids is not None:
        req['model_ids'] = mids
    if cand_dyn_onnx is not None and opp_dyn_onnx is not None:
        req['dynamic_models'] = [str(cand_dyn_onnx if c == cand_color else opp_dyn_onnx) for c in colors]
    return req, colors


def run_native_model_matchup(schedule, candidate_ref, opponent_ref, frozen_ref, simulations, opp_simulations,
                             workers, backend, max_batch, coreml_batch, wait_us, gpu_streams,
                             mcts_lanes, virtual_loss, max_turns, max_actions,
                             mcts_opponent='self-policy', opp_mcts_opponent='self-policy',
                             dtype='float16', cuda_fixed_batch=False,
                             vp_margin_alpha=0.03, opp_vp_margin_alpha=0.03):
    binary = ensure_binary()
    frozen_onnx = export_checkpoint(frozen_ref)
    same_model = candidate_ref['sha256'] == opponent_ref['sha256']

    if backend in ('coreml', 'cuda'):
        use_static = (backend == 'coreml') or cuda_fixed_batch
        eff_dtype = 'float32' if backend == 'coreml' else dtype
        eff_fixed = coreml_batch if use_static else 0
        if use_static:
            cand_onnx = export_checkpoint_coreml_static(candidate_ref, batch_size=coreml_batch, dtype=eff_dtype)
            opp_onnx = cand_onnx if same_model else export_checkpoint_coreml_static(opponent_ref, batch_size=coreml_batch, dtype=eff_dtype)
            if backend != 'coreml':
                cand_dyn_onnx = export_checkpoint_dynamic(candidate_ref, dtype=eff_dtype)
                opp_dyn_onnx = cand_dyn_onnx if same_model else export_checkpoint_dynamic(opponent_ref, dtype=eff_dtype)
            else:
                cand_dyn_onnx = None
                opp_dyn_onnx = None
        else:
            cand_onnx = export_checkpoint_dynamic(candidate_ref, dtype=eff_dtype)
            opp_onnx = cand_onnx if same_model else export_checkpoint_dynamic(opponent_ref, dtype=eff_dtype)
            cand_dyn_onnx = None
            opp_dyn_onnx = None
        return _run_native_coreml_pool(
            binary, schedule, cand_onnx, opp_onnx, frozen_onnx, simulations, opp_simulations,
            workers, coreml_batch, wait_us, gpu_streams, mcts_lanes, virtual_loss, max_turns, max_actions,
            gpu_backend=backend, mcts_opponent=mcts_opponent, opp_mcts_opponent=opp_mcts_opponent,
            fixed_batch=eff_fixed, cand_dyn_onnx=cand_dyn_onnx, opp_dyn_onnx=opp_dyn_onnx,
            vp_margin_alpha=vp_margin_alpha, opp_vp_margin_alpha=opp_vp_margin_alpha,
        )

    cand_onnx = export_checkpoint(candidate_ref)
    opp_onnx = cand_onnx if same_model else export_checkpoint(opponent_ref)
    server = None
    sock_dir = None
    socket_endpoint = None
    model_ids = None
    if backend in ('mps', 'cuda-socket'):
        sock_device = 'cuda' if backend == 'cuda-socket' else 'mps'
        resolve_torch_device(sock_device)
        sock_dir = Path(tempfile.mkdtemp(prefix='catan-matchup-'))
        socket_path = sock_dir / 'infer.sock'
        model_refs = {0: candidate_ref} if same_model else {0: candidate_ref, 1: opponent_ref}
        model_ids = (0, 0) if same_model else (0, 1)
        server = BinaryGpuBatchServer(
            socket_path=socket_path,
            model_refs=model_refs,
            device=sock_device,
            dtype=dtype,
            max_batch_rows=max_batch,
            wait_seconds=wait_us / 1_000_000.0,
        ).start()
        socket_endpoint = server.endpoint

    try:
        work_q = queue.Queue()
        for item in schedule:
            work_q.put(item)
        results = []
        res_lock = threading.Lock()

        def worker_fn(wid):
            cmd = [str(binary)]
            if socket_endpoint is not None:
                cmd.extend(['--gpu-socket', str(socket_endpoint), '--worker-id', str(wid)])
            log = (NATIVE / 'target/worker-stderr.log').open('a')
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, bufsize=1, env=runtime_environment(),
            )
            try:
                ready = json.loads(proc.stdout.readline())
                require(ready.get('type') == 'ready', f'Unexpected worker ready: {ready}')
                while True:
                    try:
                        item = work_q.get_nowait()
                    except queue.Empty:
                        break
                    req, colors = _native_eval_request(
                        item, cand_onnx, opp_onnx, frozen_onnx, simulations, opp_simulations=opp_simulations,
                        model_ids=model_ids, lanes=mcts_lanes, virtual_loss=virtual_loss,
                        max_turns=max_turns, max_actions=max_actions,
                        mcts_opponent=mcts_opponent, opp_mcts_opponent=opp_mcts_opponent,
                        vp_margin_alpha=vp_margin_alpha, opp_vp_margin_alpha=opp_vp_margin_alpha,
                    )
                    proc.stdin.write(json.dumps(req, separators=(',', ':'), allow_nan=False) + '\n')
                    proc.stdin.flush()
                    while True:
                        line = proc.stdout.readline()
                        require(bool(line), 'Native worker exited unexpectedly')
                        msg = json.loads(line)
                        if msg['type'] == 'error':
                            raise RuntimeError(f'Native worker error: {msg["error"]}')
                        if msg['type'] == 'done':
                            winner_idx = msg['winner']
                            winner_color = colors[winner_idx] if winner_idx >= 0 else None
                            cand_won = (winner_color == item['candidate_color']) if winner_color is not None else None
                            res = {
                                'index': item['index'],
                                'pair_id': item['pair']['pair_id'],
                                'candidate_color': item['candidate_color'],
                                'winner': winner_color,
                                'candidate_won': cand_won,
                                'status': msg['status'],
                                'turns': msg['turns'],
                                'actions': msg['actions'],
                                'elapsed_seconds': msg['elapsed_seconds'],
                            }
                            with res_lock:
                                results.append(res)
                                done_n = len(results)
                            outcome = 'WIN' if cand_won is True else ('LOSS' if cand_won is False else 'CUTOFF')
                            print(
                                f"  [{done_n:>{len(str(len(schedule)))}}/{len(schedule)}] "
                                f"{res['pair_id']} ({res['candidate_color']}): {outcome} in {res['turns']} turns ({res['elapsed_seconds']:.1f}s)",
                                flush=True,
                            )
                            break
            finally:
                try:
                    proc.stdin.close()
                    proc.wait(timeout=5)
                except BaseException:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
                log.close()

        n_workers = min(workers, len(schedule))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(worker_fn, wid) for wid in range(n_workers)]
            for f in futs:
                f.result()
    finally:
        if server is not None:
            server.stop()
        if sock_dir is not None:
            shutil.rmtree(sock_dir, ignore_errors=True)

    results.sort(key=lambda r: r['index'])
    return results


def _run_native_coreml_pool(binary, schedule, cand_onnx, opp_onnx, frozen_onnx, simulations, opp_simulations,
                            workers, coreml_batch, wait_us, gpu_streams, mcts_lanes,
                            virtual_loss, max_turns, max_actions,
                            gpu_backend='coreml', mcts_opponent='self-policy', opp_mcts_opponent='self-policy',
                            fixed_batch=None, cand_dyn_onnx=None, opp_dyn_onnx=None,
                            vp_margin_alpha=0.03, opp_vp_margin_alpha=0.03):
    n_workers = min(workers, len(schedule))
    if fixed_batch is None:
        fixed_batch = coreml_batch if gpu_backend == 'coreml' else 0
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
    results = []
    meta_by_gid = {}
    next_i = 0
    in_flight = 0
    started = time.perf_counter()
    last_progress = started
    progress_actions = 0
    progress_decisions = 0
    progress_sims = 0

    def dispatch_one():
        nonlocal next_i, in_flight
        item = schedule[next_i]
        next_i += 1
        req, colors = _native_eval_request(
            item, cand_onnx, opp_onnx, frozen_onnx, simulations, opp_simulations=opp_simulations,
            lanes=mcts_lanes, virtual_loss=virtual_loss, max_turns=max_turns, max_actions=max_actions,
            mcts_opponent=mcts_opponent, opp_mcts_opponent=opp_mcts_opponent,
            cand_dyn_onnx=cand_dyn_onnx, opp_dyn_onnx=opp_dyn_onnx,
            vp_margin_alpha=vp_margin_alpha, opp_vp_margin_alpha=opp_vp_margin_alpha,
        )
        meta_by_gid[req['game_id']] = (item, colors)
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
                evs = msg.get('events') or ()
                progress_actions += len(evs)
                for ev in evs:
                    m = ev.get('metrics')
                    if m:
                        progress_decisions += 1
                        progress_sims += m.get('simulations', 0) or 0
                now = time.perf_counter()
                if now - last_progress >= 10.0:
                    el = now - started
                    print(
                        f'  [progress {el:5.1f}s] {len(results)}/{len(schedule)} completed | '
                        f'{progress_actions} actions ({progress_actions / el:.1f}/s) | '
                        f'{progress_decisions} dec ({progress_decisions / el:.1f}/s) | '
                        f'{progress_sims} sims ({progress_sims / el:.0f} sims/s) | '
                        f'{in_flight} in-flight',
                        flush=True,
                    )
                    last_progress = now
            elif mtype == 'done':
                item, colors = meta_by_gid.pop(msg['game_id'])
                winner_idx = msg['winner']
                winner_color = colors[winner_idx] if winner_idx >= 0 else None
                cand_won = (winner_color == item['candidate_color']) if winner_color is not None else None
                res = {
                    'index': item['index'],
                    'pair_id': item['pair']['pair_id'],
                    'candidate_color': item['candidate_color'],
                    'winner': winner_color,
                    'candidate_won': cand_won,
                    'status': msg['status'],
                    'turns': msg['turns'],
                    'actions': msg['actions'],
                    'elapsed_seconds': msg['elapsed_seconds'],
                }
                results.append(res)
                in_flight -= 1
                outcome = 'WIN' if cand_won is True else ('LOSS' if cand_won is False else 'CUTOFF')
                print(
                    f"  [{len(results):>{len(str(len(schedule)))}}/{len(schedule)}] "
                    f"{res['pair_id']} ({res['candidate_color']}): {outcome} in {res['turns']} turns ({res['elapsed_seconds']:.1f}s)",
                    flush=True,
                )
                if next_i < len(schedule):
                    dispatch_one()
                elif in_flight == 0 and not stdin_closed:
                    proc.stdin.close()
                    stdin_closed = True
            elif mtype == 'pool_metrics':
                break
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()

    results.sort(key=lambda r: r['index'])
    return results


class NativeBridgePool:
    """Thread-safe client for `catan-native-worker --bridge-pool` (CoreML/CUDA GPU vs JSettlers)."""

    def __init__(self, binary, workers, backend, coreml_batch, wait_us, gpu_streams, fixed_batch=None):
        self.log = (NATIVE / 'target/worker-stderr.log').open('a')
        if fixed_batch is None:
            fixed_batch = coreml_batch if backend == 'coreml' else 0
        cmd = [
            str(binary), '--bridge-pool', '--workers', str(workers),
            '--gpu-backend', backend,
            '--max-batch', str(coreml_batch), '--fixed-batch', str(fixed_batch),
            '--wait-us', str(wait_us), '--gpu-streams', str(gpu_streams),
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=runtime_environment(),
        )
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending = {}
        self._seq = 0
        self.pool_metrics = {}
        ready = json.loads(self.proc.stdout.readline())
        require(ready.get('type') == 'ready', f'Unexpected bridge-pool ready: {ready}')

        def _reader():
            for line in self.proc.stdout:
                msg = json.loads(line)
                mtype = msg.get('type')
                if mtype == 'reply':
                    rid = msg['request_id']
                    with self._pending_lock:
                        q = self._pending.pop(rid, None)
                    if q is not None:
                        q.put(msg)
                elif mtype == 'pool_metrics':
                    self.pool_metrics = msg.get('metrics', {})
            with self._pending_lock:
                for q in self._pending.values():
                    q.put({'ok': False, 'error': 'Bridge pool exited unexpectedly'})
                self._pending.clear()

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def call(self, **payload):
        q = queue.Queue(maxsize=1)
        with self._write_lock:
            self._seq += 1
            rid = self._seq
            with self._pending_lock:
                self._pending[rid] = q
            payload['request_id'] = rid
            self.proc.stdin.write(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
            self.proc.stdin.flush()
        reply = q.get(timeout=180)
        require(reply.get('ok'), f'Native bridge-pool error: {reply.get("error")}')
        return reply

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except BaseException:
            self.proc.kill()
            self.proc.wait()
        self._reader_thread.join(timeout=2)
        self.proc.stdout.close()
        self.log.close()


def run_native_jsettlers_coreml(
    schedule, candidate_ref, frozen_ref, simulations, workers,
    coreml_batch, wait_us, gpu_streams, mcts_lanes, virtual_loss,
    max_turns, max_actions, gpu_backend='coreml', mcts_opponent='self-policy',
    dtype='float16', cuda_fixed_batch=False,
    vp_margin_alpha=0.03,
):
    """Run Model (native Rust + batched CoreML/CUDA GPU) vs SMART JSettlers."""
    binary = ensure_binary()
    use_static = (gpu_backend == 'coreml') or cuda_fixed_batch
    eff_dtype = 'float32' if gpu_backend == 'coreml' else dtype
    fixed_batch = coreml_batch if use_static else 0
    if use_static:
        cand_onnx = export_checkpoint_coreml_static(candidate_ref, batch_size=coreml_batch, dtype=eff_dtype)
        cand_dyn_onnx = export_checkpoint_dynamic(candidate_ref, dtype=eff_dtype) if gpu_backend != 'coreml' else None
    else:
        cand_onnx = export_checkpoint_dynamic(candidate_ref, dtype=eff_dtype)
        cand_dyn_onnx = None
    frozen_onnx = export_checkpoint(frozen_ref)
    n_workers = min(workers, len(schedule))
    bridge = NativeBridgePool(binary, n_workers, gpu_backend, coreml_batch, wait_us, gpu_streams, fixed_batch=fixed_batch)
    results = []
    res_lock = threading.Lock()

    def play_one(item):
        pair = item['pair']
        cand_color = Color[item['candidate_color']]
        opp_color = Color.RED if cand_color == Color.BLUE else Color.BLUE
        game = benchmark.make_game(pair, friendly_robber=True)
        codec = Codec(game)
        meta = job_meta(game)
        state = list(game.state.random.getstate()[1])
        colors = [c.value for c in game.state.colors]
        learner_seat = colors.index(cand_color.value)
        gid = f'js-{item["index"]:05d}'
        task = {
            'job': meta,
            'start': meta['initial'],
            'start_rng': state,
            'logs': [],
            'id': gid,
            'prefix': 0,
            'seat_seeds': [benchmark.named_seed(pair['agent_seed'], c) for c in colors],
            'rng_state': state,
        }
        bridge.call(
            op='init', game_id=gid, task=task, learner_seat=learner_seat,
            model=str(cand_onnx),
            dynamic_model=(str(cand_dyn_onnx) if cand_dyn_onnx is not None else None),
            frozen=str(frozen_onnx),
            simulations=simulations, lanes=mcts_lanes, virtual_loss=virtual_loss, eval_mode=True,
            mcts_opponent=mcts_opponent, vp_margin_alpha=vp_margin_alpha,
        )
        jsettlers = benchmark.Agent('JSETTLERS', opp_color, benchmark.named_seed(pair['agent_seed'], 'JSETTLERS'))
        started = time.perf_counter()
        actions = 0
        try:
            while game.winning_color() is None and game.state.num_turns < max_turns and actions < max_actions:
                actor = game.state.current_color()
                legal = list(game.playable_actions)
                if len(legal) == 1:
                    action = legal[0]
                elif actor == cand_color:
                    reply = bridge.call(op='choose', game_id=gid)
                    aid = reply['action_id']
                    action = next((a for a in legal if action_id(a) == aid), None)
                    require(action is not None, f'Illegal native CoreML action {aid}')
                else:
                    action = jsettlers.select(
                        game, legal, benchmark.DecisionContext(time.perf_counter(), 500, 'policy', None)
                    )
                    jsettlers.cold = False
                result = game.execute(action)
                bridge.call(op='step', game_id=gid, action_id=action_id(action), chance=codec.chance(action, result))
                actions += 1
        finally:
            try:
                bridge.call(op='close', game_id=gid)
            except Exception:
                pass

        winner = game.winning_color()
        res = {
            'index': item['index'],
            'pair_id': pair['pair_id'],
            'candidate_color': cand_color.value,
            'winner': winner.value if winner is not None else None,
            'candidate_won': (winner == cand_color) if winner is not None else None,
            'status': 'completed' if winner is not None else 'cutoff',
            'turns': game.state.num_turns,
            'actions': actions,
            'elapsed_seconds': time.perf_counter() - started,
        }
        with res_lock:
            results.append(res)
            done_n = len(results)
            wins_n = sum(1 for r in results if r['candidate_won'] is True)
            decided_n = sum(1 for r in results if r['candidate_won'] is not None)
        outcome = 'WIN' if res['candidate_won'] is True else ('LOSS' if res['candidate_won'] is False else 'CUTOFF')
        wr = f'{wins_n}/{decided_n} ({100.0*wins_n/decided_n:.1f}%)' if decided_n else '0/0'
        print(
            f"  [{done_n:>{len(str(len(schedule)))}}/{len(schedule)}] "
            f"{res['pair_id']} ({res['candidate_color']}): {outcome} in {res['turns']} turns "
            f"({res['elapsed_seconds']:.1f}s) | running win rate: {wr}",
            flush=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(play_one, item) for item in schedule]
            for f in futs:
                f.result()
    finally:
        bridge.close()
        BRIDGE.close()

    pm = bridge.pool_metrics
    if pm.get('batches'):
        print(
            f"CoreML bridge-pool stats: {pm['requests']} evals in {pm['batches']} batches "
            f"(mean_batch={pm['rows']/pm['batches']:.1f}, forward={pm['forward_seconds']:.1f}s, streams={pm.get('streams', gpu_streams)})",
            flush=True,
        )
    results.sort(key=lambda r: r['index'])
    return results


def summarize_and_publish(results, elapsed, args, candidate_ref, opponent_ref, opponent_kind, output_dir):
    total = len(results)
    wins = sum(1 for r in results if r['candidate_won'] is True)
    losses = sum(1 for r in results if r['candidate_won'] is False)
    cutoffs = sum(1 for r in results if r['candidate_won'] is None)
    decided = wins + losses
    win_rate = (wins / decided) if decided > 0 else 0.0
    match_score = (wins + 0.5 * cutoffs) / total if total > 0 else 0.0
    ci95 = wilson_ci(wins, decided)

    blue_games = [r for r in results if r['candidate_color'] == 'BLUE']
    red_games = [r for r in results if r['candidate_color'] == 'RED']
    blue_wins = sum(1 for r in blue_games if r['candidate_won'] is True)
    red_wins = sum(1 for r in red_games if r['candidate_won'] is True)
    avg_turns = sum(r['turns'] for r in results) / total if total > 0 else 0.0

    opp_label = 'JSettlers (SMART)' if opponent_kind == 'jsettlers' else opponent_ref['path']
    cand_mcts_opp = getattr(args, 'mcts_opponent', 'self-policy')
    opp_mcts_opp = getattr(args, 'opponent_mcts_opponent', 'self-policy')
    cand_alpha = getattr(args, 'vp_margin_alpha', 0.03)
    opp_alpha = getattr(args, 'opponent_vp_margin_alpha', cand_alpha)
    mode_label = 'policy-only' if args.policy_only else f'{args.simulations} simulations ({cand_mcts_opp}, alpha={cand_alpha})'
    opp_mode_label = (
        'JSettlers'
        if opponent_kind == 'jsettlers'
        else ('policy-only' if (args.opponent_simulations == 0) else f'{args.opponent_simulations} simulations ({opp_mcts_opp}, alpha={opp_alpha})')
    )

    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'candidate': candidate_ref,
        'opponent_kind': opponent_kind,
        'opponent': opponent_ref,
        'candidate_mode': mode_label,
        'opponent_mode': opp_mode_label,
        'mcts_opponent': cand_mcts_opp,
        'opponent_mcts_opponent': opp_mcts_opp,
        'vp_margin_alpha': cand_alpha,
        'opponent_vp_margin_alpha': opp_alpha,
        'backend': args.backend,
        'gpu_dtype': ('float32' if args.backend in ('cpu', 'coreml') or args.fp32 else 'float16'),
        'batch_size': args.coreml_batch if args.backend in ('coreml', 'cuda') else args.max_batch,
        'fixed_batch': (args.coreml_batch if (args.backend == 'coreml' or (args.backend == 'cuda' and args.cuda_fixed_batch)) else 0),
        'workers': args.workers,
        'paired_seats': not args.no_paired,
        'seed': args.seed,
        'games': total,
        'wins': wins,
        'losses': losses,
        'cutoffs': cutoffs,
        'win_rate_decided': win_rate,
        'match_score': match_score,
        'win_rate_wilson_95ci': ci95,
        'blue_record': {'wins': blue_wins, 'games': len(blue_games)},
        'red_record': {'wins': red_wins, 'games': len(red_games)},
        'average_turns': avg_turns,
        'elapsed_seconds': elapsed,
        'games_per_minute': (total / elapsed * 60.0) if elapsed > 0 else 0.0,
        'per_game': results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write(output_dir / 'summary.json', summary)

    ci_str = f'[{ci95[0]*100:.1f}%, {ci95[1]*100:.1f}%]' if ci95[0] is not None else 'N/A'
    md = '\n'.join([
        '# Matchup Benchmark Results',
        '',
        f'- **Candidate**: `{candidate_ref["path"]}` ({mode_label})',
        f'- **Opponent**: `{opp_label}` ({opp_mode_label})',
        f'- **Games**: {total} ({"paired BLUE/RED" if not args.no_paired else "unpaired"}, seed `{args.seed}`, workers `{args.workers}`, backend `{args.backend}`)',
        f'- **Elapsed**: {elapsed:.1f}s ({summary["games_per_minute"]:.1f} games/min, avg {avg_turns:.1f} turns/game)',
        '',
        '| Wins | Losses | Cutoffs | Win Rate (decided) | 95% Wilson CI | Match Score | As BLUE | As RED |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
        f'| {wins} | {losses} | {cutoffs} | **{win_rate*100:.1f}%** | {ci_str} | {match_score*100:.1f}% | {blue_wins}/{len(blue_games)} | {red_wins}/{len(red_games)} |',
        '',
    ])
    (output_dir / 'REPORT.md').write_text(md)

    print('\n=== Matchup Benchmark Summary ===')
    print(f'Candidate : {candidate_ref["path"]} ({mode_label})')
    print(f'Opponent  : {opp_label} ({opp_mode_label})')
    print(f'Record    : {wins}W - {losses}L - {cutoffs}C  ({total} games in {elapsed:.1f}s, {summary["games_per_minute"]:.1f} games/min)')
    print(f'Win Rate  : {win_rate*100:.1f}% (95% Wilson CI {ci_str}) | Match Score: {match_score*100:.1f}%')
    print(f'Seat Split: BLUE {blue_wins}/{len(blue_games)} | RED {red_wins}/{len(red_games)} | Avg Turns: {avg_turns:.1f}')
    print(f'Saved     : {output_dir / "summary.json"} and {output_dir / "REPORT.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=str, default='cnn', help='Candidate checkpoint path or preset (cnn, transformer, latest-cnn, latest-transformer)')
    parser.add_argument('--opponent', type=str, default='jsettlers', help='"jsettlers", preset (cnn, transformer, latest-cnn, latest-transformer), or checkpoint path')
    parser.add_argument('--games', type=int, default=20, help='Total number of games to play')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--policy-only', action='store_true', help='Pure greedy policy (0 MCTS simulations)')
    mode.add_argument('--simulations', '--candidate-simulations', dest='simulations', type=int, default=64, help='MCTS simulation budget per searched decision for the candidate (default: 64)')
    parser.add_argument('--opponent-simulations', type=int, default=None, help='Separate simulation count for the opponent model (default: matches --simulations; 0 = policy-only)')
    parser.add_argument('--backend', choices=('cpu', 'mps', 'coreml', 'cuda', 'cuda-socket'), default='cpu', help='Inference backend (coreml/mps on macOS; cuda/cuda-socket on NVIDIA GPU)')
    parser.add_argument('--max-batch', type=int, default=64)
    parser.add_argument('--coreml-batch', '--batch-size', dest='coreml_batch', type=int, default=32, help='GPU batch size for coreml (static) or cuda (max dynamic, or static with --fixed-batch)')
    parser.add_argument('--cuda-fixed-batch', '--fixed-batch', dest='cuda_fixed_batch', action='store_true', help='Use static padded batches for CUDA EP (pairs with --batch-size / --coreml-batch)')
    parser.add_argument('--wait-us', type=int, default=100)
    parser.add_argument('--gpu-streams', type=int, default=2)
    parser.add_argument('--mcts-lanes', type=int, default=1)
    parser.add_argument('--virtual-loss', type=float, default=1.0)
    parser.add_argument('--mcts-opponent', choices=('self-policy', 'adversarial-puct', 'frozen-mlp'), default='self-policy', help='Candidate opponent model inside MCTS simulations (default: self-policy)')
    parser.add_argument('--opponent-mcts-opponent', choices=('self-policy', 'adversarial-puct', 'frozen-mlp'), default='self-policy', help='Opponent model MCTS opponent in model-vs-model matchups (default: self-policy)')
    parser.add_argument('--adversarial-puct', action='store_true', help='Shorthand for --mcts-opponent adversarial-puct')
    parser.add_argument('--opponent-adversarial-puct', action='store_true', help='Shorthand for --opponent-mcts-opponent adversarial-puct')
    parser.add_argument('--vp-margin-alpha', type=float, default=0.03, help='Candidate weight on (2p-1)^2-gated VP-margin utility in MCTS (default: 0.03; 0.0 disables)')
    parser.add_argument('--opponent-vp-margin-alpha', type=float, default=None, help='Opponent VP-margin alpha in model-vs-model matchups (default: matches --vp-margin-alpha)')
    parser.add_argument('--python-search', action='store_true', help='Force Python SearchSession instead of the native Rust worker')
    parser.add_argument('--seed', type=int, default=2026111955)
    parser.add_argument('--no-paired', action='store_true', help='Disable paired BLUE/RED seats on identical boards')
    parser.add_argument('--max-turns', type=int, default=1000)
    parser.add_argument('--max-actions', type=int, default=10000)
    parser.add_argument('--fp32', action='store_true', help='Use float32 instead of default float16 on CUDA/PyTorch GPU inference')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    gpu_dtype = 'float32' if args.fp32 else 'float16'

    if args.adversarial_puct:
        args.mcts_opponent = 'adversarial-puct'
    if args.opponent_adversarial_puct:
        args.opponent_mcts_opponent = 'adversarial-puct'
    if args.opponent_vp_margin_alpha is None:
        args.opponent_vp_margin_alpha = args.vp_margin_alpha

    require(args.games > 0 and args.workers > 0, '--games and --workers must be positive')
    require(0.0 <= args.vp_margin_alpha <= 0.5, '--vp-margin-alpha must be between 0.0 and 0.5')
    require(0.0 <= args.opponent_vp_margin_alpha <= 0.5, '--opponent-vp-margin-alpha must be between 0.0 and 0.5')
    simulations = 0 if args.policy_only else args.simulations
    require(simulations >= 0, '--simulations must be >= 0')
    if args.opponent_simulations is None:
        args.opponent_simulations = simulations
    require(args.opponent_simulations >= 0, '--opponent-simulations must be >= 0')

    candidate_ref = resolve_checkpoint(args.candidate)
    opponent_kind = 'jsettlers' if args.opponent.lower() == 'jsettlers' else 'model'
    opponent_ref = None if opponent_kind == 'jsettlers' else resolve_checkpoint(args.opponent)
    frozen_ref = resolve_frozen_opponent(candidate_ref, args.mcts_opponent, args.opponent_mcts_opponent)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_dir = args.output or (ROOT / f'runs/catan-ai/matchup-benchmark/{stamp}')
    schedule = build_schedule(args.games, args.seed, paired=not args.no_paired)

    opp_sim_label = 'JSettlers (SMART)' if opponent_kind == 'jsettlers' else (
        f'{opponent_ref["path"]} ({"policy-only" if args.opponent_simulations == 0 else f"{args.opponent_simulations} sims, {args.opponent_mcts_opponent}, alpha={args.opponent_vp_margin_alpha}"})'
    )
    print(f'Candidate: {candidate_ref["path"]} ({"policy-only" if simulations == 0 else f"{simulations} sims, {args.mcts_opponent}, alpha={args.vp_margin_alpha}"})', flush=True)
    print(f'Opponent : {opp_sim_label}', flush=True)
    print(f'Running {args.games} games with {args.workers} workers (backend={args.backend})...', flush=True)

    started = time.perf_counter()
    with benchmark.exclusive_benchmark():
        if opponent_kind == 'jsettlers' and simulations > 0 and args.backend in ('coreml', 'cuda') and not args.python_search:
            results = run_native_jsettlers_coreml(
                schedule, candidate_ref, frozen_ref, simulations, args.workers,
                args.coreml_batch, args.wait_us, args.gpu_streams, args.mcts_lanes,
                args.virtual_loss, args.max_turns, args.max_actions,
                gpu_backend=args.backend, mcts_opponent=args.mcts_opponent,
                dtype=gpu_dtype, cuda_fixed_batch=args.cuda_fixed_batch,
                vp_margin_alpha=args.vp_margin_alpha,
            )
        elif (
            opponent_kind == 'model'
            and simulations > 0
            and args.opponent_simulations > 0
            and not args.python_search
        ):
            results = run_native_model_matchup(
                schedule, candidate_ref, opponent_ref, frozen_ref, simulations, args.opponent_simulations,
                args.workers, args.backend, args.max_batch, args.coreml_batch,
                args.wait_us, args.gpu_streams, args.mcts_lanes,
                args.virtual_loss, args.max_turns, args.max_actions,
                mcts_opponent=args.mcts_opponent,
                opp_mcts_opponent=args.opponent_mcts_opponent,
                dtype=gpu_dtype, cuda_fixed_batch=args.cuda_fixed_batch,
                vp_margin_alpha=args.vp_margin_alpha,
                opp_vp_margin_alpha=args.opponent_vp_margin_alpha,
            )
        else:
            if args.backend in ('cuda', 'cuda-socket') and torch.cuda.is_available():
                py_device = 'cuda'
            elif args.backend == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                py_device = 'mps'
            else:
                py_device = 'cpu'
            results = run_python_matchup(
                schedule, candidate_ref, opponent_ref, opponent_kind, frozen_ref,
                simulations, args.opponent_simulations, args.workers, py_device,
                args.max_turns, args.max_actions,
                mcts_opponent=args.mcts_opponent,
                opp_mcts_opponent=args.opponent_mcts_opponent,
                vp_margin_alpha=args.vp_margin_alpha,
                opp_vp_margin_alpha=args.opponent_vp_margin_alpha,
            )
    elapsed = time.perf_counter() - started
    summarize_and_publish(results, elapsed, args, candidate_ref, opponent_ref, opponent_kind, output_dir)


if __name__ == '__main__':
    main()
