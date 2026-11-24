"""End-to-end shared inference queue versus local CPU self-play diagnostic."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import torch
from catanatron.serialization import web_view
from agent_apis import benchmark
from agent_apis.experiment import ROOT, EXPERIMENTS, file_digest, digest
from agent_apis.observation import from_web_view, require
from agent_apis.neural import worker_benchmark as wb
from agent_apis.neural.pipeline import write
from agent_apis.neural.shared_inference import RemoteCheckpoint, serve

OUTPUT = ROOT / 'runs/catan-ai/shared-gpu-benchmark'


class TimedCheckpoint:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.calls = self.rows = 0
        self.wait_seconds = 0.

    def evaluate_batch(self, views):
        started = time.perf_counter()
        result = self.checkpoint.evaluate_batch(views)
        self.wait_seconds += time.perf_counter() - started
        self.calls += 1
        self.rows += len(views)
        return result

    def evaluate(self, view):
        return self.evaluate_batch([view])[0]

    def probabilities(self, view):
        if len(view['current_playable_actions']) == 1:
            return self.checkpoint.probabilities(view)
        return self.evaluate(view)[0]


def initialize(ref, frozen, config, barrier, pids, warmup, ids, requests, responses):
    if config['device'] == 'local':
        wb.initialize(ref, frozen, 1, barrier, pids, warmup)
        wb.MODEL = TimedCheckpoint(wb.MODEL)
        wb.OPPONENT = TimedCheckpoint(wb.OPPONENT)
        return
    torch.set_num_interop_threads(1)
    worker_id = ids.get(timeout=60)
    wb.MODEL = RemoteCheckpoint('candidate', worker_id, requests, responses[worker_id])
    if config.get('local_opponent'):
        wb.OPPONENT = TimedCheckpoint(wb.selfplay.load_checkpoint(frozen))
    else:
        wb.OPPONENT = RemoteCheckpoint('opponent', worker_id, requests, responses[worker_id])
    torch.set_num_threads(1)
    wb.THREADS = 1
    game, _, _ = wb.position(warmup)
    actor = game.state.current_color()
    view = from_web_view(web_view(game, actor), actor.value)
    wb.MODEL.evaluate(view)
    wb.OPPONENT.evaluate(view)
    pids.put(os.getpid())
    barrier.wait(timeout=120)


def run_job(job, protocol, ref, folder):
    models = {'candidate': wb.MODEL, 'opponent': wb.OPPONENT}
    before = {key: (m.calls, m.rows, m.wait_seconds) for key, m in models.items()}
    result = wb.run_job(job, protocol, ref, folder)
    result['inference'] = {key: {'calls': m.calls - before[key][0], 'rows': m.rows - before[key][1],
                               'blocking_seconds': m.wait_seconds - before[key][2]}
                           for key, m in models.items()}
    return result


def case(config, jobs, protocol, ref, frozen, repeat, workers=12, phase='screen'):
    folder = OUTPUT / f'{phase}-repeat-{repeat}-{config["name"]}'
    result_path = folder / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context('spawn')
    barrier, pids, ids = ctx.Barrier(workers + 1), ctx.Queue(), ctx.Queue()
    for i in range(workers):
        ids.put(i)
    requests, control = ctx.Queue(), ctx.Queue()
    responses = [ctx.Queue() for _ in range(workers)]
    service = None
    started = time.perf_counter()
    results, peak = [], 0
    try:
        if config['device'] != 'local':
            refs = {'candidate': ref}
            if not config.get('local_opponent'):
                refs['opponent'] = frozen
            service = ctx.Process(target=serve, args=(refs, config['device'], requests, responses, control,
                                                       64, config['wait_seconds']))
            service.start()
            kind, detail = control.get(timeout=120)
            require(kind == 'ready', f'Inference startup failed: {detail}')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=initialize,
                initargs=(ref, frozen, config, barrier, pids, jobs[0], ids, requests, responses)) as pool:
            boots = [pool.submit(wb.ready) for _ in range(workers)]
            barrier.wait(timeout=120)
            roots = [pids.get(timeout=10) for _ in range(workers)]
            if service:
                roots.append(service.pid)
            for future in boots:
                future.result()
            startup = time.perf_counter() - started
            measured = time.perf_counter()
            pending = {pool.submit(run_job, job, protocol, ref, str(folder.relative_to(ROOT))) for job in jobs}
            while pending:
                done, pending = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                peak = max(peak, wb.process_memory(roots))
                if service:
                    require(service.is_alive(), 'Shared inference service exited unexpectedly')
                for future in done:
                    results.append(future.result())
            elapsed = time.perf_counter() - measured
        service_metrics = None
        if service:
            requests.put(None)
            kind, service_metrics = control.get(timeout=60)
            require(kind == 'metrics', f'Inference service failed: {service_metrics}')
            service.join(timeout=30)
            require(service.exitcode == 0, 'Inference service did not exit cleanly')
        decisions = sum(r['searched_decisions'] for r in results)
        times = [s for r in results for s in r['decision_seconds']]
        result = {'config': config, 'phase': phase, 'repeat': repeat, 'workers': workers,
                  'cpu_threads_per_worker': 1, 'jobs': len(jobs), 'startup_seconds': startup,
                  'measured_seconds': elapsed, 'search_decisions': decisions,
                  'simulations': sum(r['simulations'] for r in results),
                  'search_decisions_per_second': decisions / elapsed,
                  'decision_p50_seconds': benchmark.quantile(times, .5),
                  'decision_p95_seconds': benchmark.quantile(times, .95),
                  'peak_sampled_worker_service_child_rss_gb': peak / 1e9,
                  'all_replay_passed': all(r['replay_passed'] for r in results),
                  'trace_sha256': digest([(r['job'], r['final_sha256']) for r in sorted(results, key=lambda r:r['job'])]),
                  'service_metrics_including_warmup': service_metrics, 'results': results}
        write(result_path, result)
        return result
    finally:
        if service and service.is_alive():
            service.terminate()
            service.join(timeout=10)
        for queue in [requests, control, ids, pids, *responses]:
            queue.close()


def configurations():
    return [
        {'name': 'local-cpu', 'device': 'local', 'wait_seconds': 0.},
        {'name': 'shared-cpu-0.5ms', 'device': 'cpu', 'wait_seconds': .0005},
        {'name': 'gpu-both-0ms', 'device': 'mps', 'wait_seconds': 0.},
        {'name': 'gpu-both-0.5ms', 'device': 'mps', 'wait_seconds': .0005},
        {'name': 'gpu-both-2ms', 'device': 'mps', 'wait_seconds': .002},
        {'name': 'gpu-cnn-local-mlp-0.5ms', 'device': 'mps', 'wait_seconds': .0005, 'local_opponent': True},
        {'name': 'gpu-cnn-local-mlp-2ms', 'device': 'mps', 'wait_seconds': .002, 'local_opponent': True},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    original = json.loads((wb.OUTPUT / 'protocol.json').read_text())
    ref, frozen, base_jobs = original['checkpoint'], original['frozen'], original['jobs']
    protocol = json.loads((EXPERIMENTS / 'fresh-xdcnn-visible.json').read_text())
    with benchmark.exclusive_benchmark():
        if args.smoke:
            for config in (configurations()[0], configurations()[3], configurations()[5]):
                result = case(config, base_jobs[:2], protocol, ref, frozen, 0, workers=2, phase='smoke')
                print(json.dumps({k:result[k] for k in ('config','search_decisions_per_second','all_replay_passed')}), flush=True)
            return
        identity = {'source_sha256': {str(p): file_digest(p) for p in (Path(__file__), ROOT/'agent_apis/neural/shared_inference.py')},
                    'original_protocol': wb.selfplay.reference(wb.OUTPUT/'protocol.json'),
                    'configurations': configurations(), 'workers': 12, 'cpu_threads': 1,
                    'screen': '24 matched continuations per config, four searched decisions each; fixed balanced order',
                    'confirmation': 'best screened GPU config versus local CPU, four repetitions of 24 workloads, three alternating passes',
                    'purpose': 'diagnostics only; training stays paused'}
        path = OUTPUT/'protocol.json'
        if path.exists():
            require(json.loads(path.read_text()) == identity, 'Shared benchmark configuration changed')
        else:
            write(path, identity)
        screen = []
        configs = configurations()
        for index in (0, 3, 5, 1, 4, 6, 2):
            config = configs[index]
            write(OUTPUT/'progress.json', {'status':'screening', 'config':config['name'], 'completed':len(screen), 'total':len(configs)})
            result = case(config, base_jobs, protocol, ref, frozen, 1)
            screen.append(result)
            print(f'screen {config["name"]}: {result["search_decisions_per_second"]:.2f} searches/s', flush=True)
        winner = max((r for r in screen if r['config']['device']=='mps'), key=lambda r:r['search_decisions_per_second'])['config']
        jobs = [{**job, 'id':f'cycle-{cycle}-'+job['id']} for cycle in range(4) for job in base_jobs]
        confirmation = []
        for repeat in (1, 2, 3):
            order = [configs[0], winner] if repeat % 2 else [winner, configs[0]]
            for config in order:
                write(OUTPUT/'progress.json', {'status':'confirmation','config':config['name'],'repeat':repeat,'completed':len(confirmation),'total':6})
                result = case(config, jobs, protocol, ref, frozen, repeat, phase='confirm')
                confirmation.append(result)
                print(f'confirm {repeat} {config["name"]}: {result["search_decisions_per_second"]:.2f} searches/s', flush=True)
        summary=[]
        for config in (configs[0],winner):
            rows=[r for r in confirmation if r['config']['name']==config['name']]
            summary.append({'config':config,'search_decisions_per_second':sum(r['search_decisions'] for r in rows)/sum(r['measured_seconds'] for r in rows),
                            'per_pass_searches_per_second':[r['search_decisions_per_second'] for r in rows],
                            'peak_rss_gb':max(r['peak_sampled_worker_service_child_rss_gb'] for r in rows),
                            'replays_passed':all(r['all_replay_passed'] for r in rows)})
        write(OUTPUT/'summary.json',{'status':'completed','screen':[{k:r[k] for k in ('config','search_decisions_per_second','all_replay_passed')} for r in screen],
                                     'confirmation':summary})
        write(OUTPUT/'progress.json',{'status':'completed','screen_cases':len(screen),'confirmation_cases':len(confirmation)})


if __name__ == '__main__':
    main()
