"""Single-thread high-worker and CPU/MPS inference diagnostics; no training."""
import json
from pathlib import Path
import random
import time

import torch
from catanatron.serialization import web_view
from agent_apis import benchmark
from agent_apis.experiment import ROOT, EXPERIMENTS, file_digest
from agent_apis.observation import from_web_view, require
from agent_apis.neural import worker_benchmark as wb
from agent_apis.neural.pipeline import write
from agent_apis.neural.provider import NeuralCheckpoint

OUTPUT = ROOT / 'runs/catan-ai/high-worker-benchmark'


def gpu_benchmark(ref, jobs):
    views = []
    for job in jobs:
        game, _, _ = wb.position(job)
        actor = game.state.current_color()
        views.append(from_web_view(web_view(game, actor), actor.value))
    rows = []
    for device in ('cpu', 'mps'):
        model = NeuralCheckpoint(ROOT / ref['path'], device=device)
        torch.set_num_threads(1)
        for batch_size in (1, 8, 32, 64):
            batch = [views[i % len(views)] for i in range(batch_size)]
            for _ in range(3):
                model.evaluate_batch(batch)
            samples = []
            for _ in range(3):
                if device == 'mps':
                    torch.mps.synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    predictions = model.evaluate_batch(batch)
                if device == 'mps':
                    torch.mps.synchronize()
                samples.append((time.perf_counter() - started) / 10)
            seconds = sorted(samples)[1]
            rows.append({'device': device, 'batch_size': batch_size,
                         'seconds_per_batch_median': seconds,
                         'positions_per_second': batch_size / seconds,
                         'predictions': predictions})
        del model
    parity = []
    for batch_size in (1, 8, 32, 64):
        cpu, gpu = [r for r in rows if r['batch_size'] == batch_size]
        value_error = max(abs(a[1] - b[1]) for a, b in zip(cpu['predictions'], gpu['predictions']))
        policy_error = max(abs(a[0][key] - b[0][key]) for a, b in zip(cpu['predictions'], gpu['predictions']) for key in a[0])
        require(value_error < .001 and policy_error < .001, 'CPU/MPS prediction mismatch')
        parity.append({'batch_size': batch_size, 'max_value_error': value_error, 'max_policy_error': policy_error})
    for row in rows:
        del row['predictions']
    result = {'checkpoint': ref, 'torch': torch.__version__, 'cpu_threads': 1,
              'scope': 'Full evaluate_batch path including observation encoding, tensor transfer, legal normalization, and result transfer; 24 matched views, three warmups and three sets of ten timed calls; median set reported. Diagnostic, not multiworker shared-GPU throughput.',
              'measurements': rows, 'parity': parity}
    write(OUTPUT / 'gpu-inference.json', result)
    print('GPU inference diagnostic completed', flush=True)


def main():
    require(json.loads((wb.OUTPUT / 'progress.json').read_text())['status'] == 'completed', 'Original sweep must finish first')
    OUTPUT.mkdir(exist_ok=True)
    original = json.loads((wb.OUTPUT / 'protocol.json').read_text())
    ref, frozen = original['checkpoint'], original['frozen']
    base_jobs = original['jobs']
    jobs = [{**job, 'id': f'cycle-{cycle}-' + job['id']} for cycle in range(4) for job in base_jobs]
    protocol = json.loads((EXPERIMENTS / 'fresh-xdcnn-visible.json').read_text())
    identity = {'source_sha256': file_digest(Path(__file__)), 'original_benchmark': wb.selfplay.reference(wb.OUTPUT / 'protocol.json'),
                'workers': [8, 10, 12, 14, 16], 'cpu_threads_per_worker': 1, 'repeats': 2,
                'workload': 'Four repetitions of the same 24 matched workloads per case; repeated seeds intentionally fixed; never training data'}
    path = OUTPUT / 'protocol.json'
    if path.exists():
        require(json.loads(path.read_text()) == identity, 'Extension protocol changed')
    else:
        write(path, identity)
    wb.OUTPUT = OUTPUT
    order = [8, 10, 12, 14, 16]
    random.Random(1803).shuffle(order)
    cases = [(w, r) for r, order_ in ((1, order), (2, list(reversed(order)))) for w in order_]
    with benchmark.exclusive_benchmark():
        results = []
        for i, (workers, repeat) in enumerate(cases):
            write(OUTPUT / 'progress.json', {'status': 'running', 'case': i+1, 'total_cases': len(cases), 'workers': workers, 'repeat': repeat})
            path = OUTPUT / f'repeat-{repeat}-workers-{workers}-threads-1/result.json'
            r = json.loads(path.read_text()) if path.exists() else wb.one_case(workers, 1, jobs, protocol, ref, frozen, repeat)
            results.append(r)
            print(f'{i+1}/{len(cases)}: workers={workers}: {r["search_decisions_per_second"]:.2f} searches/s, RSS={r["peak_sampled_worker_and_child_rss_bytes"]/1e9:.2f}GB', flush=True)
        summary = []
        for workers in sorted(order):
            selected = [r for r in results if r['workers'] == workers]
            seconds = sum(r['measured_seconds'] for r in selected)
            summary.append({'workers': workers, 'threads': 1, 'search_decisions_per_second': sum(r['search_decisions'] for r in selected)/seconds,
                            'peak_rss_gb': max(r['peak_sampled_worker_and_child_rss_bytes'] for r in selected)/1e9,
                            'replays_passed': all(r['all_replay_passed'] for r in selected),
                            'decision_p95_seconds_max': max(r['decision_p95_seconds'] for r in selected)})
        write(OUTPUT / 'summary.json', {'configurations': summary})
        gpu_benchmark(ref, base_jobs)
        write(OUTPUT / 'progress.json', {'status': 'completed', 'completed_cases': len(cases), 'total_cases': len(cases)})


if __name__ == '__main__':
    main()
