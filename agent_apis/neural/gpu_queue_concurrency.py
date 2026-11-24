"""Check whether more in-flight games improve the selected shared GPU queue."""
import json
from pathlib import Path
from agent_apis import benchmark
from agent_apis.experiment import ROOT, EXPERIMENTS, file_digest
from agent_apis.observation import require
from agent_apis.neural import shared_gpu_benchmark as shared
from agent_apis.neural.pipeline import write


def main():
    folder = shared.OUTPUT
    require(json.loads((folder/'progress.json').read_text())['status']=='completed', 'Finish fixed-worker test first')
    original = json.loads((ROOT/'runs/catan-ai/game-worker-benchmark/protocol.json').read_text())
    fixed = json.loads((folder/'summary.json').read_text())
    gpu = fixed['confirmation'][1]['config']
    cpu = shared.configurations()[0]
    ref, frozen = original['checkpoint'], original['frozen']
    jobs = [{**job, 'id':f'cycle-{cycle}-'+job['id']} for cycle in range(4) for job in original['jobs']]
    protocol = json.loads((EXPERIMENTS/'fresh-xdcnn-visible.json').read_text())
    identity = {'source_sha256':file_digest(Path(__file__)),
                'fixed_test_protocol': shared.wb.selfplay.reference(folder/'protocol.json'),
                'gpu_config':gpu,'screen_workers':[16,24,32], 'additional_control':'24 workers, no explicit batching delay', 'jobs':len(jobs),
                'confirmation':'best screened GPU worker count vs local CPU at 12 workers, two alternating passes'}
    path=folder/'concurrency-protocol.json'
    if path.exists(): require(json.loads(path.read_text())==identity,'Concurrency protocol changed')
    else: write(path,identity)
    with benchmark.exclusive_benchmark():
        screen=[]
        zero = {**gpu, 'name':'gpu-cnn-local-mlp-0ms', 'wait_seconds':0.}
        for candidate,workers in [(gpu,16),(gpu,24),(gpu,32),(zero,24)]:
            write(folder/'concurrency-progress.json',{'status':'screening','workers':workers,'completed':len(screen),'total':4})
            r=shared.case(candidate,jobs,protocol,ref,frozen,1,workers=workers,phase=f'concurrency-screen-w{workers}')
            screen.append(r)
            print(f'{candidate["name"]} workers {workers}: {r["search_decisions_per_second"]:.2f} searches/s',flush=True)
        selected=max(screen,key=lambda r:r['search_decisions_per_second'])
        winner,gpu=selected['workers'],selected['config']
        confirm=[]
        for repeat in (1,2):
            order=[(cpu,12),(gpu,winner)] if repeat==1 else [(gpu,winner),(cpu,12)]
            for config,workers in order:
                write(folder/'concurrency-progress.json',{'status':'confirmation','config':config['name'],'workers':workers,'repeat':repeat,'completed':len(confirm),'total':4})
                r=shared.case(config,jobs,protocol,ref,frozen,repeat,workers=workers,phase=f'concurrency-confirm-w{workers}')
                confirm.append(r)
                print(f'confirm {repeat} {config["name"]} workers {workers}: {r["search_decisions_per_second"]:.2f} searches/s',flush=True)
        summary=[]
        for config,workers in ((cpu,12),(gpu,winner)):
            rows=[r for r in confirm if r['config']['name']==config['name']]
            summary.append({'config':config,'workers':workers,
                            'search_decisions_per_second':sum(r['search_decisions'] for r in rows)/sum(r['measured_seconds'] for r in rows),
                            'per_pass_searches_per_second':[r['search_decisions_per_second'] for r in rows],
                            'peak_rss_gb':max(r['peak_sampled_worker_service_child_rss_gb'] for r in rows),
                            'replays_passed':all(r['all_replay_passed'] for r in rows)})
        write(folder/'concurrency-summary.json',{'screen':[{k:r[k] for k in ('workers','config','search_decisions_per_second')} for r in screen], 'confirmation':summary})
        write(folder/'concurrency-progress.json',{'status':'completed','screen_cases':4,'confirmation_cases':4})


if __name__=='__main__':main()
