"""Live shared-service checkpoint routing and batched numerical parity audit."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import torch
from catanatron.serialization import web_view
from agent_apis.experiment import ROOT
from agent_apis.observation import from_web_view, require
from agent_apis.neural import shared_gpu_benchmark as shared
from agent_apis.neural.pipeline import write
from agent_apis.neural.shared_inference import RemoteCheckpoint, serve


def main():
    original=json.loads((ROOT/'runs/catan-ai/game-worker-benchmark/protocol.json').read_text())
    refs={'candidate':original['checkpoint'],'opponent':original['frozen']}
    views=[]
    for job in original['jobs'][:8]:
        game,_,_=shared.wb.position(job)
        actor=game.state.current_color()
        views.append(from_web_view(web_view(game,actor),actor.value))
    expected={}
    for key,ref in refs.items():
        model=shared.wb.selfplay.load_checkpoint(ref)
        torch.set_num_threads(1)
        expected[key]=model.evaluate_batch(views)
    ctx=mp.get_context('spawn')
    requests,control=ctx.Queue(),ctx.Queue()
    responses=[ctx.Queue() for _ in range(16)]
    service=ctx.Process(target=serve,args=(refs,'mps',requests,responses,control,4,.002))
    service.start()
    try:
        kind,error=control.get(timeout=120)
        require(kind=='ready',str(error))
        # Each request contains 8 rows, exceeding the 4-row forward limit;
        # concurrent mixed-checkpoint requests exercise chunking and routing.
        def evaluate(index):
            key=('candidate','opponent')[index%2]
            client=RemoteCheckpoint(key,index,requests,responses[index])
            actual=client.evaluate_batch(views)
            value_error=max(abs(a[1]-b[1]) for a,b in zip(actual,expected[key]))
            policy_error=max(abs(a[0][k]-b[0][k]) for a,b in zip(actual,expected[key]) for k in a[0])
            require(value_error<.001 and policy_error<.001,'Shared-service numerical parity failed')
            return {'client':index,'checkpoint':key,'rows':len(actual),'max_value_error':value_error,'max_policy_error':policy_error}
        with ThreadPoolExecutor(max_workers=16) as pool:
            rows=list(pool.map(evaluate,range(16)))
        requests.put(None)
        kind,metrics=control.get(timeout=60)
        require(kind=='metrics',str(metrics))
        service.join(timeout=30)
        require(service.exitcode==0,'Shared-service parity process failed')
        for m in metrics.values():
            require(max(map(int,m['batch_rows_histogram']))<=4,'Service exceeded batch row cap')
        write(shared.OUTPUT/'parity.json',{'status':'passed','concurrent_clients':16,'rows':sum(r['rows'] for r in rows),
                                         'forward_row_cap':4,'results':rows,'service_metrics':metrics})
        print('Shared-service parity passed: 16 mixed-checkpoint clients, 128 rows, chunk cap 4',flush=True)
    finally:
        if service.is_alive():
            service.terminate();service.join(timeout=10)
        for queue in (requests,control,*responses):queue.close()


if __name__=='__main__':main()
