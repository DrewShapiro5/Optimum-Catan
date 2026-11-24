"""Serial device benchmarks for actual encoders, including input packing."""
from datetime import datetime,timezone
import json
import platform
import statistics
import subprocess
import time
import torch
from agent_apis.benchmark import exclusive_benchmark
from agent_apis.experiment import EXPERIMENTS,ROOT,file_digest,digest
from agent_apis.neural.models import build_model,MODEL_IDS,losses
from agent_apis.neural.train import synchronize
from agent_apis.neural.topology import topology


def measure():
    torch.set_num_threads(6)
    devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
    results={}
    for name in MODEL_IDS:
        results[name]={}
        for device in devices:
            torch.manual_seed(11);model=build_model(name).to(device)
            record={'parameters':sum(p.numel() for p in model.parameters())}
            for batch in (1,32):
                raw=torch.zeros(batch,681);samples=[]
                model.eval()
                with torch.inference_mode():
                    for i in range(35):
                        synchronize(device);start=time.perf_counter()
                        p,v=model(raw.to(device));p.softmax(-1).cpu();v.sigmoid().cpu();synchronize(device)
                        if i>=5:samples.append((time.perf_counter()-start)*1000)
                record[f'inference_batch_{batch}']={'median_ms':statistics.median(samples),
                    'p95_ms':sorted(samples)[int(.95*(len(samples)-1))],'samples_ms':samples}
            model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
            raw=torch.zeros(256,681,device=device);legal=torch.ones(256,332,device=device,dtype=torch.bool)
            target=torch.zeros(256,device=device,dtype=torch.long);outcome=torch.zeros(256,device=device)
            samples=[]
            for i in range(12):
                synchronize(device);start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                p,v=model(raw);loss=sum(losses(p,v,legal,target,outcome,validate=False));loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();synchronize(device)
                if i>=2:samples.append((time.perf_counter()-start)*1000)
            record['training_batch_256']={'median_ms':statistics.median(samples),'samples_ms':samples}
            results[name][device]=record
            print(name,device,'inference B1',record['inference_batch_1']['median_ms'],
                  'train B256',record['training_batch_256']['median_ms'],flush=True)
            del model,optimizer,raw
            if device=='mps':torch.mps.empty_cache()
    chosen={name:{'training_device':min(v,key=lambda d:v[d]['training_batch_256']['median_ms']),
                  'evaluation_device':min(v,key=lambda d:v[d]['inference_batch_1']['p95_ms'])}
            for name,v in results.items()}
    report={'created_at':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
            'hardware':{'platform':platform.platform(),'machine':platform.machine(),'cpu_threads':6,
                'cpu':subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],text=True,capture_output=True).stdout.strip()},
            'topology_sha256':topology()['sha256'],'method':'float32; real shared packing; inference includes input transfer, policy/value return; cold five excluded; training forward/backward/AdamW',
            'results':results,'selected':chosen}
    path=EXPERIMENTS/'phase5-devices.json';path.write_text(json.dumps(report,indent=2)+'\n')
    base=json.loads((EXPERIMENTS/'phase5-base.json').read_text());folder=EXPERIMENTS/'phase5-configs';folder.mkdir(exist_ok=True)
    for name in MODEL_IDS:
        for seed in (11,23,47):
            config=json.loads(json.dumps(base));config['model']={'id':name,'topology_sha256':topology()['sha256']}
            config['seed']=seed;config['runtime'].update(chosen[name]);config['runtime']['device_preflight_sha256']=file_digest(path)
            (folder/f'{name}-seed{seed}.json').write_text(json.dumps(config,indent=2)+'\n')
    print(path,flush=True)


if __name__=='__main__':
    with exclusive_benchmark():measure()
