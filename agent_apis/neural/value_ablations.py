"""Controlled 2000-update W4 Transformer ablations on one complete expert cohort."""
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import random
import signal
import time
import zipfile
import numpy as np
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT,file_digest,digest
from agent_apis.neural import fresh_cohorts as fresh,compact_iterations as compact,selfplay
from agent_apis.neural.pipeline import write
from agent_apis.neural.train import evaluate,lr_at
from agent_apis.neural.value_ablation_models import VARIANTS,build_variant,objective,EDGE_TYPES,connector_graph
from agent_apis.observation import require

OUT=ROOT/'runs/catan-ai/transformer-value-ablations'
STOP=False


def request_stop(*args):
    global STOP
    STOP=True


def sample_data(refs,protocol):
    chunks=[]
    for index,ref in enumerate(refs):
        require(not STOP,'Stop requested')
        p=ROOT/ref['path'];require(file_digest(p)==ref['sha256'],'Journal hash changed')
        with gzip.open(p,'rt') as stream:r=json.load(stream)
        require(r['status']=='completed','Incomplete game')
        chunks.append(fresh.derive(r,protocol['positions_per_game'],benchmark.named_seed(protocol['seed'],r['pair']['pair_id'])))
        if (index+1)%200==0:print('DERIVE',index+1,'/',len(refs),flush=True)
    return compact.MemoryData(chunks,refs)


def save(path,record):
    temporary=path.with_suffix('.tmp');torch.save(record,temporary);temporary.replace(path)


def metrics(model,data):
    result=evaluate(model,data,'validation','mps',batch_size=128)
    return result


def run_variant(variant,initial,meta,data,validation,holdout,train_probe,settings,identity,report):
    folder=OUT/variant;folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    model=build_variant(initial,variant).to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    seed=benchmark.named_seed(identity['seed'],'W4-training');rng=np.random.default_rng(seed)
    aug=torch.Generator().manual_seed(benchmark.named_seed(identity['seed'],'W4-augmentation'))
    torch.manual_seed(benchmark.named_seed(identity['seed'],'W4-torch'));torch.mps.manual_seed(benchmark.named_seed(identity['seed'],'W4-torch'))
    start=0;history=[];elapsed=0.;best=float('inf');best_step=0
    resume=folder/'resume.pt'
    if resume.exists():
        r=torch.load(resume,map_location='cpu',weights_only=True);require(r['identity']==identity,'Resume identity mismatch')
        model.load_state_dict(r['state_dict']);optimizer.load_state_dict(r['optimizer']);rng.bit_generator.state=r['rng'];aug.set_state(r['aug'])
        torch.random.set_rng_state(r['torch_rng']);torch.mps.set_rng_state(r['mps_rng'])
        start=r['step'];history=r['validation'];elapsed=r['elapsed_seconds'];best=r['best'];best_step=r['best_step']
    else:
        baseline=metrics(model,validation);history=[{'step':0,**baseline}];best=baseline['selection_loss']
        save(folder/'best.pt',{'experiment_format_version':1,'variant':variant,'state_dict':model.state_dict(),'step':0,'parent':identity['parent'],'metadata':meta,'identity':identity})
    began=time.perf_counter()
    def checkpoint(step):
        save(resume,{'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,'aug':aug.get_state(),
            'torch_rng':torch.random.get_rng_state(),'mps_rng':torch.mps.get_rng_state(),'step':step,'validation':history,'best':best,'best_step':best_step,
            'elapsed_seconds':elapsed+time.perf_counter()-began,'identity':identity})
    if not resume.exists():checkpoint(0)
    for step in range(start+1,2001):
        if STOP:checkpoint(step-1);raise compact.ReviewDue()
        if step%100==0:
            require(all(file_digest(ROOT/p)==sha for p,sha in identity['source_sha256'].items()),'Ablation source changed')
        old=data.sample(rng,128);new=data.sample(rng,128)
        batch={k:np.concatenate((old[k],new[k])) for k in ('features','legal','policy','value','value_weight')}
        raw,legal,targets,outcomes,vw=[torch.from_numpy(batch[k]).to('mps') for k in ('features','legal','policy','value','value_weight')]
        sym=torch.randint(12,(256,),generator=aug).to('mps');inverse=model.inputs.action_inverse[sym]
        legal=legal.gather(1,inverse);targets=targets.gather(1,inverse);model.train()
        for group in optimizer.param_groups:group['lr']=lr_at(step,settings)
        optimizer.zero_grad(set_to_none=True);logits,values=model(raw,sym)
        pl,vl,penalty=objective(logits,values,legal,targets,outcomes,vw,variant)
        (pl+vl+penalty).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%250==0:
            result=metrics(model,validation);history.append({'step':step,**result})
            if result['selection_loss']<best:
                best=result['selection_loss'];best_step=step
                save(folder/'best.pt',{'experiment_format_version':1,'variant':variant,'state_dict':model.state_dict(),'step':step,'parent':identity['parent'],'metadata':meta,'identity':identity})
            print('VALIDATE',variant,step,'loss',round(result['selection_loss'],5),'policy',round(result['policy_nll'],5),'value',round(result['value_bce'],5),flush=True)
        if step%100==0:
            checkpoint(step)
            report.update(stage='training',variant=variant,step=step,best_step=best_step,best_validation_loss=best,updated_at=datetime.now(timezone.utc).isoformat())
            write(OUT/'progress.json',report)
    save(folder/'final.pt',{'experiment_format_version':1,'variant':variant,'state_dict':model.state_dict(),'step':2000,'parent':identity['parent'],'metadata':meta,'identity':identity})
    final_train=metrics(model,train_probe)
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(selected['state_dict'])
    held=metrics(model,holdout);selected_train=metrics(model,train_probe)
    result={'variant':variant,'updates':2000,'best_step':best_step,'validation':history,'best_validation':min(history,key=lambda x:x['selection_loss']),
        'holdout_cohort7':held,'sampled_train_selected':selected_train,'sampled_train_final':final_train,'elapsed_seconds':elapsed+time.perf_counter()-began,
        'parameters':sum(p.numel() for p in model.parameters()),'best_checkpoint':selfplay.reference(folder/'best.pt'),'final_checkpoint':selfplay.reference(folder/'final.pt'),
        'identity_sha256':digest(identity)}
    write(folder/'result.json',result);model.to('cpu');del model,optimizer;torch.mps.empty_cache()
    return result


def main():
    OUT.mkdir(exist_ok=True);signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(torch.__version__=='2.15.0' and torch.backends.mps.is_available(),'Pinned MPS required')
    parent=json.loads((ROOT/'runs/catan-ai/transformer-expert-catchup-visible/cohort-008/iteration-1/training.json').read_text())['parent']
    cp=selfplay.load_checkpoint(parent);initial={k:v.detach().cpu().clone() for k,v in cp.model.state_dict().items()};meta=deepcopy(cp.metadata);del cp;torch.set_num_threads(1)
    source=ROOT/'runs/catan-ai/jsettlers-expert-cohorts-visible/cohort-008/expert/manifest.json'
    manifest=json.loads(source.read_text());protocol=json.loads((ROOT/'runs/catan-ai/jsettlers-expert-cohorts-visible/progress.json').read_text())['protocol']
    from agent_apis.neural.transformer_catchup import validate_manifest
    groups=validate_manifest(manifest)
    paths=('agent_apis/neural/value_ablations.py','agent_apis/neural/value_ablation_models.py','agent_apis/neural/models.py','agent_apis/neural/friendly_w4.py',
        'agent_apis/neural/compact_iterations.py','agent_apis/neural/fresh_cohorts.py','agent_apis/neural/topology.py','agent_apis/neural/train.py')
    types,gap=connector_graph()
    identity={'parent':parent,'source_sha256':{p:file_digest(ROOT/p) for p in paths},'manifest':selfplay.reference(source),
        'seed':protocol['seed'],'updates':2000,'batch_size':256,'validation_interval':250,'variants':list(VARIANTS),
        'weighting':'W4, correction once on outcome BCE; logit penalty estimates uniform-position regularization with same single sample weight',
        'sparse_attention':{'edge_types':list(EDGE_TYPES),'connectors':4,'expander_degree':4,'expander_spectral_gap':gap,'static_density':float((types!=0).float().mean()),
            'implementation':'dense masked SDPA over sparse interaction pattern; no sparse-kernel speed claim'}}
    if (OUT/'identity.json').exists():require(json.loads((OUT/'identity.json').read_text())==identity,'Experiment changed')
    else:
        write(OUT/'identity.json',identity)
        with zipfile.ZipFile(OUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(ROOT/p,p)
    report={'status':'running','pid':os.getpid(),'stage':'derive-data','results':{},'started_at':datetime.now(timezone.utc).isoformat()}
    write(OUT/'progress.json',report)
    try:
        data=sample_data(groups['train'],protocol);validation=sample_data(groups['validation'],protocol)
        train_refs=random.Random(20261118).sample(groups['train'],100)
        train_probe=sample_data(train_refs,protocol)
        hm=json.loads((ROOT/'runs/catan-ai/jsettlers-expert-cohorts-visible/cohort-007/expert/manifest.json').read_text())
        holdout=sample_data(validate_manifest(hm)['validation'],protocol)
        settings=deepcopy(meta['configuration']['training']);settings.update(max_updates=2000,warmup_updates=100)
        for variant in VARIANTS:
            report['results'][variant]=run_variant(variant,initial,meta,data,validation,holdout,train_probe,settings,identity,report)
            write(OUT/'progress.json',report)
        report.update(status='completed',stage='completed',pid=None)
        write(OUT/'results.json',report['results'])
    except compact.ReviewDue:report.update(status='paused',pid=None)
    except BaseException as e:
        report.update(status='needs-investigation',pid=None,error=f'{type(e).__name__}: {e}');raise
    finally:write(OUT/'progress.json',report)

if __name__=='__main__':main()
