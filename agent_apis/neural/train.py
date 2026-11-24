"""Resumable D2/W0 PyTorch training with full validation checkpoint selection."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import importlib.metadata
import numpy as np
import torch
from torch.nn import functional as F

from agent_apis.benchmark import exclusive_benchmark,named_seed,quantile
from agent_apis.experiment import (ROOT,EXPERIMENTS,source_snapshot,corpus_snapshot,
                                  file_digest,digest,stream_seeds,validate_record)
from agent_apis.observation import require
from agent_apis.neural.data import Cache,DecisionShuffle,build_cache
from agent_apis.neural.models import build_model,MODEL_IDS,losses
from agent_apis.neural.topology import topology


def synchronize(device):
    if str(device)=='mps':torch.mps.synchronize()
    elif str(device).startswith('cuda'):torch.cuda.synchronize()


def lr_at(step,settings):
    initial=settings['learning_rate'];warm=settings['warmup_updates'];maximum=settings['max_updates']
    if step<=warm:return initial*step/warm
    fraction=(step-warm)/(maximum-warm)
    low=settings['minimum_learning_rate']
    return low+.5*(initial-low)*(1+math.cos(math.pi*min(1,fraction)))


def tensors(batch,device):
    return (torch.from_numpy(batch['features']).to(device),torch.from_numpy(batch['legal']).to(device),
            torch.from_numpy(batch['action']).to(device),torch.from_numpy(batch['value']).to(device))


@torch.inference_mode()
def evaluate(model,cache,split,device,batch_size=256):
    model.eval();indices=cache.indices(split)
    totals=Counter();phase={};family={};calibration=[Counter() for _ in range(10)]
    for start in range(0,len(indices),batch_size):
        batch=cache.batch(indices[start:start+batch_size]);raw,legal,targets,outcomes=tensors(batch,device)
        policy,value=model(raw)
        logp=F.log_softmax(policy.masked_fill(~legal,float('-inf')),dim=1)
        nll=-logp.gather(1,targets[:,None]).squeeze(1).cpu().numpy()
        predicted=logp.argmax(1).cpu().numpy();prob=value.sigmoid().cpu().numpy()
        bce=F.binary_cross_entropy_with_logits(value,outcomes,reduction='none').cpu().numpy()
        eligible=batch['legal'].sum(1)>1;truth=batch['value']
        totals['rows']+=len(batch['value']);totals['nonforced']+=int(eligible.sum())
        totals['policy_nll']+=float(nll[eligible].sum());totals['correct']+=int((predicted[eligible]==batch['action'][eligible]).sum())
        totals['value_bce']+=float(bce.sum());totals['brier']+=float(np.square(prob-truth).sum())
        for destination,name in ((phase,'phase'),(family,'family')):
            for label in np.unique(batch[name]):
                select=(batch[name]==label)&eligible;c=destination.setdefault(int(label),Counter())
                c['count']+=int(select.sum());c['nll']+=float(nll[select].sum())
                c['correct']+=int((predicted[select]==batch['action'][select]).sum())
        bins=np.minimum((prob*10).astype(int),9)
        for i in range(10):
            choose=bins==i;calibration[i]['count']+=int(choose.sum())
            calibration[i]['predicted']+=float(prob[choose].sum());calibration[i]['outcome']+=float(truth[choose].sum())
    n=totals['rows'];p=totals['nonforced']
    def grouped(values,labels):
        return {labels[i]:{'count':c['count'],'nll':c['nll']/c['count'] if c['count'] else None,
                          'top1_accuracy':c['correct']/c['count'] if c['count'] else None} for i,c in values.items()}
    result={'rows':n,'nonforced':p,'policy_nll':totals['policy_nll']/p,
            'top1_accuracy':totals['correct']/p,'value_bce':totals['value_bce']/n,'brier':totals['brier']/n,
            'by_phase':grouped(phase,['setup','middle','endgame']),
            'by_family':grouped(family,cache.manifest['families']),
            'calibration':[{'low':i/10,'high':(i+1)/10,'count':c['count'],
                'mean_probability':c['predicted']/c['count'] if c['count'] else None,
                'win_frequency':c['outcome']/c['count'] if c['count'] else None} for i,c in enumerate(calibration)]}
    result['selection_loss']=result['policy_nll']+result['value_bce']
    model.train();return result


def metadata(config,cache,source,step):
    return {'format_version':1,'model_id':config['model']['id'],'training_seed':config['seed'],'step':step,
            'configuration':config,'configuration_sha256':digest(config),
            'feature_schema_version':3,'feature_schema_sha256':file_digest(EXPERIMENTS/'feature-schema-v3.json'),
            'tensor_schema':topology()['version'],'topology_sha256':topology()['sha256'],
            'model_implementation_sha256':file_digest(ROOT/'agent_apis/neural/models.py'),
            'cache_identity_sha256':cache.manifest['identity_sha256'],
            'dataset_sha256':file_digest(cache.folder/'manifest.json'),'source_sha256':source['sha256'],
            'rules':config['rules'],'value':'acting-perspective sigmoid win probability',
            'action_count':332,'inference_device':config['runtime']['evaluation_device']}


def save_checkpoint(path,model,meta,extra=None):
    temporary=path.with_suffix('.tmp')
    torch.save({'metadata':meta,'state_dict':model.state_dict(),**(extra or {})},temporary)
    temporary.replace(path)


def train(config,cache_path,output,resume=None):
    require(torch.__version__==config['runtime']['torch_version'],'PyTorch pin differs')
    from agent_apis.neural.recipes import RecipeSampler,validate_recipe,weighted_losses
    recipe=validate_recipe(config['data'])
    phase6=config.get('phase')==6
    require(phase6 or recipe=='D2-W0','Non-baseline recipes require a phase-6 configuration')
    audit=EXPERIMENTS/'phase5-augmentation-audit.json'
    require(file_digest(audit)==config['training']['augmentation_audit_sha256'] and
            json.loads(audit.read_text())['gate_passed'] and
            json.loads(audit.read_text())['topology_sha256']==topology()['sha256'],
            'Augmentation audit has not passed for this tensor schema')
    torch.set_num_threads(config['runtime']['cpu_threads'])
    seed=config['seed'];torch.manual_seed(stream_seeds(seed)['training'])
    device=config['runtime']['training_device']
    require(device!='mps' or torch.backends.mps.is_available(),'MPS is unavailable')
    cache=Cache(cache_path);model=build_model(config['model']['id']).to(device)
    settings=config['training'];optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    def make_sampler(state=None):
        if phase6:return RecipeSampler(cache,config['data'],named_seed(seed,'game-shuffle'),settings['shuffle_buffer'],state)
        return DecisionShuffle(cache.manifest['games'],named_seed(seed,'game-shuffle'),settings['shuffle_buffer'],state)
    sampler=make_sampler()
    augmentation=torch.Generator().manual_seed(named_seed(seed,'augmentation'))
    source=source_snapshot();datasets=corpus_snapshot(config)
    started=time.perf_counter();previous_seconds=0.;step=0;best_loss=math.inf;best_step=None;consumed=Counter();history=[];seen_games=set()
    game_offsets=np.asarray([g['offset'] for g in cache.manifest['games']])
    if resume:
        folder=Path(resume);state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
        require(state['metadata']['configuration_sha256']==digest(config),'Resume configuration differs')
        require(state['metadata']['dataset_sha256']==file_digest(cache.folder/'manifest.json'),'Resume dataset differs')
        original_source=json.loads((folder/'source.json').read_text())
        for path in ('agent_apis/neural/train.py','agent_apis/neural/models.py','agent_apis/neural/topology.py',
                     'agent_apis/neural/data.py','agent_apis/observation.py','experiments/catan-ai/feature-schema-v3.json')+(
                     ('agent_apis/neural/recipes.py',) if phase6 else ()):
            require(original_source['files'][path]==source['files'][path],f'Resume implementation changed: {path}')
        source=original_source
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        sampler=make_sampler(state['sampler'])
        augmentation.set_state(state['augmentation_rng']);torch.random.set_rng_state(state['torch_rng'])
        if device=='mps':torch.mps.set_rng_state(state['device_rng'])
        step=state['metadata']['step'];best_loss=state['best_loss'];best_step=state['best_step']
        consumed=Counter(state['consumed']);history=state['history'];previous_seconds=state['elapsed_seconds']
        seen_games=set(state['seen_games'])
        run_id=folder.name
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_id=f'phase{6 if phase6 else 5}-{config["model"]["id"]}-seed{seed}-{digest(config)[:12]}-{stamp}'
        folder=output/run_id;folder.mkdir(parents=True,exist_ok=False)
        (folder/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
        (folder/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    def save_resume():
        extra={'optimizer':optimizer.state_dict(),'sampler':sampler.state_dict(),'augmentation_rng':augmentation.get_state(),
               'torch_rng':torch.random.get_rng_state(),'device_rng':torch.mps.get_rng_state() if device=='mps' else torch.random.get_rng_state(),
               'best_loss':best_loss,'best_step':best_step,'consumed':dict(consumed),'history':history,
               'seen_games':sorted(seen_games),
               'elapsed_seconds':previous_seconds+time.perf_counter()-started}
        save_checkpoint(folder/'resume.pt',model,metadata(config,cache,source,step),extra)
    print(folder,flush=True)
    model.train();recent=torch.zeros(2,device=device);recent_steps=0
    try:
        with (folder/'progress.jsonl').open('a') as log:
            while step<settings['max_updates']:
                indices=sampler.take(settings['batch_size']);batch=cache.batch(indices)
                raw,legal,targets,outcomes=tensors(batch,device)
                symmetry=torch.randint(12,(len(raw),),generator=augmentation).to(device)
                targets,legal=model.inputs.augment_targets(targets,legal,symmetry)
                step+=1;lr=lr_at(step,settings)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                policy,value=model(raw,symmetry)
                if phase6:
                    pw,vw=sampler.weights(indices)
                    pl,vl=weighted_losses(policy,value,legal,targets,outcomes,
                        torch.from_numpy(pw).to(device),torch.from_numpy(vw).to(device))
                else:pl,vl=losses(policy,value,legal,targets,outcomes,validate=False)
                (pl+vl).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();recent+=torch.stack((pl.detach(),vl.detach()));recent_steps+=1
                for label,count in zip(*np.unique(batch['phase'],return_counts=True)):consumed['phase:'+['setup','middle','endgame'][int(label)]]+=int(count)
                for label,count in zip(*np.unique(batch['family'],return_counts=True)):consumed['family:'+cache.manifest['families'][int(label)]]+=int(count)
                consumed['decisions']+=len(indices)
                consumed['policy_nonforced_decisions']+=int((batch['legal'].sum(1)>1).sum())
                seen_games.update(int(i) for i in np.unique(np.searchsorted(game_offsets,indices,side='right')-1))
                if step%100==0:
                    p,v=(recent/recent_steps).cpu().tolist();synchronize(device)
                    require(math.isfinite(p+v),'Nonfinite training loss')
                    event={'step':step,'policy_loss':p,'value_loss':v,'learning_rate':lr,
                           'elapsed_seconds':previous_seconds+time.perf_counter()-started,'epoch':sampler.epoch}
                    log.write(json.dumps(event)+'\n');log.flush()
                    print(f'{config["model"]["id"]} seed {seed} step {step}/{settings["max_updates"]}: policy={p:.4f} value={v:.4f}',flush=True)
                    recent.zero_();recent_steps=0
                if step%settings['validation_interval']==0 or step==settings['max_updates']:
                    validation=evaluate(model,cache,'validation',device,settings['batch_size']);history.append({'step':step,**validation})
                    if validation['selection_loss']<best_loss:
                        best_loss=validation['selection_loss'];best_step=step
                        save_checkpoint(folder/'best.pt',model,metadata(config,cache,source,step))
                    save_resume()
                    print(f'Validation step {step}: loss={validation["selection_loss"]:.5f} top1={validation["top1_accuracy"]:.4f}; best={best_step}',flush=True)
    except BaseException:
        save_resume();raise
    save_checkpoint(folder/'final.pt',model,metadata(config,cache,source,step))
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(selected['state_dict'])
    heldout=evaluate(model,cache,'test',device,settings['batch_size'])
    elapsed=previous_seconds+time.perf_counter()-started
    record={'schema_version':1,'run_id':run_id,'status':'completed','created_at':datetime.now(timezone.utc).isoformat(),
            'configuration':{'content':config,'sha256':digest(config)},'seed':seed,'rng_seeds':stream_seeds(seed),
            'source':source,'datasets':datasets,'software':{'python':sys.version,'executable':sys.executable,
               'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))},
            'hardware':{'platform':platform.platform(),'machine':platform.machine(),'available_devices':['cpu']+(['mps'] if torch.backends.mps.is_available() else [])},
            'checkpoint':{'path':str((folder/'best.pt').relative_to(ROOT)),'sha256':file_digest(folder/'best.pt')},
            'metrics':{'training':{'updates':step,'parameters':sum(p.numel() for p in model.parameters()),
                'best_step':best_step,'best_validation_loss':best_loss,'validation_history':history,'test':heldout,
                'consumed':dict(consumed),'unique_games_consumed':len(seen_games),
                'sampling_distribution':sampler.summary if phase6 else {'recipe':'D2-W0'},
                'cache_manifest_sha256':file_digest(cache.folder/'manifest.json'),
                'final_checkpoint':{'path':str((folder/'final.pt').relative_to(ROOT)),'sha256':file_digest(folder/'final.pt')}}},
            'elapsed_seconds':elapsed}
    validate_record(record);(folder/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(folder/'run.json',flush=True);return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,default=ROOT/'runs/catan-ai')
    parser.add_argument('--resume',type=Path);args=parser.parse_args();config=json.loads(args.config.read_text())
    with exclusive_benchmark():train(config,args.cache,args.output,args.resume)


if __name__=='__main__':main()
