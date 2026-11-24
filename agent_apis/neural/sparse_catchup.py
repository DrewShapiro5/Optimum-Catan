"""Sparse typed Transformer catch-up with light value-logit regularization."""
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import json,os,signal,time,zipfile
from pathlib import Path
import numpy as np
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT,file_digest,digest
from agent_apis.neural import selfplay,compact_iterations as compact,transformer_catchup as catchup
from agent_apis.neural.friendly_w4 import losses
from agent_apis.neural.train import evaluate,lr_at,save_checkpoint
from agent_apis.neural.value_ablation_models import TypedExphormer
from agent_apis.neural.pipeline import write
from agent_apis.observation import require

OUTPUT=ROOT/'runs/catan-ai/sparse-transformer-expert-catchup-visible'
CORPUS=None

class SparseCatchup(catchup.Catchup):
    def __init__(self):
        OUTPUT.mkdir(parents=True,exist_ok=True);self.state_path=OUTPUT/'progress.json'
        cnn=json.loads((catchup.CNN/'progress.json').read_text())
        self.protocol=deepcopy(cnn['protocol'])
        self.core={p:file_digest(ROOT/p) for p in cnn['source_sha256']}
        for p in ('agent_apis/neural/sparse_catchup.py','agent_apis/neural/transformer_catchup.py','agent_apis/neural/value_ablation_models.py'):
            self.core[p]=file_digest(ROOT/p)
        self.deadline=datetime.fromisoformat(cnn['review_at']).timestamp();require(self.deadline>time.time(),'Window ended')
        self.stop_requested=False
        prior=json.loads((ROOT/'runs/catan-ai/transformer-value-ablations/typed-exphormer/result.json').read_text())['best_checkpoint']
        if self.state_path.exists():
            self.state=json.loads(self.state_path.read_text());require(self.state['source_sha256']==self.core,'Sparse sources changed')
        else:
            self.state={'status':'initializing','source_sha256':self.core,'initial_parent':prior,'cohorts':{},'protocol':self.protocol,
                'value_logit_penalty':.001,'installed_default_changed':False,'started_at':self.now(),
                'route':'selected sparse checkpoint; replay original expert then every complete expert cohort in order',
                'prior_exposure':'initial sparse checkpoint already adapted on cohort 8; catch-up revisits those games, not novel data'}
            with zipfile.ZipFile(OUTPUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in self.core:z.write(ROOT/p,p)
        self.parent=self.state['initial_parent'];self.load_sparse(self.parent)
        require(torch.__version__=='2.15.0' and torch.backends.mps.is_available(),'Pinned MPS required')
        signal.signal(signal.SIGTERM,self.request_stop);signal.signal(signal.SIGINT,self.request_stop)
        self.state.update(status='running',pid=os.getpid(),review_at=cnn['review_at']);self.state.pop('error',None);self.persist()

    def load_sparse(self,ref):
        require(file_digest(ROOT/ref['path'])==ref['sha256'],'Sparse checkpoint changed')
        saved=torch.load(ROOT/ref['path'],map_location='cpu',weights_only=True)
        if saved.get('experiment_format_version')==1:
            require(saved['variant']=='typed-exphormer','Wrong ablation checkpoint')
        else:
            require(saved['metadata']['model_id']=='TYPED-EXPHORMER' and saved['metadata']['format_version']==2,'Wrong sparse checkpoint')
            require(saved['metadata']['sparse_implementation_sha256']==file_digest(ROOT/'agent_apis/neural/value_ablation_models.py'),'Sparse architecture changed')
        model=TypedExphormer();model.load_state_dict(saved['state_dict']);model.eval()
        meta=deepcopy(saved['metadata']);meta.update(model_id='TYPED-EXPHORMER',format_version=2,
            sparse_implementation_sha256=file_digest(ROOT/'agent_apis/neural/value_ablation_models.py'),value_logit_penalty=.001)
        meta['configuration']['model']['id']='TYPED-EXPHORMER'
        return SimpleNamespace(model=model,metadata=meta)

    def execute(self):
        # Catchup.execute sets the original trainer module globals. Mirror them
        # locally at each training call instead, leaving that module untouched.
        return super().execute()

    def train_w4(self, expert, new):
        from agent_apis.neural import friendly_w4 as w4
        corpus=w4.CORPUS
        folder=w4.OUTPUT/'iteration-1';folder.mkdir(parents=True,exist_ok=True);path=folder/'training.json'
        if path.exists():
            r=json.loads(path.read_text());self.load_sparse(r['checkpoint']);return r
        self.state['stage']='train-W4';self.persist()
        checkpoint=self.load_sparse(self.parent);model=checkpoint.model.to('mps');device='mps'
        config=deepcopy(checkpoint.metadata['configuration']);config.update(phase=9,stage='friendly-W4-selfplay-continuation')
        config['rules']['friendly_robber']=True;config['data'].update(weighting='W4',source_schema=3,source_features=681,corpora=[str(corpus.relative_to(ROOT))],
            selfplay={'sampling_and_datasets_sha256':digest(new.summary),'expert_fraction':.5},
            soft_target_W4='joint action-family weighting, exactly equal to existing W4 for one-hot demonstrations')
        settings=config['training'];settings.update(max_updates=self.protocol['updates_per_iteration'],warmup_updates=100)
        meta=deepcopy(checkpoint.metadata);meta.update(configuration=config,configuration_sha256=digest(config),rules=config['rules'],
            source_sha256=digest(self.core),parent_checkpoint=self.parent,training_recipe='D2-W4',dataset_sha256=file_digest(corpus/'manifest.json'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        rng=np.random.default_rng(benchmark.named_seed(self.protocol['seed'],'W4-training'))
        aug=torch.Generator().manual_seed(benchmark.named_seed(self.protocol['seed'],'W4-augmentation'))
        torch.manual_seed(benchmark.named_seed(self.protocol['seed'],'W4-torch'))
        data_sha=digest({'expert':expert.summary,'new':new.summary,'validation':self.cache.arrays['value'].tolist(),
            'manifest':file_digest(corpus/'manifest.json'),'new_data':config['data']['selfplay']})
        step=0;history=[];seconds=0.;resume=folder/'resume.pt'
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=True)
            require(saved['core']==self.core and saved['data_sha']==data_sha,'W4 resume data changed')
            model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
            rng.bit_generator.state=saved['sampling_rng'];aug.set_state(saved['augmentation_rng']);torch.random.set_rng_state(saved['torch_rng']);torch.mps.set_rng_state(saved['mps_rng'])
            step,history,seconds=saved['step'],saved['validation'],saved['elapsed_seconds']
        else:
            history=[{'step':0,**evaluate(model,self.cache,'validation',device)}]
            if self.protocol.get('select_best_validation'):
                meta['step']=0;save_checkpoint(folder/'best.pt',model,meta)
        started=time.perf_counter()
        def save():
            torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'validation':history,
                'sampling_rng':rng.bit_generator.state,'augmentation_rng':aug.get_state(),'torch_rng':torch.random.get_rng_state(),
                'mps_rng':torch.mps.get_rng_state(),'core':self.core,'data_sha':data_sha,'elapsed_seconds':seconds+time.perf_counter()-started},folder/'resume.tmp')
            (folder/'resume.tmp').replace(resume)
        # Save only complete optimizer-update boundaries. On a mid-update
        # failure, keep the last committed checkpoint and its matching RNGs.
        if not resume.exists():save()
        try:
            while step<settings['max_updates']:
                if step%100==0:self.guard()
                old=expert.sample(rng,128);fresh=new.sample(rng,128)
                batch={k:np.concatenate((old[k],fresh[k])) for k in ('features','legal','policy','value','value_weight')}
                raw,legal,targets,outcomes,vw=[torch.from_numpy(batch[k]).to(device) for k in ('features','legal','policy','value','value_weight')]
                sym=torch.randint(12,(256,),generator=aug).to(device);inverse=model.inputs.action_inverse[sym]
                legal=legal.gather(1,inverse);targets=targets.gather(1,inverse);model.train()
                for group in optimizer.param_groups:group['lr']=lr_at(step+1,settings)
                optimizer.zero_grad(set_to_none=True);logits,values=model(raw,sym)
                pl,vl=losses(logits,values,legal,targets,outcomes,vw);(pl+vl+.001*(values.square()*vw).mean()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();step+=1
                if step%250==0 or step==settings['max_updates']:
                    validation=evaluate(model,self.cache,'validation',device)
                    if self.protocol.get('select_best_validation') and validation['selection_loss']<min(v['selection_loss'] for v in history):
                        meta['step']=step;save_checkpoint(folder/'best.pt',model,meta)
                    history.append({'step':step,**validation})
                if step%self.protocol.get('save_interval',100)==0 or step==settings['max_updates']:
                    save();self.state['training_updates']=step;self.persist();print(f'W4 TRAIN {step}/{settings["max_updates"]}',flush=True)
        except BaseException:raise
        meta['step']=step;save_checkpoint(folder/'final.pt',model,meta);model.to('cpu')
        selected=folder/('best.pt' if self.protocol.get('select_best_validation') else 'final.pt')
        result={'checkpoint':selfplay.reference(selected),'final_checkpoint':selfplay.reference(folder/'final.pt'),'updates':step,'validation':history,
            'elapsed_seconds':seconds+time.perf_counter()-started,'expert_sampling':expert.summary,'selfplay_sampling':new.summary,
            'weighting':'W4','value_correction':'1/(N*q), once, independently within each source','parent':self.parent}
        write(path,result);return result


def main():
    # Own output tree, with collection shared read-only with the CNN.
    catchup.OUTPUT=OUTPUT
    work=SparseCatchup()
    try:work.execute()
    except compact.ReviewDue:work.state.update(status='paused',stage='window-ended-or-user-paused')
    except BaseException as e:
        work.state.update(status='needs-investigation',error=f'{type(e).__name__}: {e}');raise
    finally:work.state['pid']=None;work.persist()

if __name__=='__main__':main()
