"""User-authorized 1000-game self-play-only ablation against saved mixed finals."""
from copy import deepcopy
from datetime import datetime,timezone
import gzip,json,random,time,zipfile
from pathlib import Path
import numpy as np
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT,file_digest,digest
from agent_apis.neural import fresh_cohorts as fresh,compact_iterations as compact,friendly_w4 as w4,selfplay
from agent_apis.neural.sparse_catchup import SparseCatchup
from agent_apis.neural.train import evaluate as historical_evaluate
from agent_apis.neural.pipeline import write

OUT=ROOT/'runs/catan-ai/selfplay-only-comparison'
SOURCE=ROOT/'runs/catan-ai/mixed-native-visible'

def data(refs,seed):
 chunks=[]
 for ref in refs:
  path=ROOT/ref['path'];assert file_digest(path)==ref['sha256']
  with gzip.open(path,'rt') as stream:r=json.load(stream)
  chunks.append(fresh.derive(r,64,benchmark.named_seed(seed,r['pair']['pair_id'])))
 return compact.MemoryData(chunks,refs)

@torch.inference_mode()
def corrected_evaluate(model,cache,split,device,batch_size=128):
 r=historical_evaluate(model,cache,split,device,batch_size)
 model.eval();total=0.;count=0
 for start in range(0,len(cache.arrays['value']),batch_size):
  b=cache.batch(np.arange(start,min(start+batch_size,len(cache.arrays['value']))))
  raw=torch.from_numpy(b['features']).to(device);legal=torch.from_numpy(b['legal']).to(device);target=torch.from_numpy(b['policy']).to(device)
  logits,_=model(raw);lp=torch.log_softmax(logits.masked_fill(~legal,-1e9),1);eligible=legal.sum(1)>1
  total+=float((-(target*lp).sum(1))[eligible].sum().cpu());count+=int(eligible.sum())
 r['hard_policy_nll']=r['policy_nll'];r['policy_nll']=total/count;r['selection_loss']=r['policy_nll']+r['value_bce']
 r['policy_metric']='full target cross entropy';model.train();return r

def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 state=json.load(open(SOURCE/'progress.json'));assert state['status']=='paused';seed=state['protocol']['seed']
 manifest=json.load(open(SOURCE/'cohort-001/manifest.json'))
 refs={split:[r for r in rows if '/selfplay/' in r['path']] for split,rows in manifest['splits'].items()}
 assert {k:len(v) for k,v in refs.items()}=={'train':900,'validation':50,'test':50}
 sources={p:file_digest(ROOT/p) for p in state['source_sha256']}
 sources['agent_apis/neural/selfplay_only_comparison.py']=file_digest(Path(__file__))
 progress={'status':'running','stage':'derive-data','started_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sources,'results':{},
  'authorization':'User requested retraining on only 1000 self-play games; experimental exception to normal 2000-game cohort gate',
  'comparison':'equal 2000-update final checkpoints; same incoming weights, original games and splits; 900 train/50 validation/50 test self-play games'}
 write(OUT/'progress.json',progress)
 with zipfile.ZipFile(OUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sources:z.write(ROOT/p,p)
 # Both branches use the same mixed validation set for comparable selection.
 train=data(refs['train'],seed);validation=data(manifest['splits']['validation'],seed)
 expert_val=data([r for r in manifest['splits']['validation'] if '/selfplay/' not in r['path']],seed)
 sp_val=data(refs['validation'],seed)
 train_probe=data(random.Random(20261119).sample(refs['train'],50),seed)
 # Isolated-process evaluator change: production and historical records untouched.
 w4.evaluate=corrected_evaluate
 for name,klass in [('cnn',fresh.Fresh),('sparse',SparseCatchup)]:
  baseline=state['iterations']['1']['training'][name]
  folder=OUT/name;folder.mkdir(exist_ok=True);write(folder/'manifest.json',{'completed_games':1000,'splits':refs,'validation_selection_manifest':selfplay.reference(SOURCE/'cohort-001/manifest.json')})
  work=klass.__new__(klass);work.parent=baseline['parent'];work.protocol=dict(state['protocol']);work.core=sources;work.cache=validation;work.state={};work.deadline=time.time()+3*3600;work.stop_requested=False
  def persist(work=work,name=name):
   progress.update(stage='train-'+name,training_updates=work.state.get('training_updates',0),updated_at=datetime.now(timezone.utc).isoformat());write(OUT/'progress.json',progress)
  work.persist=persist;w4.OUTPUT=folder;w4.CORPUS=folder
  # Sparse trainer's imported evaluate is a separate module global.
  import agent_apis.neural.sparse_catchup as sparse_module
  sparse_module.evaluate=corrected_evaluate
  result=work.train_w4(train,train)
  progress['results'][name]={'training':result,'mixed_training':baseline,'metrics':{}}
  for label,ref in [('incoming',baseline['parent']),('mixed-final',baseline['final_checkpoint']),('selfplay-final',result['final_checkpoint']),('selfplay-selected',result['checkpoint'])]:
   cp=selfplay.load_checkpoint(ref) if name=='cnn' else work.load_sparse(ref)
   model=cp.model.to('mps');torch.set_num_threads(1);metrics={}
   for split,d in [('expert-validation',expert_val),('selfplay-validation',sp_val),('mixed-validation',validation),('selfplay-training-sample',train_probe)]:
    r=corrected_evaluate(model,d,split,'mps');metrics[split]={k:r[k] for k in ('policy_nll','hard_policy_nll','value_bce','selection_loss','brier','top1_accuracy')}
   progress['results'][name]['metrics'][label]=metrics;write(OUT/'progress.json',progress)
   model.to('cpu');del model,cp;torch.mps.empty_cache()
 progress.update(status='completed',stage='completed');write(OUT/'progress.json',progress);write(OUT/'results.json',progress['results'])
if __name__=='__main__':main()
