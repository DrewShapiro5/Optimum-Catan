"""Bounded DAgger pilot: 20 learner games, 1000 updates, 10 tuning games."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from catanatron.models.player import Color
from catanatron.serialization import web_view,action_from_json,action_to_json,action_record_to_json
from agent_apis import benchmark
from agent_apis.benchmark import named_seed
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest,source_snapshot
from agent_apis.observation import from_web_view,encode_observation,require
from agent_apis.neural.pipeline import write,games
from agent_apis.neural.provider import NeuralCheckpoint
from agent_apis.neural.data import Cache
from agent_apis.neural.models import losses
from agent_apis.neural.train import tensors,evaluate,save_checkpoint,lr_at
from agent_apis.jsettlers.player import JSettlersPlayer,BRIDGE
from agent_apis.stochastic import action_id

OUTPUT=ROOT/'runs/catan-ai/phase9-pilot'
STATE=ROOT/'runs/catan-ai/phase9-progress.json'
PROTOCOL=EXPERIMENTS/'phase9-pilot.json'
CORE=('agent_apis/neural/phase9.py','agent_apis/neural/provider.py','agent_apis/neural/models.py',
      'agent_apis/neural/topology.py','agent_apis/neural/train.py','agent_apis/neural/data.py',
      'agent_apis/observation.py','agent_apis/jsettlers/player.py','agent_apis/jsettlers/Bridge.java',
      'agent_apis/benchmark.py','experiments/catan-ai/phase9-pilot.json')


def teacher_label(raw,actions):
    """Query only that actor's official redacted view; never invent a fallback."""
    color=actions[0].color.value;raw={**raw,'perspective':color}
    identity=JSettlersPlayer.synchronize(raw,color)
    require(identity['engine']=='SOCRobotBrain' and identity['strategy']=='SMART','Teacher identity differs')
    require(identity['state_index']==len(raw['action_records']) and
            identity['current_seat']==raw['colors'].index(color),'Teacher reconstruction differs from requested decision')
    before=identity.get('translation_rejections',0)
    selected=action_from_json(JSettlersPlayer.choose(raw))
    require(selected in actions,'Illegal teacher target')
    # Synchronizing the same pre-action state also exposes native rejection counts.
    status=JSettlersPlayer.synchronize(raw,color)
    require(status.get('translation_rejections',0)==before,'Teacher query had native translation rejection')
    return action_id(selected)


def generation_pairs(protocol):
    known=set()
    for folder in ('full-250','downsampled'):
        manifest=json.loads((ROOT/f'data/jsettlers-corpora/{folder}/manifest.json').read_text())
        known.update(g['seed'] for g in manifest['games'])
    evaluation=json.loads((EXPERIMENTS/'evaluation-seeds-v1.json').read_text())
    for entries in evaluation['sets'].values():
        for p in entries:known.update(p[k] for k in ('board_seed','simulator_seed','agent_seed'))
    pairs=[]
    for i in range(protocol['collection_games']//2):
        pair={'pair_id':f'phase9-train-{i:04d}'}
        for kind in ('board','simulator','agent'):
            seed=named_seed(protocol['seed'],f'phase9-dagger-{kind}-{i}')
            require(seed not in known,'Generation seed overlaps corpus/evaluation');known.add(seed)
            pair[kind+'_seed']=seed
        pairs.append(pair)
    return pairs


def finalize_rows(rows,winner,learner):
    """Only completed games get outcome labels; no future data enters features."""
    if winner is None:return None
    require(rows,'No valid teacher rows in completed game')
    features=np.asarray([r['features'] for r in rows],np.float32)
    legal=np.zeros((len(rows),332),bool)
    for i,row in enumerate(rows):legal[i,row['legal']]=True
    targets=np.asarray([r['teacher'] for r in rows],np.int64)
    require(np.isfinite(features).all() and legal[np.arange(len(rows)),targets].all(),'Invalid training rows')
    return {'features':features,'legal':legal,'action':targets,
            'value':np.full(len(rows),float(winner==learner),np.float32)}


class Pilot:
    def __init__(self):
        OUTPUT.mkdir(parents=True,exist_ok=True);self.protocol=json.loads(PROTOCOL.read_text())
        self.core={p:file_digest(ROOT/p) for p in CORE}
        phase6=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
        self.parent=phase6['training']['XD-CNN:11:D2-W0']['checkpoint']
        require(file_digest(ROOT/self.parent['path'])==self.parent['sha256'],'Parent checkpoint changed')
        if STATE.exists():
            self.state=json.loads(STATE.read_text());require(self.state['source_sha256']==self.core,'Phase9 sources changed')
        else:self.state={'status':'running','stage':'initializing','source_sha256':self.core,'protocol':self.protocol,
                        'parent_checkpoint':self.parent,'collection':{},'started_at':datetime.now(timezone.utc).isoformat(),
                        'final_seeds_used':False,'new_data_is_training_only':True}
        self.state.update(status='running',pid=os.getpid());self.persist()
        if not (OUTPUT/'source.json').exists():write(OUTPUT/'source.json',source_snapshot())
        self.state['native_teacher_identity']=benchmark.provider_identity('JSETTLERS')
        self.state['software']={'torch':torch.__version__,'precision':'float32','cpu_threads':6}
        self.persist()
        self.model=NeuralCheckpoint(ROOT/self.parent['path'],device='cpu')
        require(self.model.metadata['model_id']=='XD-CNN','Only XD-CNN is authorized')

    def persist(self):
        self.state['updated_at']=datetime.now(timezone.utc).isoformat();write(STATE,self.state)
    def guard(self):require(all(file_digest(ROOT/p)==sha for p,sha in self.core.items()),'Pilot implementation changed')

    def collect(self):
        for pair in generation_pairs(self.protocol):
            for color in benchmark.COLORS:
                self.guard();key=f"{pair['pair_id']}-{color.value}";path=OUTPUT/(key+'.json')
                if path.exists():
                    record=json.loads(path.read_text())
                    require(record['source_sha256']==self.core and record['pair']==pair,'Generation configuration changed')
                    if record['dataset']:require(file_digest(OUTPUT/record['dataset']['file'])==record['dataset']['sha256'],'Generated data changed')
                    self.state['collection'][key]=record;self.persist();continue
                self.state['stage']='collect:'+key;self.persist();BRIDGE.close()
                game=benchmark.make_game(pair);rng=random.Random(named_seed(pair['agent_seed'],color.value))
                rows=[];excluded=[];queried=0;started=time.perf_counter()
                opponent=benchmark.Agent('JSETTLERS',next(c for c in benchmark.COLORS if c!=color),pair['agent_seed'])
                try:
                    while game.winning_color() is None and game.state.num_turns<1000 and len(game.state.action_records)<10000:
                        actor=game.state.current_color();actions=list(game.playable_actions)
                        if actor==color:
                            raw=web_view(game,actor);safe=from_web_view(raw,actor.value)
                            features=encode_observation(safe);ids=[action_id(a) for a in actions]
                            if len(actions)==1:action=actions[0];target=ids[0]
                            else:
                                probabilities,_=self.model.evaluate(safe)
                                selected=max(probabilities,key=lambda i:(probabilities[i],-i))
                                action=next(a for a in actions if action_id(a)==selected)
                                queried+=1
                                try:target=teacher_label(raw,actions)
                                except (RuntimeError,ValueError,TimeoutError) as error:
                                    excluded.append({'step':len(game.state.action_records),'error':f'{type(error).__name__}: {error}'})
                                    target=None
                            if target is not None:rows.append({'features':features,'legal':ids,'teacher':target})
                        elif len(actions)==1:action=actions[0]
                        else:
                            context=benchmark.DecisionContext(time.perf_counter(),500,'policy',None)
                            action=opponent.select(game,actions,context);opponent.cold=False
                        require(action in actions,'Collection action is illegal');game.execute(action)
                    winner=game.winning_color();arrays=finalize_rows(rows,winner,color);dataset=None
                    if arrays is not None:
                        temp=OUTPUT/(key+'.tmp.npz');np.savez_compressed(temp,**arrays)
                        destination=OUTPUT/(key+'.npz');temp.replace(destination)
                        dataset={'file':destination.name,'sha256':file_digest(destination),'rows':len(rows)}
                    record={'pair':pair,'learner_color':color.value,'winner':winner.value if winner else None,
                            'status':'completed' if winner else 'cutoff','steps':len(game.state.action_records),'turns':game.state.num_turns,
                            'teacher_queries':queried,'excluded_queries':excluded,'dataset':dataset,'source_sha256':self.core,
                            'elapsed_seconds':time.perf_counter()-started,
                            'trajectory_sha256':digest([action_record_to_json(r) for r in game.state.action_records])}
                    write(path,record);self.state['collection'][key]=record;self.persist()
                    print(f"COLLECT {len(self.state['collection'])}/{self.protocol['collection_games']} {key}: {record['status']}; labels={len(rows)} excluded={len(excluded)}",flush=True)
                finally:BRIDGE.close()
        data=[]
        queries=sum(r['teacher_queries'] for r in self.state['collection'].values())
        excluded=sum(len(r['excluded_queries']) for r in self.state['collection'].values())
        require(queries>excluded and excluded/max(1,queries)<=self.protocol['max_query_failure_rate'],
                'Too many teacher-query failures; investigate before retraining')
        for record in self.state['collection'].values():
            if record['dataset']:
                with np.load(OUTPUT/record['dataset']['file']) as loaded:data.append({k:loaded[k].copy() for k in loaded.files})
        require(data,'No completed games with teacher labels; investigate')
        return {k:np.concatenate([d[k] for d in data]) for k in data[0]}

    def finetune(self,new):
        self.state['stage']='finetune';self.persist()
        recordpath=OUTPUT/'training.json'
        if recordpath.exists():
            r=json.loads(recordpath.read_text());require(file_digest(ROOT/r['checkpoint']['path'])==r['checkpoint']['sha256'],'Pilot checkpoint changed');return r
        device=self.protocol['training_device'];require(device!='mps' or torch.backends.mps.is_available(),'Pinned MPS unavailable; no fallback')
        cache=Cache(ROOT/'runs/catan-ai/d2-cache-89b777ce01b9ebd7');train_indices=cache.indices('train')
        model=self.model.model.to(device);model.train()
        config=deepcopy(self.model.metadata['configuration']);config.update(phase=9,stage='dagger-finetuning',experiment='phase9-dagger-pilot')
        config['data']['additional_training_data']={'source':'phase9 learner-state JSettlers labels','fraction':.5,
            'manifest_sha256':digest(self.state['collection']),'value_targets':'actual completed learner-game outcomes; censored games excluded'}
        settings=config['training'];settings.update(max_updates=self.protocol['updates'],warmup_updates=100,validation_interval=250)
        meta=deepcopy(self.model.metadata);meta.update(configuration=config,configuration_sha256=digest(config),
             source_sha256=digest(self.core),parent_checkpoint=self.parent,training_recipe='D2-W0 + 50% learner-state DAgger',phase9_pilot=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        rng=np.random.default_rng(named_seed(11,'phase9-training'));torch.manual_seed(named_seed(11,'phase9-torch'))
        augment=torch.Generator().manual_seed(named_seed(11,'phase9-augmentation'))
        initial=evaluate(model,cache,'validation',device);best=initial['selection_loss'];step=0;history=[];beststep=0;previous_seconds=0.
        resume=OUTPUT/'resume.pt'
        meta['step']=0
        if not resume.exists():save_checkpoint(OUTPUT/'best.pt',model,meta)
        else:require((OUTPUT/'best.pt').exists(),'Resume is missing selected checkpoint')
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=True)
            require(saved['core']==self.core and saved['new_data_sha256']==digest(self.state['collection']),'Resume data/source changed')
            model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
            rng.bit_generator.state=saved['sampling_rng'];augment.set_state(saved['augmentation_rng']);torch.random.set_rng_state(saved['torch_rng'])
            if device=='mps':torch.mps.set_rng_state(saved['mps_rng'])
            step=saved['step'];best=saved['best'];history=saved['history'];beststep=saved['beststep']
            previous_seconds=saved.get('elapsed_seconds',0.)
        started=time.perf_counter()
        def save():
            torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'best':best,'beststep':beststep,
                 'sampling_rng':rng.bit_generator.state,'augmentation_rng':augment.get_state(),'torch_rng':torch.random.get_rng_state(),
                 'mps_rng':torch.mps.get_rng_state() if device=='mps' else None,'core':self.core,'history':history,
                 'elapsed_seconds':previous_seconds+time.perf_counter()-started,
                 'new_data_sha256':digest(self.state['collection'])},OUTPUT/'resume.tmp')
            (OUTPUT/'resume.tmp').replace(resume)
        try:
            while step<settings['max_updates']:
                self.guard();old=cache.batch(rng.choice(train_indices,size=128));selection=rng.integers(len(new['action']),size=128)
                batch={k:np.concatenate((old[k],new[k][selection])) for k in ('features','legal','action','value')}
                raw,legal,targets,outcomes=tensors(batch,device);sym=torch.randint(12,(256,),generator=augment).to(device)
                targets,legal=model.inputs.augment_targets(targets,legal,sym);step+=1
                for group in optimizer.param_groups:group['lr']=lr_at(step,settings)
                optimizer.zero_grad(set_to_none=True);policy,value=model(raw,sym)
                pl,vl=losses(policy,value,legal,targets,outcomes);(pl+vl).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                if step%100==0:save();print(f'TRAIN {step}/{settings["max_updates"]}',flush=True)
                if step%250==0 or step==settings['max_updates']:
                    validation=evaluate(model,cache,'validation',device);history.append({'step':step,**validation})
                    if validation['selection_loss']<best:
                        best=validation['selection_loss'];beststep=step;meta['step']=step;save_checkpoint(OUTPUT/'best.pt',model,meta)
                    save();self.state['training_updates']=step;self.persist()
        except BaseException:save();raise
        meta['step']=step;save_checkpoint(OUTPUT/'final.pt',model,meta)
        model.load_state_dict(torch.load(OUTPUT/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
        test=evaluate(model,cache,'test',device)
        result={'checkpoint':{'path':str((OUTPUT/'best.pt').relative_to(ROOT)),'sha256':file_digest(OUTPUT/'best.pt')},
             'updates':step,'best_step':beststep,'best_validation_loss':best,'initial_validation':initial,'validation_history':history,
             'test':test,'new_rows':len(new['action']),'expert_fraction':.5,'elapsed_seconds':previous_seconds+time.perf_counter()-started,'configuration':config}
        write(recordpath,result);model.to('cpu');return result

    def validate(self,training):
        self.state['stage']='validation:10-games';self.persist()
        phase6=json.loads((EXPERIMENTS/'PHASE6_RESULTS.json').read_text())
        baseline_path=ROOT/phase6['benchmarks']['XD-CNN:11:D2-W0:policy:JSETTLERS:screening']['run_record']
        baseline=json.loads(baseline_path.read_text())
        require(baseline['configuration']['content']['candidate']['checkpoint']==self.parent and
                baseline['configuration']['content']['evaluation']['mode']=='policy','Wrong parent policy reference')
        for name in ('agent_apis/neural/models.py','agent_apis/neural/provider.py','agent_apis/observation.py','agent_apis/benchmark.py'):
            require(baseline['source']['files'][name]==file_digest(ROOT/name),'Historical parent decision implementation changed')
        for name,sha in baseline['metrics']['artifacts'].items():require(file_digest(baseline_path.parent/name)==sha,'Baseline logs changed')
        config=deepcopy(baseline['configuration']['content']);config.update(experiment='phase9-dagger-pilot-validation',implementation_sha256=self.core)
        config['candidate']['checkpoint']=training['checkpoint'];config['evaluation']['tier']='smoke'
        found=None
        for p in (ROOT/'runs/catan-ai').glob('phase5q-*/run.json'):
            r=json.loads(p.read_text())
            if r['configuration']['sha256']==digest(config):
                require(r['status']=='completed','Matching pilot evaluation failed')
                for name,sha in r['metrics']['artifacts'].items():require(file_digest(p.parent/name)==sha,'Validation artifacts changed')
                found=p;break
        if found is None:found=benchmark.run(config,ROOT/'runs/catan-ai')
        record=json.loads(found.read_text());new=games(found);old=games(baseline_path)[:len(new)]
        comparison=benchmark.paired_difference(new,old)
        result={'validation_run':str(found.relative_to(ROOT)),'metrics':record['metrics']['benchmark'],
                'baseline_run':str(baseline_path.relative_to(ROOT)),'paired_candidate_minus_parent':comparison,
                'scope':'Single-seed 20-game DAgger/1000-update/10-game validation pilot; original phase9 remains incomplete',
                'installed_default_changed':False,'automatic_scale_up':False,'final_seeds_used':False,'training':training}
        self.state.update(status='completed',stage='completed-pilot',pid=None,result=result);self.persist()
        write(EXPERIMENTS/'PHASE9_PILOT_RESULTS.json',self.state)
        (EXPERIMENTS/'PHASE9_PILOT_RESULTS.md').write_text('# Phase 9 DAgger pilot results\n\n'+json.dumps(result,indent=2)+'\n\nNo automatic deployment or scale-up; ten games cannot establish a robust strength improvement.\n')
        print('Pilot completed; no automatic large generation queue.',flush=True)

    def execute(self):
        new=self.collect();training=self.finetune(new);self.state['training']=training;self.persist();self.validate(training)


def main():
    with benchmark.exclusive_benchmark():
        work=Pilot()
        try:work.execute()
        except BaseException as error:
            work.state.update(status='needs-investigation',error=f'{type(error).__name__}: {error}');work.persist();raise

if __name__=='__main__':main()
