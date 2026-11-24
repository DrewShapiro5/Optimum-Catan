"""Friendly-robber JSettlers corpus followed by W4 self-play continuation."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import (web_view, state_to_json, state_from_json,
    action_to_json, action_from_json, action_record_to_json)
from agent_apis import benchmark
from agent_apis.experiment import ROOT, EXPERIMENTS, digest, file_digest, source_snapshot
from agent_apis.observation import from_web_view, encode_observation, feature_names, ResourceLedger, require
from agent_apis.observation_dataset import phase
from agent_apis.neural import selfplay
from agent_apis.neural.pipeline import write
from agent_apis.neural.recipes import balanced_mass
from agent_apis.neural.topology import topology
from agent_apis.neural.train import evaluate, lr_at, save_checkpoint
from agent_apis.jsettlers.player import BRIDGE

OUTPUT = ROOT / 'runs/catan-ai/friendly-w4'
STATE = ROOT / 'runs/catan-ai/friendly-w4-progress.json'
CORPUS = ROOT / 'data/jsettlers-friendly-250'
PROTOCOL = EXPERIMENTS / 'friendly-w4.json'
FAMILIES = sorted({a[0] for a in topology()['action_vocabulary']})
ACTION_FAMILY = np.asarray([FAMILIES.index(a[0]) for a in topology()['action_vocabulary']])


def check_arrays(data):
    n = len(data['value'])
    require(data['features'].shape == (n, 681) and data['legal'].shape == (n, 332)
        and data['policy'].shape == (n, 332), 'Wrong training shapes')
    require(np.isfinite(data['features']).all() and np.isfinite(data['policy']).all(), 'Nonfinite data')
    require((data['features'][:,feature_names().index('RULE_FRIENDLY_ROBBER')]==1).all(), 'Standard-robber data in friendly corpus')
    require((data['policy'] >= 0).all() and np.allclose(data['policy'].sum(1), 1), 'Invalid policy')
    require((data['policy'][~data['legal']] == 0).all() and np.isin(data['value'], [0, 1]).all(), 'Illegal target/outcome')


class Data:
    def __init__(self, entries, split=None):
        chunks = []; references = []
        for entry in entries:
            if not entry.get('dataset'):
                continue
            if split is not None and entry['split'] != split:
                continue
            ref = entry['dataset']
            references.append(ref)
            require(file_digest(ROOT/ref['path']) == ref['sha256'], 'Dataset checksum changed')
            with np.load(ROOT/ref['path']) as f:
                d = {k:f[k].copy() for k in f.files}
            check_arrays(d)
            if 'action' not in d:
                d['action'] = d['policy'].argmax(1).astype(np.int64)
            chunks.append(d)
        require(chunks, 'Empty dataset split')
        self.arrays = {k:np.concatenate([d[k] for d in chunks]) for k in ('features','legal','policy','value','action')}
        n = len(self.arrays['value'])
        # Validation phase tags are descriptive only and cannot affect sampling.
        self.arrays['phase'] = np.concatenate([d.get('phase', np.ones(len(d['value']), np.uint8)) for d in chunks])
        self.arrays['family'] = ACTION_FAMILY[self.arrays['action']]
        self.manifest = {'families': FAMILIES}
        family_mass = np.zeros((n, len(FAMILIES)), np.float64)
        for a, family in enumerate(ACTION_FAMILY):
            family_mass[:,family] += self.arrays['policy'][:,a]
        counts = family_mass.sum(0)
        active = counts > 0
        mass = np.zeros_like(counts)
        mass[active] = balanced_mass(counts[active])
        # Joint action-family balancing extends hard-label W4 to visit targets.
        self.factors = .5/n + .5*np.divide(mass, counts, out=np.zeros_like(mass), where=active)
        self.q = family_mass @ self.factors
        require(np.isclose(self.q.sum(), 1) and np.all(self.q > 0) and np.max(n*self.q) <= 5+1e-6, 'Invalid W4 distribution')
        self.summary = {'rows': n, 'family_target_counts': dict(zip(FAMILIES, counts.tolist())),
            'dataset_references_sha256': digest(references),
            'max_relative_sampling_probability': float(np.max(n*self.q)),
            'value_weight_expectation': float(np.sum(self.q/(n*self.q))),
            'probabilities_sha256': digest(self.q.tolist())}

    def indices(self, split):
        return np.arange(len(self.arrays['value']))

    def batch(self, indices):
        return {k:v[indices].copy() for k,v in self.arrays.items()}

    def sample(self, rng, count):
        ids = rng.choice(len(self.q), count, p=self.q)
        batch = self.batch(ids)
        # For one-hot demonstrations this leaves the target unchanged. For
        # soft visit targets, sample family/action jointly, preserving target mass.
        batch['policy'] *= (self.factors[ACTION_FAMILY][None,:]/self.q[ids,None]).astype(np.float32)
        require(np.allclose(batch['policy'].sum(1), 1, atol=2e-6), 'W4 target normalization failed')
        batch['value_weight'] = (1/(len(self.q)*self.q[ids])).astype(np.float32)
        return batch


def losses(logits, values, legal, targets, outcomes, value_weights):
    nonforced = legal.sum(1)>1
    ce = -(targets*F.log_softmax(logits.masked_fill(~legal, -1e9),1)).sum(1)
    policy = (ce*nonforced).sum()/nonforced.sum().clamp_min(1)
    value = (F.binary_cross_entropy_with_logits(values,outcomes,reduction='none')*value_weights).mean()
    return policy, value


class Workflow(selfplay.SelfPlay):
    def __init__(self):
        OUTPUT.mkdir(parents=True, exist_ok=True); CORPUS.mkdir(parents=True, exist_ok=True)
        self.protocol = json.loads(PROTOCOL.read_text())
        defaults = json.loads((EXPERIMENTS/'training-defaults.json').read_text())
        require(defaults['weighting']=='W4' and defaults['friendly_robber'], 'New training defaults changed')
        paths = (*selfplay.CORE, 'agent_apis/neural/friendly_w4.py',
            'agent_apis/neural/recipes.py','experiments/catan-ai/friendly-w4.json',
            'experiments/catan-ai/training-defaults.json')
        self.core = {p:file_digest(ROOT/p) for p in paths}
        self.core.update({str(p.relative_to(ROOT)):file_digest(p) for p in sorted((ROOT/'framework/catanatron/catanatron').rglob('*.py'))})
        prior = json.loads((ROOT/'runs/catan-ai/selfplay-small-round/iteration-1/training.json').read_text())
        self.parent = prior['checkpoint']
        if STATE.exists():
            self.state = json.loads(STATE.read_text())
            require(self.state['source_sha256']==self.core and self.state['protocol']==self.protocol and self.state['parent']==self.parent, 'W4 resume sources/protocol/parent changed')
        else:
            self.state = {'status':'running','stage':'initializing','started_at':self.now(),
                'source_sha256':self.core,'protocol':self.protocol,'parent':self.parent,
                'collection':[],'iterations':{},'installed_default_changed':False,'final_seeds_used':False}
            write(OUTPUT/'source.json', source_snapshot())
        self.state.update(status='running',pid=os.getpid());self.persist()
        require(torch.__version__=='2.15.0' and torch.backends.mps.is_available(), 'Pinned PyTorch/MPS unavailable')
        self.frozen_ref = json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
        self.frozen = selfplay.load_checkpoint(self.frozen_ref)
        selfplay.load_checkpoint(self.parent)
        self.state['native_opponent'] = benchmark.provider_identity('JSETTLERS');self.persist()
        self.reserved = set()
        for name in ('full-250','downsampled'):
            m=json.loads((ROOT/f'data/jsettlers-corpora/{name}/manifest.json').read_text())
            self.reserved.update(g['seed'] for g in m['games'])
        for entries in json.loads((EXPERIMENTS/'evaluation-seeds-v1.json').read_text())['sets'].values():
            for pair in entries:
                self.reserved.update(pair[k] for k in ('board_seed','simulator_seed','agent_seed'))
        for folder in ('phase9-pilot','phase9-selfplay/iteration-1','selfplay-small-round/iteration-1'):
            for path in (ROOT/'runs/catan-ai'/folder).glob('*.json'):
                r=json.loads(path.read_text())
                if isinstance(r,dict) and 'pair' in r:
                    self.reserved.update(r['pair'][k] for k in ('board_seed','simulator_seed','agent_seed'))

    def pair(self, attempt):
        pair={'pair_id':f'friendly-expert-{attempt:05d}'}
        for kind in ('board','simulator','agent'):
            seed=benchmark.named_seed(self.protocol['seed'],f'friendly-expert-{attempt}-{kind}')
            require(seed not in self.reserved, 'New corpus seed overlaps previous data/evaluation')
            pair[kind+'_seed']=seed
        return pair

    def generate_game(self, attempt):
        pair=self.pair(attempt);game=benchmark.make_game(pair,friendly_robber=True)
        initial=deepcopy(state_to_json(game));rows=[];actions=[];started=time.perf_counter();BRIDGE.close()
        agents={c:benchmark.Agent('JSETTLERS',c,benchmark.named_seed(pair['agent_seed'],c.value)) for c in benchmark.COLORS}
        self.state.update(stage=f'generate-friendly-expert:{attempt:05d}',active_game={'attempt':attempt,'actions':0,'turns':0});self.persist()
        try:
            while game.winning_color() is None and game.state.num_turns<1000 and len(actions)<10000:
                actor=game.state.current_color();safe=from_web_view(web_view(game,actor),actor.value)
                features=encode_observation(safe)
                ledger=ResourceLedger(actor.value);hands=ledger.update(safe)
                for c in benchmark.COLORS:
                    i=game.state.colors.index(c)
                    from catanatron.models.enums import RESOURCES
                    require(hands[c.value]==[game.state.player_state[f'P{i}_{r}_IN_HAND'] for r in RESOURCES], 'Friendly resource ledger audit failed')
                legal=np.zeros(332,bool);legal[[selfplay.action_id(a) for a in game.playable_actions]]=True
                available=list(game.playable_actions)
                chosen=available[0] if len(available)==1 else agents[actor].select(game,available,benchmark.DecisionContext(time.perf_counter(),500,'policy',None))
                agents[actor].cold=False;require(chosen in available,'Illegal native corpus action')
                label=selfplay.action_id(chosen)
                rows.append((features,legal,label,actor.value,['setup','middle','endgame'].index(phase(safe))))
                payload=action_to_json(chosen);result=game.execute(chosen)
                actions.append({'action':payload,'record':action_record_to_json(result)})
                if len(actions)%25==0:
                    self.state['active_game'].update(actions=len(actions),turns=game.state.num_turns,elapsed_seconds=time.perf_counter()-started);self.persist()
            winner=game.winning_color();final=state_to_json(game)
            replay=state_from_json(initial,[SimplePlayer(c) for c in benchmark.COLORS])
            for event in actions:
                require(action_record_to_json(replay.execute(action_from_json(event['action'])))==event['record'], 'Friendly corpus replay mismatch')
            require(state_to_json(replay)==final,'Friendly corpus final-state replay mismatch')
            replay_path=CORPUS/f'game-{attempt:05d}-replay.json.gz'
            temp=replay_path.with_suffix('.tmp')
            with gzip.open(temp,'wt') as stream:
                json.dump({'initial':initial,'actions':actions,'final':final},stream)
            temp.replace(replay_path)
            record={'pair':pair,'status':'completed' if winner else 'cutoff','winner':winner.value if winner else None,
                'steps':len(rows),'turns':game.state.num_turns,'elapsed_seconds':time.perf_counter()-started,
                'replay':selfplay.reference(replay_path),'source_sha256':digest(self.core),'dataset':None}
            if winner:
                features=np.asarray([r[0] for r in rows],np.float32);legal=np.asarray([r[1] for r in rows],bool)
                labels=np.asarray([r[2] for r in rows],np.int64)
                data={'features':features,'legal':legal,'action':labels,'policy':np.eye(332,dtype=np.float32)[labels],
                    'value':np.asarray([r[3]==winner.value for r in rows],np.float32),'phase':np.asarray([r[4] for r in rows],np.uint8)}
                check_arrays(data)
                destination=CORPUS/f'game-{attempt:05d}.npz';temporary=CORPUS/f'game-{attempt:05d}.tmp.npz'
                np.savez_compressed(temporary,**data);temporary.replace(destination)
                record['dataset']={**selfplay.reference(destination),'rows':len(rows)}
            return record
        finally:BRIDGE.close()

    def collect_expert(self):
        completed=0;records=[];seen=set()
        for attempt in range(self.protocol['max_expert_attempts']):
            if completed>=self.protocol['expert_games']:break
            self.guard();path=CORPUS/f'game-{attempt:05d}.json';pair=self.pair(attempt)
            for k in ('board_seed','simulator_seed','agent_seed'):
                require(pair[k] not in seen, 'New corpus contains duplicate seeds');seen.add(pair[k])
            if path.exists():
                record=json.loads(path.read_text());require(record['source_sha256']==digest(self.core) and record['pair']==pair,'Committed expert game changed')
                for ref in [record['replay']]+([record['dataset']] if record['dataset'] else []):
                    require(file_digest(ROOT/ref['path'])==ref['sha256'],'Expert artifact changed')
            else:
                record=self.generate_game(attempt)
                # Split assignment depends only on the next completion slot,
                # before inspecting outcomes; cutoffs do not consume a slot.
                slot=completed%20;record['split']='train' if slot<18 else 'validation' if slot==18 else 'test'
                write(path,record)
            completed+=record['status']=='completed';records.append(record)
            self.state.update(expert_completed=completed,expert_attempts=len(records),expert_cutoffs=len(records)-completed);self.persist()
            manifest={'schema_version':3,'provider':'JSETTLERS','strategy':'SMART','rules':{'friendly_robber':True,'vps_to_win':15,'discard_limit':9},
                'feature_schema_version':3,'action_count':332,'source_sha256':self.core,'games':records,'completed_games':completed,
                'sampling':'all decisions; original native action demonstrations; completed actor-relative outcomes',
                'model_inputs':'redacted schema-3 features and legal masks only; replay documents are privileged audit artifacts',
                'status':'completed' if completed==self.protocol['expert_games'] else 'running'}
            write(CORPUS/'manifest.json',manifest)
            print(f'EXPERT {completed}/{self.protocol["expert_games"]}; rows={sum(r["dataset"]["rows"] for r in records if r["dataset"])}',flush=True)
        require(completed==self.protocol['expert_games'],'Expert attempt limit reached')
        return records

    def train_w4(self, expert, new):
        folder=OUTPUT/'iteration-1';folder.mkdir(parents=True,exist_ok=True);path=folder/'training.json'
        if path.exists():
            r=json.loads(path.read_text());selfplay.load_checkpoint(r['checkpoint']);return r
        self.state['stage']='train-W4';self.persist()
        checkpoint=selfplay.load_checkpoint(self.parent);model=checkpoint.model.to('mps');device='mps'
        config=deepcopy(checkpoint.metadata['configuration']);config.update(phase=9,stage='friendly-W4-selfplay-continuation')
        config['rules']['friendly_robber']=True;config['data'].update(weighting='W4',source_schema=3,source_features=681,corpora=[str(CORPUS.relative_to(ROOT))],
            selfplay={'sampling_and_datasets_sha256':digest(new.summary),'expert_fraction':.5},
            soft_target_W4='joint action-family weighting, exactly equal to existing W4 for one-hot demonstrations')
        settings=config['training'];settings.update(max_updates=self.protocol['updates_per_iteration'],warmup_updates=100)
        meta=deepcopy(checkpoint.metadata);meta.update(configuration=config,configuration_sha256=digest(config),rules=config['rules'],
            source_sha256=digest(self.core),parent_checkpoint=self.parent,training_recipe='D2-W4',dataset_sha256=file_digest(CORPUS/'manifest.json'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        rng=np.random.default_rng(benchmark.named_seed(self.protocol['seed'],'W4-training'))
        aug=torch.Generator().manual_seed(benchmark.named_seed(self.protocol['seed'],'W4-augmentation'))
        torch.manual_seed(benchmark.named_seed(self.protocol['seed'],'W4-torch'))
        data_sha=digest({'expert':expert.summary,'new':new.summary,'validation':self.cache.arrays['value'].tolist(),
            'manifest':file_digest(CORPUS/'manifest.json'),'new_data':config['data']['selfplay']})
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
                pl,vl=losses(logits,values,legal,targets,outcomes,vw);(pl+vl).backward()
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

    def execute(self):
        records=self.collect_expert()
        expert=Data(records,'train');self.cache=Data(records,'validation')
        previous=json.loads((ROOT/'runs/catan-ai/selfplay-small-round/iteration-1/combined-collection.json').read_text())
        new=Data(previous)
        record=self.state['iterations'].setdefault('1',{})
        record.update(completed_games=100,generation_seconds=sum(r['elapsed_seconds'] for r in previous))
        record['training']=self.train_w4(expert,new);self.persist()
        del expert,new
        record['evaluation']=self.validate(1,record['training']['checkpoint'],self.parent,[self.parent],record)
        self.state.update(status='completed',stage='completed-friendly-W4',pid=None);self.persist();write(OUTPUT/'results.json',self.state)


def main():
    selfplay.OUTPUT,selfplay.STATE=OUTPUT,STATE
    with benchmark.exclusive_benchmark():
        work=Workflow()
        try:work.execute()
        except BaseException as error:
            work.state.update(status='needs-investigation',error=f'{type(error).__name__}: {error}',pid=None);work.persist();raise

if __name__=='__main__':main()
