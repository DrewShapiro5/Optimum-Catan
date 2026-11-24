"""Fresh XD-CNN; thousand-game cohorts with compact replay and sampled positions."""
from copy import deepcopy
import gzip
import json
import os
import random
import signal
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from catanatron.serialization import web_view,action_record_to_json
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest
from agent_apis.observation import from_web_view,encode_observation,require
from agent_apis.neural import compact_iterations as compact,friendly_w4 as w4,selfplay
from agent_apis.neural.models import build_model
from agent_apis.neural.pipeline import write
from agent_apis.neural.train import save_checkpoint
from agent_apis.jsettlers.player import BRIDGE

OUTPUT=ROOT/'runs/catan-ai/fresh-xdcnn-visible'
STATE=ROOT/'runs/catan-ai/fresh-xdcnn-visible-progress.json'
PROTOCOL=EXPERIMENTS/'fresh-xdcnn-visible.json'


def derive(record,limit,seed,always_include_initial=True):
    # Choose position indices without looking at outcomes, then reconstruct
    # observations before executing the recorded action. No feature persistence.
    eligible=[i for i,e in enumerate(record['events']) if e.get('v') or e.get('demonstration')]
    if always_include_initial:
        first_roll=next((i for i,e in enumerate(record['events']) if e['event'][0][1]=='ROLL'),len(record['events']))
        initial_eligible=[i for i in eligible if i<first_roll]
        later_eligible=[i for i in eligible if i>=first_roll]
        rem=max(0,limit-len(initial_eligible))
        keep=set(initial_eligible)|set(random.Random(seed).sample(later_eligible,min(rem,len(later_eligible))))
    else:
        keep=set(random.Random(seed).sample(eligible,min(limit,len(eligible))))
    game=benchmark.make_game(record['pair'],friendly_robber=True)
    require(compact.snapshot_digest(game)==record['initial_sha256'],'Initial replay mismatch')
    rows=[]
    for i,e in enumerate(record['events']):
        legal={selfplay.action_id(a):a for a in game.playable_actions};require(e['a'] in legal,'Illegal journal action')
        if i in keep:
            actor=game.state.current_color();safe=from_web_view(web_view(game,actor),actor.value)
            mask=np.zeros(332,bool);mask[list(legal)]=True;policy=np.zeros(332,np.float32)
            if e.get('v'):
                ids=[x[0] for x in e['v']];visits=np.array([x[1] for x in e['v']],float)
                require(set(ids)<=set(legal) and (visits>0).all(),'Invalid search targets');policy[ids]=visits/visits.sum()
            else:policy[e['a']]=1
            rows.append({'features':encode_observation(safe),'legal':mask,'policy':policy,'actor':actor.value})
        require(action_record_to_json(game.execute(legal[e['a']]))==e['event'],'Chance replay mismatch')
    require(compact.snapshot_digest(game)==record['final_sha256'],'Final replay mismatch')
    winner=game.winning_color();require((winner.value if winner else None)==record['winner'],'Outcome mismatch')
    return selfplay.completed_rows(rows,winner)


class Fresh(compact.Controller):
    def __init__(self,hours):
        OUTPUT.mkdir(parents=True,exist_ok=True);self.protocol=json.loads(PROTOCOL.read_text())
        paths=(*selfplay.CORE,'agent_apis/neural/compact_iterations.py','agent_apis/neural/friendly_w4.py',
            'agent_apis/neural/recipes.py','agent_apis/neural/fresh_cohorts.py',str(PROTOCOL.relative_to(ROOT)),
            'experiments/catan-ai/training-defaults.json')
        self.core={p:file_digest(ROOT/p) for p in paths}
        self.core.update({str(p.relative_to(ROOT)):file_digest(p) for p in sorted((ROOT/'framework/catanatron/catanatron').rglob('*.py'))})
        if STATE.exists():
            self.state=json.loads(STATE.read_text());require(self.state['source_sha256']==self.core,'Fresh cohort sources changed')
        else:
            self.state={'status':'initializing','source_sha256':self.core,'protocol':self.protocol,'iterations':{},
                'initialization':'random weights, no trained XD-CNN weights loaded','installed_default_changed':False}
            with zipfile.ZipFile(OUTPUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in self.core:z.write(ROOT/p,p)
        self.deadline=time.time()+hours*3600;self.stop_requested=False
        self.state.update(status='running',pid=os.getpid(),review_at=compact.datetime.fromtimestamp(self.deadline,compact.timezone.utc).isoformat())
        signal.signal(signal.SIGTERM,self.request_stop);signal.signal(signal.SIGINT,self.request_stop);self.persist()
        require(torch.__version__=='2.15.0' and torch.backends.mps.is_available(),'Pinned PyTorch/MPS required')
        initial=OUTPUT/'random-initial.pt'
        if not initial.exists():
            # Copy schema/configuration metadata only. Never load a trained state_dict.
            template=ROOT/json.loads((ROOT/'runs/catan-ai/friendly-w4/iteration-1/training.json').read_text())['checkpoint']['path']
            meta=deepcopy(torch.load(template,map_location='cpu',weights_only=True)['metadata'])
            torch.manual_seed(self.protocol['seed']);model=build_model('XD-CNN')
            config=deepcopy(meta['configuration']);config.update(seed=self.protocol['seed'],phase=9,stage='fresh-random-initialization')
            config['data'].update(weighting='W4');config['rules']['friendly_robber']=True
            meta.update(configuration=config,configuration_sha256=digest(config),step=0,training_seed=self.protocol['seed'],
                parent_checkpoint=None,initialization='torch.manual_seed then build_model(XD-CNN); no pretrained weights',source_sha256=digest(self.core))
            save_checkpoint(initial,model,meta);del model
        self.state['random_initial_checkpoint']=selfplay.reference(initial)
        self.state['initial_parent']=selfplay.reference(initial);self.persist()
        self.frozen_ref=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
        self.frozen=selfplay.load_checkpoint(self.frozen_ref) # Historical MLP simulates opponents; XD-CNN is fresh.
        self.state['native_opponent']=benchmark.provider_identity('JSETTLERS');self.persist()
        self.known_seeds=set()
        for pairs in json.loads((EXPERIMENTS/'evaluation-seeds-v1.json').read_text())['sets'].values():
            for p in pairs:self.known_seeds.update(p[k] for k in ('board_seed','simulator_seed','agent_seed'))

    def expert_game(self,path,pair):
        if path.exists():
            with gzip.open(path,'rt') as f:r=json.load(f)
            require(r['source_sha256']==digest(self.core) and r['pair']==pair,'Expert journal changed');compact.replay_journal(r);return r
        game=benchmark.make_game(pair,friendly_robber=True);initial=compact.snapshot_digest(game);events=[];started=time.perf_counter();BRIDGE.close()
        agents={c:benchmark.Agent('JSETTLERS',c,benchmark.named_seed(pair['agent_seed'],c.value)) for c in benchmark.COLORS}
        interrupted=False
        try:
            while game.winning_color() is None and game.state.num_turns<1000 and len(events)<10000:
                self.budget();actor=game.state.current_color();legal=list(game.playable_actions)
                action=legal[0] if len(legal)==1 else agents[actor].select(game,legal,benchmark.DecisionContext(time.perf_counter(),500,'policy',None))
                agents[actor].cold=False;require(action in legal,'Illegal native action')
                event={'a':selfplay.action_id(action),'demonstration':True};event['event']=action_record_to_json(game.execute(action));events.append(event)
                if len(events)%25==0:
                    compact.save_journal(path.with_suffix('.partial.gz'),{'pair':pair,'initial_sha256':initial,
                        'events':events,'status':'in-progress','source_sha256':digest(self.core)})
                    self.state['active_game']={'file':str(path.relative_to(ROOT)),'actions':len(events),'turns':game.state.num_turns};self.persist()
        except compact.ReviewDue:interrupted=True
        finally:BRIDGE.close()
        winner=game.winning_color();r={'pair':pair,'initial_sha256':initial,'final_sha256':compact.snapshot_digest(game),
            'events':events,'winner':winner.value if winner else None,'status':'interrupted' if interrupted else 'completed' if winner else 'cutoff',
            'source_sha256':digest(self.core),'elapsed_seconds':time.perf_counter()-started}
        compact.replay_journal(r);compact.save_journal(path.with_suffix('.partial.gz') if interrupted else path,r)
        if not interrupted and path.with_suffix('.partial.gz').exists():path.with_suffix('.partial.gz').unlink()
        if interrupted:raise compact.ReviewDue()
        return r

    def load_samples(self,paths):
        chunks=[];refs=[]
        for path in paths:
            self.budget()
            with gzip.open(path,'rt') as f:r=json.load(f)
            if r['status']!='completed':continue
            d=derive(r,self.protocol['positions_per_game'],benchmark.named_seed(self.protocol['seed'],r['pair']['pair_id']))
            if d is not None:chunks.append(d);refs.append(selfplay.reference(path))
        return compact.MemoryData(chunks,refs)

    def collect_expert(self):
        folder=OUTPUT/'expert';folder.mkdir(exist_ok=True);completed=0;groups={'train':[],'validation':[],'test':[]};refs={}
        targets={'train':self.protocol['expert_games']*18//20,
            'validation':self.protocol['expert_games']//20,'test':self.protocol['expert_games']//20}
        imported=folder/'revalidation.json'
        source=self.protocol.get('import_expert_manifest')
        if source:
            if imported.exists():
                audit=json.loads(imported.read_text());require(audit['source_sha256']==self.core,'Revalidation implementation changed')
            else:
                manifest=json.loads((ROOT/source).read_text());audit={'source_sha256':self.core,'source_manifest':selfplay.reference(ROOT/source),'valid':{},'rejected':[]}
                self.state['stage']='audit-prior-expert-visible-rule';self.persist()
                for split,entries in manifest['splits'].items():
                    audit['valid'][split]=[]
                    for ref in entries:
                        self.guard();path=ROOT/ref['path'];require(file_digest(path)==ref['sha256'],'Original expert journal changed')
                        with gzip.open(path,'rt') as f:record=json.load(f)
                        try:compact.replay_journal(record)
                        except ValueError as error:audit['rejected'].append({'record':ref,'reason':str(error)})
                        else:audit['valid'][split].append(ref)
                    self.state['audited_expert_games']=sum(len(v) for v in audit['valid'].values())+len(audit['rejected']);self.persist()
                write(imported,audit)
            for split,entries in audit['valid'].items():
                for ref in entries:
                    path=ROOT/ref['path'];require(file_digest(path)==ref['sha256'],'Imported valid journal changed')
                    groups[split].append(path);refs[path]=ref;completed+=1
            self.state['expert_revalidated']=completed;self.state['expert_rule_rejections']=len(audit['rejected']);self.persist()
        self.state['stage']='new-expert-cohort';self.persist()
        for attempt in range(self.protocol['expert_max_attempts']):
            if completed>=self.protocol['expert_games']:break
            self.guard();pair={'pair_id':f'fresh-visible-expert-{attempt:05d}'}
            for kind in ('board','simulator','agent'):
                pair[kind+'_seed']=benchmark.named_seed(self.protocol['seed'],f'fresh-expert-{attempt}-{kind}')
                require(pair[kind+'_seed'] not in self.known_seeds,'Reserved seed collision')
            path=folder/f'game-{attempt:05d}.json.gz';r=self.expert_game(path,pair)
            if r['status']=='completed':
                # Fill deficits while preserving every imported game's split.
                split=min((k for k in groups if len(groups[k])<targets[k]),key=lambda k:len(groups[k])/targets[k])
                groups[split].append(path);completed+=1
                refs[path]=selfplay.reference(path)
            self.state.update(expert_completed=completed,expert_attempts=attempt+1);self.persist()
            write(folder/'manifest.json',{'rules':{'friendly_robber':True},'source_sha256':self.core,'completed_games':completed,
                'splits':{k:[refs[p] for p in v] for k,v in groups.items()},'format':'gzip seeds/action IDs/chance events; no states/features'})
            print(f'FRESH EXPERT {completed}/{self.protocol["expert_games"]}',flush=True)
        require(completed==self.protocol['expert_games'],'Expert cohort incomplete')
        self.state['expert_completed']=completed;self.persist()
        write(folder/'manifest.json',{'rules':{'friendly_robber':True,'friendly_robber_basis':'visible-victory-points'},
            'source_sha256':self.core,'completed_games':completed,'revalidation':selfplay.reference(imported) if imported.exists() else None,
            'splits':{k:[refs[p] for p in v] for k,v in groups.items()},'format':'gzip seeds/action IDs/chance events; no states/features'})
        return groups

    def replay_data(self):
        # Each cohort is independently large; sample only the current cohort.
        folder=OUTPUT/f'round-{self.current_round:03d}'/'collection'
        return self.load_samples(sorted(folder.glob('*.json.gz')))

    def execute(self):
        groups=self.collect_expert();self.expert=self.load_samples(groups['train']);self.cache=self.load_samples(groups['validation'])
        # train.evaluate needs phase/family tags, already supplied by MemoryData.
        self.parent=self.state['initial_parent'];w4.OUTPUT=OUTPUT/'pretraining';w4.CORPUS=OUTPUT/'expert'
        self.state['stage']='fresh-XD-CNN-initial-training';self.persist()
        pretrain=self.train_w4(self.expert,self.expert);self.state['pretraining']=pretrain;self.persist()
        # No previous learned XD-CNN checkpoint or prior self-play data is imported.
        self.state['initial_parent']=pretrain['checkpoint']
        parent=pretrain['checkpoint'];history=[parent];iteration=1
        while True:
            self.guard();self.current_round=iteration;folder=OUTPUT/f'round-{iteration:03d}';(folder/'collection').mkdir(parents=True,exist_ok=True)
            item=self.state['iterations'].setdefault(str(iteration),{'status':'collecting','parent':parent})
            require(item['parent']==parent,'Fresh cohort lineage changed')
            if item['status']=='completed':parent=item['training']['checkpoint'];history.append(parent);iteration+=1;continue
            model=selfplay.load_checkpoint(parent);completed=0;self.state['stage']=f'round-{iteration}:large-cohort-collection';self.persist()
            for attempt in range(self.protocol['max_attempts_per_iteration']):
                if completed>=self.protocol['completed_games_per_iteration']:break
                self.guard();pair,kind,color=self.game_spec(iteration,attempt,completed)
                old=history[benchmark.named_seed(pair['agent_seed'],'history')%len(history)]
                enemy=selfplay.load_checkpoint(old) if kind=='historical' else model
                r=self.play(folder/'collection'/f'game-{attempt:05d}.json.gz',pair,color,model,enemy,kind,history,'search')
                completed+=r['status']=='completed';item.update(completed_games=completed,attempted_games=attempt+1);self.persist()
            require(completed>=self.protocol['completed_games_per_iteration'],'Incomplete cohort')
            fresh=self.replay_data();self.parent=parent;w4.OUTPUT=folder
            item['training']=self.train_w4(self.expert,fresh);del fresh;self.persist()
            item['evaluation']=self.evaluate_policy(folder,parent,item['training']['checkpoint'])
            item['status']='completed';self.persist();parent=item['training']['checkpoint'];history.append(parent);iteration+=1


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=3);args=parser.parse_args()
    require(0<args.hours<=3,'Windows are bounded to three hours')
    compact.OUTPUT=OUTPUT;compact.STATE=STATE;selfplay.OUTPUT=OUTPUT;selfplay.STATE=STATE
    with benchmark.exclusive_benchmark():
        work=Fresh(args.hours)
        try:work.execute()
        except compact.ReviewDue:
            work.state.update(status='paused',stage='window-ended-or-user-paused',pid=None);work.persist()
        except BaseException as e:
            work.state.update(status='needs-investigation',error=f'{type(e).__name__}: {e}',pid=None);work.persist();raise

if __name__=='__main__':main()
