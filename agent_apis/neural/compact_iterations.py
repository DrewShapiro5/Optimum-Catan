"""Time-bounded W4 iterations with action/event-only persistent game journals."""
import argparse
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
from catanatron.serialization import action_record_to_json, state_to_json, web_view, map_to_json
from agent_apis import benchmark
from agent_apis.experiment import ROOT,EXPERIMENTS,digest,file_digest
from agent_apis.observation import from_web_view,encode_observation,require
from agent_apis.neural import selfplay,friendly_w4 as w4
from agent_apis.neural.pipeline import write
from agent_apis.jsettlers.player import BRIDGE

OUTPUT=ROOT/'runs/catan-ai/compact-w4'
STATE=ROOT/'runs/catan-ai/compact-w4-progress.json'
PROTOCOL=EXPERIMENTS/'compact-w4.json'


class ReviewDue(Exception):pass


def snapshot_digest(game):
    doc=state_to_json(game)
    doc['game']={k:v for k,v in doc['game'].items() if k!='id'}
    return digest(doc)


def save_journal(path,record):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            stream.write(json.dumps(record,separators=(',',':'),allow_nan=False).encode())
    temporary.replace(path)


def replay_journal(record,derive=False):
    """Replay recorded legal action IDs; reproduce every actual chance event."""
    game=benchmark.make_game(record['pair'],friendly_robber=True)
    require(snapshot_digest(game)==record['initial_sha256'],'Initial board/RNG/rules changed')
    rows=[]
    for event in record['events']:
        legal={selfplay.action_id(a):a for a in game.playable_actions}
        require(event['a'] in legal,'Illegal replay action')
        if event.get('v'):
            ids=[x[0] for x in event['v']];visits=np.asarray([x[1] for x in event['v']],np.float64)
            require(len(ids)==len(set(ids)) and set(ids)<=set(legal) and (visits>0).all(),'Invalid sparse search targets')
            if derive:
                actor=game.state.current_color()
                safe=from_web_view(web_view(game,actor),actor.value)
                mask=np.zeros(332,bool);mask[list(legal)]=True
                policy=np.zeros(332,np.float32);policy[ids]=visits/visits.sum()
                rows.append({'features':encode_observation(safe),'legal':mask,'policy':policy,'actor':actor.value})
        result=game.execute(legal[event['a']])
        require(action_record_to_json(result)==event['event'],'Chance-event replay mismatch')
    if 'final_sha256' in record:
        require(snapshot_digest(game)==record['final_sha256'],'Final-state/RNG replay mismatch')
    else:
        require(record['status']=='in-progress','Missing committed final checksum')
    winner=game.winning_color()
    if 'winner' in record:
        require((winner.value if winner else None)==record['winner'],'Replay outcome mismatch')
    if derive:return selfplay.completed_rows(rows,winner)
    return game


class MemoryData(w4.Data):
    """Same W4 objective as friendly_w4.Data, with derived arrays only in RAM."""
    def __init__(self,chunks,references):
        require(chunks,'No completed searched examples')
        for d in chunks:w4.check_arrays(d)
        self.arrays={k:np.concatenate([d[k] for d in chunks]) for k in ('features','legal','policy','value')}
        self.arrays['action']=self.arrays['policy'].argmax(1).astype(np.int64)
        self.arrays['family']=w4.ACTION_FAMILY[self.arrays['action']]
        self.arrays['phase']=np.ones(len(self.arrays['value']),np.uint8)
        self.manifest={'families':w4.FAMILIES}
        n=len(self.arrays['value']);counts=np.zeros(len(w4.FAMILIES));family_mass=np.zeros((n,len(counts)))
        for action,family in enumerate(w4.ACTION_FAMILY):family_mass[:,family]+=self.arrays['policy'][:,action]
        counts=family_mass.sum(0);active=counts>0;mass=np.zeros_like(counts)
        mass[active]=w4.balanced_mass(counts[active])
        self.factors=.5/n+.5*np.divide(mass,counts,out=np.zeros_like(mass),where=active)
        self.q=family_mass@self.factors
        require(np.isclose(self.q.sum(),1) and (self.q>0).all() and np.max(n*self.q)<=5+1e-6,'Invalid W4 replay distribution')
        self.summary={'rows':n,'dataset_references_sha256':digest(references),
            'max_relative_sampling_probability':float(np.max(n*self.q)),
            'value_weight_expectation':float(np.sum(self.q/(n*self.q))),
            'probabilities_sha256':digest(self.q.tolist())}


class Controller(w4.Workflow):
    def __init__(self,hours):
        OUTPUT.mkdir(parents=True,exist_ok=True)
        self.protocol=json.loads(PROTOCOL.read_text())
        paths=(*selfplay.CORE,'agent_apis/neural/friendly_w4.py','agent_apis/neural/recipes.py',
            'agent_apis/neural/compact_iterations.py','experiments/catan-ai/compact-w4.json',
            'experiments/catan-ai/training-defaults.json')
        engine=sorted((ROOT/'framework/catanatron/catanatron').rglob('*.py'))
        self.core={p:file_digest(ROOT/p) for p in paths}
        self.core.update({str(p.relative_to(ROOT)):file_digest(p) for p in engine})
        self.protocol['seed']=2026111703
        prior=json.loads((ROOT/'runs/catan-ai/friendly-w4/iteration-1/training.json').read_text())
        if STATE.exists():
            self.state=json.loads(STATE.read_text())
            require(self.state['source_sha256']==self.core,'Compact runner sources changed')
        else:
            self.state={'status':'running','stage':'initializing','source_sha256':self.core,
                'protocol':self.protocol,'initial_parent':prior['checkpoint'],'iterations':{},
                'installed_default_changed':False,'final_seeds_used':False}
            # One immutable source archive, not one snapshot per game/iteration.
            with zipfile.ZipFile(OUTPUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for p in self.core:archive.write(ROOT/p,p)
            self.state['source_archive']=selfplay.reference(OUTPUT/'source.zip')
        now=time.time();self.state.update(status='running',pid=os.getpid(),window_started_at=datetime.fromtimestamp(now,timezone.utc).isoformat(),
            review_at=datetime.fromtimestamp(now+hours*3600,timezone.utc).isoformat())
        self.deadline=now+hours*3600;self.stop_requested=False
        signal.signal(signal.SIGTERM,self.request_stop);signal.signal(signal.SIGINT,self.request_stop)
        self.persist()
        require(torch.__version__=='2.15.0' and torch.backends.mps.is_available(),'Pinned MPS unavailable')
        defaults=json.loads((EXPERIMENTS/'training-defaults.json').read_text());require(defaults['weighting']=='W4' and defaults['friendly_robber'],'W4/friendly defaults required')
        self.frozen_ref=json.loads((EXPERIMENTS/'phase5-frozen-opponent.json').read_text())['checkpoint']
        self.frozen=selfplay.load_checkpoint(self.frozen_ref)
        self.manifest=json.loads((w4.CORPUS/'manifest.json').read_text())
        self.expert=w4.Data(self.manifest['games'],'train');self.cache=w4.Data(self.manifest['games'],'validation')
        self.state['native_opponent']=benchmark.provider_identity('JSETTLERS');self.persist()
        self.known_seeds=set()
        for folder in ('full-250','downsampled'):
            self.known_seeds.update(g['seed'] for g in json.loads((ROOT/f'data/jsettlers-corpora/{folder}/manifest.json').read_text())['games'])
        for pairs in json.loads((EXPERIMENTS/'evaluation-seeds-v1.json').read_text())['sets'].values():
            for pair in pairs:self.known_seeds.update(pair[k] for k in ('board_seed','simulator_seed','agent_seed'))
        for g in self.manifest['games']:self.known_seeds.update(g['pair'][k] for k in ('board_seed','simulator_seed','agent_seed'))

    def request_stop(self,*args):self.stop_requested=True

    def budget(self):
        if self.stop_requested or time.time()>=self.deadline:raise ReviewDue()

    def guard(self):
        self.budget();super().guard()

    def game_spec(self,iteration,attempt,completed_slot=None):
        return selfplay.SelfPlay.game_spec(self,iteration,attempt,completed_slot)

    def play(self,path,pair,candidate_color,model,enemy_model,kind,history,mode):
        model_ref=selfplay.reference(model.path) if hasattr(model,'path') else None
        enemy_ref=selfplay.reference(enemy_model.path) if hasattr(enemy_model,'path') else None
        if path.exists():
            with gzip.open(path,'rt') as stream:record=json.load(stream)
            require(record['pair']==pair and record['source_sha256']==digest(self.core),'Committed compact game changed')
            require(record['candidate_checkpoint']==model_ref and record['opponent_checkpoint']==enemy_ref,'Compact game model lineage changed')
            replay_journal(record)
            return record
        game=benchmark.make_game(pair,friendly_robber=True);initial=snapshot_digest(game);BRIDGE.close()
        other=next(c for c in benchmark.COLORS if c!=candidate_color)
        seats={candidate_color:selfplay.SearchSeat(model,self.frozen,benchmark.named_seed(pair['agent_seed'],candidate_color.value),self.protocol,game.id)}
        if kind!='JSETTLERS':seats[other]=selfplay.SearchSeat(enemy_model,self.frozen,benchmark.named_seed(pair['agent_seed'],other.value),self.protocol,game.id)
        native=benchmark.Agent('JSETTLERS',other,pair['agent_seed']) if kind=='JSETTLERS' else None
        record={'pair':pair,'candidate_color':candidate_color.value,'opponent_kind':kind,'mode':mode,
            'candidate_checkpoint':model_ref,'opponent_checkpoint':enemy_ref,
            'frozen_search_opponent':getattr(self,'frozen_ref',None),
            'source_sha256':digest(self.core),'initial_sha256':initial,'events':[],
            'board_sha256':digest(map_to_json(game.state.board.map)),'simulator_seed':pair['simulator_seed']}
        started=time.perf_counter();searched=simulations=0;status='cutoff';interrupted=False
        partial=path.with_suffix('.partial.gz')
        self.state['active_game']={'file':str(path.relative_to(ROOT)),'turns':0,'actions':0};self.persist()
        try:
            while game.winning_color() is None and game.state.num_turns<1000 and len(record['events'])<10000:
                self.budget();actor=game.state.current_color();legal=list(game.playable_actions);visits=None
                if len(legal)==1:action=legal[0]
                elif actor in seats:
                    safe=from_web_view(web_view(game,actor),actor.value)
                    if mode=='policy':
                        probabilities=model.probabilities(safe);selected=max(probabilities,key=lambda i:(probabilities[i],-i))
                        action=next(a for a in legal if selfplay.action_id(a)==selected)
                    else:
                        action,target,metrics=seats[actor].choose(safe);simulations+=metrics['simulations']
                        if actor==candidate_color or kind=='current':
                            visits=[[int(i),int(e.visits)] for i,e in sorted(seats[actor].tree.nodes[()].edges.items()) if e.visits]
                            searched+=1
                else:
                    action=native.select(game,legal,benchmark.DecisionContext(time.perf_counter(),500,'policy',None));native.cold=False
                require(action in legal,'Illegal compact collection action')
                event={'a':selfplay.action_id(action)}
                if visits:event['v']=visits
                event['event']=action_record_to_json(game.execute(action));record['events'].append(event)
                if len(record['events'])%10==0:
                    self.state['active_game'].update(turns=game.state.num_turns,actions=len(record['events']),elapsed_seconds=time.perf_counter()-started,searched=searched)
                    # Periodic compressed action/event journal supports recovery
                    # after abrupt power loss, without any saved board states.
                    save_journal(partial,{**record,'status':'in-progress'})
                    self.persist()
            winner=game.winning_color();status='completed' if winner else 'cutoff'
        except ReviewDue:
            winner=game.winning_color();status='interrupted';interrupted=True
        finally:BRIDGE.close()
        record.update(status=status,winner=winner.value if winner else None,turns=game.state.num_turns,
            final_sha256=snapshot_digest(game),elapsed_seconds=time.perf_counter()-started,
            simulations=simulations,searched_rows=searched,match_score=float(winner==candidate_color) if winner else .5)
        replay_journal(record)
        save_journal(partial if interrupted else path,record)
        if not interrupted and partial.exists():partial.unlink()
        if interrupted:raise ReviewDue()
        return record

    def replay_data(self):
        chunks=[];references=[]
        previous=json.loads((ROOT/'runs/catan-ai/selfplay-small-round/iteration-1/combined-collection.json').read_text())
        for r in previous:
            if not r['dataset']:continue
            ref=r['dataset'];require(file_digest(ROOT/ref['path'])==ref['sha256'],'Historical self-play changed')
            with np.load(ROOT/ref['path']) as f:chunks.append({k:f[k].copy() for k in ('features','legal','policy','value')})
            references.append(ref)
        for path in sorted(OUTPUT.glob('round-*/collection/*.json.gz')):
            with gzip.open(path,'rt') as stream:r=json.load(stream)
            if r['status']!='completed':continue
            data=replay_journal(r,derive=True)
            if data is not None:chunks.append(data);references.append(selfplay.reference(path))
        return MemoryData(chunks,references)

    def evaluate_policy(self,folder,parent,candidate):
        pairs=json.loads((EXPERIMENTS/'evaluation-seeds-v1.json').read_text())['sets']['tuning'][:self.protocol['evaluation_pairs']]
        results={}
        for label,ref in [('parent',parent),('candidate',candidate)]:
            model=selfplay.load_checkpoint(ref);games=[]
            sub=folder/'evaluation'/label;sub.mkdir(parents=True,exist_ok=True)
            for pair in pairs:
                for color in benchmark.COLORS:
                    self.guard()
                    game=self.play(sub/f'{pair["pair_id"]}-{color.value}.json.gz',pair,color,model,None,'JSETTLERS',[],mode='policy')
                    require(game['status']!='interrupted','Interrupted evaluation')
                    games.append({k:game[k] for k in ('candidate_color','board_sha256','simulator_seed','match_score','status','winner') }|{'pair_id':pair['pair_id']})
                    self.state['evaluation_games_current_condition']=len(games);self.persist()
            results[label]=games
        result={'mode':'noise-free pure policy; not a fixed-search-budget strength claim',
            'paired_difference':benchmark.paired_difference(results['candidate'],results['parent']),
            'games':results,'installed_default_changed':False,'final_seeds_used':False}
        write(folder/'policy-evaluation.json',result);return result

    def execute(self):
        parent=self.state['initial_parent'];history=[parent];iteration=1
        while True:
            self.guard();key=str(iteration);folder=OUTPUT/f'round-{iteration:03d}'
            (folder/'collection').mkdir(parents=True,exist_ok=True)
            item=self.state['iterations'].setdefault(key,{'status':'collecting','parent':parent})
            require(item['parent']==parent,'Round lineage changed')
            if item['status']=='completed':
                parent=item['training']['checkpoint'];history.append(parent);iteration+=1;continue
            model=selfplay.load_checkpoint(parent);completed=0
            self.state['stage']=f'round-{iteration}:collection';self.persist()
            for attempt in range(self.protocol['max_attempts_per_iteration']):
                if completed>=self.protocol['completed_games_per_iteration']:break
                self.guard();pair,kind,color=self.game_spec(iteration,attempt,completed)
                old=history[benchmark.named_seed(pair['agent_seed'],'history')%len(history)]
                enemy=selfplay.load_checkpoint(old) if kind=='historical' else model
                record=self.play(folder/'collection'/f'game-{attempt:05d}.json.gz',pair,color,model,enemy,kind,history,mode='search')
                completed+=record['status']=='completed'
                item.update(completed_games=completed,attempted_games=attempt+1,cutoffs=attempt+1-completed)
                self.persist();print(f'ROUND {iteration} collection {completed}/{self.protocol["completed_games_per_iteration"]}',flush=True)
            require(completed>=self.protocol['completed_games_per_iteration'],'Too many cutoff games')
            self.guard();new=self.replay_data();self.parent=parent
            self.state['stage']=f'round-{iteration}:W4-training';self.persist()
            # Reuse the audited W4 trainer; change only the output namespace.
            w4.OUTPUT=folder
            item['training']=self.train_w4(self.expert,new);del new
            self.persist();self.guard()
            self.state['stage']=f'round-{iteration}:policy-evaluation';self.persist()
            item['evaluation']=self.evaluate_policy(folder,parent,item['training']['checkpoint'])
            item.update(status='completed',finished_at=self.now());self.persist()
            print(f'ROUND {iteration} completed',flush=True)
            parent=item['training']['checkpoint'];history.append(parent);iteration+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--hours',type=float,default=3)
    args=parser.parse_args();require(0<args.hours<=3,'Each authorized window is at most three hours')
    selfplay.STATE=STATE;selfplay.OUTPUT=OUTPUT
    with benchmark.exclusive_benchmark():
        work=Controller(args.hours)
        try:work.execute()
        except ReviewDue:
            work.state.update(status='awaiting-review',stage='three-hour-review-due' if not work.stop_requested else 'paused-by-user',pid=None)
            work.persist();print('Stopped at a resumable boundary for review.',flush=True)
        except BaseException as error:
            work.state.update(status='needs-investigation',error=f'{type(error).__name__}: {error}',pid=None);work.persist();raise

if __name__=='__main__':main()
