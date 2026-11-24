"""Bounded parallel native expert cohorts; unchanged pinned W4 trainer."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import time
import zipfile
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT, digest, file_digest
from agent_apis.neural import compact_iterations as compact, fresh_cohorts as fresh, friendly_w4 as w4, selfplay
from agent_apis.neural.parallel_cohorts import CohortLedger
from agent_apis.neural.pipeline import write
from agent_apis.observation import require

OUTPUT=ROOT/'runs/catan-ai/jsettlers-expert-cohorts-visible'
WORKER=None


def split_for(slot):
    return 'train' if slot % 20 < 18 else 'validation' if slot % 20 == 18 else 'test'


class ExpertWorker(fresh.Fresh):
    def __init__(self,core,deadline,stop,parent):
        self.core,self.deadline,self.stop,self.parent_pid=core,deadline,stop,parent
        self.stop_requested=False;self.state={};self.progress=None
    def budget(self):
        if self.stop.is_set() or os.getppid()!=self.parent_pid:raise compact.ReviewDue()
        super().budget()
    def persist(self):
        if self.progress:write(self.progress,{**self.state,'pid':os.getpid(),'cpu_threads':torch.get_num_threads(),'updated_at':self.now()})


def initialize(core,deadline,stop,parent):
    global WORKER
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    WORKER=ExpertWorker(core,deadline,stop,parent)
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,WORKER.request_stop)


def collect_game(assignment,folder):
    folder=ROOT/folder;path=folder/f"game-{assignment['attempt']:05d}.json.gz"
    partial=path.with_suffix('.partial.gz')
    WORKER.progress=folder/'worker-progress'/f"game-{assignment['attempt']:05d}.json"
    if partial.exists() and not path.exists():
        history=folder/'partial-history';history.mkdir(exist_ok=True)
        destination=history/f'{partial.stem}-{file_digest(partial)}.gz'
        if not destination.exists():shutil.copyfile(partial,destination)
    try:record=WORKER.expert_game(path,assignment['pair'])
    except compact.ReviewDue:
        return {'status':'interrupted','journal':selfplay.reference(partial),'replay_verified':True}
    require(torch.get_num_threads()==1,'Expert worker thread count changed')
    return {'status':record['status'],'journal':selfplay.reference(path),'replay_verified':True}


class ExpertCohorts(fresh.Fresh):
    def __init__(self,hours,output=OUTPUT,smoke=False):
        self.output=output;output.mkdir(parents=True,exist_ok=True)
        self.state_path=output/'progress.json';self.smoke=smoke
        prior=json.loads(fresh.STATE.read_text())
        require(prior['status']=='paused','Pause self-play before expert collection')
        self.core=dict(prior['source_sha256'])
        self.core.update({p:file_digest(ROOT/p) for p in ('agent_apis/neural/expert_cohorts.py','agent_apis/neural/parallel_cohorts.py')})
        self.protocol=dict(prior['protocol']);self.protocol.update(seed=20261118041,route='additional JSettlers-vs-JSettlers expert-only cohorts',
            checkpoint_for_next_iteration='best validation checkpoint, earliest tie',
            new_data_sampling='up to 64 seeded positions per game from current expert cohort only',
            import_expert_manifest=None)
        self.parent=prior['initial_parent']
        for item in prior['iterations'].values():
            if item.get('status')=='completed':self.parent=item['training']['checkpoint']
        if self.state_path.exists():
            self.state=json.loads(self.state_path.read_text())
            previous=self.state['source_sha256']
            self.accepted_source_digests={digest(previous),digest(self.core)}
            if previous!=self.core:
                migration=json.loads((output/'repair'/'authorized-migration.json').read_text())
                require(migration['previous']==previous and migration['current']==self.core,'Unapproved source migration')
                self.state['source_migration']=migration
            require(self.state['protocol']==self.protocol,'Expert protocol changed')
            self.state['source_sha256']=self.core
        else:
            self.state={'source_sha256':self.core,'protocol':self.protocol,'initial_parent':self.parent,'iterations':{},
                'installed_default_changed':False,'smoke_only':smoke,'collection_runtime':{'workers':12,'cpu_threads_per_worker':1,'inference_device':'cpu'},
                'native_opponent':benchmark.provider_identity('JSETTLERS')}
            with zipfile.ZipFile(output/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in self.core:z.write(ROOT/p,p)
        self.accepted_source_digests=getattr(self,'accepted_source_digests',{digest(self.core)})
        if self.state.get('source_migration'):
            self.accepted_source_digests.add(digest(self.state['source_migration']['previous']))
        self.parent=self.state['initial_parent']
        require(file_digest(ROOT/self.parent['path'])==self.parent['sha256'],'Parent checkpoint changed')
        self.deadline=time.time()+hours*3600;self.stop_requested=False
        self.state.pop('error',None)
        self.state.update(status='running',pid=os.getpid(),window_started_at=self.now(),review_at=datetime.fromtimestamp(self.deadline,timezone.utc).isoformat())
        signal.signal(signal.SIGTERM,self.request_stop);signal.signal(signal.SIGINT,self.request_stop)
        self.persist()
        self.known_seeds=set()
        for pairs in json.loads(benchmark.SEED_FILE.read_text())['sets'].values():
            for pair in pairs:self.known_seeds.update(pair[k] for k in ('board_seed','simulator_seed','agent_seed'))
        require(torch.__version__=='2.15.0','Pinned torch required')
        if not smoke:require(torch.backends.mps.is_available(),'Pinned MPS optimizer required')

    def persist(self):
        self.state['updated_at']=self.now();write(self.state_path,self.state)

    def assignment(self,iteration,attempt,slot):
        pair={'pair_id':f'expert-continuation-{iteration}-{attempt:05d}'}
        for kind in ('board','simulator','agent'):
            pair[kind+'_seed']=benchmark.named_seed(self.protocol['seed'],f'expert-continuation-{iteration}-{attempt}-{kind}')
            require(pair[kind+'_seed'] not in self.known_seeds,'Reserved seed collision')
        return {'pair':pair,'split':split_for(slot)}

    def collect(self,iteration,item):
        folder=self.output/f'cohort-{iteration:03d}'/'expert';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'plan.json';total=24 if self.smoke else 2000
        identity={'iteration':iteration,'parent':self.parent,'games':total,'source_sha256':digest(self.core)}
        plan=json.loads(path.read_text()) if path.exists() else {'identity':identity,'ledger':None}
        require({k:v for k,v in plan['identity'].items() if k!='source_sha256'}=={k:v for k,v in identity.items() if k!='source_sha256'} and plan['identity']['source_sha256'] in self.accepted_source_digests,'Expert plan identity changed')
        ledger=CohortLedger(total,4000,plan['ledger']);plan['ledger']=ledger.data
        make=lambda attempt,slot:self.assignment(iteration,attempt,slot)
        for key,a in ledger.assignments.items():
            require(all(a[k]==v for k,v in make(int(key),a['slot']).items()),'Expert assignment changed')
            journal=folder/f'game-{int(key):05d}.json.gz'
            if journal.exists():
                if a.get('result'):require(file_digest(journal)==a['result']['journal']['sha256'],'Committed journal changed')
                with gzip.open(journal,'rt') as f:r=json.load(f)
                require(r['pair']==a['pair'] and r['source_sha256'] in self.accepted_source_digests,'Expert journal lineage changed')
                compact.replay_journal(r)
                ledger.finish(int(key),r['status'],{'journal':selfplay.reference(journal),'replay_verified':True})
            else:require(a['status'] not in ('completed','cutoff'),'Missing committed journal')
        write(path,plan)
        ctx=mp.get_context('spawn');stop=ctx.Event();active={}
        self.state['stage']=f'cohort-{iteration}:collect-JSettlers-expert';self.persist()
        with ProcessPoolExecutor(max_workers=12,mp_context=ctx,initializer=initialize,initargs=(self.core,self.deadline,stop,os.getpid())) as pool:
            try:
                while len(ledger.completed_slots)<total:
                    if self.stop_requested or time.time()>=self.deadline:stop.set()
                    if not stop.is_set():
                        self.guard()
                        while len(active)<12:
                            a=ledger.reserve(set(active.values()),make)
                            if a is None:break
                            write(path,plan)
                            active[pool.submit(collect_game,a,str(folder.relative_to(ROOT)))]=a['attempt']
                    if not active:
                        if stop.is_set():raise compact.ReviewDue()
                        require(len(ledger.completed_slots)==total,'Expert attempt budget exhausted');break
                    done,_=wait(active,timeout=.5,return_when=FIRST_COMPLETED)
                    for future in done:
                        attempt=active.pop(future);result=future.result();ledger.finish(attempt,result['status'],result)
                        write(path,plan)
                    if done:
                        item.update(completed_games=len(ledger.completed_slots),dispatched_attempts=ledger.data['next_attempt'])
                        self.persist();print(f'EXPERT COHORT {iteration}: {len(ledger.completed_slots)}/{total}',flush=True)
            finally:stop.set()
        require(len(ledger.completed_slots)==total,'Incomplete expert cohort')
        item.update(completed_games=total,dispatched_attempts=ledger.data['next_attempt']);self.persist()
        groups={k:[] for k in ('train','validation','test')}
        for a in ledger.assignments.values():
            if a['status']=='completed':groups[a['split']].append(a['result']['journal'])
        if not self.smoke:require([len(groups[k]) for k in groups]==[1800,100,100],'Incorrect whole-game split')
        manifest={'completed_games':total,'splits':groups,'source_sha256':self.core,'rules':{'friendly_robber':True,'friendly_robber_basis':'visible-victory-points'},'provider':'JSETTLERS-vs-JSETTLERS'}
        write(folder/'manifest.json',manifest)
        return folder,groups

    def execute(self):
        iteration=1
        while True:
            self.guard()
            item=self.state['iterations'].setdefault(str(iteration),{'status':'collecting','parent':self.parent})
            require(item['parent']==self.parent,'Expert checkpoint lineage changed')
            if item['status']=='completed':self.parent=item['training']['checkpoint'];iteration+=1;continue
            folder,groups=self.collect(iteration,item)
            if self.smoke:
                self.state.update(status='smoke-completed',stage='smoke-completed');self.persist();return
            require(item['completed_games']==2000,'Training requires 2000 completed games')
            for entries in groups.values():
                for ref in entries:require(file_digest(ROOT/ref['path'])==ref['sha256'],'Training journal checksum changed')
            self.state['stage']=f'cohort-{iteration}:derive-seeded-samples';self.persist()
            data=self.load_samples([ROOT/r['path'] for r in groups['train']])
            self.cache=self.load_samples([ROOT/r['path'] for r in groups['validation']])
            w4.CORPUS=folder;w4.OUTPUT=folder.parent
            item['status']='training';self.persist()
            # Both independent sampler halves are expert-only. The pinned W4
            # sampler/loss applies its inverse-probability correction once.
            item['training']=self.train_w4(data,data)
            item['status']='completed';self.parent=item['training']['checkpoint'];self.state['latest_checkpoint']=self.parent
            self.persist();del data;self.cache=None;iteration+=1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hours',type=float,default=3)
    p.add_argument('--smoke-seconds',type=float)
    args=p.parse_args();require(0<args.hours<=3,'At most three hours')
    smoke=args.smoke_seconds is not None
    if smoke:require(0<args.smoke_seconds<=60,'Smoke is at most 60 seconds')
    with benchmark.exclusive_benchmark():
        work=ExpertCohorts(args.smoke_seconds/3600 if smoke else args.hours,OUTPUT.with_name(OUTPUT.name+'-smoke') if smoke else OUTPUT,smoke)
        try:work.execute()
        except compact.ReviewDue:work.state.update(status='paused',stage='window-ended-or-user-paused')
        except BaseException as e:
            work.state.update(status='needs-investigation',error=f'{type(e).__name__}: {e}');raise
        finally:work.state['pid']=None;work.persist()

if __name__=='__main__':main()
