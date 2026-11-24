"""Transformer W4 continuation over corrected whole-game expert cohorts."""
from copy import deepcopy
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import signal
import time
import zipfile
import torch
from agent_apis import benchmark
from agent_apis.experiment import ROOT, file_digest, digest
from agent_apis.neural import fresh_cohorts as fresh, compact_iterations as compact, friendly_w4 as w4, selfplay
from agent_apis.neural.pipeline import write
from agent_apis.observation import require

OUTPUT=ROOT/'runs/catan-ai/transformer-expert-catchup-visible'
CNN=ROOT/'runs/catan-ai/jsettlers-expert-cohorts-visible'
PRIOR=ROOT/'runs/catan-ai/phase5-TRANSFORMER-seed11-ce053344d384-20261105T002653706179Z/run.json'


def validate_manifest(manifest):
    require(manifest['completed_games']==2000,'Require 2000 completed games')
    require(manifest['rules'].get('friendly_robber') and manifest['rules'].get('friendly_robber_basis')=='visible-victory-points','Require visible-point friendly rules')
    groups=manifest['splits']
    require({k:len(v) for k,v in groups.items()}=={'train':1800,'validation':100,'test':100},'Wrong whole-game splits')
    paths=[r['path'] for refs in groups.values() for r in refs]
    require(len(set(paths))==2000,'Duplicate/cross-split journals')
    return groups


class Catchup(fresh.Fresh):
    def __init__(self):
        OUTPUT.mkdir(parents=True,exist_ok=True)
        self.state_path=OUTPUT/'progress.json'
        cnn=json.loads((CNN/'progress.json').read_text())
        self.protocol=deepcopy(cnn['protocol'])
        self.core={p:file_digest(ROOT/p) for p in cnn['source_sha256']}
        self.core['agent_apis/neural/transformer_catchup.py']=file_digest(Path(__file__))
        self.deadline=datetime.fromisoformat(cnn['review_at']).timestamp()
        require(self.deadline>time.time(),'CNN window already ended')
        self.stop_requested=False
        prior=json.loads(PRIOR.read_text())['checkpoint']
        if self.state_path.exists():
            self.state=json.loads(self.state_path.read_text())
            require(self.state['source_sha256']==self.core,'Catchup sources changed')
        else:
            self.state={'status':'initializing','source_sha256':self.core,'initial_parent':prior,'cohorts':{},
                'route':'historical seed11 Transformer continued with W4 over corrected expert cohorts',
                'installed_default_changed':False,'historical_weighting':'W0 preserved; all new training W4',
                'protocol':self.protocol,'started_at':self.now()}
            with zipfile.ZipFile(OUTPUT/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in self.core:z.write(ROOT/p,p)
        self.parent=self.state['initial_parent']
        model=selfplay.load_checkpoint(self.parent)
        require(model.metadata['model_id']=='TRANSFORMER','Expected Transformer parent')
        del model
        require(torch.__version__=='2.15.0' and torch.backends.mps.is_available(),'Pinned MPS required')
        signal.signal(signal.SIGTERM,self.request_stop);signal.signal(signal.SIGINT,self.request_stop)
        self.state.update(status='running',pid=os.getpid(),review_at=cnn['review_at']);self.state.pop('error',None);self.persist()

    def persist(self):
        self.state['updated_at']=self.now();write(self.state_path,self.state)

    def execute(self):
        index=0
        while True:
            self.guard()
            key='original-expert' if index==0 else f'cohort-{index:03d}'
            source=ROOT/'runs/catan-ai/fresh-xdcnn-visible/expert/manifest.json' if index==0 else CNN/key/'expert/manifest.json'
            item=self.state['cohorts'].get(key)
            if item and item['status']=='completed':
                self.parent=item['training']['checkpoint'];index+=1;continue
            if not source.exists():
                self.state['stage']=f'waiting-for-{key}-2000-games';self.persist();time.sleep(5);continue
            manifest=json.loads(source.read_text());groups=validate_manifest(manifest)
            source_ref=selfplay.reference(source)
            if item is None:
                item={'status':'deriving','parent':self.parent,'source_manifest':source_ref};self.state['cohorts'][key]=item
            require(item['parent']==self.parent and item['source_manifest']==source_ref,'Catchup lineage/data changed')
            # Verify all artifacts, including held-out games, without deriving test features.
            for refs in groups.values():
                for ref in refs:
                    self.budget();require(file_digest(ROOT/ref['path'])==ref['sha256'],'Journal checksum changed')
            self.state['stage']=f'{key}:derive-seeded-samples';self.persist()
            data=self.load_samples([ROOT/r['path'] for r in groups['train']])
            self.cache=self.load_samples([ROOT/r['path'] for r in groups['validation']])
            folder=OUTPUT/key;folder.mkdir(exist_ok=True)
            # Local manifest pins input refs without altering shared corpus files.
            write(folder/'manifest.json',manifest)
            w4.CORPUS=folder;w4.OUTPUT=folder
            item['status']='training';self.persist()
            item['training']=self.train_w4(data,data)
            item['status']='completed';self.parent=item['training']['checkpoint'];self.state['latest_checkpoint']=self.parent
            self.persist();del data;self.cache=None;index+=1


def main():
    # Separate process and artifact folders permit simultaneous CNN collection/training.
    work=Catchup()
    try:work.execute()
    except compact.ReviewDue:work.state.update(status='paused',stage='window-ended-or-user-paused')
    except BaseException as e:
        work.state.update(status='needs-investigation',error=f'{type(e).__name__}: {e}');raise
    finally:work.state['pid']=None;work.persist()

if __name__=='__main__':main()
