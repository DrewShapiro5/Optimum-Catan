"""Checksummed derived D2 cache; whole-game order and bounded decision shuffle."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import shutil
import numpy as np

from agent_apis.experiment import ROOT,EXPERIMENTS,file_digest,digest
from agent_apis.observation import feature_names,require
from agent_apis.observation_dataset import _frames,phase
from agent_apis.neural.topology import topology

CACHE_ROOT=ROOT/'runs/catan-ai'


def cache_identity():
    files=['agent_apis/observation.py','agent_apis/observation_dataset.py',
           'agent_apis/dataset_bundle.py','agent_apis/neural/data.py',
           'experiments/catan-ai/feature-schema-v3.json']
    manifests=['data/jsettlers-corpora/full-250/manifest.json','data/jsettlers-corpora/downsampled/manifest.json']
    engine=sorted((ROOT/'framework/catanatron/catanatron').rglob('*.py'))
    return {'schema':1,'view':'D2-complete','feature_schema':3,
            'inputs':{p:file_digest(ROOT/p) for p in files+manifests},
            'engine_sha256':digest({str(p.relative_to(ROOT)):file_digest(p) for p in engine})}


def build_cache():
    identity=cache_identity();folder=CACHE_ROOT/('d2-cache-'+digest(identity)[:16])
    if (folder/'manifest.json').exists():return folder
    partial=folder.with_name(folder.name+'.partial')
    require(not partial.exists(),'A partial cache exists; inspect it before restarting')
    entries=[];total=0
    for directory in ('data/jsettlers-corpora/full-250','data/jsettlers-corpora/downsampled'):
        manifest=json.loads((ROOT/directory/'manifest.json').read_text())
        for entry in manifest['games']:
            path=ROOT/directory/entry['file']
            require(file_digest(path)==entry['sha256'],f'Changed source shard {path}')
            entries.append({**entry,'directory':directory,'offset':total})
            total+=entry['steps']
    require(total==489337 and len(entries)==1267,'Unexpected corpus size')
    require(shutil.disk_usage(ROOT).free>total*(681*4+64)+1_000_000_000,'Insufficient free space for derived cache')
    partial.mkdir(parents=True)
    specs={'features':(np.float32,(total,681)),'legal':(np.uint8,(total,42)),
           'action':(np.int16,(total,)),'value':(np.uint8,(total,)),
           'phase':(np.uint8,(total,)),'family':(np.uint8,(total,)),
           'retained':(np.uint8,(total,))}
    arrays={name:np.lib.format.open_memmap(partial/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
            for name,(dtype,shape) in specs.items()}
    vocabulary=topology()['action_vocabulary'];families=sorted({a[0] for a in vocabulary})
    counts={};games=[]
    for ordinal,entry in enumerate(entries):
        count=0;phase_counts=Counter();family_counts=Counter()
        for _,_,view,row,saved in _frames(ROOT/entry['directory']/entry['file']):
            i=entry['offset']+count
            features=np.asarray(row['features'],dtype=np.float32)
            require(np.isfinite(features).all() and len(features)==681,'Invalid audited features')
            legal=np.zeros(332,dtype=np.uint8);legal[row['legal_action_ids']]=1
            require(legal[row['action_id']]==1 and row['value_target'] in (0,1) and not row['truncated'],'Invalid training target')
            label=phase(view);family=vocabulary[row['action_id']][0]
            arrays['features'][i]=features;arrays['legal'][i]=np.packbits(legal,bitorder='little')
            arrays['action'][i]=row['action_id'];arrays['value'][i]=row['value_target']
            arrays['phase'][i]=['setup','middle','endgame'].index(label)
            arrays['family'][i]=families.index(family);arrays['retained'][i]=saved is not None
            phase_counts[label]+=1;family_counts[family]+=1;count+=1
        require(count==entry['steps'],f'Reconstructed decision count differs: {entry["file"]}')
        games.append({k:entry[k] for k in ('game_id','seed','split','offset','steps','directory','file','sha256')})
        c=counts.setdefault(entry['split'],{'games':0,'decisions':0,'phases':Counter(),'families':Counter()})
        c['games']+=1;c['decisions']+=count;c['phases'].update(phase_counts);c['families'].update(family_counts)
        if (ordinal+1)%25==0:print(f'Cached {ordinal+1}/{len(entries)} games',flush=True)
    for a in arrays.values():a.flush()
    checksums={name+'.npy':file_digest(partial/(name+'.npy')) for name in arrays}
    manifest={'schema_version':1,'identity':identity,'identity_sha256':digest(identity),
              'created_at':datetime.now(timezone.utc).isoformat(),'feature_names':list(feature_names()),
              'topology_sha256':topology()['sha256'],'families':families,'counts':counts,
              'games':games,'rows':total,'files':checksums}
    (partial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    partial.rename(folder)
    print(folder,flush=True)
    return folder


class Cache:
    def __init__(self,folder,verify=True):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text())
        require(self.manifest['identity_sha256']==digest(cache_identity()),'Derived cache is stale')
        if verify:
            for name,checksum in self.manifest['files'].items():require(file_digest(self.folder/name)==checksum,f'Corrupt cache {name}')
        self.arrays={name:np.load(self.folder/(name+'.npy'),mmap_mode='r')
                     for name in ('features','legal','action','value','phase','family','retained')}
    def indices(self,split):
        return np.concatenate([np.arange(g['offset'],g['offset']+g['steps'],dtype=np.int64)
                               for g in self.manifest['games'] if g['split']==split])
    def batch(self,indices):
        a=self.arrays
        return {'features':np.array(a['features'][indices],copy=True),
                'legal':np.unpackbits(a['legal'][indices],axis=1,bitorder='little')[:,:332].astype(bool),
                'action':np.asarray(a['action'][indices],dtype=np.int64),
                'value':np.asarray(a['value'][indices],dtype=np.float32),
                'phase':np.array(a['phase'][indices]),'family':np.array(a['family'][indices])}


class DecisionShuffle:
    """Each epoch visits every training decision exactly once, with a bounded buffer."""
    def __init__(self,games,seed,buffer_size=8192,state=None):
        self.games=[g for g in games if g['split']=='train'];self.rng=random.Random(seed)
        self.buffer_size=buffer_size;self.epoch=0;self.buffer=[];self.order=[];self.game_cursor=0;self.row_cursor=0
        if state:
            for k in ('epoch','buffer','order','game_cursor','row_cursor'):setattr(self,k,state[k])
            self.rng.setstate(state['rng'])
    def _source(self):
        while self.game_cursor<len(self.order):
            g=self.games[self.order[self.game_cursor]]
            if self.row_cursor<g['steps']:
                i=g['offset']+self.row_cursor;self.row_cursor+=1;return i
            self.game_cursor+=1;self.row_cursor=0
        return None
    def _epoch(self):
        self.order=list(range(len(self.games)));self.rng.shuffle(self.order)
        self.game_cursor=self.row_cursor=0;self.epoch+=1
        self.buffer=[]
        for _ in range(self.buffer_size):
            value=self._source()
            if value is None:break
            self.buffer.append(value)
    def take(self,count):
        result=[]
        for _ in range(count):
            if not self.buffer:self._epoch()
            j=self.rng.randrange(len(self.buffer));result.append(self.buffer[j])
            value=self._source()
            if value is None:self.buffer[j]=self.buffer[-1];self.buffer.pop()
            else:self.buffer[j]=value
        return np.asarray(result,dtype=np.int64)
    def state_dict(self):
        return {k:getattr(self,k) for k in ('epoch','buffer','order','game_cursor','row_cursor')}|{'rng':self.rng.getstate()}


if __name__=='__main__':
    print(build_cache())
