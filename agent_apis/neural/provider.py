"""Checkpoint-verified observation-only pure-policy and neural-search providers.

Keep torch imports lazy so optional training dependencies do not break discovery
of unrelated providers on a fresh installation.
"""
import importlib.util
import json
from pathlib import Path
import random
import time
from functools import lru_cache
import threading
import numpy as np
from catanatron.models.player import Player,Color
from catanatron.serialization import action_from_json,action_to_json,web_view
from catanatron.gym.envs.action_space import from_action_space
from agent_apis.experiment import ROOT,EXPERIMENTS,file_digest
from agent_apis.observation import encode_observation,from_web_view,require,validate_rules

DEFAULT=EXPERIMENTS/'phase5-selected.json'
COLORS=(Color.BLUE,Color.RED)


class NeuralCheckpoint:
    def __init__(self,path,device=None):
        import torch
        from agent_apis.neural.models import build_model
        from agent_apis.neural.topology import topology
        saved=torch.load(path,map_location='cpu',weights_only=True);meta=saved['metadata']
        is_exphormer = meta.get('model_id') == 'TYPED-EXPHORMER'
        expected_fmt = 2 if is_exphormer else 1
        require(meta['format_version']==expected_fmt and meta['feature_schema_version']==3 and meta['action_count']==332,'Incompatible checkpoint schema')
        require(meta['topology_sha256']==topology()['sha256'],'Checkpoint tensor topology differs')
        require(meta['feature_schema_sha256']==file_digest(EXPERIMENTS/'feature-schema-v3.json'),'Checkpoint feature schema differs')
        require(meta['model_implementation_sha256']==file_digest(ROOT/'agent_apis/neural/models.py'),'Checkpoint model implementation differs')
        from agent_apis.neural.recipes import validate_recipe
        recipe=validate_recipe(meta['configuration']['data'])
        require(recipe=='D2-W0' or meta['configuration'].get('phase')==6 or
                (meta['configuration'].get('phase')==9 and recipe=='D2-W4'),
                'Expected a baseline, phase-6, or W4 self-play checkpoint')
        self.metadata=meta;self.path=Path(path);self.device=device or meta['inference_device']
        require(self.device!='mps' or (hasattr(torch.backends,'mps') and torch.backends.mps.is_available()),'Requested MPS inference is unavailable; configure CPU explicitly')
        require(not str(self.device).startswith('cuda') or torch.cuda.is_available(),'Requested CUDA inference is unavailable; configure CPU explicitly')
        torch.set_num_threads(meta['configuration']['runtime']['cpu_threads'])
        if is_exphormer:
            from agent_apis.neural.value_ablation_models import TypedExphormer
            self.model=TypedExphormer().to(self.device)
        else:
            self.model=build_model(meta['model_id']).to(self.device)
        self.model.load_state_dict(saved['state_dict']);self.model.eval()

    def evaluate_batch(self,views):
        import torch
        from agent_apis.stochastic import action_id,SearchObservation
        legal=[];features=[]
        for view in views:
            validate_rules(view)
            ids=[action_id(action_from_json(a)) for a in view['current_playable_actions']]
            require(ids,'No legal neural actions');legal.append(ids);features.append(view.features if isinstance(view,SearchObservation) else encode_observation(view))
        raw=torch.tensor(features,dtype=torch.float32,device=self.device)
        with torch.inference_mode():
            logits,value=self.model(raw)
            scores=logits.cpu().numpy().astype(np.float64);values=value.sigmoid().cpu().numpy()
        result=[]
        for row,ids,probability in zip(scores,legal,values):
            selected=row[ids];probability=float(probability)
            require(np.isfinite(selected).all() and np.isfinite(probability) and 0<=probability<=1,'Nonfinite neural prediction')
            weights=np.exp(selected-selected.max());weights/=weights.sum()
            result.append(({int(i):float(p) for i,p in zip(ids,weights)},probability))
        return result

    def evaluate(self,view):
        return self.evaluate_batch([view])[0]

    def probabilities(self,view):
        if len(view['current_playable_actions'])==1:
            from agent_apis.stochastic import action_id
            return {action_id(action_from_json(view['current_playable_actions'][0])):1.}
        return self.evaluate(view)[0]


class NeuralRuntime:
    def __init__(self,path,mode,device=None):
        self.policy=NeuralCheckpoint(path,device=device);self.mode=mode;self.opponent=None;self.opponent_sha=None
        self.last_search_metrics=None
        self.search_session=None
        self.lock=threading.RLock()
    def reset_search(self):
        if self.search_session is not None:self.search_session.clear()
    def benchmark_choose(self,view,context,rng):
        validate_rules(view)
        if len(view['current_playable_actions'])==1:return view['current_playable_actions'][0]
        if self.mode=='policy':
            require(context.mode=='policy','Pure policy requires policy mode')
            probabilities,value=self.policy.evaluate(view)
            selected=max(probabilities,key=lambda i:(probabilities[i],-i))
            action_json=action_to_json(from_action_space(selected,Color[view['perspective']],COLORS,'BASE'))
            from agent_apis import debug_telemetry
            debug_telemetry.publish_policy_result(
                view, probabilities, value, action_json, session_id=getattr(context, 'session_id', None)
            )
            return action_json
        require(context.mode in ('time','simulations'),'Neural search requires an explicit search budget')
        settings=context.settings['search'];frozen=settings['opponent_checkpoint']
        require(frozen and file_digest(ROOT/frozen['path'])==frozen['sha256'],'Frozen opponent checkpoint missing or changed')
        with self.lock:
            if self.opponent is None:
                self.opponent=NeuralCheckpoint(ROOT/frozen['path'],device='cpu');self.opponent_sha=frozen['sha256']
                meta=self.opponent.metadata
                require(meta['model_id']=='MLP' and meta['training_seed']==11,'Frozen search opponent must be seed-11 D2/W0 MLP')
                require(meta['configuration']['data']['composition']=='D2' and meta['configuration']['data']['weighting']=='W0',
                        'Frozen search opponent must retain its D2/W0 recipe')
        require(self.opponent_sha==frozen['sha256'],'Frozen opponent changed within a match')
        from agent_apis.stochastic_search import StochasticSearch
        count=context.simulations if context.mode=='simulations' else 1000000
        if settings.get('reuse_subtree',False):
            from agent_apis.neural.subtree_reuse import SearchSession
            with self.lock:
                if self.search_session is None:self.search_session=SearchSession()
                result=self.search_session.run(view,rng.getrandbits(64),self.policy,self.opponent,
                    settings.get('workers',1),count,context,settings['c_puct'],settings.get('max_actions',128),
                    session_id=getattr(context,'session_id',None),
                    model_revision=(self.policy.metadata['configuration_sha256'],self.policy.metadata['step'],self.opponent_sha),
                    mcts_opponent=settings.get('mcts_opponent','frozen-mlp'))
        elif settings.get('workers',1)>1:
            from agent_apis.neural.parallel_search import parallel_search
            result=parallel_search(view,rng.getrandbits(64),self.policy,self.opponent,
                settings['workers'],count,context,settings['c_puct'],settings.get('max_actions',128))
        else:
            result=StochasticSearch(view,seed=rng.getrandbits(64),evaluator=self.policy,opponent=self.opponent,
                c_puct=settings['c_puct'],max_root_depth=None,
                max_actions=settings.get('max_actions',128),merged=True,
                mcts_opponent=settings.get('mcts_opponent','frozen-mlp')).run(count,context)
        self.last_search_metrics=result['metrics']
        return result['action']


class NeuralPolicyPlayer(Player):
    LABEL='Neural policy (phase 5)';MODE='policy'
    @staticmethod
    def availability():
        if importlib.util.find_spec('torch') is None:return False,'Install the experiment optional dependency'
        return True,''
    @classmethod
    def from_checkpoint(cls,path):return NeuralRuntime(path,cls.MODE)
    @classmethod
    def choose(cls,observation):
        started=time.perf_counter()
        require(DEFAULT.exists(),'No selected neural checkpoint is configured yet')
        settings=json.loads(DEFAULT.read_text());reference=settings['checkpoint']
        require(file_digest(ROOT/reference['path'])==reference['sha256'],'Selected checkpoint changed')
        opponent=settings['search'].get('opponent_checkpoint')
        color=observation.get('perspective') or observation['colors'][observation['current_player_index']]
        session_key=(observation.get('game',{}).get('id'),color) if cls.MODE=='search' else None
        runtime=cached_runtime(str(ROOT/reference['path']),reference['sha256'],cls.MODE,
                               opponent['sha256'] if opponent else None,session_key)
        from agent_apis.benchmark import DecisionContext
        safe=from_web_view(observation,color)
        context=DecisionContext(started,500,'policy' if cls.MODE=='policy' else 'time',None,settings)
        context.session_id=observation.get('game',{}).get('id')
        return runtime.benchmark_choose(safe,context,random.Random(11))
    def decide(self,game,playable_actions):
        if len(playable_actions)==1:return playable_actions[0]
        return action_from_json(self.choose(web_view(game,self.color)))


class NeuralSearchPlayer(NeuralPolicyPlayer):
    LABEL='Neural stochastic search (phase 5)';MODE='search'


@lru_cache(4)
def cached_runtime(path,checksum,mode,opponent_checksum,session_key=None,device=None):
    require(file_digest(path)==checksum,'Cached model checkpoint changed')
    return NeuralRuntime(path,mode,device=device)
