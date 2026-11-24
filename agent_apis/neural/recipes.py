"""Phase-6 training distributions; all retrospective fields stay outside model inputs."""
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from agent_apis.experiment import digest
from agent_apis.observation import require
from agent_apis.neural.data import DecisionShuffle

RECIPES=('D0-W0','D1-W0','D2-W0','D2-W1','D2-W2','D2-W3-V0','D2-W3-V1','D2-W4')


def validate_recipe(data):
    composition=data['composition'];weighting=data['weighting']
    variant=data.get('value_weighting','V0')
    name=f'{composition}-{weighting}'+(f'-{variant}' if weighting=='W3' else '')
    require(name in RECIPES,f'Unsupported training recipe: {name}')
    require(variant in ('V0','V1') and (weighting=='W3' or variant=='V0'),'Unexpected value weighting')
    return name


def balanced_mass(counts,cap=9.):
    """Equal family mass, redistributed when the extra stream's row cap binds."""
    counts=np.asarray(counts,dtype=np.float64)
    require(np.all(counts>0),'Empty family pool')
    limits=cap*counts/counts.sum();result=np.zeros(len(counts));active=np.ones(len(counts),dtype=bool)
    while active.any():
        share=(1-result.sum())/active.sum()
        clipped=active&(limits<share)
        if not clipped.any():result[active]=share;break
        result[clipped]=limits[clipped];active[clipped]=False
    require(np.isclose(result.sum(),1),'Family probabilities do not sum to one')
    return result


class RecipeSampler:
    def __init__(self,cache,data,seed,buffer_size=8192,state=None):
        self.recipe=validate_recipe(data);self.weighting=data['weighting'];self.variant=data.get('value_weighting','V0')
        self.cache=cache;self.rng=np.random.default_rng(seed);self.epoch=0;self.draws=0
        self.games=[g for g in cache.manifest['games'] if g['split']=='train' and
                    (data['composition']!='D0' or g['directory'].endswith('/full-250'))]
        all_indices=np.concatenate([np.arange(g['offset'],g['offset']+g['steps']) for g in self.games])
        if data['composition']=='D1':all_indices=all_indices[cache.arrays['retained'][all_indices].astype(bool)]
        self.indices=all_indices;self.n=len(all_indices)
        require(self.n>0,'Empty training distribution')
        self.probabilities=np.full(self.n,1/self.n,dtype=np.float64)
        self.policy_weights=np.ones(self.n);self.value_weights=np.ones(self.n)
        legal=np.unpackbits(cache.arrays['legal'][all_indices],axis=1,bitorder='little')[:,:332]
        nonforced=legal.sum(1)>1
        self.shuffle=None
        if self.weighting=='W0' and data['composition']!='D1':
            self.shuffle=DecisionShuffle(self.games,seed,buffer_size,state.get('shuffle') if state else None)
        if self.weighting=='W1':
            cursor=0
            for game in self.games:
                self.probabilities[cursor:cursor+game['steps']]=1/(len(self.games)*game['steps']);cursor+=game['steps']
        elif self.weighting=='W2':
            phases=cache.arrays['phase'][all_indices];labels,counts=np.unique(phases,return_counts=True)
            require(len(labels)==3,'Phase mixture requires all three pools')
            self.probabilities=.5/self.n+.5/(3*counts[np.searchsorted(labels,phases)])
            self.value_weights=1/(self.n*self.probabilities)
        elif self.weighting=='W3':
            progress=np.empty(self.n);cursor=0
            # Feature 638 is the permitted decision actor color, not a future field.
            require(cache.manifest['feature_names'][638]=='ACTOR_IS_BLUE','Actor feature ordering changed')
            for game in self.games:
                actors=cache.arrays['features'][game['offset']:game['offset']+game['steps'],638]
                for actor in np.unique(actors):
                    rows=np.flatnonzero(actors==actor)
                    # Zero-based ordinal / final ordinal, including forced decisions.
                    progress[cursor+rows]=np.arange(len(rows))/max(1,len(rows)-1)
                cursor+=game['steps']
            self.policy_weights=1+2*np.exp(-progress/.10)+2*np.exp(-(1-progress)/.10)
            self.policy_weights/=self.policy_weights[nonforced].mean()
            if self.variant=='V1':
                self.value_weights=1+np.exp(-(1-progress)/.10)
                self.value_weights/=self.value_weights.mean()
        elif self.weighting=='W4':
            families=cache.arrays['family'][all_indices];labels,counts=np.unique(families,return_counts=True)
            mass=balanced_mass(counts)
            self.probabilities=.5/self.n+.5*mass[np.searchsorted(labels,families)]/counts[np.searchsorted(labels,families)]
            require(np.max(self.n*self.probabilities)<=5+1e-10,'Rare-family oversampling cap exceeded')
            self.value_weights=1/(self.n*self.probabilities)
        require(np.isclose(self.probabilities.sum(),1) and np.all(self.probabilities>0),'Invalid sampling probabilities')
        # Constant conditional normalization keeps expected policy gradient scale stable.
        eligible_mass=float(self.probabilities[nonforced].sum())
        normalization=float(np.sum(self.probabilities[nonforced]*self.policy_weights[nonforced])/eligible_mass)
        self.policy_weights/=normalization
        self.lookup=np.full(cache.manifest['rows'],-1,dtype=np.int64);self.lookup[all_indices]=np.arange(self.n)
        self.summary={'recipe':self.recipe,'unique_games':len(self.games),'available_decisions':self.n,
                      'probabilities_sha256':digest(self.probabilities.tolist()),
                      'maximum_relative_sampling_probability':float(np.max(self.n*self.probabilities)),
                      'expected_phase_proportions':self.proportions('phase'),
                      'expected_family_proportions':self.proportions('family'),
                      'policy_weight_mean_nonforced':normalization,
                      'value_weight_expectation':float(self.probabilities@self.value_weights)}
        if state:
            require(state['recipe']==self.recipe and state['probabilities_sha256']==self.summary['probabilities_sha256'],'Sampler resume distribution changed')
            self.rng.bit_generator.state=state['rng'];self.draws=state['draws'];self.epoch=state['epoch']

    def proportions(self,name):
        values=self.cache.arrays[name][self.indices]
        labels=['setup','middle','endgame'] if name=='phase' else self.cache.manifest['families']
        return {labels[int(k)]:float(self.probabilities[values==k].sum()) for k in np.unique(values)}

    def take(self,count):
        if self.shuffle:
            rows=self.shuffle.take(count);self.epoch=self.shuffle.epoch
        else:
            positions=self.rng.choice(self.n,size=count,p=self.probabilities)
            rows=self.indices[positions];self.epoch=(self.draws+count)//self.n
        self.draws+=count
        return rows

    def weights(self,rows):
        positions=self.lookup[rows];require(np.all(positions>=0),'Sample outside training recipe')
        return self.policy_weights[positions].astype(np.float32),self.value_weights[positions].astype(np.float32)

    def state_dict(self):
        return deepcopy({'recipe':self.recipe,'rng':self.rng.bit_generator.state,'draws':self.draws,'epoch':self.epoch,
                'probabilities_sha256':self.summary['probabilities_sha256'],
                'shuffle':self.shuffle.state_dict() if self.shuffle else None})


def weighted_losses(policy,value_logit,legal,targets,outcomes,policy_weights,value_weights):
    masked=policy.masked_fill(~legal,float('-inf'));nonforced=legal.sum(1)>1
    per_row=F.cross_entropy(masked,targets,reduction='none')
    # Use the eligible count; re-normalizing sampled weights would undo the
    # precomputed scale and bias the inverse-probability value estimator.
    pl=(per_row*nonforced*policy_weights).sum()/nonforced.sum().clamp_min(1)
    vl=(F.binary_cross_entropy_with_logits(value_logit,outcomes,reduction='none')*value_weights).mean()
    return pl,vl
