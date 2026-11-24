"""Isolated Transformer value and typed Exphormer-style attention ablations."""
import math
import networkx as nx
import numpy as np
import torch
from torch import nn
from agent_apis.neural.models import StructuralTransformer
from agent_apis.neural.topology import topology

VARIANTS=('control','logit-penalty','value-dropout','value-depth1','value-depth2','typed-exphormer')
EDGE_TYPES=('blocked','self','tile-to-node','node-to-tile','node-to-road','road-to-node',
    'expander','to-connector','from-connector','connector-peer','player-to-node',
    'node-to-player','player-to-road','road-to-player','port')


class ValueBranch(nn.Module):
    def __init__(self,output,depth=0,dropout=0.):
        super().__init__();self.output=output;self.dropout=nn.Dropout(dropout)
        self.layers=nn.ModuleList(nn.Linear(128,128) for _ in range(depth))
        # Residual zero initialization preserves the incoming function exactly.
        for layer in self.layers:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,x):
        for layer in self.layers:x=x+torch.nn.functional.gelu(layer(x))
        return self.output(self.dropout(x))


def connector_graph(seed=20261118):
    t=topology();n=152;types=torch.zeros(n,n,dtype=torch.long)
    overlay=nx.random_regular_graph(4,148,seed=seed)
    eigen=np.linalg.eigvalsh(nx.to_numpy_array(overlay)/4)
    gap=1-max(abs(eigen[0]),abs(eigen[-2]))
    assert nx.is_connected(overlay) and gap>.05
    for a,b in overlay.edges:types[a,b]=types[b,a]=6
    for tile,ns in enumerate(t['tile_nodes']):
        for node in ns:types[tile,19+node]=2;types[19+node,tile]=3
    for road,(a,b) in enumerate(t['edges']):
        for node in (a,b):types[19+node,73+road]=4;types[73+road,19+node]=5
    types[:148,148:]=7;types[148:,:148]=8;types[148:,148:]=9
    types.fill_diagonal_(1)
    return types,float(gap)


class TypedExphormer(StructuralTransformer):
    def __init__(self):
        super().__init__();types,gap=connector_graph()
        self.register_buffer('edge_types',types,persistent=False);self.spectral_gap=gap
        self.connectors=nn.Parameter(torch.randn(4,128)*.02)
        self.edge_bias=nn.Embedding(len(EDGE_TYPES),4);nn.init.zeros_(self.edge_bias.weight)
    def attention_graph(self,x):
        batch=len(x['global']);types=self.edge_types[None].expand(batch,-1,-1).clone()
        own_node=x['owner_node']!=0;own_road=x['owner_road']!=0
        types[:,145:147,19:73]=torch.where(own_node,10,types[:,145:147,19:73])
        types[:,19:73,145:147]=torch.where(own_node.transpose(1,2),11,types[:,19:73,145:147])
        types[:,145:147,73:145]=torch.where(own_road,12,types[:,145:147,73:145])
        types[:,73:145,145:147]=torch.where(own_road.transpose(1,2),13,types[:,73:145,145:147])
        types[:,19:73,19:73]=torch.where(x['port_adj']!=0,14,types[:,19:73,19:73])
        bias=self.edge_bias(types).permute(0,3,1,2)
        # Retain pretrained distance/relation and dynamic public relation biases.
        bias[:,:,:148,:148]=bias[:,:,:148,:148]+(self.distance(self.inputs.distance)+self.relation(self.inputs.relation)).permute(2,0,1)[None]
        dynamic=bias.new_zeros(batch,152,152)
        dynamic[:,145:147,19:73]=x['owner_node'];dynamic[:,19:73,145:147]=x['owner_node'].transpose(1,2)
        dynamic[:,145:147,73:145]=x['owner_road'];dynamic[:,73:145,145:147]=x['owner_road'].transpose(1,2)
        bias=bias+dynamic[:,None]*self.ownership_bias[None,:,None,None]
        ports=bias.new_zeros(batch,152,152);ports[:,19:73,19:73]=x['port_adj']
        bias=bias+ports[:,None]*self.port_bias[None,:,None,None]
        return types,bias.masked_fill(types[:,None]==0,float('-inf'))
    def forward(self,raw,symmetry=None):
        x=self.inputs(raw,symmetry);states=self.embed(x)
        entities=torch.cat((*states[:4],states[4][:,None],self.connectors[None].expand(len(raw),-1,-1)),1)
        _,bias=self.attention_graph(x)
        for block in self.blocks:entities=block(entities,bias)
        return self.heads((entities[:,:19],entities[:,19:73],entities[:,73:145],entities[:,145:147],entities[:,147]))


def build_variant(state_dict,variant):
    assert variant in VARIANTS
    torch.manual_seed(2026111807)
    model=TypedExphormer() if variant=='typed-exphormer' else StructuralTransformer()
    missing,unexpected=model.load_state_dict(state_dict,strict=False)
    expected={'connectors','edge_bias.weight'} if variant=='typed-exphormer' else set()
    assert set(missing)==expected and not unexpected,(missing,unexpected)
    if variant.startswith('value-depth'):
        model.heads.value=ValueBranch(model.heads.value,depth=int(variant[-1]))
    elif variant=='value-dropout':model.heads.value=ValueBranch(model.heads.value,dropout=.2)
    return model


def objective(logits,values,legal,targets,outcomes,value_weights,variant):
    from agent_apis.neural.friendly_w4 import losses
    pl,vl=losses(logits,values,legal,targets,outcomes,value_weights)
    # Correct once for W4 when estimating the uniform-position logit penalty.
    penalty=(values.square()*value_weights).mean()*.01 if variant=='logit-penalty' else values.new_zeros(())
    return pl,vl,penalty
