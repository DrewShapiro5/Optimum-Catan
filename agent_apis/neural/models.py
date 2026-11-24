"""Four encoders with common information, 332 masked logits and sigmoid value."""
import torch
from torch import nn
from torch.nn import functional as F
from agent_apis.neural.topology import topology

MODEL_IDS=('MLP','XD-CNN','H-GNN','TRANSFORMER')


class Inputs(nn.Module):
    def __init__(self):
        super().__init__();t=topology();self.topology=t
        for key in ('feature_inverse','action_forward','action_inverse','edge_inverse','tile_indices',
                    'dice_indices','node_indices','edge_indices','port_indices','player_indices',
                    'global_indices','flat_indices','port_edges','node_cell','tile_cell','road_cell',
                    'road_orientation','entity_types','distance','relation'):
            self.register_buffer(key,torch.tensor(t[key],dtype=torch.long),persistent=False)
        self.register_buffer('scales',torch.tensor(t['scales']),persistent=False)
        nr=torch.tensor(t['node_road']);nt=torch.tensor(t['node_tile'])
        self.register_buffer('node_road',nr,persistent=False);self.register_buffer('node_tile',nt,persistent=False)
        a=torch.zeros(72,54);b=torch.zeros(72,54)
        for e,(x,y) in enumerate(t['edges']):a[e,x]=1;b[e,y]=1
        self.register_buffer('edge_a',a,persistent=False);self.register_buffer('edge_b',b,persistent=False)
        for ori in range(3):
            self.register_buffer(f'road_group{ori}',(self.road_orientation==ori).nonzero().flatten(),persistent=False)

    def forward(self,raw,symmetry=None):
        batch=len(raw)
        if symmetry is None:symmetry=torch.zeros(batch,dtype=torch.long,device=raw.device)
        transformed=raw.gather(1,self.feature_inverse[symmetry])
        ports=raw[:,self.port_indices]
        edge_ports=raw.new_zeros(batch,72,6)
        edge_ports[:,self.port_edges]=ports
        edge_ports=edge_ports.gather(1,self.edge_inverse[symmetry,:,None].expand(-1,-1,6))
        x=transformed/self.scales
        dice=F.one_hot(transformed[:,self.dice_indices].long(),13).to(raw.dtype)
        tile=torch.cat((x[:,self.tile_indices],dice),-1)
        node=torch.cat((x[:,self.node_indices],torch.matmul(self.node_road,edge_ports)),-1)
        road=torch.cat((x[:,self.edge_indices],edge_ports),-1)
        ids=self.player_indices
        player=torch.cat((x[:,ids.clamp_min(0)]*(ids>=0),
                          (ids>=0).to(raw.dtype)[None].expand(batch,-1,-1)),-1)
        owner_node=torch.stack((node[:,:,0]+node[:,:,1],node[:,:,2]+node[:,:,3]),1)
        owner_road=road[:,:,:2].transpose(1,2)
        flags=edge_ports.sum(-1)
        port_adj=torch.matmul(self.edge_a.T,self.edge_b[None]*flags[:,:,None])
        port_adj=port_adj+port_adj.transpose(1,2)
        flat=torch.cat((x[:,self.flat_indices],dice.flatten(1),edge_ports.flatten(1)),1)
        return {'tile':tile,'node':node,'road':road,'player':player,'global':x[:,self.global_indices],
                'owner_node':owner_node,'owner_road':owner_road,'port_adj':port_adj,'flat':flat}

    def augment_targets(self,targets,legal,symmetry):
        return self.action_forward[symmetry,targets],legal.gather(1,self.action_inverse[symmetry])


class StructuredHeads(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(width*5,width),nn.LayerNorm(width),nn.GELU())
        self.node=nn.Linear(width*2,2);self.road=nn.Linear(width*2,1);self.tile=nn.Linear(width*2,3)
        scalar=[];indices=[];coords={tuple(c):i for i,c in enumerate(topology()['coordinates'])}
        edges={tuple(e):i for i,e in enumerate(topology()['edges'])}
        kinds=[k for k,v in topology()['action_vocabulary']]
        scalar_ids=[i for i,k in enumerate(kinds) if k not in ('BUILD_SETTLEMENT','BUILD_CITY','BUILD_ROAD','MOVE_ROBBER')]
        for i,(kind,value) in enumerate(topology()['action_vocabulary']):
            if i in scalar_ids:indices.append(scalar_ids.index(i))
            elif kind in ('BUILD_SETTLEMENT','BUILD_CITY'):indices.append(len(scalar_ids)+value*2+int(kind=='BUILD_CITY'))
            elif kind=='BUILD_ROAD':indices.append(len(scalar_ids)+108+edges[tuple(value)])
            else:indices.append(len(scalar_ids)+180+coords[tuple(value[0])]*3+[None,'BLUE','RED'].index(value[1]))
        self.scalar=nn.Linear(width,len(scalar_ids));self.value=nn.Linear(width,1)
        self.register_buffer('action_columns',torch.tensor(indices),persistent=False)

    def forward(self,tokens):
        tile,node,road,players,global_=tokens
        context=self.context(torch.cat((tile.mean(1),node.mean(1),road.mean(1),players.mean(1),global_),1))
        def joined(x):return torch.cat((x,context[:,None].expand(-1,x.shape[1],-1)),-1)
        scores=torch.cat((self.scalar(context),self.node(joined(node)).flatten(1),
                          self.road(joined(road)).flatten(1),self.tile(joined(tile)).flatten(1)),1)
        return scores[:,self.action_columns],self.value(context).squeeze(-1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.inputs=Inputs()
        dim=len(topology()['flat_indices'])+19*13+72*6
        layers=[]
        for _ in range(3):layers.extend((nn.Linear(dim,256),nn.GELU()));dim=256
        self.encoder=nn.Sequential(*layers);self.policy=nn.Linear(256,332);self.value=nn.Linear(256,1)

    def forward(self,raw,symmetry=None):
        x=self.encoder(self.inputs(raw,symmetry)['flat'])
        return self.policy(x),self.value(x).squeeze(-1)


class EntityInputs(nn.Module):
    def __init__(self):
        super().__init__();t=topology()
        self.projections=nn.ModuleList(nn.Linear(n,128) for n in (21,10,8,2*len(t['player_indices'][0]),len(t['global_indices'])))
        self.type_embedding=nn.Embedding(5,128)
    def forward(self,x):
        result=[]
        for i,name in enumerate(('tile','node','road','player','global')):
            projected=self.projections[i](x[name])+self.type_embedding.weight[i]
            result.append(projected)
        return result


def normalized(matrix):
    return matrix/matrix.sum(-1,keepdim=True).clamp_min(1)


class MessageBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.ModuleList(nn.LayerNorm(128) for _ in range(5))
        self.transforms=nn.ModuleList(nn.Linear(128,128,bias=False) for _ in range(17))
        self.feed=nn.ModuleList(nn.Sequential(nn.LayerNorm(128),nn.Linear(128,256),nn.GELU(),nn.Linear(256,128)) for _ in range(5))
    def forward(self,states,relations):
        t,n,e,p,g=states
        norm=[module(x) for module,x in zip(self.norm,states)]
        t0,n0,e0,p0,g0=norm
        messages=[torch.zeros_like(s) for s in states]
        nt,nr,on,oe,port=relations
        operations=[(1,nt,t0),(0,nt.T,n0),(1,nr,e0),(2,nr.T,n0),
                    (1,on.transpose(1,2),p0),(3,on,n0),(2,oe.transpose(1,2),p0),(3,oe,e0),(1,port,n0)]
        for i,(target,adj,source) in enumerate(operations):
            messages[target]=messages[target]+self.transforms[i](torch.matmul(normalized(adj),source))
        for i,source in enumerate(norm[:4]):
            messages[i]=messages[i]+self.transforms[9+i](g0)[:,None]
            messages[4]=messages[4]+self.transforms[13+i](source.mean(1))
        updated=[x+msg for x,msg in zip(states,messages)]
        return [x+feed(x) for x,feed in zip(updated,self.feed)]


class HGNN(nn.Module):
    def __init__(self):
        super().__init__();self.inputs=Inputs();self.embed=EntityInputs()
        self.blocks=nn.ModuleList(MessageBlock() for _ in range(6));self.heads=StructuredHeads()
    def forward(self,raw,symmetry=None):
        x=self.inputs(raw,symmetry);states=self.embed(x)
        relations=(self.inputs.node_tile,self.inputs.node_road,x['owner_node'],x['owner_road'],x['port_adj'])
        for block in self.blocks:states=block(states,relations)
        return self.heads(states)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(128);self.qkv=nn.Linear(128,384);self.output=nn.Linear(128,128)
        self.norm2=nn.LayerNorm(128);self.ff=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,128))
    def forward(self,x,bias):
        batch,n,_=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(batch,n,3,4,32).permute(2,0,3,1,4).unbind(0)
        attention=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)
        x=x+self.output(attention.transpose(1,2).reshape(batch,n,128))
        return x+self.ff(self.norm2(x))


class StructuralTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.inputs=Inputs();self.embed=EntityInputs();self.heads=StructuredHeads()
        self.distance=nn.Embedding(14,4);self.relation=nn.Embedding(5,4)
        self.ownership_bias=nn.Parameter(torch.zeros(4));self.port_bias=nn.Parameter(torch.zeros(4))
        self.blocks=nn.ModuleList(AttentionBlock() for _ in range(4))
    def forward(self,raw,symmetry=None):
        x=self.inputs(raw,symmetry);states=self.embed(x)
        entities=torch.cat((*states[:4],states[4][:,None]),1)
        bias=(self.distance(self.inputs.distance)+self.relation(self.inputs.relation)).permute(2,0,1)[None]
        dynamic=raw.new_zeros(len(raw),148,148)
        dynamic[:,145:147,19:73]=x['owner_node'];dynamic[:,19:73,145:147]=x['owner_node'].transpose(1,2)
        dynamic[:,145:147,73:145]=x['owner_road'];dynamic[:,73:145,145:147]=x['owner_road'].transpose(1,2)
        bias=bias+dynamic[:,None]*self.ownership_bias[None,:,None,None]
        port=raw.new_zeros(len(raw),148,148);port[:,19:73,19:73]=x['port_adj']
        bias=bias+port[:,None]*self.port_bias[None,:,None,None]
        for block in self.blocks:entities=block(entities,bias)
        return self.heads((entities[:,:19],entities[:,19:73],entities[:,73:145],entities[:,145:147],entities[:,147]))


class ChannelNorm(nn.Module):
    def __init__(self):super().__init__();self.norm=nn.LayerNorm(64)
    def forward(self,x):return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class CrossDimBlock(nn.Module):
    def __init__(self):
        super().__init__();self.scalar_norm=nn.LayerNorm(128);self.broadcast=nn.Linear(128,64)
        self.norm1=ChannelNorm();self.norm2=ChannelNorm()
        self.conv1=nn.Conv2d(64,64,(5,3),padding=(2,1),bias=False)
        self.conv2=nn.Conv2d(64,64,(5,3),padding=(2,1),bias=False)
        self.scalar=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,128))
    def forward(self,x,s,mask):
        x=x+self.broadcast(self.scalar_norm(s))[:,:,None,None]*mask
        z=self.conv1(F.gelu(self.norm1(x))*mask)*mask
        x=x+self.conv2(F.gelu(self.norm2(z))*mask)*mask
        mean=(x*mask).sum((2,3))/mask.sum()
        variance=(x.square()*mask).sum((2,3))/mask.sum()-mean.square()
        s=s+self.scalar(torch.cat((self.scalar_norm(s),mean,variance.clamp_min(0)),1))
        return x,s


class XDCNN(nn.Module):
    def __init__(self):
        super().__init__();self.inputs=Inputs();t=topology();self.h,self.w=t['grid']
        self.initial=nn.Conv2d(60,64,1)
        self.scalar=nn.Sequential(nn.Linear(29+2*94,128),nn.LayerNorm(128),nn.GELU())
        self.blocks=nn.ModuleList(CrossDimBlock() for _ in range(6))
        self.entity=nn.ModuleList(nn.Linear(64,128) for _ in range(5))
        self.players=nn.Linear(94,128);self.heads=StructuredHeads()
        mask=torch.zeros(1,1,self.h*self.w);mask[0,0,self.inputs.node_cell]=1
        self.register_buffer('valid',mask.reshape(1,1,self.h,self.w),persistent=False)
    def forward(self,raw,symmetry=None):
        x=self.inputs(raw,symmetry);batch=len(raw)
        planes=raw.new_zeros(batch,self.h*self.w,60)
        planes[:,self.inputs.tile_cell,:21]=x['tile'];planes[:,self.inputs.tile_cell,55]=1
        planes[:,self.inputs.node_cell,21:31]=x['node'];planes[:,self.inputs.node_cell,56]=1
        for ori in range(3):
            selected=getattr(self.inputs,f'road_group{ori}')
            cells=self.inputs.road_cell[selected]
            planes[:,cells,31+8*ori:39+8*ori]=x['road'][:,selected]
            planes[:,cells,57+ori]=1
        spatial=self.initial(planes.transpose(1,2).reshape(batch,60,self.h,self.w))*self.valid
        scalar=self.scalar(torch.cat((x['global'],x['player'].flatten(1)),1))
        for block in self.blocks:spatial,scalar=block(spatial,scalar,self.valid)
        flat=spatial.flatten(2).transpose(1,2)
        tile=self.entity[0](flat[:,self.inputs.tile_cell]);node=self.entity[1](flat[:,self.inputs.node_cell])
        road=flat.new_zeros(batch,72,128)
        for ori in range(3):
            selected=getattr(self.inputs,f'road_group{ori}')
            road[:,selected]=self.entity[2+ori](flat[:,self.inputs.road_cell[selected]])
        return self.heads((tile,node,road,self.players(x['player']),scalar))


def build_model(model_id):
    return {'MLP':MLP,'XD-CNN':XDCNN,'H-GNN':HGNN,'TRANSFORMER':StructuralTransformer}[model_id]()


def losses(policy,value_logit,legal,targets,outcomes,validate=True):
    if validate and not torch.all(legal.any(1)):raise ValueError('A sample has no legal action')
    if validate and not torch.all(legal.gather(1,targets[:,None])):raise ValueError('Illegal demonstrated target')
    masked=policy.masked_fill(~legal,float('-inf'))
    nonforced=legal.sum(1)>1
    per_row=F.cross_entropy(masked,targets,reduction='none')
    policy_loss=(per_row*nonforced).sum()/nonforced.sum().clamp_min(1)
    value_loss=F.binary_cross_entropy_with_logits(value_logit,outcomes)
    return policy_loss,value_loss
