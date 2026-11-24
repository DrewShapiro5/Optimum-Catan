"""Frozen BASE topology, exact D6 action permutations, and typed port locations."""
from functools import lru_cache
import random
import re
import numpy as np

from catanatron.models.map import CatanMap,BASE_MAP_TEMPLATE,PORT_DIRECTION_TO_NODEREFS
from catanatron.models.enums import NodeRef
from agent_apis.dataset import action_vocabulary
from agent_apis.observation import feature_names,scales,require
from agent_apis.experiment import digest

VERSION='audited-v3-typed-ports-d6-v1'
OFFSETS={NodeRef.NORTH:(0,-2),NodeRef.NORTHEAST:(1,-1),NodeRef.SOUTHEAST:(1,1),
         NodeRef.SOUTH:(0,2),NodeRef.SOUTHWEST:(-1,1),NodeRef.NORTHWEST:(-1,-1)}


def rotate(cube,k,reflection=False):
    x,y,z=cube
    if reflection:x,y=y,x
    for _ in range(k):x,y,z=-z,-x,-y
    return x,y,z


@lru_cache(1)
def topology():
    m=CatanMap.from_template(BASE_MAP_TEMPLATE,rng=random.Random(0))
    names=list(feature_names());lookup={n:i for i,n in enumerate(names)}
    tiles=sorted(m.land_tiles.values(),key=lambda t:t.id)
    coordinates={t.id:c for c,t in m.land_tiles.items()}
    coordinate_id={c:i for i,c in coordinates.items()}
    edges=sorted({tuple(sorted(e)) for t in tiles for e in t.edges.values()})
    edge_id={e:i for i,e in enumerate(edges)}
    tile_nodes=[list(t.nodes.values()) for t in tiles]
    node_tiles=[[] for _ in range(54)]
    for i,nodes in enumerate(tile_nodes):
        for n in nodes:node_tiles[n].append(i)
    ports=[]
    for p in sorted(m.ports_by_id.values(),key=lambda p:p.id):
        refs=PORT_DIRECTION_TO_NODEREFS[p.direction]
        ports.append(edge_id[tuple(sorted(p.nodes[r] for r in refs))])
    node_xy={}
    for i,t in enumerate(tiles):
        x,y,z=coordinates[i]
        for ref,n in t.nodes.items():
            dx,dy=OFFSETS[ref];point=(x-y+dx,3*z+dy)
            require(n not in node_xy or node_xy[n]==point,'Inconsistent brick geometry')
            node_xy[n]=point
    ys=sorted({p[1] for p in node_xy.values()});xs=sorted({p[0] for p in node_xy.values()})
    node_cell=[ys.index(node_xy[n][1])*len(xs)+xs.index(node_xy[n][0]) for n in range(54)]
    tile_anchor=[min(nodes,key=lambda n:(node_xy[n][1],node_xy[n][0])) for nodes in tile_nodes]
    road_anchor=[];road_orientation=[]
    orientations=sorted({tuple(np.subtract(node_xy[b],node_xy[a])) if node_xy[a]<node_xy[b]
                         else tuple(np.subtract(node_xy[a],node_xy[b])) for a,b in edges})
    # Consistent orientation by lexicographic coordinate order (three edge slopes).
    require(len(orientations)==3,'Expected three road directions')
    for a,b in edges:
        lo,hi=sorted((a,b),key=lambda n:node_xy[n])
        road_anchor.append(lo)
        road_orientation.append(orientations.index(tuple(np.subtract(node_xy[hi],node_xy[lo]))))
    for ori in range(3):
        cells=[node_cell[road_anchor[i]] for i in range(72) if road_orientation[i]==ori]
        require(len(cells)==len(set(cells)),'Road brick collision within type plane')
    require(len(set(tile_anchor))==19,'Tile brick collision')
    vocabulary=action_vocabulary()
    action_lookup={str([kind,value]):i for i,(kind,value) in enumerate(vocabulary)}
    feature_forward=[];action_forward=[];node_forward=[];edge_forward=[];tile_forward=[]
    for reflection in (False,True):
        for k in range(6):
            tm=[coordinate_id[rotate(coordinates[i],k,reflection)] for i in range(19)]
            # A vertex is uniquely identified by the geometric center of its three
            # incident hexes, including water; use integer cube vertex coordinates.
            vertex_cube={}
            for coord,t in m.tiles.items():
                for n in t.nodes.values():vertex_cube.setdefault(n,[]).append(coord)
            signature={tuple(sorted(cs)):n for n,cs in vertex_cube.items() if n<54}
            nm=[]
            for n in range(54):
                cs=tuple(sorted(rotate(c,k,reflection) for c in vertex_cube[n]))
                # Water topology is a complete ring, so all transformed incident
                # hex sets for land intersections remain representable.
                require(cs in signature,'Vertex symmetry left BASE topology')
                nm.append(signature[cs])
            em=[edge_id[tuple(sorted((nm[a],nm[b])))] for a,b in edges]
            fm=[]
            for name in names:
                out=name
                match=re.match(r'TILE(\d+)_(.*)',name)
                if match:out=f'TILE{tm[int(match[1])]}_{match[2]}'
                match=re.match(r'NODE(\d+)_(.*)',name)
                if match:out=f'NODE{nm[int(match[1])]}_{match[2]}'
                match=re.match(r'EDGE\((\d+), (\d+)\)_(.*)',name)
                if match:out=f'EDGE{tuple(sorted((nm[int(match[1])],nm[int(match[2])])))}_{match[3]}'
                fm.append(lookup[out])
            am=[]
            for kind,value in vocabulary:
                transformed=value
                if kind in ('BUILD_SETTLEMENT','BUILD_CITY'):transformed=nm[value]
                elif kind=='BUILD_ROAD':transformed=sorted((nm[value[0]],nm[value[1]]))
                elif kind=='MOVE_ROBBER':transformed=[list(rotate(tuple(value[0]),k,reflection)),value[1]]
                # Absolute victim colors and actor-relative player features stay
                # unchanged under geometric transforms.
                am.append(action_lookup[str([kind,transformed])])
            feature_forward.append(fm);action_forward.append(am)
            node_forward.append(nm);edge_forward.append(em);tile_forward.append(tm)
    ff=np.asarray(feature_forward);af=np.asarray(action_forward)
    for p in list(ff)+list(af):require(len(set(p.tolist()))==len(p),'Nonbijective symmetry')
    tile_fields=['IS_'+r for r in ('WOOD','BRICK','SHEEP','WHEAT','ORE')]+['IS_DESERT','PROBA','HAS_ROBBER']
    tile_indices=[[lookup[f'TILE{i}_{f}'] for f in tile_fields] for i in range(19)]
    dice_indices=[lookup[f'TILE{i}_DICE_NUMBER'] for i in range(19)]
    node_indices=[[lookup[f'NODE{n}_P{i}_{kind}'] for i in range(2) for kind in ('SETTLEMENT','CITY')] for n in range(54)]
    edge_indices=[[lookup[f'EDGE{e}_P{i}_ROAD'] for i in range(2)] for e in edges]
    port_indices=[[lookup[f'PORT{p}_IS_{r}'] for r in ('WOOD','BRICK','SHEEP','WHEAT','ORE')]+[lookup[f'PORT{p}_IS_THREE_TO_ONE']] for p in range(9)]
    player_names=sorted({re.sub(r'^P[01]_(?:INFERRED_)?','',n) for n in names if re.match(r'^P[01]_',n)})
    player_indices=[]
    for p in range(2):
        row=[]
        for field in player_names:
            row.append(lookup.get(f'P{p}_{field}',lookup.get(f'P{p}_INFERRED_{field}',-1)))
        player_indices.append(row)
    global_indices=[i for i,n in enumerate(names) if not re.match(r'^(TILE|NODE|EDGE|PORT|P[01]_)',n)]
    flat_indices=[i for i,n in enumerate(names) if not n.startswith('PORT') and i not in dice_indices]
    incidence=np.zeros((54,72),np.float32)
    for i,(a,b) in enumerate(edges):incidence[a,i]=incidence[b,i]=1
    tile_incidence=np.zeros((54,19),np.float32)
    for t,ns in enumerate(tile_nodes):tile_incidence[ns,t]=1
    entity_types=[0]*19+[1]*54+[2]*72+[3]*2+[4]
    n=148;distance=np.full((n,n),99,np.int64);np.fill_diagonal(distance,0)
    relation=np.zeros((n,n),np.int64)
    for t,ns in enumerate(tile_nodes):
        for node in ns:distance[t,19+node]=distance[19+node,t]=1;relation[t,19+node]=relation[19+node,t]=1
    for e,(a,b) in enumerate(edges):
        for node in (a,b):distance[73+e,19+node]=distance[19+node,73+e]=1;relation[73+e,19+node]=relation[19+node,73+e]=2
    for k in range(145):distance[:145,:145]=np.minimum(distance[:145,:145],distance[:145,k,None]+distance[None,k,:145])
    distance=np.minimum(distance,13)
    relation[147,:]=relation[:,147]=3
    relation[145:147,:]=np.maximum(relation[145:147,:],4)
    relation[:,145:147]=np.maximum(relation[:,145:147],4)
    result={'version':VERSION,'feature_names':names,'scales':scales(),'edges':edges,'tile_nodes':tile_nodes,
        'coordinates':[coordinates[i] for i in range(19)],'port_edges':ports,
        'feature_inverse':np.argsort(ff,axis=1).tolist(),'action_forward':af.tolist(),
        'action_inverse':np.argsort(af,axis=1).tolist(),'node_forward':node_forward,
        'edge_forward':edge_forward,'edge_inverse':np.argsort(edge_forward,axis=1).tolist(),
        'tile_forward':tile_forward,'tile_indices':tile_indices,'dice_indices':dice_indices,
        'node_indices':node_indices,'edge_indices':edge_indices,'port_indices':port_indices,
        'player_indices':player_indices,'global_indices':global_indices,'flat_indices':flat_indices,
        'node_road':incidence.tolist(),'node_tile':tile_incidence.tolist(),
        'entity_types':entity_types,'distance':distance.tolist(),'relation':relation.tolist(),
        'grid':[len(ys),len(xs)],'node_cell':node_cell,'tile_cell':[node_cell[a] for a in tile_anchor],
        'road_cell':[node_cell[a] for a in road_anchor],'road_orientation':road_orientation,
        'action_vocabulary':vocabulary}
    result['sha256']=digest(result)
    return result
