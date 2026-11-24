"""Python/native wire codec. No experimental-library or model-runtime imports."""
import copy
import json
from catanatron.models.enums import RESOURCES, Action, ActionType as AT, ActionPrompt as AP
from catanatron.models.player import Color
from catanatron.serialization import web_view
from catanatron.gym.envs.action_space import get_action_array
from agent_apis.observation import feature_names, from_web_view, encode_observation
DEVS = ['KNIGHT','MONOPOLY','YEAR_OF_PLENTY','ROAD_BUILDING','VICTORY_POINT']
KINDS = ['END_TURN','ROLL','BUILD_SETTLEMENT','BUILD_ROAD','BUILD_CITY','BUY_DEVELOPMENT_CARD','DISCARD_RESOURCE','MOVE_ROBBER','PLAY_KNIGHT_CARD','PLAY_YEAR_OF_PLENTY','PLAY_MONOPOLY','PLAY_ROAD_BUILDING','MARITIME_TRADE']
PROMPTS = [AP.BUILD_INITIAL_SETTLEMENT,AP.BUILD_INITIAL_ROAD,AP.PLAY_TURN,AP.MOVE_ROBBER,AP.DISCARD]
class Codec:
    def __init__(self,g):
        b=g.state.board;self.tiles=list(b.map.land_tiles);self.tile_index={v:i for i,v in enumerate(self.tiles)}
        self.edges=sorted(tuple(sorted(e)) for e in b.buildable_subgraph.edges)
        self.edge_index={e:i for i,e in enumerate(self.edges)}
        self.n=max(b.map.land_nodes)+1
        assert self.n==54 and len(self.edges)==72
        self.map={'edges':self.edges,'tiles':[{'nodes':list(b.map.land_tiles[t].nodes.values()),
            'resource':RESOURCES.index(b.map.land_tiles[t].resource) if b.map.land_tiles[t].resource else -1,
            'number':b.map.land_tiles[t].number or 0} for t in self.tiles],
            'ports':[[] for _ in range(self.n)],
            'adjacent':[list(b.buildable_subgraph.neighbors(n)) for n in range(self.n)]}
        for r,nodes in b.map.port_nodes.items():
            for n in nodes: self.map['ports'][n].append(RESOURCES.index(r) if r else -1)
    def action(self,g,a):
        k=KINDS.index(a.action_type.value);v=a.value;x=y=-1
        if k in (2,4): x=v
        elif k==3:x=self.edge_index[tuple(sorted(v))]
        elif k in (6,10):x=RESOURCES.index(v)
        elif k==7:x=self.tile_index[v[0]];y=g.state.colors.index(v[1]) if v[1] else -1
        elif k==9:x=RESOURCES.index(v[0]);y=RESOURCES.index(v[1]) if len(v)>1 else -1
        elif k==12:x=RESOURCES.index(v[0]);y=RESOURCES.index(v[-1])
        return k,x,y
    def snapshot(self,g):
        s=g.state;b=s.board;ci=s.color_to_index;ps=s.player_state
        players=[]
        for p,c in enumerate(s.colors):
            get=lambda key:ps[f'P{p}_{key}']
            players.append({'hand':[get(f'{r}_IN_HAND') for r in RESOURCES],
                'dev':[get(f'{d}_IN_HAND') for d in DEVS], 'played':[get(f'PLAYED_{d}') for d in DEVS],
                'owned':[get(f'{d}_OWNED_AT_START') for d in DEVS[:4]],'rolled':get('HAS_ROLLED'),
                'used_dev':get('HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'),'vp':get('VICTORY_POINTS'),
                'actual':get('ACTUAL_VICTORY_POINTS'),'pieces':[get(f'{v}_AVAILABLE') for v in ('ROADS','SETTLEMENTS','CITIES')],
                'has_road':get('HAS_ROAD'),'has_army':get('HAS_ARMY'),'road_length':get('LONGEST_ROAD_LENGTH'),
                'settlements':list(s.buildings_by_color[c]['SETTLEMENT']),'cities':list(s.buildings_by_color[c]['CITY'])})
        st={'players':players,'bank':list(s.resource_freqdeck),'deck':[DEVS.index(d) for d in s.development_listdeck],
            'buildings':[[ci[b.buildings[n][0]], 1 if b.buildings[n][1]=='SETTLEMENT' else 2] if n in b.buildings else [-1,0] for n in range(self.n)],
            'roads':[ci[b.roads[e]] if e in b.roads else -1 for e in self.edges],
            'components':[[sorted(c) for c in b.connected_components.get(color,[]) ] for color in s.colors],
            'buildable':[n in b.board_buildable_ids for n in range(self.n)],
            'road_lengths':[[ci[c],n] for c,n in b.road_lengths.items()],
            'road_color':ci[b.road_color] if b.road_color is not None else -1,'road_length':b.road_length,
            'robber':self.tile_index[b.robber_coordinate], 'current':s.current_player_index,'turn':s.current_turn_index,
            'turns':s.num_turns,'prompt':PROMPTS.index(s.current_prompt),'initial':s.is_initial_build_phase,
            'discarding':s.is_discarding,'moving':s.is_moving_knight,'free_roads':s.free_roads_available,
            'road_building':s.is_road_building,'discards':list(s.discard_counts),
            'discard_limit':s.discard_limit,'friendly':s.friendly_robber,'target':g.vps_to_win}
        return {'map':self.map,'state':st}
    def chance(self,a,record):
        if a.action_type==AT.ROLL:return sum(record.result)
        if a.action_type==AT.BUY_DEVELOPMENT_CARD:return DEVS.index(record.result)
        if a.action_type==AT.MOVE_ROBBER and record.result is not None:return RESOURCES.index(record.result)
        return -1

def canonical(doc):
    doc=copy.deepcopy(doc)
    for cs in doc['state']['components']:
        for c in cs: c.sort()
    return json.loads(json.dumps(doc))

def job_meta(g):
 c=Codec(g);view=from_web_view(web_view(g,g.state.current_color()),g.state.current_color().value);raw=encode_observation(view);static=[x if n.startswith(('PORT','RULE_')) or n.startswith('TILE') and not n.endswith('HAS_ROBBER') else 0 for n,x in zip(feature_names(),raw)]
 ids=[{'action':dict(zip(('k','a','b'),c.action(g,Action(g.state.current_color(),k,v)))),'id':i,'rate':sum(x is not None for x in v[:4]) if k.value=='MARITIME_TRADE' else 0} for i,(k,v) in enumerate(get_action_array((Color.BLUE,Color.RED),'BASE'))]
 return {'initial':c.snapshot(g),'coordinates':[list(t) for t in c.tiles],'names':list(feature_names()),'static_features':static,'tile_ids':[g.state.board.map.land_tiles[t].id for t in c.tiles],'colors':[c.value for c in g.state.colors],'seed':11,'action_ids':ids}

