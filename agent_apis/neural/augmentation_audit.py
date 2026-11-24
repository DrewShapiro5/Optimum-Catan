"""Independent rule-engine audit of rotations/reflections, including moved ports."""
import copy
from collections import defaultdict
import torch
import numpy as np
from catanatron.models.map import CatanMap,PORT_DIRECTION_TO_NODEREFS
from catanatron.models.tiles import LandTile,Port,Water
from catanatron.models.board import Board
from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import Color
from catanatron.serialization import map_from_json,map_to_json
from catanatron.gym.envs.action_space import to_action_space
from agent_apis.neural.topology import topology,rotate
from agent_apis.neural.models import Inputs


def transformed_game(game,symmetry):
    t=topology();reflection=symmetry>=6;k=symmetry%6
    nm=t['node_forward'][symmetry]
    source=game.state.board.map;base=map_from_json(map_to_json(source));tiles={}
    for coordinate,tile in source.tiles.items():
        destination=rotate(coordinate,k,reflection);geometry=base.tiles[destination]
        if isinstance(tile,LandTile):
            tiles[destination]=LandTile(geometry.id,tile.resource,tile.number,geometry.nodes,geometry.edges)
        elif isinstance(tile,Port):
            refs=PORT_DIRECTION_TO_NODEREFS[tile.direction]
            edge=sorted(nm[tile.nodes[r]] for r in refs)
            direction=next(d for d,rs in PORT_DIRECTION_TO_NODEREFS.items()
                           if sorted(geometry.nodes[r] for r in rs)==edge)
            tiles[destination]=Port(tile.id,tile.resource,direction,geometry.nodes,geometry.edges)
        else:tiles[destination]=Water(geometry.nodes,geometry.edges)
    result=game.copy();result.state.board=Board(CatanMap.from_tiles(tiles))
    for step,record in enumerate(game.state.action_records):
        action=record.action;kind=action.action_type.value;value=action.value
        if kind=='BUILD_SETTLEMENT':result.state.board.build_settlement(action.color,nm[value],step<8)
        elif kind=='BUILD_ROAD':result.state.board.build_road(action.color,tuple(nm[n] for n in value))
        elif kind=='BUILD_CITY':result.state.board.build_city(action.color,nm[value])
    result.state.board.robber_coordinate=rotate(game.state.board.robber_coordinate,k,reflection)
    result.state.buildings_by_color={c:defaultdict(list,{kind:[tuple(sorted(nm[n] for n in edge)) for edge in locations]
            if kind=='ROAD' else [nm[n] for n in locations] for kind,locations in buildings.items()})
        for c,buildings in game.state.buildings_by_color.items()}
    result.playable_actions=generate_playable_actions(result.state)
    return result


def check_position(game,features):
    t=topology();inputs=Inputs();original={to_action_space(a,(Color.BLUE,Color.RED),'BASE') for a in game.playable_actions}
    for symmetry in range(12):
        transformed=transformed_game(game,symmetry)
        actual={to_action_space(a,(Color.BLUE,Color.RED),'BASE') for a in transformed.playable_actions}
        expected={t['action_forward'][symmetry][a] for a in original}
        if actual!=expected:raise AssertionError(f'Legal-mask symmetry differs: {symmetry}, {actual^expected}')
        packed=inputs(torch.tensor([features],dtype=torch.float32),torch.tensor([symmetry]))
        for tile in transformed.state.board.map.tiles_by_id.values():
            expected_number=tile.number or 0
            if packed['tile'][0,tile.id,8:].argmax().item()!=expected_number:raise AssertionError('Tile-number correlation changed')
        expected_ports=np.zeros((72,6),dtype=np.float32)
        edge_id={tuple(e):i for i,e in enumerate(t['edges'])}
        resources=['WOOD','BRICK','SHEEP','WHEAT','ORE',None]
        for port in transformed.state.board.map.ports_by_id.values():
            refs=PORT_DIRECTION_TO_NODEREFS[port.direction]
            edge=tuple(sorted(port.nodes[r] for r in refs))
            expected_ports[edge_id[edge],resources.index(port.resource)]=1
        np.testing.assert_array_equal(packed['road'][0,:,2:].numpy(),expected_ports)
    return 12
