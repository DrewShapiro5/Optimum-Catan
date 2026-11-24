"""Observation-built sampled worlds and explicit, normalized chance classes."""
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
import json
import random

from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionRecord, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.state import PLAYER_INITIAL_STATE
from catanatron.serialization import (action_to_json, state_from_json, state_to_json,
                                     client_view)
from catanatron.gym.envs.action_space import to_action_space,get_action_array

from agent_apis.development_belief import CARDS
from agent_apis.experiment import digest
from agent_apis.observation import (ResourceLedger, development_knowledge,
                                   encode_observation, from_web_view, require, PLAY_TYPES)

COLORS = (Color.BLUE, Color.RED)


class SearchObservation(dict):
    """Internal immutable-by-contract view; never construct from external JSON."""
    @property
    def features(self):
        if not hasattr(self,'_features'):self._features=encode_observation(self)
        return self._features


def policy_observation(view):
    """Remove irrelevant roll labels before ANY evaluator/opponent callback.

    Current hands/bank and phase flags carry their sufficient effects. Public
    shortage effects are retained separately in the search history path.
    """
    result = dict(view)
    result['action_records'] = [
        [[action[0],action[1],None],None] if action[1]=='ROLL' else [action,value]
        for action,value in view['action_records']]
    return result


class NoFutureRandom(random.Random):
    """All engine chances MUST be supplied explicitly, including on copied worlds."""
    def random(self):
        raise RuntimeError('Implicit engine chance is forbidden in search')

    def getrandbits(self, k):
        raise RuntimeError('Implicit engine chance is forbidden in search')


@lru_cache(maxsize=1)
def _action_ids():
    return {item:i for i,item in enumerate(get_action_array(COLORS,'BASE'))}

def action_id(action):
    return _action_ids()[(action.action_type,action.value)]


def world_from_observation(view, allocation):
    """Construct a fresh simulated Game; never clone an authoritative Game."""
    own = view['perspective']
    other = next(c for c in view['colors'] if c != own)
    ledger = ResourceLedger(own)
    hands = ledger.update(view)
    state = {}
    for seat, color in enumerate(view['colors']):
        prefix = f'P{seat}_'
        for field, default in PLAYER_INITIAL_STATE.items():
            state[prefix+field] = view['player_state'].get(prefix+field, default)
        from catanatron.models.enums import RESOURCES
        for resource, number in zip(RESOURCES, hands[color]):
            state[prefix+resource+'_IN_HAND'] = number
        if color == other:
            for i, card in enumerate(CARDS):
                state[prefix+card+'_IN_HAND'] = allocation['opponent_old'][i]+allocation['opponent_new'][i]
                if card != 'VICTORY_POINT':
                    state[prefix+card+'_OWNED_AT_START'] = allocation['opponent_start_flags'][i]
            state[prefix+'ACTUAL_VICTORY_POINTS'] = state[prefix+'VICTORY_POINTS'] + state[prefix+'VICTORY_POINT_IN_HAND']
    records = []
    for step, (action, result) in enumerate(view['action_records']):
        action = list(action)
        if action[1] == 'BUY_DEVELOPMENT_CARD' and action[0] == other:
            result = allocation['opponent_draws'][step]
            action[2] = result
        records.append([action, result])
    deck = [card for card, count in zip(CARDS, allocation['deck']) for _ in range(count)]
    doc = {k: view[k] for k in ('map','board','colors','buildings_by_color',
        'resource_freqdeck','num_turns','current_player_index','current_turn_index',
        'current_prompt','is_initial_build_phase','is_discarding','discard_counts',
        'is_moving_knight','is_road_building','free_roads_available','is_resolving_trade')}
    doc.update(schema_version=2, game={**view['game'], 'id':'sampled-world', 'seed':0},
               player_state=state, action_records=records, development_listdeck=deck,
               random_state=random.Random(0).getstate(), current_trade=[0]*11,
               acceptees=[False]*2)
    game = state_from_json(doc, [SimplePlayer(Color[c]) for c in view['colors']])
    game.random = game.state.random = NoFutureRandom(0)
    require(game.winning_color() is None, 'Sampled an already terminal root world')
    expected = sorted(view['current_playable_actions'], key=lambda a: json.dumps(a))
    actual = sorted([action_to_json(a) for a in game.playable_actions], key=lambda a: json.dumps(a))
    require(actual == expected, 'Sampled world disagrees with supplied root legal actions')
    return game


def world_from_template(template,view,allocation):
    """Reuse decoded public geometry/history; replace every sampled secret."""
    from catanatron.models.enums import Action
    game=template.copy();state=game.state
    other=next(c for c in view['colors'] if c!=view['perspective'])
    seat=view['colors'].index(other);prefix=f'P{seat}_'
    for i,card in enumerate(CARDS):
        state.player_state[prefix+card+'_IN_HAND']=allocation['opponent_old'][i]+allocation['opponent_new'][i]
        if card!='VICTORY_POINT':state.player_state[prefix+card+'_OWNED_AT_START']=allocation['opponent_start_flags'][i]
    state.player_state[prefix+'ACTUAL_VICTORY_POINTS']=state.player_state[prefix+'VICTORY_POINTS']+state.player_state[prefix+'VICTORY_POINT_IN_HAND']
    state.development_listdeck=[card for card,n in zip(CARDS,allocation['deck']) for _ in range(n)]
    for step,card in allocation['opponent_draws'].items():
        old=state.action_records[step].action
        state.action_records[step]=ActionRecord(Action(old.color,old.action_type,card),card)
    require(game.winning_color() is None,'Sampled an already terminal root world')
    return game


@lru_cache(maxsize=8192)
def _record_document(record):
    # ActionRecord/Action are immutable. Callers treat these serialized values
    # as read-only; redaction constructs replacement entries rather than edits.
    from catanatron.serialization import action_record_to_json
    return action_record_to_json(record)


def record_document(record):
    try:return _record_document(record)
    except TypeError:
        # Some recorded outcomes contain dictionaries and cannot be cache keys.
        from catanatron.serialization import action_record_to_json
        return action_record_to_json(record)


def observation(game, color):
    """A player's OWN view; do not expose another actor's private legal actions."""
    from catanatron.serialization import map_to_json,board_to_json,action_record_to_json,_plain
    from catanatron.models.enums import RESOURCES,DEVELOPMENT_CARDS
    state=game.state
    public_map=getattr(state.board.map,'_search_public_map',None)
    if public_map is None:
        public_map=map_to_json(state.board.map)
        state.board.map._search_public_map=public_map
    player_state=dict(state.player_state)
    for seat,other in enumerate(state.colors):
        player_state[f'P{seat}_NUM_RESOURCES_IN_HAND']=sum(player_state[f'P{seat}_{r}_IN_HAND'] for r in RESOURCES)
        player_state[f'P{seat}_NUM_DEVELOPMENT_CARDS_IN_HAND']=sum(player_state[f'P{seat}_{c}_IN_HAND'] for c in DEVELOPMENT_CARDS)
    view={name:getattr(state,name) for name in ('num_turns','current_player_index','current_turn_index',
        'is_initial_build_phase','is_discarding','is_moving_knight','is_road_building',
        'free_roads_available','is_resolving_trade')}
    view.update(map=public_map,board=board_to_json(state.board),colors=[c.value for c in state.colors],
        player_state=player_state,development_listdeck=len(state.development_listdeck),
        game={'vps_to_win':game.vps_to_win,'discard_limit':state.discard_limit,'friendly_robber':state.friendly_robber},
        current_prompt=state.current_prompt.value,discard_counts=list(state.discard_counts),
        buildings_by_color={c.value:{k:_plain(v) for k,v in kinds.items()} for c,kinds in state.buildings_by_color.items()},
        resource_freqdeck=list(state.resource_freqdeck),
        action_records=[record_document(r) for r in state.action_records],
        current_playable_actions=[action_to_json(a) for a in game.playable_actions] if state.current_color().value==color else [],
        state_index=len(state.action_records))
    return SearchObservation(policy_observation(from_web_view(view,color)))


def observation_key(view):
    # Full history tree paths live outside this key. Its observation component
    # contains only normalized sufficient effects, never irrelevant dice labels.
    return digest({'features': view.features if isinstance(view,SearchObservation) else encode_observation(view),
                   'knowledge': development_knowledge(view)['events'],
                   'legal': sorted(view['current_playable_actions'], key=lambda a: json.dumps(a))})


def visible_successor_key(game,color):
    winner = game.winning_color()
    return digest({'observation':observation_key(observation(game,color)),
                   'announced_terminal_winner':winner.value if winner is not None else None})


def _world_history(game):
    records=game.state.action_records
    cached=getattr(game,'_search_history',None)
    if cached is not None and cached[0]==(id(records),len(records)):return cached[1:]
    turns=Counter();knowledge=[]
    for record in records:
        action=record.action;color=action.color.value;kind=action.action_type.value
        if kind=='END_TURN':turns[color]+=1
        elif kind=='BUY_DEVELOPMENT_CARD':knowledge.append((color,'buy',turns[color],record.result))
        elif kind in PLAY_TYPES:knowledge.append((color,'play',turns[color],PLAY_TYPES[kind]))
    result=(tuple(knowledge),dict(turns))
    game._search_history=((id(records),len(records)),*result)
    return result


def _extend_world_history(parent,child):
    knowledge,turns=_world_history(parent)
    record=child.state.action_records[-1];action=record.action
    color=action.color.value;kind=action.action_type.value
    if kind=='END_TURN':turns={**turns,color:turns.get(color,0)+1}
    elif kind=='BUY_DEVELOPMENT_CARD':knowledge=knowledge+((color,'buy',turns.get(color,0),record.result),)
    elif kind in PLAY_TYPES:knowledge=knowledge+((color,'play',turns.get(color,0),PLAY_TYPES[kind]),)
    child._search_history=((id(child.state.action_records),len(child.state.action_records)),knowledge,turns)


def canonical_successor(game):
    """All dynamic world facts, including private knowledge, without future order."""
    state = game.state
    knowledge,_ = _world_history(game)
    return digest({'player_state': state.player_state,
                   'bank': state.resource_freqdeck,
                   'development_composition': sorted(Counter(state.development_listdeck).items()),
                   'board': {'buildings': sorted((n,c.value,k) for n,(c,k) in state.board.buildings.items()),
                             'roads': sorted((tuple(e),c.value) for e,c in state.board.roads.items()),
                             'robber': state.board.robber_coordinate},
                   'colors': [c.value for c in state.colors],
                   'actor': state.current_player_index, 'turn_owner': state.current_turn_index,
                   'turns':state.num_turns, 'prompt':state.current_prompt.value,
                   'setup':state.is_initial_build_phase, 'discarding':state.is_discarding,
                   'discards':state.discard_counts, 'knight':state.is_moving_knight,
                   'road_building':state.is_road_building, 'free_roads':state.free_roads_available,
                   'resolving_trade':state.is_resolving_trade,
                   'rules':[game.vps_to_win,state.discard_limit,state.friendly_robber],
                   'knowledge':knowledge})


@dataclass(frozen=True)
class ElementaryOutcome:
    result: object
    probability: float


@dataclass(frozen=True)
class OutcomeClass:
    key: str
    probability: float
    members: tuple
    bank_shortages: tuple = ()


def elementary_outcomes(game, action):
    kind = action.action_type
    if kind == ActionType.ROLL:
        return [ElementaryOutcome((a,b),1/36) for a in range(1,7) for b in range(1,7)]
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        counts = Counter(game.state.development_listdeck)
        total = sum(counts.values())
        require(total > 0, 'Cannot draw from an empty sampled deck')
        return [ElementaryOutcome(card, counts[card]/total) for card in CARDS if counts[card]]
    if kind == ActionType.MOVE_ROBBER and action.value[1] is not None:
        from catanatron.models.enums import RESOURCES
        seat = game.state.color_to_index[action.value[1]]
        counts = [game.state.player_state[f'P{seat}_{r}_IN_HAND'] for r in RESOURCES]
        require(sum(counts) > 0, 'Cannot steal from an empty hand')
        return [ElementaryOutcome(r,n/sum(counts)) for r,n in zip(RESOURCES,counts) if n]
    return [ElementaryOutcome(None,1.)]


def successor(game, action, outcome):
    require(action in game.playable_actions, 'Illegal simulated action')
    child = game.copy()
    # Game.copy shares RNG references. NoFutureRandom makes accidental use fail.
    child.random = child.state.random
    if action.action_type in (ActionType.ROLL,ActionType.BUY_DEVELOPMENT_CARD,ActionType.MOVE_ROBBER):
        child.execute(action, action_record=ActionRecord(action,outcome.result))
    else:
        child.execute(action)
    _extend_world_history(game,child)
    return child


@lru_cache(maxsize=512)
def _production_table(layout, buildings, robber):
    """Immutable dice entitlements, independent of hands, roads and bank stock."""
    pieces = dict(buildings)
    table = []
    for number in range(2,13):
        payouts = [[0]*5 for _ in COLORS]
        if number != 7:
            for coordinate, resource, dice, nodes in layout:
                if dice != number or coordinate == robber or resource is None:continue
                for node in nodes:
                    if node in pieces:
                        color,kind=pieces[node]
                        payouts[tuple(c.value for c in COLORS).index(color)][resource] += 2 if kind=='CITY' else 1
        table.append(tuple(tuple(p) for p in payouts))
    return tuple(table)


def roll_classes(game):
    from catanatron.models.enums import RESOURCES
    board=game.state.board
    # Maps are immutable and shared by Game.copy. Cache layout on that map.
    layout=getattr(board.map,'_search_dice_layout',None)
    if layout is None:
        layout=tuple((tuple(coordinate),RESOURCES.index(tile.resource) if tile.resource else None,
                      tile.number,tuple(tile.nodes.values())) for coordinate,tile in board.map.land_tiles.items())
        board.map._search_dice_layout=layout
    buildings=tuple(sorted((node,(color.value,kind)) for node,(color,kind) in board.buildings.items()))
    table=_production_table(layout,buildings,tuple(board.robber_coordinate))
    groups={}
    for outcome in elementary_outcomes(game,next(a for a in game.playable_actions if a.action_type==ActionType.ROLL)):
        number=sum(outcome.result)
        payouts=table[number-2]
        denied=tuple(sorted(RESOURCES[i] for i in range(5)
                     if sum(p[i] for p in payouts)>game.state.resource_freqdeck[i])) if number!=7 else ()
        effective=tuple(tuple(0 if RESOURCES[i] in denied else n for i,n in enumerate(p)) for p in payouts)
        signature=('seven',) if number==7 else (effective,denied)
        groups.setdefault(signature,[]).append(outcome)
    return [(tuple(members),() if signature==('seven',) else signature[1]) for signature,members in groups.items()]


def enumerate_classes(game, action, merged=True):
    raw = elementary_outcomes(game,action)
    require(abs(sum(o.probability for o in raw)-1) < 1e-12, 'Invalid elementary probability mass')
    if merged and action.action_type==ActionType.ROLL:
        classes=[]
        for members,denied in roll_classes(game):
            child=successor(game,action,members[0])
            key=digest([canonical_successor(child),denied])
            classes.append(OutcomeClass(key,sum(o.probability for o in members),members,denied))
        return classes
    groups,shortages = {},{}
    for index,outcome in enumerate(raw):
        require(outcome.probability > 0, 'Nonpositive outcome')
        child = successor(game,action,outcome)
        denied = ()
        if action.action_type == ActionType.ROLL and sum(outcome.result)!=7:
            from catanatron.apply_action import yield_resources
            _,depleted = yield_resources(game.state.board,game.state.resource_freqdeck,sum(outcome.result))
            denied = tuple(sorted(depleted))
        key = digest([canonical_successor(child),denied]) if merged else f'raw-{index}'
        groups.setdefault(key,[]).append(outcome)
        shortages[key] = denied
    classes = [OutcomeClass(key,sum(o.probability for o in members),tuple(members),shortages[key])
               for key,members in groups.items()]
    require(abs(sum(c.probability for c in classes)-1) < 1e-12
            and sum(len(c.members) for c in classes)==len(raw), 'Merged distribution lost probability mass')
    return classes


def sample_class(classes,rng):
    threshold = rng.random()
    for outcome in classes:
        threshold -= outcome.probability
        if threshold < 0:
            return outcome
    return classes[-1]  # floating-point rounding at the final cumulative mass
