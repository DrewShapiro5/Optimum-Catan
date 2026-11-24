"""Schema 3: two-player observation-only features and exact resource accounting.

This module never accepts an authoritative Game. The adapter/replay layer owns
redaction; bank conservation is explicitly permitted by the experiment contract.
"""
from collections import Counter
from functools import lru_cache

import json
from pathlib import Path
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS

SCHEMA_VERSION = 3
COLORS = ('BLUE', 'RED')
PLAYABLE = tuple(c for c in DEVELOPMENT_CARDS if c != 'VICTORY_POINT')
PROMPTS = ('BUILD_INITIAL_SETTLEMENT', 'BUILD_INITIAL_ROAD', 'PLAY_TURN',
           'DISCARD', 'MOVE_ROBBER')
PLAY_TYPES = {'PLAY_KNIGHT_CARD': 'KNIGHT', 'PLAY_YEAR_OF_PLENTY': 'YEAR_OF_PLENTY',
              'PLAY_MONOPOLY': 'MONOPOLY', 'PLAY_ROAD_BUILDING': 'ROAD_BUILDING'}


def from_web_view(view, perspective):
    """Whitelist the live agent boundary, collapsing deck composition to total.

    Pass a web_view already requested as this perspective. Opponent hand fields,
    private draws, seeds, RNG, winners and unrelated UI metadata are excluded.
    """
    colors = view['colors']
    require(perspective in colors, 'Perspective is not seated')
    seat = colors.index(perspective)
    common = {'VICTORY_POINTS', 'ROADS_AVAILABLE', 'SETTLEMENTS_AVAILABLE',
              'CITIES_AVAILABLE', 'HAS_ROAD', 'HAS_ARMY', 'HAS_ROLLED',
              'HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN', 'LONGEST_ROAD_LENGTH'}
    common |= {'PLAYED_' + c for c in DEVELOPMENT_CARDS}
    own = common | {'ACTUAL_VICTORY_POINTS'} | {r + '_IN_HAND' for r in RESOURCES}
    own |= {c + '_IN_HAND' for c in DEVELOPMENT_CARDS}
    own |= {c + '_OWNED_AT_START' for c in PLAYABLE}
    other = common | {'NUM_RESOURCES_IN_HAND', 'NUM_DEVELOPMENT_CARDS_IN_HAND'}
    state = {f'P{i}_{key}': view['player_state'][f'P{i}_{key}']
             for i in range(2) for key in (own if i == seat else other)}
    fields = ('map', 'board', 'colors', 'buildings_by_color', 'resource_freqdeck',
              'num_turns', 'current_player_index', 'current_turn_index',
              'current_prompt', 'is_initial_build_phase', 'is_discarding',
              'discard_counts', 'is_moving_knight', 'is_road_building',
              'free_roads_available', 'is_resolving_trade', 'current_playable_actions')
    result = {k: view[k] for k in fields}
    result['player_state'] = state
    result['perspective'] = perspective
    result['game'] = {k: view['game'][k] for k in
                      ('vps_to_win', 'discard_limit', 'friendly_robber')}
    deck = view['development_listdeck']
    require(isinstance(deck, (int, dict)), 'Ordered development deck is forbidden')
    result['development_listdeck'] = sum(deck.values()) if isinstance(deck, dict) else deck
    records = view['action_records']
    require(view.get('state_index', len(records)) == len(records), 'Incomplete action history')
    result['state_index'] = len(records)
    result['action_records'] = [
        [[a[0], a[1], None], None] if a[0] != perspective and a[1] == 'BUY_DEVELOPMENT_CARD'
        else [a, None] if a[0] != perspective and a[1] == 'MOVE_ROBBER'
        else [a, value] for a, value in records]
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_rules(view):
    require(len(view['colors']) == 2 and set(view['colors']) == set(COLORS),
            'Schema 3 requires BLUE/RED two-player play')
    require(view['map']['template'] == 'BASE', 'Schema 3 requires BASE topology')
    rules = view['game']
    require(rules['vps_to_win'] == 15 and rules['discard_limit'] == 9
            and type(rules['friendly_robber']) is bool, 'Unsupported game rules')
    require(not view['is_resolving_trade'], 'Player trading is unsupported')
    require(view['current_prompt'] in PROMPTS, 'Unsupported decision phase')
    require(len(view['action_records']) == view['state_index'], 'Incomplete action history')


class ResourceLedger:
    """Exact sufficient-state ledger using public bank + private own snapshots.

    update may skip events: conservation recovers their net effects exactly.
    History records validate the cursor but are not used to guess secret thefts.
    A fresh ledger can recover from the latest permitted snapshot alone.
    """
    def __init__(self, color):
        require(color in COLORS, 'Unknown ledger perspective')
        self.color = color
        self.hands = {c: [0] * 5 for c in COLORS}
        self.bank = [19] * 5
        self.cursor = 0
        self.last_effect = {c: [0] * 5 for c in COLORS}

    def update(self, view):
        validate_rules(view)
        require(view['perspective'] == self.color, 'Ledger perspective mismatch')
        seat = view['colors'].index(self.color)
        own = [view['player_state'][f'P{seat}_{r}_IN_HAND'] for r in RESOURCES]
        bank = list(view['resource_freqdeck'])
        require(len(bank) == 5, 'Expected five bank counts')
        require(all(type(x) is int and 0 <= x <= 19 for x in own + bank),
                'Invalid own/bank resource count')
        other = next(c for c in COLORS if c != self.color)
        inferred = [19 - b - x for b, x in zip(bank, own)]
        require(all(0 <= x <= 19 for x in inferred), 'Negative inferred resource count')
        require(sum(inferred) == view['player_state'][
            f'P{1-seat}_NUM_RESOURCES_IN_HAND'], 'Opponent total disagrees with conservation')
        hands = {self.color: own, other: inferred}
        self.last_effect = {c: [a-b for a, b in zip(hands[c], self.hands[c])]
                            for c in COLORS}
        for i in range(5):
            require(sum(self.last_effect[c][i] for c in COLORS) == self.bank[i] - bank[i],
                    'Resource conservation failed')
        self.hands, self.bank = hands, bank
        self.cursor = view['state_index']
        return self.hands


def development_knowledge(view):
    """Observable purchase/play constraints, without strategic inference.

    Normalized events preserve purchase/play order and per-seat turn age. Other
    actions (including raw dice labels) never enter this knowledge representation.
    """
    turns = Counter()
    purchases = Counter()
    new = {c: Counter() for c in COLORS}
    events = []
    own = view['perspective']
    for action, result in view['action_records']:
        color, kind, value = action
        if kind == 'END_TURN':
            turns[color] += 1
            new[color].clear()
        elif kind == 'BUY_DEVELOPMENT_CARD':
            card = result if color == own else None
            require(color != own or card in DEVELOPMENT_CARDS, 'Missing own draw identity')
            purchases[color] += 1
            new[color][card] += 1
            events.append([color, 'buy', turns[color], card])
        elif kind in PLAY_TYPES:
            events.append([color, 'play', turns[color], PLAY_TYPES[kind]])
    return {'events': events, 'purchases': purchases, 'new': new}


@lru_cache(1)
def base_names():
    # Metadata is frozen; loading it must not instantiate a random engine Game.
    schema = Path(__file__).resolve().parents[1] / 'experiments/catan-ai/feature-schema-v3.json'
    return tuple(json.loads(schema.read_text())['feature_names'][:633])


@lru_cache(1)
def feature_names():
    extras = [f'P1_INFERRED_{r}_IN_HAND' for r in RESOURCES]
    extras += ['ACTOR_IS_BLUE', 'OBSERVER_IS_BLUE', 'DECISION_ACTOR_IS_SELF',
               'TURN_OWNER_IS_SELF', 'IS_SETUP', 'IS_FREE_ROAD', 'FREE_ROADS_AVAILABLE',
               'IS_MOVING_KNIGHT', 'DISCARD_REMAINING_SELF', 'DISCARD_REMAINING_OTHER',
               'RULE_VPS_TO_WIN', 'RULE_DISCARD_LIMIT', 'RULE_FRIENDLY_ROBBER',
               'RULE_PLAYER_TRADING', 'RULE_PLAYERS', 'RULE_RESOURCE_SUPPLY']
    extras += [f'PHASE_{p}' for p in PROMPTS]
    extras += [f'P{i}_DEV_PURCHASES' for i in range(2)]
    extras += ['P1_NEW_DEVS', 'P1_OLDER_DEVS']
    extras += [f'P0_NEW_{c}' for c in DEVELOPMENT_CARDS]
    extras += [f'P0_OLDER_{c}' for c in DEVELOPMENT_CARDS]
    extras += [f'P0_{c}_OWNED_AT_START' for c in PLAYABLE]
    extras += [f'P0_{c}_PLAYABLE' for c in PLAYABLE]
    return base_names() + tuple(extras)


def scales():
    """Fixed scales; categorical dice identities stay integer category codes."""
    result = []
    for name in feature_names():
        scale = 1
        if name.endswith('_DICE_NUMBER'):
            scale = 1
        elif 'IN_HAND' in name or name.startswith('BANK_'):
            scale = (95 if 'NUM_RESOURCES' in name else
                     25 if ('DEV' in name or any(c in name for c in DEVELOPMENT_CARDS)) else 19)
        elif name.endswith('_PLAYED') or 'DEV_PURCHASES' in name or '_NEW_' in name or '_OLDER_' in name or name in ('P1_NEW_DEVS', 'P1_OLDER_DEVS'):
            scale = 25
        elif name.endswith('_VPS') or name == 'RULE_VPS_TO_WIN':
            scale = 20
        elif 'ROADS_LEFT' in name or 'LONGEST_ROAD_LENGTH' in name:
            scale = 15
        elif 'SETTLEMENTS_LEFT' in name:
            scale = 5
        elif 'CITIES_LEFT' in name:
            scale = 4
        elif name.startswith('DISCARD_REMAINING'):
            scale = 95
        elif name == 'FREE_ROADS_AVAILABLE':
            scale = 2
        elif name == 'RULE_DISCARD_LIMIT':
            scale = 95
        elif name == 'RULE_PLAYERS':
            scale = 2
        elif name == 'RULE_RESOURCE_SUPPLY':
            scale = 19
        result.append(scale)
    return result


def encode_observation(view, ledger=None):
    """Encode ONLY an already redacted observation. Return raw schema-3 values.

    Legal actions are boundary-supplied rule-engine output, never reconstructed
    using a hidden state. Unknown/privileged extra fields are ignored.
    """
    validate_rules(view)
    own = view['perspective']
    ledger = ledger or ResourceLedger(own)
    hands = ledger.update(view)
    other = next(c for c in COLORS if c != own)
    ordered = [own, other]
    state = view['player_state']
    sample = {}
    for i, color in enumerate(ordered):
        key = f'P{view["colors"].index(color)}_'
        for out, src in [('PUBLIC_VPS', 'VICTORY_POINTS'), ('HAS_ARMY', 'HAS_ARMY'),
                         ('HAS_ROAD', 'HAS_ROAD'), ('ROADS_LEFT', 'ROADS_AVAILABLE'),
                         ('SETTLEMENTS_LEFT', 'SETTLEMENTS_AVAILABLE'),
                         ('CITIES_LEFT', 'CITIES_AVAILABLE'), ('HAS_ROLLED', 'HAS_ROLLED'),
                         ('LONGEST_ROAD_LENGTH', 'LONGEST_ROAD_LENGTH')]:
            sample[f'P{i}_{out}'] = state[key + src]
        sample[f'P{i}_NUM_RESOURCES_IN_HAND'] = sum(hands[color])
        sample[f'P{i}_NUM_DEVS_IN_HAND'] = (
            sum(state[key + c + '_IN_HAND'] for c in DEVELOPMENT_CARDS) if i == 0
            else state[key + 'NUM_DEVELOPMENT_CARDS_IN_HAND'])
        for card in PLAYABLE:
            sample[f'P{i}_{card}_PLAYED'] = state[key + 'PLAYED_' + card]
        if i == 0:
            sample['P0_ACTUAL_VPS'] = state[key + 'ACTUAL_VICTORY_POINTS']
            for card in list(RESOURCES) + list(DEVELOPMENT_CARDS):
                sample[f'P0_{card}_IN_HAND'] = state[key + card + '_IN_HAND']
            sample['P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'] = state[key + 'HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN']
    for name in base_names():
        if name.startswith(('EDGE', 'NODE')):
            sample[name] = 0
    for color, buildings in view['buildings_by_color'].items():
        i = ordered.index(color)
        for kind in ('SETTLEMENT', 'CITY'):
            for node in buildings.get(kind, []):
                sample[f'NODE{node}_P{i}_{kind}'] = 1
        for a, b in buildings.get('ROAD', []):
            sample[f'EDGE{tuple(sorted((a, b)))}_P{i}_ROAD'] = 1
    for tile in view['map']['tiles']:
        if tile['type'] == 'LAND':
            prefix = f'TILE{tile["id"]}_'
            for r in RESOURCES:
                sample[prefix + 'IS_' + r] = tile['resource'] == r
            sample[prefix + 'IS_DESERT'] = tile['resource'] is None
            number = tile['number'] or 0
            sample[prefix + 'DICE_NUMBER'] = number
            sample[prefix + 'PROBA'] = (6 - abs(7-number))/36 if number else 0
            sample[prefix + 'HAS_ROBBER'] = tile['coordinate'] == view['board']['robber_coordinate']
        elif tile['type'] == 'PORT':
            prefix = f'PORT{tile["id"]}_'
            for r in RESOURCES:
                sample[prefix + 'IS_' + r] = tile['resource'] == r
            sample[prefix + 'IS_THREE_TO_ONE'] = tile['resource'] is None
    for r, value in zip(RESOURCES, view['resource_freqdeck']):
        sample['BANK_' + r] = value
    require(type(view['development_listdeck']) is int, 'Only total remaining development deck is permitted')
    sample['BANK_DEV_CARDS'] = view['development_listdeck']
    legal_types = {a[1] for a in view['current_playable_actions']}
    sample['IS_MOVING_ROBBER'] = 'MOVE_ROBBER' in legal_types
    sample['IS_DISCARDING'] = 'DISCARD_RESOURCE' in legal_types
    for r, value in zip(RESOURCES, hands[other]):
        sample[f'P1_INFERRED_{r}_IN_HAND'] = value
    actor = view['colors'][view['current_player_index']]
    seat = view['colors'].index(own)
    sample.update(ACTOR_IS_BLUE=actor == 'BLUE', OBSERVER_IS_BLUE=own == 'BLUE',
                  DECISION_ACTOR_IS_SELF=actor == own,
                  TURN_OWNER_IS_SELF=view['colors'][view['current_turn_index']] == own,
                  IS_SETUP=view['is_initial_build_phase'], IS_FREE_ROAD=view['is_road_building'],
                  FREE_ROADS_AVAILABLE=view['free_roads_available'],
                  IS_MOVING_KNIGHT=view['is_moving_knight'],
                  DISCARD_REMAINING_SELF=view['discard_counts'][seat],
                  DISCARD_REMAINING_OTHER=view['discard_counts'][1-seat],
                  RULE_VPS_TO_WIN=15, RULE_DISCARD_LIMIT=9,
                  RULE_FRIENDLY_ROBBER=int(view['game']['friendly_robber']),
                  RULE_PLAYER_TRADING=0, RULE_PLAYERS=2, RULE_RESOURCE_SUPPLY=19)
    for prompt in PROMPTS:
        sample['PHASE_' + prompt] = view['current_prompt'] == prompt
    knowledge = development_knowledge(view)
    for i, color in enumerate(ordered):
        sample[f'P{i}_DEV_PURCHASES'] = knowledge['purchases'][color]
    sample['P1_NEW_DEVS'] = sum(knowledge['new'][other].values())
    sample['P1_OLDER_DEVS'] = sample['P1_NUM_DEVS_IN_HAND'] - sample['P1_NEW_DEVS']
    key = f'P{seat}_'
    for card in DEVELOPMENT_CARDS:
        sample['P0_NEW_' + card] = knowledge['new'][own][card]
        sample['P0_OLDER_' + card] = sample[f'P0_{card}_IN_HAND'] - sample['P0_NEW_' + card]
        require(sample['P0_OLDER_' + card] >= 0, 'Inconsistent own card age/history')
    for card in PLAYABLE:
        sample[f'P0_{card}_OWNED_AT_START'] = state[key + card + '_OWNED_AT_START']
        kind = next(k for k, c in PLAY_TYPES.items() if c == card)
        sample[f'P0_{card}_PLAYABLE'] = kind in legal_types if actor == own else False
    require(sample['P1_OLDER_DEVS'] >= 0, 'Inconsistent opponent card age/history')
    return [float(sample[name]) for name in feature_names()]


def normalized_features(features):
    require(len(features) == len(feature_names()), 'Feature schema length mismatch')
    return [x / scale for x, scale in zip(features, scales())]
