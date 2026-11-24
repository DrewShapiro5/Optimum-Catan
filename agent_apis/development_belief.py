"""Exact draw-prior conditioning on observable development-card rule constraints.

Choices of actions are not evidence about card types. Purchases/plays/ages,
known own draws, finite supply and observed nontermination are constraints.
"""
from collections import Counter, namedtuple

from catanatron.models.board import Board
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import DEVELOPMENT_CARDS
from catanatron.models.player import Color
from catanatron.serialization import map_from_json

from agent_apis.observation import PLAY_TYPES, require, validate_rules

CARDS = tuple(DEVELOPMENT_CARDS)
SUPPLY = tuple(starting_devcard_bank().count(c) for c in CARDS)
ZERO = (0,) * len(CARDS)
VP = CARDS.index('VICTORY_POINT')


def friendly_robber_evidence(view, scores):
    """Visible-point protection provides no evidence about hidden VP cards.

    Historical actual-point-rule corpora remain archived under their pinned
    sources. Current search never infers private cards from friendly legality.
    """
    return {}, 0, SUPPLY[VP]


def changed(values, index, delta):
    result = list(values)
    result[index] += delta
    return tuple(result)


def public_score_history(view):
    """Replay only public pieces/knights, using the engine's public Board rules."""
    board = Board(map_from_json(view['map']))
    scores, knights = Counter(), Counter()
    army = None
    result = []
    for step, (action, _) in enumerate(view['action_records']):
        color, kind, value = action
        previous_road = board.road_color
        if kind == 'BUILD_SETTLEMENT':
            board.build_settlement(Color[color], value, step < 8)
            scores[color] += 1
        elif kind == 'BUILD_CITY':
            board.build_city(Color[color], value)
            scores[color] += 1
        elif kind == 'BUILD_ROAD':
            board.build_road(Color[color], tuple(value))
        if kind in ('BUILD_SETTLEMENT', 'BUILD_ROAD') and step >= 8:
            new_road = board.road_color
            if new_road is not None and new_road != previous_road:
                scores[new_road.value] += 2
                if previous_road is not None:
                    scores[previous_road.value] -= 2
        if kind == 'PLAY_KNIGHT_CARD':
            knights[color] += 1
            if knights[color] >= 3 and (army is None or knights[color] > knights[army]):
                if army != color:
                    scores[color] += 2
                    if army is not None:
                        scores[army] -= 2
                    army = color
        result.append(dict(scores))
    for i, color in enumerate(view['colors']):
        require(scores[color] == view['player_state'][f'P{i}_VICTORY_POINTS'],
                'Public score/history mismatch')
    return result


class DevelopmentBelief:
    """Dynamic program over old/new opponent holdings, then posterior sampling.

    Past unknown draws are assigned jointly. They are not sampled independently
    from current card totals. The DP includes probabilities of known own draws.
    Public plays only constrain legality; there is no strategic action likelihood.
    """
    def __init__(self, view):
        validate_rules(view)
        self.view = view
        self.own = view['perspective']
        self.other = next(c for c in view['colors'] if c != self.own)
        self.events = []
        own_bought, other_played = ZERO, ZERO
        own_old, own_new = ZERO, ZERO
        played_in_turn = Counter()
        public_plays = {c:Counter() for c in view['colors']}
        own_played_since_end = Counter()
        self.opponent_played_since_end = Counter()
        scores = public_score_history(view)
        self.friendly_minimums, self.minimum_vp, self.maximum_vp = friendly_robber_evidence(view, scores)
        previous_limits = None
        for step, ((action, result), public) in enumerate(zip(view['action_records'], scores)):
            color, kind, value = action
            event, index = 'none', None
            before_bought, before_played = own_bought, other_played
            if kind == 'BUY_DEVELOPMENT_CARD':
                if color == self.own:
                    require(result in CARDS, 'Own historical draw identity is missing')
                    index = CARDS.index(result)
                    event = 'own_buy'
                    own_bought = changed(own_bought, index, 1)
                    own_new = changed(own_new, index, 1)
                else:
                    event = 'other_buy'
            elif kind in PLAY_TYPES:
                index = CARDS.index(PLAY_TYPES[kind])
                public_plays[color][CARDS[index]] += 1
                require(not played_in_turn[color], 'More than one development card in a turn')
                played_in_turn[color] = 1
                if color == self.own:
                    require(own_old[index] > 0, 'Own play has no eligible old card')
                    own_old = changed(own_old, index, -1)
                    own_played_since_end[CARDS[index]] += 1
                else:
                    event = 'other_play'
                    other_played = changed(other_played, index, 1)
                    self.opponent_played_since_end[CARDS[index]] += 1
            elif kind == 'END_TURN':
                played_in_turn[color] = 0
                if color == self.own:
                    own_old = tuple(a+b for a, b in zip(own_old, own_new))
                    own_new = ZERO
                    own_played_since_end.clear()
                else:
                    event = 'other_end'
                    self.opponent_played_since_end.clear()
            # A running game cannot previously have ended from hidden VP cards.
            require(public.get(self.own, 0) + own_old[VP] + own_new[VP] < 15,
                    'Observed history already ended for the root player')
            limit = 14 - public.get(self.other, 0)
            if event != 'none' or limit != previous_limits or step in self.friendly_minimums:
                self.events.append((event, index, step, before_bought, before_played, limit))
                previous_limits = limit
        seat = view['colors'].index(self.own)
        for i,color in enumerate(view['colors']):
            for card in CARDS:
                require(public_plays[color][card] == view['player_state'][f'P{i}_PLAYED_{card}'],
                        'Public card-play counts disagree with history')
            require(bool(played_in_turn[color]) == bool(view['player_state'][f'P{i}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN']),
                    'Development-card turn restriction disagrees with history')
        for i,card in enumerate(CARDS):
            if card != 'VICTORY_POINT':
                require(bool(own_old[i] or own_played_since_end[card]) ==
                        bool(view['player_state'][f'P{seat}_{card}_OWNED_AT_START']),
                        'Own card age flags disagree with history')
        expected = tuple(view['player_state'][f'P{seat}_{c}_IN_HAND'] for c in CARDS)
        require(tuple(a+b for a, b in zip(own_old, own_new)) == expected,
                'Own development holdings disagree with purchase/play history')
        self.own_old, self.own_new = own_old, own_new
        self.own_bought, self.other_played = own_bought, other_played
        self.other_total = view['player_state'][f'P{1-seat}_NUM_DEVELOPMENT_CARDS_IN_HAND']
        self.remaining_total = view['development_listdeck']
        require(type(self.remaining_total) is int, 'Only total remaining deck is permitted')

        cache = {}
        hits = 0
        CacheInfo = namedtuple('CacheInfo', 'hits misses maxsize currsize')
        def mass(position, old, new):
            # An observed END_TURN is still an event, so a legal long game can
            # contain more events than Python permits recursive calls. Keep
            # exactly the same transitions, summation order and sampling RNG;
            # evaluate the acyclic posterior graph with an explicit DFS stack.
            nonlocal hits
            initial = (position, old, new)
            if initial in cache:
                hits += 1
                return cache[initial]
            stack = [(initial, None)]
            while stack:
                key, children = stack[-1]
                if key in cache:
                    stack.pop()
                    continue
                pos, old_cards, new_cards = key
                if pos == len(self.events):
                    deck = self.deck_counts(own_bought, other_played, old_cards, new_cards)
                    cache[key] = float(sum(old_cards)+sum(new_cards) == self.other_total
                        and sum(deck) == self.remaining_total and min(deck) >= 0
                        and self.minimum_vp <= old_cards[VP]+new_cards[VP] <= min(self.maximum_vp,
                            14-view['player_state'][f'P{1-seat}_VICTORY_POINTS']))
                    stack.pop()
                    continue
                if children is None:
                    children = [(probability, (pos+1, next_old, next_new))
                        for _, probability, next_old, next_new in self.transitions(pos, old_cards, new_cards)]
                    hits += sum(child in cache for _, child in children)
                    stack[-1] = (key, children)
                missing = next((child for _, child in children if child not in cache), None)
                if missing is not None:
                    stack.append((missing, None))
                    continue
                cache[key] = sum(probability*cache[child] for probability, child in children)
                stack.pop()
            return cache[initial]
        mass.cache_info = lambda: CacheInfo(hits, len(cache), None, len(cache))
        self.mass = mass
        self.evidence = self.mass(0, ZERO, ZERO)
        require(self.evidence > 0, 'No development-card world satisfies permitted history')

    @staticmethod
    def deck_counts(bought, played, old, new):
        return tuple(s-b-p-o-n for s, b, p, o, n in zip(SUPPLY, bought, played, old, new))

    def transitions(self, position, old, new):
        event, index, step, bought, played, limit = self.events[position]
        deck = self.deck_counts(bought, played, old, new)
        if min(deck) < 0:
            return []
        possibilities = []
        if event == 'other_buy':
            if sum(deck):
                possibilities = [(i, n/sum(deck), old, changed(new, i, 1))
                                 for i, n in enumerate(deck) if n]
        elif event == 'own_buy':
            if deck[index] > 0:
                possibilities = [(index, deck[index]/sum(deck), old, new)]
        elif event == 'other_play':
            if old[index] > 0:
                possibilities = [(None, 1., changed(old, index, -1), new)]
        elif event == 'other_end':
            possibilities = [(None, 1., tuple(a+b for a, b in zip(old, new)), ZERO)]
        else:
            possibilities = [(None, 1., old, new)]
        minimum = self.friendly_minimums.get(step, 0)
        return [x for x in possibilities if minimum <= x[2][VP]+x[3][VP] <= limit]

    def sample(self, rng):
        old, new = ZERO, ZERO
        draws = {}
        for position, event in enumerate(self.events):
            weighted = [(item, item[1]*self.mass(position+1, item[2], item[3]))
                        for item in self.transitions(position, old, new)]
            total = sum(w for _, w in weighted)
            require(total > 0, 'Inconsistent posterior transition')
            threshold = rng.random()*total
            selected = weighted[-1][0]
            for item, weight in weighted:
                threshold -= weight
                if threshold < 0:
                    selected = item
                    break
            index, _, old, new = selected
            if event[0] == 'other_buy':
                draws[event[2]] = CARDS[index]
        deck = self.deck_counts(self.own_bought, self.other_played, old, new)
        return {'opponent_old': old, 'opponent_new': new, 'opponent_draws': draws,
                'deck': deck,
                'opponent_start_flags': tuple(bool(old[i] or self.opponent_played_since_end[c])
                                              for i, c in enumerate(CARDS))}
