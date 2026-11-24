"""POMCP-style root history search against a frozen observation-only opponent."""
from dataclasses import dataclass, field
import math
import random
import time

from catanatron.serialization import action_from_json, action_to_json, map_from_json
from catanatron.models.player import Player
from catanatron.models.map import number_probability

from agent_apis.benchmark import named_seed
from agent_apis.experiment import digest
from agent_apis.development_belief import DevelopmentBelief
from agent_apis.observation import encode_observation, feature_names, from_web_view, require, validate_rules
from agent_apis.stochastic import (action_id, canonical_successor, enumerate_classes,
    observation, observation_key, policy_observation, sample_class, successor, visible_successor_key, world_from_observation, world_from_template)


def root_value(actor_probability, actor, root):
    require(0 <= actor_probability <= 1, 'Value must be a win probability')
    return actor_probability if actor == root else 1-actor_probability


class HeuristicOpponent:
    """Frozen phase-4 policy. Replace with one fixed D2/W0 MLP in phase 5."""
    VERSION = 'public-production-build-prior-v1'

    def probabilities(self, view):
        public_map = map_from_json(view['map'])
        def production(node):
            return sum(number_probability(t.number) for t in public_map.adjacent_tiles[node]
                       if t.resource is not None)
        scores = {}
        weights = {'BUILD_CITY':3., 'BUILD_SETTLEMENT':3., 'BUILD_ROAD':.6,
                   'BUY_DEVELOPMENT_CARD':1.1,'END_TURN':-1.,'ROLL':1.,
                   'PLAY_KNIGHT_CARD':1.,'PLAY_MONOPOLY':1.2,'PLAY_YEAR_OF_PLENTY':1.2,
                   'PLAY_ROAD_BUILDING':1.,'MARITIME_TRADE':-.4}
        for payload in view['current_playable_actions']:
            action = action_from_json(payload)
            kind, value = payload[1:]
            score = weights.get(kind,0.)
            if kind in ('BUILD_SETTLEMENT','BUILD_CITY'):
                score += 3*production(value)
            elif kind == 'BUILD_ROAD':
                score += .3*sum(production(n) for n in value)
            scores[action_id(action)] = math.exp(score)
        total = sum(scores.values())
        require(total > 0, 'Opponent has no legal actions')
        return {i:s/total for i,s in scores.items()}


class ReferenceEvaluator:
    """Deterministic observation-only score; not a learned value or strength claim."""
    VERSION = 'score-resource-logistic-v1'

    def __init__(self):
        self.policy = HeuristicOpponent()

    def value(self, view):
        values = dict(zip(feature_names(), encode_observation(view)))
        difference = values['P0_ACTUAL_VPS']-values['P1_PUBLIC_VPS']
        difference += .02*(values['P0_NUM_RESOURCES_IN_HAND']-values['P1_NUM_RESOURCES_IN_HAND'])
        return 1/(1+math.exp(-difference/4))

    def evaluate(self, view):
        return self.policy.probabilities(view), self.value(view)


@dataclass
class ChanceChild:
    successor: object
    visits: int = 0
    total: float = 0.


@dataclass
class ChanceWorld:
    classes: list
    children: dict = field(default_factory=dict)


@dataclass
class ActionEdge:
    action: object
    prior: float
    visits: int = 0
    avail: int = 0
    total: float = 0.
    worlds: dict = field(default_factory=dict)

    @property
    def mean(self):
        return self.total/self.visits if self.visits else 0.


@dataclass
class DecisionNode:
    observation: str
    edges: dict
    visits: int = 0


class StochasticSearch:
    def __init__(self, view, seed=11, evaluator=None, opponent=None, merged=True,
                 c_puct=1.5, max_root_depth=None, max_actions=128, adversarial_puct=False,
                 mcts_opponent=None):
        validate_rules(view)
        require(math.isfinite(c_puct) and c_puct > 0 and (max_root_depth is None or max_root_depth >= 1) and max_actions >= 1,
                'Invalid search limits or exploration constant')
        require(view['perspective'] == view['colors'][view['current_player_index']],
                'Search must be requested by the decision actor')
        self.view, self.root = policy_observation(view), view['perspective']
        self.evaluator = evaluator or ReferenceEvaluator()
        self.opponent = opponent or HeuristicOpponent()
        self.merged, self.c_puct = merged,c_puct
        self.max_root_depth, self.max_actions = max_root_depth,max_actions
        if mcts_opponent is None:
            mcts_opponent = 'adversarial-puct' if adversarial_puct else 'frozen-mlp'
        require(mcts_opponent in ('frozen-mlp', 'self-policy', 'adversarial-puct'),
                f'Unsupported mcts_opponent: {mcts_opponent}')
        self.mcts_opponent = mcts_opponent
        self.adversarial_puct = (mcts_opponent == 'adversarial-puct')
        self.world_rng = random.Random(named_seed(seed,'worlds'))
        self.chance_rng = random.Random(named_seed(seed,'chance'))
        self.opponent_rng = random.Random(named_seed(seed,'opponent'))
        self.nodes, self.opponent_nodes, self.opponent_edges = {},{},{}
        self.world_templates = {}
        self.decoded_world = None
        self.view_cache = {};self.view_keys = {};self.world_keys = {}
        self.metrics = {'simulations':0,'raw_outcomes':0,'merged_classes':0,
                        'chance_distributions':0,'retained_chance_children':0,
                        'max_action_depth':0,'max_root_depth':0,'max_completed_turn_depth':0,
                        'capped_simulations':0,'worlds_sampled':0,'evaluations':0}

    def _evaluate(self,view):
        self.metrics['evaluations'] += 1
        priors,value = self.evaluator.evaluate(view)
        require(math.isfinite(value) and 0 <= value <= 1,'Invalid evaluator value')
        legal = {action_id(action_from_json(a)) for a in view['current_playable_actions']}
        require(legal and set(priors)==legal and all(math.isfinite(p) and p>=0 for p in priors.values()),
                'Evaluator priors must cover exactly the legal action vocabulary')
        require(abs(sum(priors.values())-1)<1e-8,'Priors require one global legal normalization')
        return priors,root_value(value,view['perspective'],self.root)

    def _node(self,view,key=None):
        priors,value = self._evaluate(view)
        node = DecisionNode(observation_key(view) if key is None else key,{
            action_id(action_from_json(a)):ActionEdge(action_from_json(a),priors[action_id(action_from_json(a))])
            for a in view['current_playable_actions']})
        return node,value

    def _view(self,game,color):
        key=(game,color)
        if key not in self.view_cache:self.view_cache[key]=observation(game,color)
        return self.view_cache[key]

    def _view_key(self,game,color):
        key=(game,color)
        if key not in self.view_keys:self.view_keys[key]=observation_key(self._view(game,color))
        return self.view_keys[key]

    def _step(self,game,edge):
        # Separate action-edge evidence from per-world chance-child evidence.
        if game not in self.world_keys:self.world_keys[game]=canonical_successor(game)
        key = self.world_keys[game]
        world = edge.worlds.get(key)
        if world is None:
            classes = enumerate_classes(game,edge.action,self.merged)
            world = edge.worlds[key] = ChanceWorld(classes)
            self.metrics['chance_distributions'] += 1
            self.metrics['raw_outcomes'] += sum(len(c.members) for c in classes)
            self.metrics['merged_classes'] += len(classes)
        selected = sample_class(world.classes,self.chance_rng)
        child = world.children.get(selected.key)
        if child is None:
            child = world.children[selected.key] = ChanceChild(successor(game,edge.action,selected.members[0]))
            self.metrics['retained_chance_children'] += 1
        # Retain only sampled child Games. Enumeration's temporary Games die.
        return child.successor,child,selected.bank_shortages

    def _simulate(self,game,path,root_depth,action_depth):
        self.metrics['max_action_depth'] = max(self.metrics['max_action_depth'],action_depth)
        self.metrics['max_root_depth'] = max(self.metrics['max_root_depth'],root_depth)
        self.metrics['max_completed_turn_depth'] = max(self.metrics['max_completed_turn_depth'],
                                                     game.state.num_turns-self.view['num_turns'])
        winner = game.winning_color()
        if winner is not None:
            return float(winner.value == self.root)
        if action_depth >= self.max_actions:
            self.metrics['capped_simulations'] += 1
            return .5  # unresolved computation cap, never a training outcome
        actor = game.state.current_color().value
        if actor == self.root:
            view = self._view(game,self.root)
            if self.max_root_depth is not None and root_depth >= self.max_root_depth:
                return self._evaluate(view)[1]
            node = self.nodes.get(path)
            if node is None:
                node,value = self._node(view,self._view_key(game,self.root))
                self.nodes[path] = node
                node.visits += 1
                return value
            require(node.observation == self._view_key(game,self.root), 'Root history aliases distinct observations')
            require(set(node.edges)=={action_id(a) for a in game.playable_actions},
                    'Hidden world changed root legal actions directly')
            edge = max(node.edges.values(), key=lambda e:(
                (e.mean if e.visits else .5)+self.c_puct*e.prior*math.sqrt(max(1,node.visits))/(1+e.visits),
                -action_id(e.action)))
            next_root_depth = root_depth+1
        elif self.mcts_opponent == 'adversarial-puct':
            legal = {action_id(a):a for a in game.playable_actions}
            if len(legal) == 1:
                only_id, only_act = next(iter(legal.items()))
                edge = self.opponent_edges.setdefault((path, only_id), ActionEdge(only_act, 1.0))
                next_root_depth = root_depth
                node = None
            else:
                node = self.opponent_nodes.get(path)
                if node is None:
                    own_view = self._view(game, actor)
                    node, _ = self._node(own_view, self._view_key(game, actor))
                    self.opponent_nodes[path] = node
                else:
                    missing = [i for i in legal if i not in node.edges]
                    if missing:
                        own_view = self._view(game, actor)
                        priors, _ = self._evaluate(own_view)
                        for i in missing:
                            node.edges[i] = ActionEdge(legal[i], priors[i])
                legal_edges = [node.edges[i] for i in legal]
                for e in legal_edges:
                    e.avail += 1
                edge = max(legal_edges, key=lambda e:(
                    (1.0 - e.mean if e.visits else .5) + self.c_puct * e.prior * math.sqrt(max(1, e.avail)) / (1 + e.visits),
                    -action_id(e.action)))
                next_root_depth = root_depth
        elif self.mcts_opponent == 'self-policy':
            legal = {action_id(a):a for a in game.playable_actions}
            if len(legal) == 1:
                only_id, only_act = next(iter(legal.items()))
                edge = self.opponent_edges.setdefault((path, only_id), ActionEdge(only_act, 1.0))
            else:
                own_view = self._view(game, actor)
                priors, _ = self._evaluate(own_view)
                threshold = self.opponent_rng.random()
                selected = sorted(priors)[-1]
                for i in sorted(priors):
                    threshold -= priors[i]
                    if threshold < 0:
                        selected = i
                        break
                edge = self.opponent_edges.setdefault((path, selected), ActionEdge(legal[selected], priors[selected]))
            next_root_depth = root_depth
            node = None
        else:
            # Policy receives ONLY its own observation, not root private cards.
            own_view = self._view(game,actor)
            priors = self.opponent.probabilities(own_view)
            legal = {action_id(a):a for a in game.playable_actions}
            require(set(priors)==set(legal) and all(p>=0 for p in priors.values())
                    and abs(sum(priors.values())-1)<1e-8,'Invalid frozen opponent distribution')
            threshold = self.opponent_rng.random()
            selected = sorted(priors)[-1]
            for i in sorted(priors):
                threshold -= priors[i]
                if threshold < 0:
                    selected = i
                    break
            edge = self.opponent_edges.setdefault((path,selected),ActionEdge(legal[selected],priors[selected]))
            next_root_depth = root_depth
            node = None
        child,chance_child,bank_shortages = self._step(game,edge)
        public_action = action_id(edge.action)  # roll/draw outcomes are not action IDs
        visible = digest({'observation':self._view_key(child,self.root),
            'announced_terminal_winner':child.winning_color().value if child.winning_color() is not None else None})
        next_path = path+((public_action,visible,bank_shortages),)
        value = self._simulate(child,next_path,next_root_depth,action_depth+1)
        # Root utility stays fixed over sequential actions and opponent phases.
        edge.visits += 1
        edge.total += value
        chance_child.visits += 1
        chance_child.total += value
        if node is not None:
            node.visits += 1
        return value

    def run(self,simulations=128,context=None):
        require(simulations > 0, 'Positive simulation count required')
        started = time.perf_counter()
        legal = [action_from_json(a) for a in self.view['current_playable_actions']]
        require(legal,'No legal root actions')
        if len(legal)==1:
            self.metrics.update(forced=True,elapsed_seconds=time.perf_counter()-started)
            return {'action':action_to_json(legal[0]),'root_statistics':[], 'metrics':self.metrics}
        self.metrics['forced'] = False
        root = getattr(self,'retained_root',None)
        if root is None:root,value = self._node(self.view)
        else:
            value = None
            require(root.observation==observation_key(self.view),'Retained root observation changed')
            require(set(root.edges)=={action_id(a) for a in legal},'Retained root legality changed')
        self.nodes[()] = root
        self.belief = DevelopmentBelief(self.view)
        self.metrics['preparation_seconds'] = time.perf_counter()-started
        from agent_apis import debug_telemetry
        target_sims = simulations if (context is None or context.mode != 'time') else None
        session_id = getattr(self, 'session_id', None) or (getattr(context, 'session_id', None) if context is not None else None)
        debug_telemetry.publish_search_progress(
            root.edges,
            simulations_done=0,
            simulations_target=target_sims,
            root_value=value,
            session_id=session_id,
            force=True,
        )
        longest_iteration=0.
        for _ in range(simulations):
            if context is not None and context.mode == 'time':
                admission=min(context.admission_deadline,context.deadline-1.25*longest_iteration)
                if time.perf_counter()>=admission:break
            iteration_start=time.perf_counter()
            allocation = self.belief.sample(self.world_rng)
            allocation_key=(allocation['opponent_old'],allocation['opponent_new'],tuple(allocation['opponent_draws'].items()),allocation['deck'],allocation['opponent_start_flags'])
            prototype=self.world_templates.get(allocation_key)
            if prototype is not None:
                self.metrics['root_world_cache_hits']=self.metrics.get('root_world_cache_hits',0)+1
            if prototype is None:
                if self.decoded_world is None:
                    self.decoded_world=world_from_observation(self.view,allocation)
                    self.metrics['root_world_decodes']=self.metrics.get('root_world_decodes',0)+1
                    prototype=self.decoded_world
                else:prototype=world_from_template(self.decoded_world,self.view,allocation)
                if len(self.world_templates)<128:self.world_templates[allocation_key]=prototype
            world=prototype
            self.metrics['worlds_sampled'] += 1
            self._simulate(world,(),0,0)
            self.metrics['simulations'] += 1
            debug_telemetry.publish_search_progress(
                root.edges,
                simulations_done=self.metrics['simulations'],
                simulations_target=target_sims,
                root_value=value,
                session_id=session_id,
            )
            longest_iteration=max(longest_iteration,time.perf_counter()-iteration_start)
        # Final action selection never uses exploration bonuses or sampled worlds.
        selected = max(root.edges.values(),key=lambda e:(e.visits,e.mean,-action_id(e.action)))
        debug_telemetry.publish_search_progress(
            root.edges,
            simulations_done=self.metrics['simulations'],
            simulations_target=target_sims,
            root_value=value,
            selected_action_id=action_id(selected.action),
            session_id=session_id,
            force=True,
        )
        elapsed = time.perf_counter()-started
        self.metrics.update(elapsed_seconds=elapsed,
                            simulations_per_second=self.metrics['simulations']/elapsed,
                            decision_nodes=len(self.nodes),
                            decoded_root_worlds=len(self.world_templates),
                            cached_observations=len(self.view_cache),
                            belief_dp_states=self.belief.mass.cache_info().currsize)
        return {'action':action_to_json(selected.action),
                'root_statistics':[{'action_id':i,'visits':e.visits,'mean_value':e.mean,'prior':e.prior}
                                   for i,e in sorted(root.edges.items())], 'metrics':self.metrics}


class StochasticReferencePlayer(Player):
    """Phase-4 diagnostic root-sampled search; no learned components or strength claim."""
    LABEL = 'Stochastic reference (phase 4)'

    @staticmethod
    def availability():
        return True,''

    @staticmethod
    def benchmark_choose(safe_observation,context,rng):
        require(context.mode in ('time','simulations'),'Reference search requires time/simulations mode')
        count = context.simulations if context.mode == 'simulations' else 1000000
        return StochasticSearch(safe_observation,seed=rng.getrandbits(64)).run(count,context)['action']

    @staticmethod
    def choose(observation):
        from agent_apis.benchmark import DecisionContext
        started = time.perf_counter()
        color = observation.get('perspective') or observation['colors'][observation['current_player_index']]
        safe = from_web_view(observation,color)
        return StochasticSearch(safe).run(1000000,DecisionContext(started,500,'time',None))['action']

    def decide(self,game,playable_actions):
        from catanatron.serialization import web_view
        return action_from_json(self.choose(web_view(game,self.color)))
