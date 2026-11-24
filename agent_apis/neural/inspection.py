"""Pinned, user-requested web inspection agents; no training-default changes."""
from dataclasses import dataclass
import json
import random
import time
from typing import Optional
from catanatron.serialization import action_from_json, web_view
from agent_apis.experiment import ROOT, EXPERIMENTS, file_digest
from agent_apis.observation import from_web_view, require
from agent_apis.neural.provider import NeuralPolicyPlayer, cached_runtime

CONFIG = EXPERIMENTS / 'web-cnn-inspection.json'


class InspectionPolicyPlayer(NeuralPolicyPlayer):
    LABEL = 'CNN best measured — policy'
    MODE = 'policy'

    @staticmethod
    def availability():
        ready, reason = NeuralPolicyPlayer.availability()
        if not ready:
            return ready, reason
        try:
            config = json.loads(CONFIG.read_text())
            for ref in (config['checkpoint'], config['search']['opponent_checkpoint']):
                if file_digest(ROOT / ref['path']) != ref['sha256']:
                    return False, 'Inspection checkpoint checksum differs'
        except (OSError, KeyError, ValueError) as error:
            return False, f'Inspection checkpoint unavailable: {error}'
        return True, ''

    @classmethod
    def choose(cls, observation, simulations=None):
        started = time.perf_counter()
        config = json.loads(CONFIG.read_text())
        ref = config['checkpoint']
        require(file_digest(ROOT / ref['path']) == ref['sha256'], 'Inspection checkpoint changed')
        color = observation.get('perspective') or observation['colors'][observation['current_player_index']]
        safe = from_web_view(observation, color)
        frozen = config['search']['opponent_checkpoint']
        sims = None if simulations is None else int(simulations)
        if sims is not None:
            require(0 <= sims <= 4096, f'simulations must be between 0 and 4096, got {sims}')
        effective_mode = 'policy' if (cls.MODE == 'policy' or sims == 0) else 'search'
        session = (observation.get('game', {}).get('id'), color) if effective_mode == 'search' else None
        runtime = cached_runtime(str(ROOT/ref['path']), ref['sha256'], effective_mode, frozen['sha256'], session)
        require(runtime.policy.metadata['model_id'] == 'XD-CNN', 'Inspection agent must use XD-CNN')
        from agent_apis.benchmark import DecisionContext
        if effective_mode == 'policy':
            context = DecisionContext(started, config['budget_ms'], 'policy', None, config)
        elif sims is not None:
            context = DecisionContext(started, 120000, 'simulations', sims, config)
        else:
            context = DecisionContext(started, config['budget_ms'], 'time', None, config)
        context.session_id = observation.get('game', {}).get('id')
        return runtime.benchmark_choose(safe, context, random.Random(11))


class InspectionSearchPlayer(InspectionPolicyPlayer):
    LABEL = 'CNN best measured — search (500 ms)'
    MODE = 'search'

    @dataclass(frozen=True)
    class Params:
        simulations: Optional[int] = None

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return action_from_json(self.choose(web_view(game, self.color), simulations=self.params.simulations))
