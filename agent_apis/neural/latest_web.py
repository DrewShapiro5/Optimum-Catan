"""Latest multi-cohort XD-CNN and TypedExphormer web gameplay agents with tunable MCTS simulations."""
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import time

from catanatron.serialization import action_from_json, web_view

from agent_apis.experiment import EXPERIMENTS, ROOT, file_digest
from agent_apis.neural.provider import NeuralCheckpoint, NeuralPolicyPlayer, cached_runtime
from agent_apis.observation import from_web_view, require

RUNS_ROOT = ROOT / 'runs/catan-ai'
COHORTS_ROOT = RUNS_ROOT / 'coreml-cohorts-2000'
V2_COHORTS_ROOT = RUNS_ROOT / 'coreml-cohorts-v2-selfpolicy'
STATIC_COHORT_DIRS = {COHORTS_ROOT.name, V2_COHORTS_ROOT.name}
FALLBACK_CNN = RUNS_ROOT / 'selfplay-only-comparison/cnn/iteration-1/best.pt'
FALLBACK_TRANSFORMER = RUNS_ROOT / 'selfplay-only-comparison/sparse/iteration-1/best.pt'
BEST_CNN = COHORTS_ROOT / 'cohort-006/cnn/iteration-1/best.pt'
BEST_TRANSFORMER = COHORTS_ROOT / 'cohort-006/transformer/iteration-1/best.pt'
FROZEN_CONFIG = EXPERIMENTS / 'phase5-frozen-opponent.json'
_DYNAMIC_PLAYER_CLASSES = {}


def latest_cnn_checkpoint():
    """Return the newest trained cohort CNN best.pt, falling back to pre-self-play CNN."""
    if COHORTS_ROOT.is_dir():
        candidates = sorted(COHORTS_ROOT.glob('cohort-*/cnn/iteration-1/best.pt'))
        if candidates:
            return candidates[-1]
    return FALLBACK_CNN


def latest_transformer_checkpoint():
    """Return the newest trained cohort Transformer best.pt, falling back to pre-self-play sparse transformer."""
    if COHORTS_ROOT.is_dir():
        candidates = sorted(COHORTS_ROOT.glob('cohort-*/transformer/iteration-1/best.pt'))
        if candidates:
            return candidates[-1]
    return FALLBACK_TRANSFORMER


def latest_v2_cnn_checkpoint():
    """Return the newest trained v2 self-policy cohort CNN best.pt."""
    if V2_COHORTS_ROOT.is_dir():
        candidates = sorted(V2_COHORTS_ROOT.glob('cohort-*/cnn/iteration-1/best.pt'))
        if candidates:
            return candidates[-1]
    return FALLBACK_CNN


def latest_v2_transformer_checkpoint():
    """Return the newest trained v2 self-policy cohort Transformer best.pt."""
    if V2_COHORTS_ROOT.is_dir():
        candidates = sorted(V2_COHORTS_ROOT.glob('cohort-*/transformer/iteration-1/best.pt'))
        if candidates:
            return candidates[-1]
    return FALLBACK_TRANSFORMER


def _build_config(checkpoint_path, model_id='XD-CNN', mcts_opponent='frozen-mlp', vp_margin_alpha=0.03):
    frozen = json.loads(FROZEN_CONFIG.read_text())['checkpoint'] if mcts_opponent == 'frozen-mlp' else None
    return {
        'checkpoint': {
            'path': str(checkpoint_path.relative_to(ROOT)),
            'sha256': file_digest(checkpoint_path),
        },
        'search': {
            'c_puct': 1.5,
            'chance': 'true-probability-merged',
            'opponent': 'frozen-MLP-D2-W0' if mcts_opponent == 'frozen-mlp' else mcts_opponent,
            'mcts_opponent': mcts_opponent,
            'vp_margin_alpha': float(vp_margin_alpha),
            'root_selection': ['visits', 'mean_value', 'stable_action_id'],
            'max_root_depth': None,
            'max_actions': 128,
            'opponent_checkpoint': frozen,
            'workers': 1,
            'revision': 'optimized-unbounded-own-actions-v1',
            'reuse_subtree': True,
        },
        'budget_ms': 120000,
        'model': model_id,
        'recipe': 'D2-W4',
    }


def _check_latest_availability(ckpt_fn, label, require_frozen=False):
    ready, reason = NeuralPolicyPlayer.availability()
    if not ready:
        return ready, reason
    ckpt = ckpt_fn()
    if not ckpt.is_file():
        return False, f'Latest {label} checkpoint missing: {ckpt}'
    if require_frozen:
        try:
            frozen = json.loads(FROZEN_CONFIG.read_text())['checkpoint']
            if file_digest(ROOT / frozen['path']) != frozen['sha256']:
                return False, 'Frozen search opponent checksum differs'
        except (OSError, KeyError, ValueError) as error:
            return False, f'Latest {label} config unavailable: {error}'
    return True, ''


def _choose_latest(observation, ckpt_fn, expected_model_id, simulations=128, mcts_opponent='frozen-mlp', vp_margin_alpha=0.03):
    started = time.perf_counter()
    sims = 128 if simulations is None else int(simulations)
    require(0 <= sims <= 4096, f'simulations must be between 0 and 4096, got {sims}')
    alpha = 0.03 if vp_margin_alpha is None else float(vp_margin_alpha)
    require(0.0 <= alpha <= 0.5, f'vp_margin_alpha must be between 0.0 and 0.5, got {alpha}')
    ckpt = ckpt_fn()
    require(ckpt.is_file(), f'Latest {expected_model_id} checkpoint missing: {ckpt}')
    config = _build_config(ckpt, model_id=expected_model_id, mcts_opponent=mcts_opponent, vp_margin_alpha=alpha)
    ref = config['checkpoint']
    frozen = config['search']['opponent_checkpoint']
    color = observation.get('perspective') or observation['colors'][observation['current_player_index']]
    safe = from_web_view(observation, color)
    mode = 'policy' if sims == 0 else 'search'
    session = (observation.get('game', {}).get('id'), color) if mode == 'search' else None
    frozen_sha = frozen['sha256'] if (mode == 'search' and frozen is not None) else None
    runtime = cached_runtime(str(ckpt), ref['sha256'], mode, frozen_sha, session, device='cpu')
    require(
        runtime.policy.metadata['model_id'] == expected_model_id,
        f'Latest web agent must use {expected_model_id}',
    )
    import torch
    with runtime.lock:
        if mode == 'search' and mcts_opponent == 'frozen-mlp' and runtime.opponent is None:
            require(
                frozen is not None and file_digest(ROOT / frozen['path']) == frozen['sha256'],
                'Frozen search opponent checkpoint missing or changed',
            )
            runtime.opponent = NeuralCheckpoint(ROOT / frozen['path'], device='cpu')
            runtime.opponent_sha = frozen['sha256']
    torch.set_num_threads(1)
    from agent_apis.benchmark import DecisionContext
    context = DecisionContext(
        started,
        config['budget_ms'],
        'policy' if sims == 0 else 'simulations',
        None if sims == 0 else sims,
        config,
    )
    context.session_id = observation.get('game', {}).get('id')
    return runtime.benchmark_choose(safe, context, random.Random(11))


class LatestCnnSearchPlayer(NeuralPolicyPlayer):
    LABEL = 'CNN latest best — search'
    MODE = 'search'

    @dataclass(frozen=True)
    class Params:
        simulations: int = 128
        mcts_opponent: str = 'self-policy'
        vp_margin_alpha: float = 0.03

    @staticmethod
    def availability():
        return _check_latest_availability(latest_cnn_checkpoint, 'CNN')

    @classmethod
    def choose(cls, observation, simulations=128, mcts_opponent='self-policy', vp_margin_alpha=0.03):
        return _choose_latest(
            observation,
            latest_cnn_checkpoint,
            'XD-CNN',
            simulations=simulations,
            mcts_opponent=mcts_opponent,
            vp_margin_alpha=vp_margin_alpha,
        )

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return action_from_json(
            self.choose(
                web_view(game, self.color),
                simulations=self.params.simulations,
                mcts_opponent=self.params.mcts_opponent,
                vp_margin_alpha=self.params.vp_margin_alpha,
            )
        )


class LatestTransformerSearchPlayer(NeuralPolicyPlayer):
    LABEL = 'Transformer latest best — search'
    MODE = 'search'

    @dataclass(frozen=True)
    class Params:
        simulations: int = 128
        mcts_opponent: str = 'self-policy'
        vp_margin_alpha: float = 0.03

    @staticmethod
    def availability():
        return _check_latest_availability(latest_transformer_checkpoint, 'Transformer')

    @classmethod
    def choose(cls, observation, simulations=128, mcts_opponent='self-policy', vp_margin_alpha=0.03):
        return _choose_latest(
            observation,
            latest_transformer_checkpoint,
            'TYPED-EXPHORMER',
            simulations=simulations,
            mcts_opponent=mcts_opponent,
            vp_margin_alpha=vp_margin_alpha,
        )

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return action_from_json(
            self.choose(
                web_view(game, self.color),
                simulations=self.params.simulations,
                mcts_opponent=self.params.mcts_opponent,
                vp_margin_alpha=self.params.vp_margin_alpha,
            )
        )


class BestCnnSearchPlayer(NeuralPolicyPlayer):
    LABEL = 'CNN all-time best (cohort 6) — search'
    MODE = 'search'

    @dataclass(frozen=True)
    class Params:
        simulations: int = 128
        mcts_opponent: str = 'self-policy'
        vp_margin_alpha: float = 0.03

    @staticmethod
    def availability():
        return _check_latest_availability(lambda: BEST_CNN, 'All-time best CNN')

    @classmethod
    def choose(cls, observation, simulations=128, mcts_opponent='self-policy', vp_margin_alpha=0.03):
        return _choose_latest(
            observation,
            lambda: BEST_CNN,
            'XD-CNN',
            simulations=simulations,
            mcts_opponent=mcts_opponent,
            vp_margin_alpha=vp_margin_alpha,
        )

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return action_from_json(
            self.choose(
                web_view(game, self.color),
                simulations=self.params.simulations,
                mcts_opponent=self.params.mcts_opponent,
                vp_margin_alpha=self.params.vp_margin_alpha,
            )
        )


class BestTransformerSearchPlayer(NeuralPolicyPlayer):
    LABEL = 'Transformer all-time best (cohort 6) — search'
    MODE = 'search'

    @dataclass(frozen=True)
    class Params:
        simulations: int = 128
        mcts_opponent: str = 'self-policy'
        vp_margin_alpha: float = 0.03

    @staticmethod
    def availability():
        return _check_latest_availability(lambda: BEST_TRANSFORMER, 'All-time best Transformer')

    @classmethod
    def choose(cls, observation, simulations=128, mcts_opponent='self-policy', vp_margin_alpha=0.03):
        return _choose_latest(
            observation,
            lambda: BEST_TRANSFORMER,
            'TYPED-EXPHORMER',
            simulations=simulations,
            mcts_opponent=mcts_opponent,
            vp_margin_alpha=vp_margin_alpha,
        )

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return action_from_json(
            self.choose(
                web_view(game, self.color),
                simulations=self.params.simulations,
                mcts_opponent=self.params.mcts_opponent,
                vp_margin_alpha=self.params.vp_margin_alpha,
            )
        )


class V2BestCnnSearchPlayer(NeuralPolicyPlayer):
    LABEL = 'CNN v2 best (self-policy) — search'
    MODE = 'search'

    @dataclass(frozen=True)
    class Params:
        simulations: int = 128
        mcts_opponent: str = 'self-policy'
        vp_margin_alpha: float = 0.03

    @staticmethod
    def availability():
        return _check_latest_availability(latest_v2_cnn_checkpoint, 'v2 best CNN')

    @classmethod
    def choose(cls, observation, simulations=128, mcts_opponent='self-policy', vp_margin_alpha=0.03):
        return _choose_latest(
            observation,
            latest_v2_cnn_checkpoint,
            'XD-CNN',
            simulations=simulations,
            mcts_opponent=mcts_opponent,
            vp_margin_alpha=vp_margin_alpha,
        )

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return action_from_json(
            self.choose(
                web_view(game, self.color),
                simulations=self.params.simulations,
                mcts_opponent=self.params.mcts_opponent,
                vp_margin_alpha=self.params.vp_margin_alpha,
            )
        )


class V2BestTransformerSearchPlayer(NeuralPolicyPlayer):
    LABEL = 'Transformer v2 best (self-policy) — search'
    MODE = 'search'

    @dataclass(frozen=True)
    class Params:
        simulations: int = 128
        mcts_opponent: str = 'self-policy'
        vp_margin_alpha: float = 0.03

    @staticmethod
    def availability():
        return _check_latest_availability(latest_v2_transformer_checkpoint, 'v2 best Transformer')

    @classmethod
    def choose(cls, observation, simulations=128, mcts_opponent='self-policy', vp_margin_alpha=0.03):
        return _choose_latest(
            observation,
            latest_v2_transformer_checkpoint,
            'TYPED-EXPHORMER',
            simulations=simulations,
            mcts_opponent=mcts_opponent,
            vp_margin_alpha=vp_margin_alpha,
        )

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        return action_from_json(
            self.choose(
                web_view(game, self.color),
                simulations=self.params.simulations,
                mcts_opponent=self.params.mcts_opponent,
                vp_margin_alpha=self.params.vp_margin_alpha,
            )
        )


def _cohort_set_slug(dir_name: str) -> str:
    slug = dir_name
    for prefix in ('coreml-cohorts-', 'cohorts-'):
        if slug.startswith(prefix) and len(slug) > len(prefix):
            slug = slug[len(prefix):]
            break
    return slug


def _cohort_number_from_path(ckpt_path: Path):
    for part in ckpt_path.parts:
        m = re.fullmatch(r'cohort-(\d+)', part)
        if m:
            return int(m.group(1))
    return None


def _detect_cohort_mcts_opponent(cohort_dir: Path, latest_ckpt: Path) -> str:
    valid = ('self-policy', 'adversarial-puct', 'frozen-mlp')
    cohort_root = latest_ckpt.parents[3] if len(latest_ckpt.parents) >= 4 else cohort_dir
    for candidate in (cohort_root / 'manifest.json', cohort_dir / 'multi_cohort_summary.json'):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text())
            opp = (
                data.get('collection_config', {}).get('mcts_opponent')
                or data.get('mcts_opponent')
            )
            if opp in valid:
                return opp
        except (OSError, ValueError, AttributeError):
            pass
    if 'adversarial-puct' in cohort_dir.name:
        return 'adversarial-puct'
    return 'self-policy'


def _make_cohort_set_player(cohort_dir: Path, arch: str, model_id: str, label: str, default_opp: str):
    cache_key = (cohort_dir.name, arch)
    cached = _DYNAMIC_PLAYER_CLASSES.get(cache_key)
    if cached is not None:
        cached.LABEL = label
        return cached

    def _latest_ckpt():
        candidates = sorted(cohort_dir.glob(f'cohort-*/{arch}/iteration-1/best.pt'))
        return candidates[-1] if candidates else (FALLBACK_CNN if arch == 'cnn' else FALLBACK_TRANSFORMER)

    @dataclass(frozen=True)
    class _Params:
        simulations: int = 128
        mcts_opponent: str = default_opp
        vp_margin_alpha: float = 0.03

    class _DynamicCohortSearchPlayer(NeuralPolicyPlayer):
        LABEL = label
        MODE = 'search'
        Params = _Params

        @staticmethod
        def availability():
            return _check_latest_availability(_latest_ckpt, label)

        @classmethod
        def choose(cls, observation, simulations=128, mcts_opponent=default_opp, vp_margin_alpha=0.03):
            return _choose_latest(
                observation,
                _latest_ckpt,
                model_id,
                simulations=simulations,
                mcts_opponent=mcts_opponent,
                vp_margin_alpha=vp_margin_alpha,
            )

        def decide(self, game, playable_actions):
            if len(playable_actions) == 1:
                return playable_actions[0]
            return action_from_json(
                self.choose(
                    web_view(game, self.color),
                    simulations=self.params.simulations,
                    mcts_opponent=self.params.mcts_opponent,
                    vp_margin_alpha=self.params.vp_margin_alpha,
                )
            )

    safe_name = re.sub(r'[^A-Za-z0-9]+', '_', cohort_dir.name).strip('_')
    _DynamicCohortSearchPlayer.__name__ = f'Cohort_{safe_name}_{arch}_SearchPlayer'
    _DYNAMIC_PLAYER_CLASSES[cache_key] = _DynamicCohortSearchPlayer
    return _DynamicCohortSearchPlayer


def discover_cohort_set_agents():
    """Auto-discover all new multi-cohort sets under runs/catan-ai for the WebUI."""
    if not RUNS_ROOT.is_dir():
        return []
    discovered_dirs = []
    for entry in RUNS_ROOT.iterdir():
        if not entry.is_dir() or entry.name in STATIC_COHORT_DIRS:
            continue
        cnn_ckpts = sorted(entry.glob('cohort-*/cnn/iteration-1/best.pt'))
        tr_ckpts = sorted(entry.glob('cohort-*/transformer/iteration-1/best.pt'))
        if not cnn_ckpts and not tr_ckpts:
            continue
        newest_mtime = max(p.stat().st_mtime for p in (cnn_ckpts[-1:] + tr_ckpts[-1:]))
        discovered_dirs.append((newest_mtime, entry, cnn_ckpts, tr_ckpts))
    discovered_dirs.sort(key=lambda item: (-item[0], item[1].name))

    records = []
    for _, cohort_dir, cnn_ckpts, tr_ckpts in discovered_dirs:
        slug = _cohort_set_slug(cohort_dir.name)
        key_slug = re.sub(r'[^A-Za-z0-9]+', '_', slug).strip('_').upper()
        for arch, display_arch, model_id, ckpts in (
            ('cnn', 'CNN', 'XD-CNN', cnn_ckpts),
            ('transformer', 'Transformer', 'TYPED-EXPHORMER', tr_ckpts),
        ):
            if not ckpts:
                continue
            latest = ckpts[-1]
            cohort_num = _cohort_number_from_path(latest)
            cohort_tag = f'cohort {cohort_num}' if cohort_num is not None else 'latest'
            default_opp = _detect_cohort_mcts_opponent(cohort_dir, latest)
            label = f'{display_arch} {slug} ({cohort_tag}) — search'
            key = f'{arch.upper()}_COHORT_{key_slug}_SEARCH'
            cls = _make_cohort_set_player(cohort_dir, arch, model_id, label, default_opp)
            entry_dict = {
                'key': key,
                'name': label,
                'description': (
                    f'Auto-discovered {slug} cohort set ({cohort_dir.name}, {cohort_tag}) '
                    f'{model_id} W4 best checkpoint. Two-player, 15 VP; visible-point friendly robber supported.'
                ),
                'protocol': 'catan-ai-schema-3',
                'searchable': True,
                'default_simulations': 128,
                'default_mcts_opponent': default_opp,
                'supported_rules': {
                    'players': 2,
                    'map': 'BASE',
                    'vps_to_win': 15,
                    'discard_limit': 9,
                },
                'capabilities': ['decide', 'checkpoint'],
            }
            records.append((entry_dict, cls))
    return records
