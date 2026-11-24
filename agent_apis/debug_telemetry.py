"""Live MCTS policy telemetry and step-by-step control for the WebUI debug window.

Keeps normal human-vs-bot `_browser_view` privacy completely untouched while
exposing real-time root policy distributions, development card hands, legal
trades, and wait-between-actions stepping through dedicated debug endpoints.
"""

from contextlib import contextmanager
import threading
import time

from catanatron.serialization import action_to_json


DEV_CARD_KEYS = (
    "KNIGHT",
    "MONOPOLY",
    "YEAR_OF_PLENTY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)
RESOURCE_KEYS = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


class _DebugTracker:
    """Mutable live telemetry state for one in-flight bot decision."""

    def __init__(self, game_id, state_index, color, player_label, dev_cards, resources, legal_actions, simulations=None, can_set_simulations=False, mcts_opponent=None, can_set_mcts_opponent=False):
        self.game_id = game_id
        self.state_index = state_index
        self.color = color
        self.player_label = player_label
        self.dev_cards = dev_cards
        self.resources = resources
        self.legal_actions = legal_actions
        self.simulations = simulations
        self.can_set_simulations = can_set_simulations
        self.mcts_opponent = mcts_opponent
        self.can_set_mcts_opponent = can_set_mcts_opponent
        self.last_publish = 0.0
        self.has_published_search = False


_LOCK = threading.RLock()
_SNAPSHOTS = {}          # game_id -> latest snapshot dict
_PENDING_ACTIONS = {}    # game_id -> {"state_index": int, "action": list}
_GAME_LOCKS = {}         # game_id -> threading.Lock
_ACTIVE_TRACKERS = {}    # game_id -> _DebugTracker
_TLS = threading.local()


def game_lock(game_id):
    with _LOCK:
        lock = _GAME_LOCKS.get(game_id)
        if lock is None:
            lock = _GAME_LOCKS[game_id] = threading.Lock()
        return lock


def current_tracker():
    return getattr(_TLS, "tracker", None)


@contextmanager
def bind_tracker(tracker):
    """Propagate a parent thread's tracker into a search worker thread."""
    previous = getattr(_TLS, "tracker", None)
    _TLS.tracker = tracker
    try:
        yield
    finally:
        _TLS.tracker = previous


def _extract_hands(game, color):
    seat = game.state.colors.index(color)
    prefix = f"P{seat}_"
    ps = game.state.player_state
    dev_cards = {k: int(ps.get(f"{prefix}{k}_IN_HAND", 0)) for k in DEV_CARD_KEYS}
    resources = {k: int(ps.get(f"{prefix}{k}_IN_HAND", 0)) for k in RESOURCE_KEYS}
    return dev_cards, resources


def _player_label(player):
    label = getattr(player, "LABEL", None)
    if label:
        return str(label)
    return type(player).__name__


def game_simulations_info(game):
    """Return (current_simulations, can_set_simulations) for searchable bot seats."""
    current = game.state.current_player()
    candidates = [current] + [p for p in game.state.players if p is not current]
    for p in candidates:
        params = getattr(p, "params", None)
        if p.is_bot and params is not None and hasattr(params, "simulations"):
            return params.simulations, True
    return None, False


def game_mcts_opponent_info(game):
    """Return (current_mcts_opponent, can_set_mcts_opponent) for searchable bot seats."""
    current = game.state.current_player()
    candidates = [current] + [p for p in game.state.players if p is not current]
    for p in candidates:
        params = getattr(p, "params", None)
        if p.is_bot and params is not None and hasattr(params, "mcts_opponent"):
            return params.mcts_opponent, True
    return None, False


def _uniform_action_entries(legal_actions):
    n = len(legal_actions)
    p = (1.0 / n) if n else 0.0
    return [
        {
            "action": a,
            "action_id": None,
            "policy": p,
            "prior": None,
            "visits": 0,
            "mean_value": None,
            "selected": False,
        }
        for a in legal_actions
    ]


def snapshot_from_game(game, status="idle"):
    """Build a baseline debug snapshot directly from the current Game state."""
    player = game.state.current_player()
    color = player.color
    dev_cards, resources = _extract_hands(game, color)
    legal_actions = [action_to_json(a) for a in game.playable_actions]
    state_index = len(game.state.action_records)
    sims, can_set_sims = game_simulations_info(game)
    mcts_opp, can_set_opp = game_mcts_opponent_info(game)
    return {
        "game_id": game.id,
        "state_index": state_index,
        "color": color.value,
        "is_bot": bool(player.is_bot),
        "player_label": _player_label(player),
        "status": status,
        "mode": "forced" if len(legal_actions) == 1 else ("bot" if player.is_bot else "human"),
        "simulations_done": 0,
        "simulations_target": sims,
        "simulations": sims,
        "can_set_simulations": can_set_sims,
        "mcts_opponent": mcts_opp,
        "can_set_mcts_opponent": can_set_opp,
        "root_value": None,
        "updated_at": time.time(),
        "dev_cards": dev_cards,
        "resources": resources,
        "actions": _uniform_action_entries(legal_actions) if player.is_bot else [],
        "chosen_action": None,
    }


def get_snapshot(game_id):
    with _LOCK:
        snap = _SNAPSHOTS.get(game_id)
        return dict(snap) if snap is not None else None


def get_pending_action(game_id, state_index):
    with _LOCK:
        entry = _PENDING_ACTIONS.get(game_id)
        if entry is not None and entry["state_index"] == state_index:
            return entry["action"]
        return None


def clear_pending_action(game_id):
    with _LOCK:
        _PENDING_ACTIONS.pop(game_id, None)


def _resolve_tracker(session_id=None):
    tracker = getattr(_TLS, "tracker", None)
    if tracker is not None:
        return tracker
    if session_id is not None:
        with _LOCK:
            return _ACTIVE_TRACKERS.get(session_id)
    return None


def publish_search_progress(
    root_edges,
    simulations_done,
    simulations_target=None,
    root_value=None,
    selected_action_id=None,
    session_id=None,
    force=False,
):
    """Publish live MCTS root visit/prior policy distribution during search."""
    tracker = _resolve_tracker(session_id)
    if tracker is None:
        return
    now = time.perf_counter()
    if not force and tracker.has_published_search and (now - tracker.last_publish) < 0.06:
        return
    tracker.last_publish = now
    tracker.has_published_search = True

    total_visits = sum(e.visits for e in root_edges.values())
    if selected_action_id is None and root_edges:
        if total_visits > 0:
            best_id = max(
                root_edges.keys(),
                key=lambda i: (root_edges[i].visits, root_edges[i].mean, -i),
            )
        else:
            best_id = max(root_edges.keys(), key=lambda i: (root_edges[i].prior, -i))
    else:
        best_id = selected_action_id

    actions = []
    for aid, edge in root_edges.items():
        policy = (edge.visits / total_visits) if total_visits > 0 else float(edge.prior)
        actions.append(
            {
                "action": action_to_json(edge.action),
                "action_id": int(aid),
                "policy": float(policy),
                "prior": float(edge.prior),
                "visits": int(edge.visits),
                "mean_value": float(edge.mean) if edge.visits > 0 else None,
                "selected": aid == best_id,
            }
        )
    actions.sort(key=lambda r: (-r["policy"], -(r["prior"] or 0.0), r["action_id"] or 0))

    est_value = root_value
    if total_visits > 0:
        est_value = sum(e.total for e in root_edges.values()) / total_visits

    chosen = next((r["action"] for r in actions if r["selected"]), None)
    snap = {
        "game_id": tracker.game_id,
        "state_index": tracker.state_index,
        "color": tracker.color,
        "is_bot": True,
        "player_label": tracker.player_label,
        "status": "searching",
        "mode": "search",
        "simulations_done": int(simulations_done),
        "simulations_target": int(simulations_target) if simulations_target is not None else None,
        "simulations": tracker.simulations,
        "can_set_simulations": tracker.can_set_simulations,
        "mcts_opponent": tracker.mcts_opponent,
        "can_set_mcts_opponent": tracker.can_set_mcts_opponent,
        "root_value": float(est_value) if est_value is not None else None,
        "updated_at": time.time(),
        "dev_cards": dict(tracker.dev_cards),
        "resources": dict(tracker.resources),
        "actions": actions,
        "chosen_action": chosen,
    }
    with _LOCK:
        _SNAPSHOTS[tracker.game_id] = snap


def publish_policy_result(view, probabilities, value, selected_action_json, session_id=None):
    """Publish pure-policy (0-simulation) distribution."""
    tracker = _resolve_tracker(session_id)
    if tracker is None:
        return
    from catanatron.serialization import action_from_json
    from agent_apis.stochastic import action_id

    tracker.has_published_search = True
    actions = []
    for payload in view["current_playable_actions"]:
        aid = action_id(action_from_json(payload))
        prob = float(probabilities.get(aid, 0.0))
        actions.append(
            {
                "action": payload,
                "action_id": int(aid),
                "policy": prob,
                "prior": prob,
                "visits": 0,
                "mean_value": None,
                "selected": payload == selected_action_json,
            }
        )
    actions.sort(key=lambda r: (-r["policy"], r["action_id"] or 0))
    snap = {
        "game_id": tracker.game_id,
        "state_index": tracker.state_index,
        "color": tracker.color,
        "is_bot": True,
        "player_label": tracker.player_label,
        "status": "searching",
        "mode": "policy",
        "simulations_done": 0,
        "simulations_target": 0,
        "simulations": tracker.simulations,
        "can_set_simulations": tracker.can_set_simulations,
        "mcts_opponent": tracker.mcts_opponent,
        "can_set_mcts_opponent": tracker.can_set_mcts_opponent,
        "root_value": float(value) if value is not None else None,
        "updated_at": time.time(),
        "dev_cards": dict(tracker.dev_cards),
        "resources": dict(tracker.resources),
        "actions": actions,
        "chosen_action": selected_action_json,
    }
    with _LOCK:
        _SNAPSHOTS[tracker.game_id] = snap


@contextmanager
def track_decision(game, hold=False):
    """Context manager wrapping a bot decision to record live telemetry."""
    player = game.state.current_player()
    color = player.color
    dev_cards, resources = _extract_hands(game, color)
    legal_actions = [action_to_json(a) for a in game.playable_actions]
    state_index = len(game.state.action_records)

    sims, can_set_sims = game_simulations_info(game)
    mcts_opp, can_set_opp = game_mcts_opponent_info(game)
    tracker = _DebugTracker(
        game_id=game.id,
        state_index=state_index,
        color=color.value,
        player_label=_player_label(player),
        dev_cards=dev_cards,
        resources=resources,
        legal_actions=legal_actions,
        simulations=sims,
        can_set_simulations=can_set_sims,
        mcts_opponent=mcts_opp,
        can_set_mcts_opponent=can_set_opp,
    )
    initial_snap = {
        "game_id": game.id,
        "state_index": state_index,
        "color": color.value,
        "is_bot": bool(player.is_bot),
        "player_label": tracker.player_label,
        "status": "searching",
        "mode": "forced" if len(legal_actions) == 1 else "search",
        "simulations_done": 0,
        "simulations_target": sims,
        "simulations": sims,
        "can_set_simulations": can_set_sims,
        "mcts_opponent": mcts_opp,
        "can_set_mcts_opponent": can_set_opp,
        "root_value": None,
        "updated_at": time.time(),
        "dev_cards": dev_cards,
        "resources": resources,
        "actions": _uniform_action_entries(legal_actions),
        "chosen_action": None,
    }
    with _LOCK:
        _SNAPSHOTS[game.id] = initial_snap
        _ACTIVE_TRACKERS[game.id] = tracker

    previous = getattr(_TLS, "tracker", None)
    _TLS.tracker = tracker
    outcome = {"action_json": None}
    try:
        yield outcome
    finally:
        _TLS.tracker = previous
        with _LOCK:
            if _ACTIVE_TRACKERS.get(game.id) is tracker:
                _ACTIVE_TRACKERS.pop(game.id, None)
            snap = dict(_SNAPSHOTS.get(game.id, initial_snap))
            chosen = outcome["action_json"]
            if chosen is not None:
                snap["chosen_action"] = chosen
                if not tracker.has_published_search:
                    # Forced move or non-neural bot (e.g. JSettlers / AlphaBeta):
                    # assign 1.0 to the chosen move and 0.0 to other legal moves.
                    snap["mode"] = "forced" if len(legal_actions) == 1 else "bot"
                    snap["actions"] = [
                        {
                            "action": a,
                            "action_id": None,
                            "policy": 1.0 if a == chosen else 0.0,
                            "prior": None,
                            "visits": 0,
                            "mean_value": None,
                            "selected": a == chosen,
                        }
                        for a in legal_actions
                    ]
                    snap["actions"].sort(key=lambda r: -r["policy"])
                else:
                    # Ensure chosen action is flagged selected
                    for row in snap["actions"]:
                        row["selected"] = row["action"] == chosen
                snap["status"] = "awaiting_step" if hold else "completed"
                snap["updated_at"] = time.time()
                _SNAPSHOTS[game.id] = snap
                if hold:
                    _PENDING_ACTIONS[game.id] = {
                        "state_index": state_index,
                        "action": chosen,
                    }
                else:
                    _PENDING_ACTIONS.pop(game.id, None)
            else:
                snap["status"] = "idle"
                snap["updated_at"] = time.time()
                _SNAPSHOTS[game.id] = snap
