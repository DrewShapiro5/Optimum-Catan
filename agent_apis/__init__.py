"""Shared agent discovery and legal-action selection for web and scripts."""

import importlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("agents.json")


def _entries():
    return json.loads(MANIFEST.read_text())["agents"]


def _implementation(entry):
    module, name = entry["implementation"].split(":")
    return getattr(importlib.import_module(module), name)


def _all_records():
    static_entries = _entries()
    static_keys = {entry["key"] for entry in static_entries}
    records = [(entry, _implementation(entry)) for entry in static_entries]
    try:
        from .neural.latest_web import discover_cohort_set_agents

        dynamic = [
            (entry, cls)
            for entry, cls in discover_cohort_set_agents()
            if entry["key"] not in static_keys
        ]
    except Exception:
        dynamic = []
    if not dynamic:
        return records
    # Insert auto-discovered cohort sets right after JSETTLERS so JSETTLERS stays index 0
    # and newly trained cohort sets appear at the top of the model list.
    insert_at = 1 if records and records[0][0].get("key") == "JSETTLERS" else 0
    return records[:insert_at] + dynamic + records[insert_at:]


def list_agents():
    result = []
    for entry, cls in _all_records():
        ready, reason = cls.availability()
        result.append(
            {k: v for k, v in entry.items() if k != "implementation"}
            | {
                "serverKey": entry["key"],
                "available": ready,
                "reason": reason,
                "endpoint": f"/api/agents/{entry['key']}/decide",
            }
        )
    return result


def register_agents(registry=None):
    if registry is None:
        from catanatron.registry import REGISTRY

        registry = REGISTRY
    for entry, cls in _all_records():
        registry.register(entry["key"], cls, replace=True)
    return registry


def choose_action(key, observation):
    match = next(((e, cls) for e, cls in _all_records() if e["key"] == key.upper()), None)
    if match is None:
        raise ValueError(f"Unknown agent: {key}")
    _, cls = match
    ready, reason = cls.availability()
    if not ready:
        raise RuntimeError(reason)
    return cls.choose(observation)


def synchronize(key, observation, color=None):
    """Update an agent's parallel state without asking it to choose a move."""
    match = next(((e, cls) for e, cls in _all_records() if e["key"] == key.upper()), None)
    if match is None:
        raise ValueError(f"Unknown agent: {key}")
    _, cls = match
    ready, reason = cls.availability()
    if not ready:
        raise RuntimeError(reason)
    sync = getattr(cls, "synchronize", None)
    if sync is None:
        return {"state_index": observation["state_index"], "stateful": False}
    return sync(observation, color)


def synchronize_game(game):
    """Fan out a committed game update to the registered, seated providers.

    This integration hook lives here so neither the game engine nor its UI
    needs provider-specific knowledge. The same hook is available to scripts.
    """
    from catanatron.serialization import web_view

    providers = [(cls, entry["key"]) for entry, cls in _all_records()]
    updates = {}
    for player in game.state.players:
        for cls, key in providers:
            if isinstance(player, cls) and hasattr(cls, "synchronize"):
                updates[player.color.value] = synchronize(
                    key, web_view(game, player.color), player.color.value
                )
                break
    return updates
