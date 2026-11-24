"""Bounded JSettlers self-play data and streaming Catanatron training/replay APIs.

Run with the project environment: python -m agent_apis.dataset generate --help.
"""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import time

from catanatron import Game
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.action_space import get_action_array, to_action_space
from catanatron.models.enums import Action
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import (
    action_from_json, action_record_to_json, action_to_json, client_view,
    state_from_json, state_to_json,
)

MAX_GAMES = 1000
MAX_BYTES = 1_000_000_000  # Decimal GB; also cap the uncompressed JSON payload.
COLORS = (Color.BLUE, Color.RED)
VERSION = 1
UPSTREAM_REVISION = "7a756baac6ad675f0930605bfb3a8e1bd7a796a5"


def encode(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def action_vocabulary():
    return [action_to_json(Action(Color.BLUE, kind, value))[1:]
            for kind, value in get_action_array(COLORS, "BASE")]


def observation(game):
    """Compact pre-action state. Static map lives in the game's header.

    Model inputs use numeric features. This state supports inspection without
    repeating map, history, or geometry at every decision.
    """
    doc = client_view(state_to_json(game), game.state.current_color())
    for key in ("map", "action_records", "game", "schema_version"):
        doc.pop(key)
    # The exact remaining composition can disclose opponents' hidden dev cards.
    doc["development_listdeck"] = sum(doc["development_listdeck"].values())
    doc["perspective"] = game.state.current_color().value
    return doc


def replay_game(path):
    """Yield (mutable Game BEFORE action, row); next iteration advances it.

    Replays consume the saved engine RNG and verify the recorded chance result.
    The original bot's Java randomness is unnecessary: its actions are recorded.
    """
    with gzip.open(path, "rt") as stream:
        header = json.loads(next(stream))
        if header.get("schema_version") == 2:
            from agent_apis.dataset_bundle import replay_saved
            yield from replay_saved(path)
            return
        initial = header["initial_state"]
        players = [SimplePlayer(Color[c]) for c in initial["colors"]]
        game = state_from_json(initial, players)
        for line in stream:
            row = json.loads(line)
            yield game, row
            record = game.execute(action_from_json(row["action"]))
            if action_record_to_json(record) != row["record"]:
                raise ValueError(f"Chance outcome mismatch: {path}, step {row['step']}")
        if state_to_json(game) != header["final_state"]:
            raise ValueError(f"Final state mismatch: {path}")


def iter_samples(directory, split=None):
    """Stream safe training columns; never expose privileged replay fields.

    features -> policy/value input, action_id -> cross entropy target,
    legal_action_ids -> sparse mask, value_target -> actor's terminal outcome.
    Train/validation/test membership is assigned to whole games.
    """
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    for entry in manifest["games"]:
        if split is not None and entry["split"] != split:
            continue
        with gzip.open(Path(directory) / entry["file"], "rt") as stream:
            next(stream)
            for line in stream:
                row = json.loads(line)
                yield {key: row[key] for key in (
                    "game_id", "step", "num_turns", "actor", "features",
                    "action_id", "legal_action_ids", "value_target", "reward",
                    "terminated", "truncated", "next_actor",
                )}


def _generate_game(index, seed, max_turns, max_actions, byte_budget):
    from agent_apis import register_agents
    from agent_apis.jsettlers.player import BRIDGE
    from catanatron.registry import REGISTRY

    started = time.monotonic()
    # Release previous games' native sessions and bound JVM memory use.
    BRIDGE.close()
    try:
        register_agents()
        game = Game(REGISTRY.build_all(["JSETTLERS"] * 2), seed=seed,
                    vps_to_win=15, discard_limit=9)
        # Feature extraction materializes empty defaultdict building lists.
        # Include those defaults in the checkpoint, so replay has identical
        # serialized cache keys even for players who never build a city.
        for color in game.state.colors:
            for kind in ("SETTLEMENT", "CITY", "ROAD"):
                game.state.buildings_by_color[color][kind]
        initial = json.loads(encode(state_to_json(game)))
        features = get_feature_ordering(2, "BASE")
        rows = []
        raw_estimate = 0
        while (game.winning_color() is None and game.state.num_turns < max_turns
               and len(rows) < max_actions):
            actor = game.state.current_color()
            legal = list(game.playable_actions)
            selected = game.state.current_player().decide(game, legal)
            row = {
                "type": "step", "game_id": game.id, "step": len(rows),
                "num_turns": game.state.num_turns, "actor": actor.value,
                "observation": observation(game),
                "features": create_sample_vector(game, actor, features),
                "action": action_to_json(selected),
                "action_id": to_action_space(selected, COLORS, "BASE"),
                "legal_action_ids": sorted({to_action_space(a, COLORS, "BASE") for a in legal}),
            }
            record = game.execute(selected)
            row["record"] = action_record_to_json(record)
            row["next_actor"] = game.state.current_color().value
            rows.append(row)
            raw_estimate += len(encode(row)) + 200
            if raw_estimate > byte_budget:
                return {"index": index, "seed": seed, "error": "game_exceeds_byte_budget"}
        winner = game.winning_color()
        if winner is None:
            return {"index": index, "seed": seed, "error": "turn_or_action_limit",
                    "num_turns": game.state.num_turns, "steps": len(rows)}
        for step, row in enumerate(rows):
            terminal = step == len(rows) - 1
            row.update(value_target=1 if row["actor"] == winner.value else -1,
                       reward=(1 if row["actor"] == winner.value else -1) if terminal else 0,
                       terminated=terminal, truncated=False)
        header = {
            "type": "game", "schema_version": VERSION, "index": index,
            "seed": seed, "winner": winner.value, "num_turns": game.state.num_turns,
            "steps": len(rows), "initial_state": initial,
            "final_state": json.loads(encode(state_to_json(game))),
        }
        # Verify every executed action/chance result against an independent game.
        replay = state_from_json(initial, [SimplePlayer(c) for c in COLORS])
        for row in rows:
            record = replay.execute(action_from_json(row["action"]))
            if action_record_to_json(record) != row["record"]:
                raise ValueError("Generated trajectory failed deterministic replay")
        if state_to_json(replay) != header["final_state"]:
            differing = [k for k, v in state_to_json(replay).items()
                         if v != header["final_state"][k]]
            raise ValueError(f"Generated final state failed deterministic replay: {differing}")
        raw = encode(header) + b"".join(encode(row) for row in rows)
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
            stream.write(raw)
        payload = buffer.getvalue()
        # Assign split by seed before looking at any game outcome.
        bucket = int(hashlib.sha256(str(seed).encode()).hexdigest()[:8], 16) % 100
        return {
            "index": index, "seed": seed, "game_id": game.id,
            "file": f"game-{index:04d}.jsonl.gz",
            "split": "train" if bucket < 90 else "validation" if bucket < 95 else "test",
            "winner": winner.value, "num_turns": game.state.num_turns,
            "steps": len(rows), "bytes": len(payload), "uncompressed_bytes": len(raw),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "action_counts": dict(Counter(row["action"][1] for row in rows)),
            "seconds": round(time.monotonic() - started, 3), "payload": payload,
        }
    except Exception as error:
        return {"index": index, "seed": seed,
                "error": f"{type(error).__name__}: {error}"}
    finally:
        BRIDGE.close()


def generate(directory, games=1000, max_bytes=MAX_BYTES, workers=4,
             seed=20261104, max_turns=1000, max_actions=10000):
    if not 1 <= games <= MAX_GAMES or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("Require 1..1000 games and 1..1,000,000,000 bytes")
    if workers < 1 or max_turns < 1 or max_actions < 1:
        raise ValueError("Workers, max turns and max actions must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Output directory must be empty; existing datasets are never overwritten")
    from agent_apis.jsettlers.player import JSettlersPlayer
    ready, reason = JSettlersPlayer.availability()
    if not ready:
        raise RuntimeError(reason)
    manifest = {
        "schema_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "JSETTLERS", "strategy": "SMART", "jsettlers_revision": UPSTREAM_REVISION,
        "rules": {"players": 2, "map": "BASE", "vps_to_win": 15,
                  "discard_limit": 9, "friendly_robber": False, "player_trading": False},
        "limits": {"games": games, "bytes": max_bytes, "max_turns": max_turns,
                   "max_actions": max_actions, "byte_basis": "stored AND uncompressed JSON"},
        "base_seed": seed, "feature_names": get_feature_ordering(2, "BASE"),
        "action_vocabulary": action_vocabulary(),
        "value_target": "+1 winner / -1 loser, from the acting player's perspective",
        "policy_target": "one-hot JSettlers action; these are demonstrations, not MCTS visit counts",
        "games": [], "failures": [], "stop_reason": "running",
    }
    stored = raw_bytes = 0
    # Metadata uses a fixed reserve so the dataset never crosses either cap.
    reserve = min(max_bytes, max(32_768, min(8_000_000, max_bytes // 10)))
    next_index = 0
    def save_manifest():
        manifest["num_games"] = len(manifest["games"])
        manifest["num_samples"] = sum(g["steps"] for g in manifest["games"])
        manifest["shard_bytes"] = stored
        manifest["uncompressed_shard_bytes"] = raw_bytes
        data = encode(manifest)
        if len(data) > reserve or max(stored, raw_bytes) + len(data) > max_bytes:
            raise ValueError("Metadata exceeds the reserved byte budget")
        # In-place writing avoids temporarily duplicating the complete manifest.
        (directory / "manifest.json").write_bytes(data)

    save_manifest()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = set()
        while next_index < games or pending:
            while next_index < games and len(pending) < workers:
                pending.add(pool.submit(_generate_game, next_index, seed + next_index,
                                        max_turns, max_actions, max_bytes - reserve))
                next_index += 1
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                if "error" in result:
                    manifest["failures"].append(result)
                    print(f"Skipped seed {result['seed']}: {result['error']}", flush=True)
                    save_manifest()
                    continue
                payload = result.pop("payload")
                if (stored + result["bytes"] > max_bytes - reserve
                        or raw_bytes + result["uncompressed_bytes"] > max_bytes - reserve):
                    manifest["stop_reason"] = "byte_limit"
                    next_index = games
                    # Results already in flight are discarded without writing.
                    for task in pending:
                        task.cancel()
                    pending.clear()
                    break
                (directory / result["file"]).write_bytes(payload)
                stored += result["bytes"]
                raw_bytes += result["uncompressed_bytes"]
                manifest["games"].append(result)
                save_manifest()
                count = len(manifest["games"])
                if count == 1 or count % 10 == 0:
                    print(f"{count} games; {manifest['num_samples']:,} samples; "
                          f"{stored / 1e6:.1f} MB stored / {raw_bytes / 1e6:.1f} MB raw", flush=True)
            if manifest["stop_reason"] == "byte_limit":
                break
    manifest["games"].sort(key=lambda g: g["index"])
    if manifest["stop_reason"] == "running":
        manifest["stop_reason"] = "game_limit" if not manifest["failures"] else "attempt_limit"
    save_manifest()
    return manifest


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["games"]:
        raise ValueError("Dataset has no completed games")
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    assert total <= manifest["limits"]["bytes"] <= MAX_BYTES
    assert len(manifest["games"]) <= manifest["limits"]["games"] <= MAX_GAMES
    assert manifest["action_vocabulary"] == action_vocabulary()
    assert manifest["feature_names"] == get_feature_ordering(2, "BASE")
    counts = Counter()
    raw_bytes = (directory / "manifest.json").stat().st_size
    for entry in manifest["games"]:
        path = directory / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_size == entry["bytes"]
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
        assert len(raw) == entry["uncompressed_bytes"]
        raw_bytes += len(raw)
        header = json.loads(raw.split(b"\n", 1)[0])
        assert header["winner"] == entry["winner"]
        assert header["num_turns"] == entry["num_turns"]
        game_counts = Counter()
        next_actor = None
        for game, row in replay_game(path):
            actor = game.state.current_color()
            if next_actor is not None:
                assert next_actor == actor.value
            next_actor = row["next_actor"]
            assert row["game_id"] == game.id == entry["game_id"]
            assert row["step"] == sum(game_counts.values())
            assert row["num_turns"] == game.state.num_turns
            assert row["actor"] == actor.value
            assert row["observation"] == observation(game)
            assert row["features"] == create_sample_vector(game, actor)
            assert row["action_id"] == to_action_space(action_from_json(row["action"]), COLORS, "BASE")
            assert row["legal_action_ids"] == sorted({to_action_space(a, COLORS, "BASE") for a in game.playable_actions})
            assert row["action_id"] in row["legal_action_ids"]
            assert row["value_target"] == (1 if actor.value == entry["winner"] else -1)
            terminal = row["step"] == entry["steps"] - 1
            assert row["terminated"] == terminal and row["truncated"] is False
            assert row["reward"] == (row["value_target"] if terminal else 0)
            game_counts[row["action"][1]] += 1
        assert dict(game_counts) == entry["action_counts"]
        assert sum(game_counts.values()) == entry["steps"]
        assert next_actor == game.state.current_color().value
        assert game.winning_color().value == entry["winner"]
        assert game.state.num_turns == entry["num_turns"]
        counts.update(game_counts)
    assert raw_bytes <= manifest["limits"]["bytes"]
    assert sum(counts.values()) == manifest["num_samples"]
    return {"games": len(manifest["games"]), "samples": sum(counts.values()),
            "stored_bytes": total, "uncompressed_bytes": raw_bytes,
            "action_counts": dict(counts), "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate")
    create.add_argument("--output", type=Path, default=Path("data/jsettlers-1v1"))
    create.add_argument("--games", type=int, default=1000)
    create.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    create.add_argument("--workers", type=int, default=4)
    create.add_argument("--seed", type=int, default=20261104)
    create.add_argument("--max-turns", type=int, default=1000)
    create.add_argument("--max-actions", type=int, default=10000)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.directory), indent=2))
    else:
        result = generate(args.output, args.games, args.max_bytes, args.workers,
                          args.seed, args.max_turns, args.max_actions)
        print(json.dumps({k: result[k] for k in ("num_games", "num_samples", "stop_reason")}, indent=2))


if __name__ == "__main__":
    main()
