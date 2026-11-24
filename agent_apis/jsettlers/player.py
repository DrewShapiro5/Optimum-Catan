"""Stateful API adapter for the upstream JSettlers SOCRobotBrain.

Only this package knows both game representations. Each seat has a persistent
native client view, with opponent cards redacted. No heuristic action scorer.
"""

import atexit
from functools import lru_cache
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading

import networkx as nx
from catanatron.models.player import Player
from catanatron.models.map import Port, PORT_DIRECTION_TO_NODEREFS
from catanatron.serialization import web_view, map_from_json, action_to_json

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "integrations/jsettlers/lib"
RESOURCES = ["BRICK", "ORE", "SHEEP", "WHEAT", "WOOD"]
DEVS = ["KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING", "VICTORY_POINT"]


class Bridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.reader.join(timeout=2)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def request(self, **payload):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.process = subprocess.Popen(
                    [
                        shutil.which("java"),
                        "-Xmx768m",
                        "-Djava.awt.headless=true",
                        "-cp",
                        str(LIB / "*"),
                        "Bridge",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    text=True,
                    bufsize=1,
                )
                replies = queue.Queue()

                def read(stream):
                    for line in stream:
                        replies.put(line)
                    replies.put(None)

                self.reader = threading.Thread(
                    target=read, args=(self.process.stdout,), daemon=True
                )
                self.reader.start()
                self.replies = replies
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                line = self.replies.get(timeout=45)
                if line is None:
                    raise RuntimeError("JSettlers adapter exited")
                reply = json.loads(line)
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply
            except queue.Empty as error:
                self.close()
                raise TimeoutError("JSettlers decision exceeded 45 seconds") from error
            except Exception:
                self.close()
                raise


BRIDGE = Bridge()
atexit.register(BRIDGE.close)


def _graph(hexes, edges, ports):
    graph = nx.Graph()
    for coord, nodes in hexes:
        graph.add_node(("h", coord), kind="hex")
        for node in nodes:
            graph.add_node(("n", node), kind="node")
            graph.add_edge(("h", coord), ("n", node), kind="incidence")
    for a, b in edges:
        graph.add_edge(
            ("n", a), ("n", b), kind="port" if frozenset((a, b)) in ports else "road"
        )
    return graph


@lru_cache(maxsize=16)
def _mapping(map_json):
    """Derive the coordinate bijection by topology, including port locations."""
    cmap = map_from_json(json.loads(map_json))
    if len(cmap.land_tiles) != 19:
        raise ValueError("JSettlers supports the 19-hex base board (2 or 4 players)")
    native = BRIDGE.request(op="geometry")
    ports = {}
    for tile in cmap.tiles.values():
        if isinstance(tile, Port):
            refs = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            ports[frozenset(tile.nodes[r] for r in refs)] = (
                0 if tile.resource is None else RESOURCES.index(tile.resource) + 1
            )
    edges = set()
    for tile in cmap.land_tiles.values():
        edges.update(tuple(sorted(e)) for e in tile.edges.values())
    native_edges = {e["coord"]: tuple(e["nodes"]) for e in native["edges"]}
    source = _graph(
        [(t.id, t.nodes.values()) for t in cmap.land_tiles.values()], edges, ports
    )
    target = _graph(
        [(h["coord"], h["nodes"]) for h in native["hexes"]],
        native_edges.values(),
        {frozenset(native_edges[e]) for e in native["port_edges"]},
    )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        source,
        target,
        node_match=nx.algorithms.isomorphism.categorical_node_match("kind", ""),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("kind", ""),
    )
    mapping = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        raise ValueError(
            "Board topology or ports do not match a native JSettlers base board"
        )
    nodes = {k[1]: v[1] for k, v in mapping.items() if k[0] == "n"}
    hexes = {k[1]: v[1] for k, v in mapping.items() if k[0] == "h"}
    edge_lookup = {frozenset(v): k for k, v in native_edges.items()}
    edge_map = {
        tuple(sorted(e)): edge_lookup[frozenset(nodes[n] for n in e)] for e in edges
    }
    native_ports = {frozenset(nodes[n] for n in pair): r for pair, r in ports.items()}
    return {
        "nodes": nodes,
        "hexes": hexes,
        "edges": edge_map,
        "ports": [
            native_ports[frozenset(native_edges[e])] for e in native["port_edges"]
        ],
        "layout": [
            [
                hexes[t.id],
                0 if t.resource is None else RESOURCES.index(t.resource) + 1,
                t.number or -1,
            ]
            for t in cmap.land_tiles.values()
        ],
        "coordinates": {tuple(c): hexes[t.id] for c, t in cmap.land_tiles.items()},
    }


def _phase(obs):
    kinds = {a[1] for a in obs.get("current_playable_actions", [])}
    if obs.get("winning_color"):
        return "over"
    if obs["is_initial_build_phase"]:
        return "initial_road" if "BUILD_ROAD" in kinds else "initial_settlement"
    if obs["is_discarding"]:
        return "discard"
    if obs["is_road_building"]:
        return "free_road"
    if kinds == {"MOVE_ROBBER"}:
        return "robber"
    if "ROLL" in kinds:
        return "roll"
    return "main"


def _state(obs, color=None):
    color = (
        getattr(color, "value", color) or obs.get("perspective") or obs["current_color"]
    )
    if color not in obs["colors"]:
        raise ValueError("Invalid agent perspective")
    if len(obs["colors"]) not in (2, 4):
        raise ValueError("JSettlers supports 2 or 4 players")
    mapping = _mapping(json.dumps(obs["map"], sort_keys=True))
    ps = obs["player_state"]
    players = []
    if f"P{obs['colors'].index(color)}_WOOD_IN_HAND" not in ps:
        raise ValueError("Observation must include the requesting seat’s resource hand")
    for i, c in enumerate(obs["colors"]):
        p = f"P{i}_"
        known = p + "WOOD_IN_HAND" in ps
        hand = [ps.get(p + r + "_IN_HAND", 0) for r in RESOURCES] + [
            0 if known else ps[p + "NUM_RESOURCES_IN_HAND"]
        ]
        old = [
            (
                ps.get(p + d + "_IN_HAND", 0)
                if d == "VICTORY_POINT" or ps.get(p + d + "_OWNED_AT_START")
                else 0
            )
            for d in DEVS
        ]
        new = [ps.get(p + d + "_IN_HAND", 0) - old[j] for j, d in enumerate(DEVS)]
        old.append(0 if known else ps[p + "NUM_DEVELOPMENT_CARDS_IN_HAND"])
        new.append(0)
        players.append(
            dict(
                hand=hand,
                old=old,
                new=new,
                knights=ps[p + "PLAYED_KNIGHT"],
                played=ps[p + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
                army=ps[p + "HAS_ARMY"],
                road=ps[p + "HAS_ROAD"],
                longest=obs["longest_roads_by_player"][c],
            )
        )
    pieces = []
    events = []
    for record_index, (action, result) in enumerate(obs["action_records"]):
        c, kind, value = action
        event = _legal(action, mapping, obs["colors"])
        event["seat"] = obs["colors"].index(c)
        event["initial"] = record_index < len(players) * 4
        if kind == "ROLL":
            event["dice"] = sum(result)
        if kind == "MOVE_ROBBER":
            event["stolen"] = RESOURCES.index(result) + 1 if result in RESOURCES else 6
        events.append(event)
        if kind not in ("BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"):
            continue
        coord = (
            mapping["edges"][tuple(sorted(value))]
            if kind == "BUILD_ROAD"
            else mapping["nodes"][value]
        )
        pieces.append(
            [
                obs["colors"].index(c),
                ["BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"].index(kind),
                coord,
            ]
        )
    state = dict(
        seat=obs["colors"].index(color),
        current=obs["current_turn_index"],
        vps=obs["game"]["vps_to_win"],
        hexes=mapping["layout"],
        ports=mapping["ports"],
        players=players,
        pieces=pieces,
        events=events,
        robber=mapping["coordinates"][tuple(obs["robber_coordinate"])],
        phase=_phase(obs),
        dice=(
            next(
                (
                    sum(result)
                    for action, result in reversed(obs["action_records"])
                    if action[1] == "ROLL"
                ),
                0,
            )
            if ps[f"P{obs['current_turn_index']}_HAS_ROLLED"]
            else 0
        ),
        index=obs["state_index"],
        turn=obs["num_turns"],
        round=obs["num_turns"] // len(players) + 1,
        deck=(
            sum(obs["development_listdeck"].values())
            if isinstance(obs["development_listdeck"], dict)
            else len(obs["development_listdeck"])
        ),
    )
    return f"{obs['game']['id']}:{color}", state, mapping


def _legal(action, mapping, colors):
    _, kind, value = action
    result = {"kind": kind}
    if kind in ("BUILD_SETTLEMENT", "BUILD_CITY"):
        result["coord"] = mapping["nodes"][value]
    elif kind == "BUILD_ROAD":
        result["coord"] = mapping["edges"][tuple(sorted(value))]
    elif kind == "MOVE_ROBBER":
        result.update(
            coord=mapping["coordinates"][tuple(value[0])],
            victim=-1 if value[1] is None else colors.index(value[1]),
        )
    elif kind == "MARITIME_TRADE":
        result.update(
            give=[value[:4].count(r) for r in RESOURCES],
            get=[int(value[4] == r) for r in RESOURCES],
        )
    elif kind == "PLAY_YEAR_OF_PLENTY":
        result["resources"] = [value.count(r) for r in RESOURCES]
    elif kind in ("PLAY_MONOPOLY", "DISCARD_RESOURCE"):
        result["resource"] = RESOURCES.index(value) + 1
    return result


class JSettlersPlayer(Player):
    """Full upstream SOCRobotBrain with native SMART planning and strategies."""

    LABEL = "JSettlers"

    @staticmethod
    def availability():
        if not shutil.which("java"):
            return False, "Java 8+ is required"
        if (
            not (LIB / "jsettlers-adapter.jar").is_file()
            or not (LIB / "gson-2.8.6.jar").is_file()
        ):
            return False, "Run integrations/jsettlers/build.sh"
        return True, ""

    @staticmethod
    def synchronize(observation, color=None):
        session, state, _ = _state(observation, color)
        return BRIDGE.request(op="sync", session=session, state=state)

    @staticmethod
    def choose(observation):
        # Accept both direct Python views and JSON observations.
        obs = json.loads(json.dumps(observation))
        actions = obs.get("current_playable_actions", [])
        if not actions:
            raise ValueError("No legal actions")
        if len(actions) > 2048:
            raise ValueError("Too many legal actions")
        session, state, mapping = _state(obs)
        phase = _phase(obs)
        reply = BRIDGE.request(
            op="decide",
            session=session,
            state=state,
            phase=phase,
            legal=[_legal(a, mapping, obs["colors"]) for a in actions],
            discard_count=obs["current_discard_count"],
        )
        index = reply.get("action_index")
        if not isinstance(index, int) or not 0 <= index < len(actions):
            raise RuntimeError("Invalid native action index")
        return actions[index]

    def decide(self, game, playable_actions):
        ready, reason = self.availability()
        if not ready:
            raise RuntimeError(reason)
        obs = web_view(game, self.color)
        obs["current_playable_actions"] = [action_to_json(a) for a in playable_actions]
        selected = self.choose(obs)
        return next(a for a in playable_actions if action_to_json(a) == selected)
