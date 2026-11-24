"""Integration checks against the compiled upstream Java robot, not a mock."""

from collections import Counter
import unittest

from agent_apis import choose_action, list_agents, register_agents, synchronize
from agent_apis.jsettlers.player import BRIDGE, RESOURCES, _mapping
from catanatron import Game
from catanatron.registry import REGISTRY
from catanatron.serialization import web_view
from play_catan import make_app


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_agents()

    def game(self, count=2, seed=14):
        return Game(
            REGISTRY.build_all(["JSETTLERS"] * count),
            seed=seed,
            vps_to_win=15 if count == 2 else 10,
            discard_limit=9 if count == 2 else 7,
        )

    def advance(self, game):
        action = game.state.current_player().decide(game, game.playable_actions)
        self.assertIn(action, game.playable_actions)
        game.execute(action)
        return action

    def assert_native_state(self, game, color):
        import json

        obs = web_view(game, color)
        state = synchronize("JSETTLERS", obs, color.value)
        mapping = _mapping(json.dumps(obs["map"], sort_keys=True))
        self.assertEqual(state["engine"], "SOCRobotBrain")
        self.assertEqual(state["strategy"], "SMART")
        self.assertEqual(state["state_index"], len(game.state.action_records))
        self.assertEqual(state["current_seat"], game.state.current_turn_index)
        self.assertEqual(state["vps_to_win"], game.vps_to_win)
        self.assertEqual(
            state["robber"], mapping["coordinates"][tuple(obs["robber_coordinate"])]
        )
        self.assertEqual(
            {tuple(h) for h in state["hexes"]},
            {(h, r, max(0, n)) for h, r, n in mapping["layout"]},
        )
        for seat, player in enumerate(state["players"]):
            color_at_seat = game.state.colors[seat]
            hand = [game.state.player_state[f"P{seat}_{r}_IN_HAND"] for r in RESOURCES]
            self.assertEqual(sum(player["hand"]), sum(hand))
            # Publicly known cards are conservative; hidden losses become UNKNOWN.
            self.assertTrue(all(0 <= player["hand"][r] <= hand[r] for r in range(5)))
            if color_at_seat == color:
                self.assertEqual(player["hand"], hand + [0])
            buildings = game.state.buildings_by_color[color_at_seat]
            self.assertEqual(player["settlements"], len(buildings["SETTLEMENT"]))
            self.assertEqual(player["cities"], len(buildings["CITY"]))
            self.assertEqual(player["roads"], len(buildings["ROAD"]))
            ports = game.state.board.get_player_port_resources(color_at_seat)
            self.assertEqual(player["ports"], [r in ports for r in [None] + RESOURCES])
        return state

    def test_java_library_and_catalog(self):
        entry = list_agents()[0]
        self.assertTrue(entry["available"])
        self.assertEqual(entry["name"], "JSettlers")
        self.assertIn("sync", entry["capabilities"])

    def test_http_discovery_sync_and_idempotent_decision(self):
        app = make_app("sqlite:///:memory:")
        game = self.game()
        observation = web_view(game, game.state.current_color())
        with app.test_client() as client:
            self.assertTrue(client.get("/api/agents").json["agents"][0]["available"])
            synced = client.post(
                "/api/agents/JSETTLERS/sync", json={"observation": observation}
            )
            self.assertEqual(synced.status_code, 200)
            self.assertEqual(synced.json["engine"], "SOCRobotBrain")
            reply = client.post(
                "/api/agents/JSETTLERS/decide", json={"observation": observation}
            )
            self.assertEqual(reply.status_code, 200)
            self.assertEqual(
                reply.json["action"], choose_action("JSETTLERS", observation)
            )
            self.assertIn(reply.json["action"], observation["current_playable_actions"])
            self.assertEqual(
                client.post(
                    "/api/agents/missing/decide", json={"observation": observation}
                ).status_code,
                400,
            )
            self.assertEqual(
                client.post("/api/agents/JSETTLERS/sync", json={}).status_code, 400
            )

    def test_web_updates_parallel_state_after_human_moves(self):
        app = make_app("sqlite:///:memory:")
        with app.test_client() as client:
            gid = client.post(
                "/api/games",
                json={
                    "players": ["HUMAN", "JSETTLERS"],
                    "vps_to_win": 15,
                    "discard_limit": 9,
                },
            ).json["game_id"]
            for i in range(30):
                obs = client.get(f"/api/games/{gid}/states/latest").json
                # Empty requests let bots act; humans explicitly submit legal moves.
                action = (
                    {}
                    if obs["current_color"] in obs["bot_colors"]
                    else obs["current_playable_actions"][0]
                )
                reply = client.post(f"/api/games/{gid}/actions", json=action)
                self.assertEqual(reply.status_code, 200)
                self.assertNotEqual(reply.headers.get("X-Agent-Sync"), "failed")
                seat_color = reply.json["bot_colors"][0]
                # Status-only Java request proves the API hook already applied it.
                native = BRIDGE.request(op="status", session=f"{gid}:{seat_color}")
                self.assertEqual(native["state_index"], i + 1)

    def test_native_bank_trade_and_building_regression(self):
        game = self.game()
        actions = [self.advance(game).action_type.value for _ in range(60)]
        self.assertIn("MARITIME_TRADE", actions)
        self.assertTrue(
            {"BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "BUY_DEVELOPMENT_CARD"}
            & set(actions[9:])
        )
        for color in game.state.colors:
            self.assert_native_state(game, color)

    def test_native_empty_bank_trade_uses_upstream_rejection_handler(self):
        import json
        from pathlib import Path
        observation = json.loads(Path('tests/fixtures/jsettlers-empty-bank.json').read_text())
        BRIDGE.close()
        selected = choose_action('JSETTLERS', observation)
        self.assertIn(selected, observation['current_playable_actions'])
        status = BRIDGE.request(op='status', session=f"{observation['game']['id']}:{observation['current_color']}")
        self.assertEqual(status['engine'], 'SOCRobotBrain')
        self.assertEqual(status['strategy'], 'SMART')
        self.assertEqual(status['translation_rejections'], 9)
        self.assertEqual(selected[1], 'END_TURN')
        self.assertEqual(choose_action('JSETTLERS', observation), selected)

    def test_four_player_second_initial_road(self):
        # This layout exposed an early clear of native settlement potentials.
        game = self.game(count=4, seed=4)
        for _ in range(17):
            self.advance(game)
        self.assertFalse(game.state.is_initial_build_phase)
        for color in game.state.colors:
            self.assert_native_state(game, color)

    def test_restart_rehydrates_initial_placements_and_free_roads(self):
        game = self.game()
        for _ in range(8):
            BRIDGE.close()
            self.advance(game)
        recovered_free_roads = 0
        for _ in range(1200):
            if game.winning_color():
                break
            if game.state.is_road_building:
                BRIDGE.close()
                recovered_free_roads += 1
            self.advance(game)
        self.assertGreater(recovered_free_roads, 0)
        for color in game.state.colors:
            self.assert_native_state(game, color)

    def test_full_games_and_state_convergence(self):
        kinds = Counter()
        for count in (2, 4):
            with self.subTest(players=count):
                game = self.game(count)
                for i in range(3000):
                    if game.winning_color():
                        break
                    action = self.advance(game)
                    kinds[action.action_type.value] += 1
                    if i % 45 == 0:
                        for color in game.state.colors:
                            self.assert_native_state(game, color)
                self.assertIsNotNone(game.winning_color())
                # Rebuilding from the complete public history must converge too.
                BRIDGE.close()
                for color in game.state.colors:
                    self.assert_native_state(game, color)
        self.assertTrue(
            {
                "BUILD_CITY",
                "MARITIME_TRADE",
                "PLAY_KNIGHT_CARD",
                "PLAY_YEAR_OF_PLENTY",
                "PLAY_ROAD_BUILDING",
                "DISCARD_RESOURCE",
            }
            <= kinds.keys()
        )


if __name__ == "__main__":
    unittest.main()
