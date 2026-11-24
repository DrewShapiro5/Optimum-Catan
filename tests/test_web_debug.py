import unittest
from catanatron.web.models import get_game_state, upsert_game_state
from play_catan import make_app
from agent_apis import debug_telemetry


class WebDebugEndpointsTests(unittest.TestCase):
    def test_prepare_holds_mcts_decision_and_step_executes_it(self):
        app = make_app("sqlite:///:memory:")
        client = app.test_client()
        resp = client.post(
            "/api/games",
            json={
                "players": ["HUMAN", "STOCHASTIC_REFERENCE"],
                "vps_to_win": 15,
                "discard_limit": 9,
                "friendly_robber": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        gid = resp.json["game_id"]

        # Advance until it's the bot's turn (valid unmodified history).
        with app.app_context():
            g = get_game_state(gid)
            bot = next(p.color for p in g.state.players if p.is_bot)
            while g.state.current_color() != bot:
                g.execute(g.playable_actions[0])
            before_index = len(g.state.action_records)
            upsert_game_state(g)

        # Prepare should run MCTS, populate policy distribution & dev cards, and NOT advance state.
        prep = client.post(f"/api/games/{gid}/debug/prepare")
        self.assertEqual(prep.status_code, 200)
        snap = prep.json
        self.assertEqual(snap["status"], "awaiting_step")
        self.assertEqual(snap["state_index"], before_index)
        self.assertEqual(
            set(snap["dev_cards"].keys()),
            {"KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING", "VICTORY_POINT"},
        )
        self.assertGreater(len(snap["actions"]), 1)
        self.assertAlmostEqual(
            sum(a["policy"] for a in snap["actions"]), 1.0, places=5
        )
        self.assertIsNotNone(snap["chosen_action"])
        self.assertGreater(snap["simulations_done"], 0)

        # Confirm game state has NOT advanced yet.
        latest = client.get(f"/api/games/{gid}/states/latest").json
        self.assertEqual(latest["state_index"], before_index)

        # GET /debug/state returns the held awaiting_step snapshot.
        polled = client.get(f"/api/games/{gid}/debug/state").json
        self.assertEqual(polled["status"], "awaiting_step")
        self.assertEqual(polled["chosen_action"], snap["chosen_action"])

        # POST /debug/step commits the exact prepared action and advances state_index by 1.
        stepped = client.post(f"/api/games/{gid}/debug/step")
        self.assertEqual(stepped.status_code, 200)
        after = stepped.json
        self.assertEqual(after["state_index"], before_index + 1)
        self.assertEqual(after["action_records"][-1][0], snap["chosen_action"])

        # Pending action is cleared and status is completed.
        after_debug = client.get(f"/api/games/{gid}/debug/state").json
        self.assertIn(after_debug["status"], ("completed", "idle"))
        self.assertIsNone(debug_telemetry.get_pending_action(gid, before_index))

    def test_snapshot_reports_bot_dev_cards(self):
        app = make_app("sqlite:///:memory:")
        client = app.test_client()
        resp = client.post(
            "/api/games",
            json={
                "players": ["HUMAN", "STOCHASTIC_REFERENCE"],
                "vps_to_win": 15,
                "discard_limit": 9,
                "friendly_robber": True,
            },
        )
        gid = resp.json["game_id"]
        with app.app_context():
            g = get_game_state(gid)
            bot = next(p.color for p in g.state.players if p.is_bot)
            bot_seat = g.state.colors.index(bot)
            while g.state.current_color() != bot:
                g.execute(g.playable_actions[0])
            g.state.player_state[f"P{bot_seat}_KNIGHT_IN_HAND"] = 2
            g.state.player_state[f"P{bot_seat}_VICTORY_POINT_IN_HAND"] = 1
            snap = debug_telemetry.snapshot_from_game(g)
        self.assertEqual(snap["dev_cards"]["KNIGHT"], 2)
        self.assertEqual(snap["dev_cards"]["VICTORY_POINT"], 1)
        self.assertEqual(snap["dev_cards"]["MONOPOLY"], 0)

    def test_update_debug_simulations_persists_on_searchable_bot(self):
        app = make_app("sqlite:///:memory:")
        client = app.test_client()
        resp = client.post(
            "/api/games",
            json={
                "players": [
                    "HUMAN",
                    {"key": "CNN_LATEST_SEARCH", "params": {"simulations": 32}},
                ],
                "vps_to_win": 15,
                "discard_limit": 9,
                "friendly_robber": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        gid = resp.json["game_id"]

        initial = client.get(f"/api/games/{gid}/debug/state").json
        self.assertTrue(initial["can_set_simulations"])
        self.assertEqual(initial["simulations"], 32)

        updated = client.post(
            f"/api/games/{gid}/debug/simulations",
            json={"simulations": 256},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json["simulations"], 256)

        with app.app_context():
            g = get_game_state(gid)
            bot = next(p for p in g.state.players if p.is_bot)
            self.assertEqual(bot.params.simulations, 256)


if __name__ == "__main__":
    unittest.main()
