"""Verify the real browser API supports two human turns and saved games."""

from pathlib import Path
import tempfile
import unittest

from play_catan import DIST, make_app, new_game


class LauncherTests(unittest.TestCase):
    def test_four_humans_complete_initial_placement(self):
        app = make_app("sqlite:///:memory:")
        client = app.test_client()
        response = client.post("/api/games", json={"players": ["HUMAN"] * 4})
        self.assertEqual(response.status_code, 200)
        game_id = response.get_json()["game_id"]
        endpoint = f"/api/games/{game_id}"
        state = client.get(endpoint + "/states/latest").get_json()
        self.assertEqual(set(state["colors"]), {"RED", "BLUE", "ORANGE", "WHITE"})
        self.assertEqual(state["bot_colors"], [])
        actions_per_color = dict.fromkeys(state["colors"], 0)
        for _ in range(16):
            self.assertTrue(state["is_initial_build_phase"])
            actions_per_color[state["current_color"]] += 1
            response = client.post(endpoint + "/actions", json=state["current_playable_actions"][0])
            self.assertEqual(response.status_code, 200)
            state = response.get_json()
        self.assertEqual(set(actions_per_color.values()), {4})
        self.assertFalse(state["is_initial_build_phase"])

    def test_two_humans_can_place_and_take_turns(self):
        app = make_app("sqlite:///:memory:")
        client = app.test_client()
        game_id = new_game(app)
        endpoint = f"/api/games/{game_id}"
        state = client.get(endpoint + "/states/latest").get_json()
        self.assertEqual(set(state["colors"]), {"RED", "BLUE"})
        self.assertEqual(state["bot_colors"], [])
        self.assertEqual(state["game"]["vps_to_win"], 15)
        self.assertEqual(state["game"]["discard_limit"], 9)
        self.assertTrue(state["game"]["friendly_robber"])
        initial_colors = set()
        for _ in range(8):
            self.assertTrue(state["is_initial_build_phase"])
            initial_colors.add(state["current_color"])
            response = client.post(endpoint + "/actions", json=state["current_playable_actions"][0])
            self.assertEqual(response.status_code, 200)
            state = response.get_json()
        self.assertEqual(initial_colors, {"RED", "BLUE"})
        self.assertFalse(state["is_initial_build_phase"])
        first = state["current_color"]
        for _ in range(20):
            actions = state["current_playable_actions"]
            action = next((a for a in actions if a[1] == "END_TURN"), actions[0])
            response = client.post(endpoint + "/actions", json=action)
            self.assertEqual(response.status_code, 200)
            state = response.get_json()
            if action[1] == "END_TURN":
                break
        else:
            self.fail("Could not end the first human's turn")
        self.assertNotEqual(state["current_color"], first)
        self.assertTrue(any(action[1] == "ROLL" for action in state["current_playable_actions"]))

    def test_game_survives_app_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            uri = f"sqlite:///{Path(directory) / 'games.sqlite3'}"
            game_id = new_game(make_app(uri))
            response = make_app(uri).test_client().get(f"/api/games/{game_id}/states/latest")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["bot_colors"], [])

    def test_browser_routes_and_assets(self):
        client = make_app("sqlite:///:memory:").test_client()
        for route in ("/", "/games/example"):
            response = client.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'<div id="root">', response.data)
            response.close()
        asset = next((DIST / "assets").glob("*.js"))
        response = client.get(f"/assets/{asset.name}")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(client.get("/api/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
