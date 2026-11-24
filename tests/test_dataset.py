"""Real-JSettlers trajectory, storage-budget, and training-boundary checks."""

from pathlib import Path
import tempfile
import unittest

from agent_apis.dataset import (
    MAX_BYTES, _generate_game, generate, iter_samples, verify,
)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = _generate_game(0, 100, 1000, 10000, MAX_BYTES)
        if "error" in cls.trajectory:
            raise RuntimeError(cls.trajectory)

    def test_full_game_replay_and_training_boundary(self):
        entry = {k: v for k, v in self.trajectory.items() if k != "payload"}
        from agent_apis.dataset import action_vocabulary, encode
        from catanatron.features import get_feature_ordering
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / entry["file"]).write_bytes(self.trajectory["payload"])
            manifest = {
                "games": [entry], "num_samples": entry["steps"],
                "limits": {"bytes": MAX_BYTES, "games": 1},
                "feature_names": get_feature_ordering(2, "BASE"),
                "action_vocabulary": action_vocabulary(),
            }
            (directory / "manifest.json").write_bytes(encode(manifest))
            self.assertTrue(verify(directory)["verified"])
            samples = list(iter_samples(directory, entry["split"]))
            self.assertEqual(len(samples), entry["steps"])
            self.assertEqual({s["value_target"] for s in samples}, {-1, 1})
            self.assertTrue(all(s["action_id"] in s["legal_action_ids"] for s in samples))
            for key in ("record", "initial_state", "final_state", "observation", "seed"):
                self.assertNotIn(key, samples[0])
            self.assertEqual(list(iter_samples(directory, "nonexistent")), [])
            # Hash checking detects any corrupted/edited shard before replay.
            with (directory / entry["file"]).open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(AssertionError):
                verify(directory)

    def test_budget_rejects_whole_game_and_never_exceeds_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = generate(directory, games=1, max_bytes=100_000, workers=1, seed=100)
            self.assertEqual(manifest["num_games"], 0)
            self.assertEqual(len(manifest["failures"]), 1)
            self.assertLessEqual(sum(p.stat().st_size for p in directory.iterdir()), 100_000)
            with self.assertRaises(ValueError):
                generate(directory, games=1)

    def test_aggregate_byte_limit_stops_between_complete_games(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = generate(directory, games=6, max_bytes=6_000_000,
                                workers=1, seed=100)
            self.assertEqual(manifest["stop_reason"], "byte_limit")
            self.assertGreater(manifest["num_games"], 0)
            self.assertLess(manifest["num_games"], 6)
            report = verify(directory)
            self.assertLessEqual(report["stored_bytes"], 6_000_000)
            self.assertLessEqual(report["uncompressed_bytes"], 6_000_000)

    def test_hard_limits_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            for games, byte_limit in ((1001, MAX_BYTES), (1, MAX_BYTES + 1), (0, 1000)):
                with self.assertRaises(ValueError):
                    generate(temp, games=games, max_bytes=byte_limit)
        result = _generate_game(0, 100, 1000, 1, MAX_BYTES)
        self.assertEqual(result["error"], "turn_or_action_limit")
        self.assertNotIn("payload", result)


if __name__ == "__main__":
    unittest.main()
