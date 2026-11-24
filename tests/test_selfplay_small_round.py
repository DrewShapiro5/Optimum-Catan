import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from agent_apis.neural import selfplay, selfplay_small_round as small


class SmallRoundTests(unittest.TestCase):
    def test_imported_data_preserves_original_budget_and_checksums(self):
        protocol = json.loads(small.PROTOCOL.read_text())
        manifest = json.loads((small.ROOT / protocol['seed_manifest']).read_text())
        records = small.seed_records(manifest)
        self.assertEqual(len(records), 27)
        self.assertEqual(sum(r['dataset']['rows'] for r in records), 5908)
        self.assertEqual(manifest['original_protocol']['simulations'], 512)
        self.assertEqual(protocol['simulations'], 128)
        broken = deepcopy(manifest)
        broken['games'][0]['simulations_per_decision'] = 128
        with self.assertRaises(ValueError):
            small.seed_records(broken)

    def test_new_seeds_continue_after_imported_mixture_slots(self):
        protocol = json.loads(small.PROTOCOL.read_text())
        work = small.SmallRound.__new__(small.SmallRound)
        work.protocol = protocol
        work.known_seeds = set()
        work.offset = 27
        control = selfplay.SelfPlay.__new__(selfplay.SelfPlay)
        control.protocol = protocol
        control.known_seeds = set()
        self.assertEqual(work.game_spec(1, 0, 0), control.game_spec(1, 27, 27))
        self.assertEqual(work.game_spec(1, 3, 1), control.game_spec(1, 30, 28))

    def test_protocol_stops_after_one_bounded_round(self):
        protocol = json.loads(small.PROTOCOL.read_text())
        self.assertEqual(protocol['iterations'], 1)
        self.assertEqual(protocol['completed_games_per_iteration'], 100)
        self.assertEqual(protocol['updates_per_iteration'], 1000)
        self.assertEqual(protocol['evaluation_pairs'], 5)
        self.assertTrue(protocol['friendly_robber'])
        self.assertFalse(protocol['automatic_scale_up'])


if __name__ == '__main__':
    unittest.main()
