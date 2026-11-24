import copy
import json
from pathlib import Path
import unittest

from catanatron import Game
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import web_view

from agent_apis.observation import (ResourceLedger, development_knowledge,
                                   encode_observation, feature_names,
                                   from_web_view, normalized_features)
from agent_apis.observation_dataset import audit_game, iter_observed_samples


def initial_view():
    game = Game([SimplePlayer(Color.BLUE), SimplePlayer(Color.RED)], seed=11,
                vps_to_win=15, discard_limit=9)
    actor = game.state.current_color().value
    return from_web_view(web_view(game, Color[actor]), actor)


def set_hands(view, own, other):
    seat = view['colors'].index(view['perspective'])
    for resource, value in zip(('WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE'), own):
        view['player_state'][f'P{seat}_{resource}_IN_HAND'] = value
    view['player_state'][f'P{1-seat}_NUM_RESOURCES_IN_HAND'] = sum(other)
    view['resource_freqdeck'] = [19-a-b for a, b in zip(own, other)]


class ObservationTests(unittest.TestCase):
    def test_conservation_tracks_transfers_and_recovers_skipped_events(self):
        view = initial_view()
        own = view['perspective']
        other = next(c for c in view['colors'] if c != own)
        ledger = ResourceLedger(own)
        ledger.update(view)
        set_hands(view, [2, 1, 0, 0, 0], [1, 3, 0, 0, 0])
        ledger.update(view)  # production/setup receipt
        self.assertEqual(ledger.hands[other], [1, 3, 0, 0, 0])
        bank = list(ledger.bank)
        set_hands(view, [3, 1, 0, 0, 0], [0, 3, 0, 0, 0])
        ledger.update(view)  # redacted theft; bank is unchanged
        self.assertEqual(ledger.bank, bank)
        self.assertEqual(ledger.last_effect[own], [1, 0, 0, 0, 0])
        self.assertEqual(ledger.last_effect[other], [-1, 0, 0, 0, 0])
        set_hands(view, [0, 2, 1, 3, 0], [0, 0, 0, 2, 4])
        ledger.update(view)  # multiple omitted spends/discards/trades/transfers
        fresh = ResourceLedger(own)
        fresh.update(view)
        self.assertEqual(ledger.hands, fresh.hands)
        self.assertEqual(ledger.bank, fresh.bank)

    def test_invalid_counts_and_wrong_rules_are_rejected(self):
        view = initial_view()
        set_hands(view, [1, 0, 0, 0, 0], [0, 0, 0, 0, 0])
        view['resource_freqdeck'][0] = 19
        with self.assertRaisesRegex(ValueError, 'Negative inferred'):
            ResourceLedger(view['perspective']).update(view)
        view = initial_view()
        view['game']['discard_limit'] = 7
        with self.assertRaisesRegex(ValueError, 'Unsupported game rules'):
            encode_observation(view)

    def test_hidden_fields_and_deck_composition_cannot_change_features(self):
        view = initial_view()
        baseline = encode_observation(view)
        dirty = copy.deepcopy(view)
        dirty.update(random_state=[123], winning_color='RED', future_actions=['win'])
        dirty['game']['seed'] = 123
        other_seat = 1 - dirty['colors'].index(dirty['perspective'])
        dirty['player_state'][f'P{other_seat}_KNIGHT_IN_HAND'] = 100
        dirty['player_state'][f'P{other_seat}_WOOD_IN_HAND'] = 100
        dirty['development_listdeck'] = {'KNIGHT': 20, 'VICTORY_POINT': 5}
        safe = from_web_view(dirty, dirty['perspective'])
        self.assertNotIn('random_state', safe)
        self.assertNotIn('seed', safe['game'])
        self.assertNotIn(f'P{other_seat}_KNIGHT_IN_HAND', safe['player_state'])
        self.assertEqual(encode_observation(safe), baseline)
        dirty['development_listdeck'] = {'MONOPOLY': 25}
        self.assertEqual(encode_observation(from_web_view(dirty, dirty['perspective'])), baseline)
        dirty['development_listdeck'] = ['KNIGHT'] * 25
        with self.assertRaisesRegex(ValueError, 'Ordered development deck'):
            from_web_view(dirty, dirty['perspective'])

    def test_card_age_and_public_constraints_are_rebuilt_from_history(self):
        view = initial_view()
        own = view['perspective']
        other = next(c for c in view['colors'] if c != own)
        seat = view['colors'].index(own)
        view['action_records'] = [
            [[own, 'BUY_DEVELOPMENT_CARD', 'KNIGHT'], 'KNIGHT'],
            [[own, 'END_TURN', None], None],
            [[other, 'BUY_DEVELOPMENT_CARD', 'MONOPOLY'], 'MONOPOLY'],
            [[own, 'BUY_DEVELOPMENT_CARD', 'KNIGHT'], 'KNIGHT']]
        view['state_index'] = 4
        view['player_state'][f'P{seat}_KNIGHT_IN_HAND'] = 2
        view['player_state'][f'P{seat}_KNIGHT_OWNED_AT_START'] = True
        view['player_state'][f'P{1-seat}_NUM_DEVELOPMENT_CARDS_IN_HAND'] = 1
        view['development_listdeck'] = 22
        safe = from_web_view(view, own)
        self.assertIsNone(safe['action_records'][2][0][2])
        self.assertIsNone(safe['action_records'][2][1])
        features = dict(zip(feature_names(), encode_observation(safe)))
        self.assertEqual(features['P0_NEW_KNIGHT'], 1)
        self.assertEqual(features['P0_OLDER_KNIGHT'], 1)
        self.assertEqual(features['P1_NEW_DEVS'], 1)
        self.assertEqual(features['P1_OLDER_DEVS'], 0)
        self.assertEqual(development_knowledge(safe)['events'][1][-1], None)
        # No history beyond the declared observation cursor can enter the encoder.
        safe['state_index'] = 3
        with self.assertRaisesRegex(ValueError, 'Incomplete action history'):
            from_web_view(safe, own)

    def test_schema_keeps_exact_dice_as_categories_and_fixes_scales(self):
        raw = encode_observation(initial_view())
        self.assertEqual(len(raw), 681)
        normalized = dict(zip(feature_names(), normalized_features(raw)))
        values = dict(zip(feature_names(), raw))
        for i in range(19):
            self.assertEqual(normalized[f'TILE{i}_DICE_NUMBER'], values[f'TILE{i}_DICE_NUMBER'])
        self.assertEqual(normalized['BANK_WOOD'], 1)
        self.assertEqual(normalized['BANK_DEV_CARDS'], 1)

    def test_complete_view_reconstructs_omitted_decisions_and_retained_view_matches(self):
        directory = Path('data/jsettlers-corpora/downsampled')
        if not (directory / 'manifest.json').exists():
            self.skipTest('Local corpus not installed')
        from tempfile import TemporaryDirectory
        import shutil
        manifest = json.loads((directory / 'manifest.json').read_text())
        entry = manifest['games'][0]
        with TemporaryDirectory() as temp:
            subset = Path(temp)
            shutil.copyfile(directory / entry['file'], subset / entry['file'])
            (subset / 'manifest.json').write_text(json.dumps({'games': [entry]}))
            complete = list(iter_observed_samples(subset, view='complete'))
            retained = list(iter_observed_samples(subset, view='retained'))
            self.assertEqual(len(complete), entry['steps'])
            self.assertEqual(len(retained), entry['saved_steps'])
            self.assertLess(len(retained), len(complete))
            self.assertEqual(retained, [complete[r['step']] for r in retained])
            self.assertEqual(len(list(iter_observed_samples(subset, split='nonexistent'))), 0)
            self.assertTrue(all(r['action_id'] in r['legal_action_ids'] for r in complete))
            self.assertTrue(all('seed' not in r and 'observation' not in r for r in complete))
            result = audit_game(subset / entry['file'])
            self.assertEqual(result['retained_encoder_checks'], len(retained))


if __name__ == '__main__':
    unittest.main()
