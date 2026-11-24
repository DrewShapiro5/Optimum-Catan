"""Search targets, exploration, rule propagation, and training recovery."""
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from catanatron.models.player import Color
from catanatron.serialization import web_view
from agent_apis import benchmark
from agent_apis.development_belief import DevelopmentBelief, VP
from agent_apis.observation import encode_observation, feature_names, from_web_view
from agent_apis.stochastic import observation, world_from_observation
from agent_apis.stochastic_search import ReferenceEvaluator, HeuristicOpponent
from agent_apis.neural import selfplay
from agent_apis.neural.models import build_model
from tests.test_stochastic_search import buy_position, root_view


class SelfPlayTests(unittest.TestCase):
    def test_tempered_acting_preserves_untempered_training_visits(self):
        rows = [{'action_id': 2, 'visits': 1}, {'action_id': 8, 'visits': 3}]
        first, ids, early = selfplay.visit_target(rows, 1.)
        late, _, concentrated = selfplay.visit_target(rows, .25)
        np.testing.assert_array_equal(first, late)
        np.testing.assert_allclose(first[ids], [.25, .75])
        np.testing.assert_allclose(early, [.25, .75])
        np.testing.assert_allclose(concentrated, [1/82, 81/82])
        with self.assertRaises(ValueError):
            selfplay.visit_target([{'action_id': 2, 'visits': 0}], 1.)

    def test_completed_outcomes_use_each_seats_perspective_and_exclude_cutoffs(self):
        pi = np.zeros(332, np.float32)
        pi[2] = 1
        legal = pi.astype(bool)
        rows = [{'features': [0.]*681, 'legal': legal, 'policy': pi, 'actor': c.value}
                for c in (Color.BLUE, Color.RED)]
        self.assertIsNone(selfplay.completed_rows(rows, None))
        win = selfplay.completed_rows(rows, Color.BLUE)
        np.testing.assert_array_equal(win['value'], [1, 0])
        rows[0]['legal'] = np.zeros(332, bool)
        with self.assertRaises(ValueError):
            selfplay.completed_rows(rows, Color.BLUE)

    def test_soft_policy_gradient_respects_distribution_legality_and_forced_moves(self):
        logits = torch.tensor([[.2, .4, 5.], [.7, -.1, 4.]], requires_grad=True)
        legal = torch.tensor([[True, True, False], [True, False, False]])
        target = torch.tensor([[.25, .75, 0.], [1., 0., 0.]])
        pl, vl = selfplay.soft_losses(logits, torch.zeros(2), legal, target, torch.tensor([1., 0.]))
        expected = -(target[0, :2] * torch.log_softmax(logits[0, :2], 0)).sum()
        torch.testing.assert_close(pl, expected)
        (pl + vl).backward()
        self.assertEqual(logits.grad[0, 2], 0)
        torch.testing.assert_close(logits.grad[1], torch.zeros(3))

    def test_soft_targets_follow_all_audited_action_symmetries(self):
        model = build_model('XD-CNN')
        pi = torch.zeros(12, 332)
        pi[:, 3], pi[:, 17] = .2, .8
        sym = torch.arange(12)
        transformed = pi.gather(1, model.inputs.action_inverse[sym])
        for row in range(12):
            self.assertAlmostEqual(float(transformed[row, model.inputs.action_forward[row, 3]]), .2)
            self.assertAlmostEqual(float(transformed[row, model.inputs.action_forward[row, 17]]), .8)
        torch.testing.assert_close(transformed.sum(1), torch.ones(12))

    def test_noise_is_confined_to_current_root_and_refreshes_base_priors(self):
        view = root_view(buy_position()[0])
        tree = selfplay.ExploratorySearch(view, seed=31, session_id='noise')
        result = tree.run(40)
        self.assertEqual(result['metrics']['simulations'], 40)
        priors, _ = tree.evaluator.evaluate(view)
        expected_noise = np.random.default_rng(benchmark.named_seed(31, 'root-noise')).dirichlet(
            np.full(len(priors), 10/len(priors)))
        for (i, prior), noise in zip(sorted(priors.items()), expected_noise):
            self.assertAlmostEqual(tree.nodes[()].edges[i].prior, .75*prior+.25*noise)
        for path, node in tree.nodes.items():
            if path:
                safe = tree._view(tree.positions[path][1], tree.root)
                base, _ = tree.evaluator.evaluate(safe)
                for i, edge in node.edges.items():
                    self.assertAlmostEqual(edge.prior, base[i])
        with patch.object(tree, '_simulate', return_value=.5):
            tree.run(1, noise_fraction=0)
        for i, edge in tree.nodes[()].edges.items():
            self.assertAlmostEqual(edge.prior, priors[i])

    def test_friendly_robber_is_encoded_and_survives_search_reconstruction(self):
        pair = {'board_seed': 11, 'simulator_seed': 12, 'agent_seed': 13}
        friendly = benchmark.make_game(pair)
        standard = benchmark.make_game(pair, friendly_robber=False)
        self.assertTrue(friendly.friendly_robber)
        safe = from_web_view(web_view(friendly, friendly.state.current_color()), friendly.state.current_color().value)
        old = from_web_view(web_view(standard, standard.state.current_color()), standard.state.current_color().value)
        encoded = np.asarray(encode_observation(safe))
        expected = np.asarray(encode_observation(old))
        flag = feature_names().index('RULE_FRIENDLY_ROBBER')
        self.assertEqual(encoded[flag], 1)
        expected[flag] = 1
        np.testing.assert_array_equal(encoded, expected)
        allocation = DevelopmentBelief(safe).sample(random.Random(1))
        world = world_from_observation(safe, allocation)
        self.assertTrue(world.friendly_robber)
        self.assertTrue(world.state.friendly_robber)

    def test_opponent_mixture_and_fresh_disjoint_seed_streams(self):
        work = selfplay.SelfPlay.__new__(selfplay.SelfPlay)
        work.protocol = json.loads(selfplay.PROTOCOL.read_text())
        work.known_seeds = set()
        specs = [work.game_spec(1, i) for i in range(40)]
        self.assertEqual(sum(kind == 'current' for _, kind, _ in specs), 20)
        self.assertEqual(sum(kind == 'historical' for _, kind, _ in specs), 10)
        self.assertEqual(sum(kind == 'JSETTLERS' for _, kind, _ in specs), 10)
        seeds = [p[k] for p, _, _ in specs for k in ('board_seed', 'simulator_seed', 'agent_seed')]
        self.assertEqual(len(seeds), len(set(seeds)))
        # A cutoff consumes a seed attempt, without consuming its mixture slot.
        first = work.game_spec(1, 0, 0)
        retry = work.game_spec(1, 1, 0)
        self.assertEqual(first[1:], retry[1:])
        self.assertNotEqual(first[0], retry[0])
        work.known_seeds.add(seeds[0])
        with self.assertRaises(ValueError):
            work.game_spec(1, 0)

    def test_exploration_action_is_used_for_next_reuse_match(self):
        view = root_view(buy_position()[0])
        protocol = json.loads(selfplay.PROTOCOL.read_text())
        protocol['simulations'] = 8
        seat = selfplay.SearchSeat(ReferenceEvaluator(), HeuristicOpponent(), 7, protocol, 'session')
        action, target, metrics = seat.choose(view)
        self.assertEqual(seat.tree.selected_action, selfplay.action_id(action))
        self.assertEqual(metrics['simulations'], 8)
        self.assertAlmostEqual(float(target.sum()), 1)

    def test_visible_friendly_robber_does_not_disclose_hidden_vp(self):
        fixture = selfplay.ROOT / 'tests/fixtures/friendly-robber-hidden-vp.json'
        view = json.loads(fixture.read_text())
        from catanatron.serialization import map_from_json
        tiles = map_from_json(view['map']).land_tiles
        occupied = set(view['buildings_by_color']['BLUE']['SETTLEMENT'])
        view['current_playable_actions'] = [a for a in view['current_playable_actions']
            if not occupied.intersection(tiles[tuple(a[2][0])].nodes.values())]
        belief = DevelopmentBelief(view)
        self.assertEqual(belief.minimum_vp, 0)
        self.assertFalse(belief.friendly_minimums)
        sampled=set()
        for seed in range(100):
            allocation=belief.sample(random.Random(seed))
            sampled.add(allocation['opponent_old'][VP])
            world=world_from_observation(view,allocation)
            self.assertTrue(all(a.value[1] is None for a in world.playable_actions))
        self.assertEqual(sampled,{0,1})


if __name__ == '__main__':
    unittest.main()
