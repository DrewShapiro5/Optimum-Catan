import queue
import unittest
from unittest.mock import patch
import numpy as np
from agent_apis.neural.shared_inference import RemoteCheckpoint
from agent_apis.stochastic import action_id
from catanatron.serialization import action_from_json
from tests.test_stochastic_search import hidden_card_view


class SharedInferenceTests(unittest.TestCase):
    def client(self, response):
        requests, responses = queue.Queue(), queue.Queue()
        responses.put(response)
        return RemoteCheckpoint('candidate', 3, requests, responses, timeout=.01), requests

    def test_routes_encoded_rows_and_normalizes_only_legal_actions(self):
        view = hidden_card_view([])
        scores = np.zeros((2, 332), dtype=np.float32)
        legal = [action_id(action_from_json(a)) for a in view['current_playable_actions']]
        scores[:, legal[0]] = 2.
        client, requests = self.client((0, None, scores, np.array([.2, .8], dtype=np.float32)))
        result = client.evaluate_batch([view, view])
        worker, sequence, model, features = requests.get_nowait()
        self.assertEqual((worker, sequence, model), (3, 0, 'candidate'))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(features), 2)
        self.assertEqual(set(result[0][0]), set(legal))
        self.assertAlmostEqual(sum(result[0][0].values()), 1.)
        self.assertAlmostEqual(result[1][1], .8, places=6)
        self.assertEqual((client.calls, client.rows), (1, 2))

    def test_rejects_misrouted_and_failed_responses(self):
        for response in ((9, None, [], []), (0, 'model failed', None, None)):
            client, _ = self.client(response)
            with self.assertRaises(ValueError):
                client.evaluate(hidden_card_view([]))

    def test_rejects_missing_rows_and_nonfinite_predictions(self):
        for scores, values in ((np.zeros((0, 332)), np.zeros(0)),
                               (np.zeros((1, 332)), np.array([np.nan]))):
            client, _ = self.client((0, None, scores, values))
            with self.assertRaises(ValueError):
                client.evaluate(hidden_card_view([]))

    def test_timeout_is_bounded(self):
        client = RemoteCheckpoint('candidate', 0, queue.Queue(), queue.Queue(), timeout=.001)
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            client.evaluate(hidden_card_view([]))


if __name__ == '__main__':
    unittest.main()
