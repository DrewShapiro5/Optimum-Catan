from copy import deepcopy
import json
import unittest
from agent_apis.neural.parallel_cohorts import CohortLedger


def assignment(attempt, slot):
    return {'pair': {'seed': attempt}, 'kind': ('current','current','historical','JSETTLERS')[slot % 4]}


class LedgerTests(unittest.TestCase):
    def test_out_of_order_completion_and_cutoff_retry_preserve_mixture(self):
        ledger = CohortLedger(8, 20)
        active = set()
        for _ in range(4):
            active.add(ledger.reserve(active, assignment)['attempt'])
        ledger.finish(2, 'completed'); active.remove(2)
        ledger.finish(0, 'cutoff'); active.remove(0)
        retry = ledger.reserve(active, assignment)
        self.assertEqual((retry['slot'], retry['kind'], retry['attempt']), (0, 'current', 4))
        self.assertNotEqual(retry['pair'], ledger.assignments['0']['pair'])
        active.add(4)
        for attempt in list(active):
            ledger.finish(attempt, 'completed')
        active.clear()
        while len(ledger.completed_slots) < 8:
            a = ledger.reserve(set(), assignment)
            ledger.finish(a['attempt'], 'completed')
        self.assertIsNone(ledger.reserve(set(), assignment))
        self.assertEqual(ledger.completed_slots, set(range(8)))
        kinds = [a['kind'] for a in ledger.assignments.values() if a['status']=='completed']
        self.assertEqual({k:kinds.count(k) for k in set(kinds)}, {'current':4,'historical':2,'JSETTLERS':2})

    def test_resume_uses_durable_pending_assignments_and_original_seeds(self):
        ledger = CohortLedger(4, 8)
        first = deepcopy(ledger.reserve(set(), assignment))
        second = deepcopy(ledger.reserve({0}, assignment))
        ledger.finish(1, 'completed')
        loaded = CohortLedger(4, 8, json.loads(json.dumps(ledger.data)))
        resumed = loaded.reserve(set(), assignment)
        self.assertEqual(resumed, first)
        self.assertEqual(loaded.completed_slots, {second['slot']})
        self.assertEqual(loaded.data['next_attempt'], 2)
        self.assertIsNotNone(loaded.reserve({0}, assignment))

    def test_interrupted_slot_is_reused_without_spending_an_attempt(self):
        ledger = CohortLedger(1, 2)
        a = deepcopy(ledger.reserve(set(), assignment))
        ledger.finish(0, 'interrupted')
        self.assertEqual(ledger.reserve(set(), assignment), a)
        self.assertEqual(ledger.data['next_attempt'], 1)

    def test_duplicate_completion_and_attempt_exhaustion_are_rejected(self):
        ledger = CohortLedger(1, 2)
        ledger.reserve(set(), assignment); ledger.finish(0, 'cutoff')
        ledger.reserve(set(), assignment); ledger.finish(1, 'cutoff')
        self.assertIsNone(ledger.reserve(set(), assignment))
        ledger.finish(1, 'completed')
        with self.assertRaises(ValueError):ledger.finish(0, 'completed')


if __name__=='__main__':unittest.main()
