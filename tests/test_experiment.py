import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_apis.experiment import (EXPERIMENTS, corpus_snapshot, digest,
                                   file_digest, rng_streams, stream_seeds,
                                   validate_record)


class ExperimentTests(unittest.TestCase):
    def test_rng_streams_are_repeatable_isolated_and_do_not_reseed_global(self):
        state = random.getstate()
        first, second = rng_streams(11), rng_streams(11)
        self.assertEqual(len(set(stream_seeds(11).values())), 5)
        first['search'].random()
        self.assertEqual(first['training'].random(), second['training'].random())
        self.assertEqual(random.getstate(), state)
        self.assertNotEqual(stream_seeds(11), stream_seeds(23))
        with self.assertRaises(ValueError):
            stream_seeds(-1)

    def test_config_hash_ignores_key_order_and_detects_changes(self):
        self.assertEqual(digest({'a': 1, 'b': 2}), digest({'b': 2, 'a': 1}))
        self.assertNotEqual(digest({'seed': 11}), digest({'seed': 23}))

    def test_corrupt_dataset_is_rejected_before_run(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            shard = root / 'game.gz'
            shard.write_bytes(b'fixture')
            manifest = {'schema_version': 2, 'feature_names': ['x'] * 633,
                        'action_vocabulary': list(range(332)), 'num_games': 1,
                        'num_samples': 1, 'provider': 'fixture', 'strategy': 'fixture',
                        'jsettlers_revision': 'fixture', 'games': [
                            {'file': 'game.gz', 'game_id': 'a', 'seed': 1,
                             'split': 'train', 'saved_steps': 1, 'sha256': file_digest(shard)}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            config = json.loads((EXPERIMENTS / 'phase1.json').read_text())
            config['data']['corpora'] = ['.']
            with patch('agent_apis.experiment.ROOT', root):
                self.assertEqual(corpus_snapshot(config)[0]['num_games'], 1)
                shard.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    corpus_snapshot(config)

    def test_schema_requires_provenance(self):
        from jsonschema import ValidationError
        with self.assertRaises(ValidationError):
            validate_record({'schema_version': 1})


if __name__ == '__main__':
    unittest.main()
