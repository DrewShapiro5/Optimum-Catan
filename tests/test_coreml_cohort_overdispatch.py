"""Tests for tail over-dispatch in coreml_cohort_pipeline.collect_neural_block."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agent_apis.neural import coreml_cohort_pipeline as pipeline


class _FakeRecorder:
    def __init__(self, game, record, simulations, setup_simulations=None):
        self.record = dict(record)

    def append(self, events):
        self.record.setdefault('events', []).extend(events)

    def finish(self, msg):
        return {
            **self.record,
            'status': msg.get('status', 'completed'),
            'winner': msg.get('winner_color', 'BLUE'),
            'turns': msg.get('turns', 80),
        }


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.dispatched = []
        self.stdin_closed = False
        self._stdout_lines = [json.dumps({'type': 'ready'}) + '\n']
        self.stdin = self
        self.stdout = self

    def write(self, text):
        req = json.loads(text)
        self.dispatched.append(req)
        gid = req['game_id']
        # Simulate game-00000 stalling until cutoff, while others complete normally.
        if gid.endswith('-00000'):
            self._stdout_lines.append(
                json.dumps({'type': 'done', 'game_id': gid, 'status': 'cutoff', 'winner_color': None, 'turns': 1000}) + '\n'
            )
        else:
            self._stdout_lines.append(
                json.dumps({'type': 'done', 'game_id': gid, 'status': 'completed', 'winner_color': 'BLUE', 'turns': 90}) + '\n'
            )

    def flush(self):
        pass

    def close(self):
        self.stdin_closed = True

    def readline(self):
        return self._stdout_lines.pop(0)

    def __iter__(self):
        while self._stdout_lines:
            yield self._stdout_lines.pop(0)
        yield json.dumps({'type': 'pool_metrics', 'metrics': {'batches': 1, 'rows': 1}}) + '\n'

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


class OverDispatchTests(unittest.TestCase):
    def test_collect_neural_block_over_dispatches_and_stops_at_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            block_dir = root / 'games'
            onnx = root / 'model.onnx'
            onnx.write_bytes(b'onnx')
            saved = []
            popens = []

            def fake_popen(cmd, **kwargs):
                p = _FakePopen(cmd, **kwargs)
                popens.append(p)
                return p

            with (
                patch.object(pipeline, 'NATIVE', root),
                patch.object(pipeline.subprocess, 'Popen', side_effect=fake_popen),
                patch.object(pipeline.benchmark, 'make_game', return_value=type('G', (), {'state': type('S', (), {'random': type('R', (), {'getstate': lambda self: (0, [1, 2])})(), 'colors': []})()})()),
                patch.object(pipeline, 'job_meta', return_value={'initial': {}}),
                patch.object(pipeline, 'file_digest', return_value='deadbeef'),
                patch.object(pipeline, 'JournalRecorder', _FakeRecorder),
                patch.object(pipeline.compact, 'save_journal', side_effect=lambda p, r: (saved.append(p.name), p.write_bytes(b'gz'))),
                patch.object(pipeline.selfplay, 'reference', side_effect=lambda p: {'path': str(p), 'sha256': 'ok'}),
            ):
                (root / 'target').mkdir()
                completed, _ = pipeline.collect_neural_block(
                    'cnn_vs_cnn',
                    block_dir,
                    target_games=10,
                    seed=123,
                    binary=root / 'worker',
                    model_a_ref={'sha256': 'a'},
                    model_b_ref={'sha256': 'b'},
                    onnx_a=onnx,
                    onnx_b=onnx,
                    frozen_ref={'sha256': 'f'},
                    frozen_onnx=onnx,
                    protocol={'simulations': 128},
                    workers=256,
                    coreml_batch=128,
                    gpu_streams=1,
                    mcts_lanes=1,
                    virtual_loss=1.0,
                    wait_us=50,
                    tail_over_dispatch=32,
                )

            self.assertEqual(len(completed), 10)
            self.assertEqual(len(saved), 10)
            self.assertEqual(len(popens), 1)
            # Initial wave launches target_games + tail_over_dispatch (10 + 32 = 42) games.
            self.assertIn('42', popens[0].cmd)
            self.assertGreaterEqual(len(popens[0].dispatched), 42)
            self.assertTrue(popens[0].stdin_closed)

    def test_resampling_memory_data_reuses_derived_cache_and_parallel_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = []
            for i in range(7):
                p = root / f'game-{i:05d}.json.gz'
                p.write_bytes(b'gz')
                refs.append({'path': p.name, 'sha256': f'sha-{i}'})

            derive_calls = []

            def fake_derive(rec):
                pid = rec['pair']['pair_id']
                derive_calls.append(pid)
                arr = {
                    'features': np.zeros((4, 681), np.float32),
                    'legal': np.ones((4, 332), bool),
                    'policy': np.full((4, 332), 1.0 / 332, np.float32),
                    'value': np.array([1.0, 0.0, 1.0, 0.0], np.float32),
                }
                return arr, 1

            class _FakePool:
                def __init__(self, max_workers=None, mp_context=None, initializer=None):
                    self.max_workers = max_workers

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def map(self, fn, items, chunksize=1):
                    return [fn(x) for x in items]

            def fake_gzip_open(path, mode='rt'):
                idx = int(Path(path).stem.split('-')[1].split('.')[0])
                return io.StringIO(json.dumps({'pair': {'pair_id': f'p-{idx}'}, 'events': []}))

            cache = {}
            with (
                patch.object(pipeline, 'ROOT', root),
                patch.object(pipeline, 'file_digest', side_effect=lambda p: f"sha-{int(Path(p).stem.split('-')[1].split('.')[0])}"),
                patch.object(pipeline.gzip, 'open', side_effect=fake_gzip_open),
                patch.object(pipeline, '_derive_all_game_positions', side_effect=fake_derive),
                patch.object(pipeline.w4, 'check_arrays', return_value=None),
                patch.object(pipeline, 'ProcessPoolExecutor', _FakePool),
            ):
                d1 = pipeline.ResamplingMemoryData(refs[:5], base_seed=1, limit=2, derived_cache=cache)
                self.assertEqual(len(derive_calls), 5)
                self.assertEqual(set(cache), {f'sha-{i}' for i in range(5)})

                derive_calls.clear()
                d2 = pipeline.ResamplingMemoryData(refs, base_seed=2, limit=2, derived_cache=cache)
                # First 5 came from RAM cache; only the 2 new games were derived.
                self.assertEqual(derive_calls, ['p-5', 'p-6'])
                self.assertEqual(d1.summary['pool_rows'], 20)
                self.assertEqual(d2.summary['pool_rows'], 28)


if __name__ == '__main__':
    unittest.main()
