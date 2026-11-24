"""Process-shared, checkpoint-separated neural inference for bounded experiments.

Clients send permitted encoded observations, never simulator/private game state.
One owner process runs the models. Each client permits one outstanding request;
responses carry sequence IDs and service failures propagate to the caller.
"""
from collections import Counter
from queue import Empty
import time
import traceback

import numpy as np
import torch
from catanatron.serialization import action_from_json
from agent_apis.experiment import ROOT, file_digest
from agent_apis.observation import encode_observation, require, validate_rules
from agent_apis.stochastic import SearchObservation, action_id
from agent_apis.neural.provider import NeuralCheckpoint


class RemoteCheckpoint:
    def __init__(self, key, worker_id, requests, responses, timeout=90):
        self.key, self.worker_id = key, worker_id
        self.requests, self.responses, self.timeout = requests, responses, timeout
        self.sequence = 0
        self.calls = self.rows = 0
        self.wait_seconds = 0.

    def evaluate_batch(self, views):
        legal, features = [], []
        for view in views:
            validate_rules(view)
            ids = [action_id(action_from_json(a)) for a in view['current_playable_actions']]
            require(ids, 'No legal neural actions')
            legal.append(ids)
            features.append(view.features if isinstance(view, SearchObservation) else encode_observation(view))
        require(features, 'Empty remote inference request')
        sequence = self.sequence
        self.sequence += 1
        started = time.perf_counter()
        self.requests.put((self.worker_id, sequence, self.key, np.asarray(features, dtype=np.float32)))
        try:
            received, error, scores, values = self.responses.get(timeout=self.timeout)
        except Empty as exc:
            raise RuntimeError('Shared inference service timed out') from exc
        require(received == sequence, 'Shared inference response routing mismatch')
        require(error is None, f'Shared inference service failed: {error}')
        require(len(scores) == len(legal) == len(values), 'Shared inference lost rows')
        self.calls += 1
        self.rows += len(views)
        self.wait_seconds += time.perf_counter() - started
        result = []
        for row, ids, value in zip(scores, legal, values):
            selected = row[ids].astype(np.float64)
            probability = float(value)
            require(np.isfinite(selected).all() and np.isfinite(probability) and 0 <= probability <= 1,
                    'Nonfinite shared inference prediction')
            weights = np.exp(selected - selected.max())
            weights /= weights.sum()
            result.append(({int(i): float(p) for i, p in zip(ids, weights)}, probability))
        return result

    def evaluate(self, view):
        return self.evaluate_batch([view])[0]

    def probabilities(self, view):
        if len(view['current_playable_actions']) == 1:
            return {action_id(action_from_json(view['current_playable_actions'][0])): 1.}
        return self.evaluate(view)[0]


def serve(refs, device, requests, responses, control, max_batch_rows=64, wait_seconds=.0005):
    """Drain a bounded window, batch only same-checkpoint rows, return in order."""
    active = []
    try:
        torch.set_num_interop_threads(1)
        models = {}
        for key, ref in refs.items():
            require(file_digest(ROOT / ref['path']) == ref['sha256'], 'Shared checkpoint changed')
            models[key] = NeuralCheckpoint(ROOT / ref['path'], device=device)
        torch.set_num_threads(1)
        control.put(('ready', None))
        metrics = {key: {'requests': 0, 'rows': 0, 'batches': 0, 'forward_seconds': 0.,
                         'queue_window_seconds': 0., 'batch_rows_histogram': Counter()} for key in refs}
        stopping = False
        while not stopping:
            first = requests.get()
            if first is None:
                break
            active = [first]
            rows = len(first[3])
            started = time.perf_counter()
            deadline = started + wait_seconds
            while rows < max_batch_rows:
                try:
                    message = requests.get(timeout=max(0., deadline - time.perf_counter()))
                except Empty:
                    break
                if message is None:
                    stopping = True
                    break
                active.append(message)
                rows += len(message[3])
            window = time.perf_counter() - started
            grouped = {}
            for request in active:
                grouped.setdefault(request[2], []).append(request)
            for key, group in grouped.items():
                # Requests can contain several rows. Chunk the flattened tensor
                # so the configured bound applies to actual model forwards.
                features = np.concatenate([request[3] for request in group], axis=0)
                all_scores, all_values = [], []
                m = metrics[key]
                m['queue_window_seconds'] += window
                m['requests'] += len(group)
                m['rows'] += len(features)
                for offset in range(0, len(features), max_batch_rows):
                    part = features[offset:offset + max_batch_rows]
                    begin = time.perf_counter()
                    raw = torch.from_numpy(part).to(device)
                    with torch.inference_mode():
                        scores, values = models[key].model(raw)
                        all_scores.append(scores.cpu().numpy())
                        all_values.append(values.sigmoid().cpu().numpy())
                    m['forward_seconds'] += time.perf_counter() - begin
                    m['batches'] += 1
                    m['batch_rows_histogram'][len(part)] += 1
                scores, values = np.concatenate(all_scores), np.concatenate(all_values)
                offset = 0
                for worker_id, sequence, _, features in group:
                    end = offset + len(features)
                    responses[worker_id].put((sequence, None, scores[offset:end], values[offset:end]))
                    offset = end
            active = []
        control.put(('metrics', metrics))
    except BaseException:
        error = traceback.format_exc()
        for worker_id, sequence, _, _ in active:
            responses[worker_id].put((sequence, error, None, None))
        control.put(('error', error))
        raise
