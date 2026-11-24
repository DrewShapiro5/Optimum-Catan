# Shared GPU inference queue versus CPU — November 18, 2026

**The best tested GPU configuration improved end-to-end diagnostic throughput by 8.8% over the twelve-worker CPU baseline.** It requires 32 concurrent game workers, a shared MPS owner for the XD-CNN, a local CPU opponent MLP in every game worker, and a 0.5 ms batching window. At the same twelve-worker count the GPU hybrid was 24% slower. No training controller default or playing checkpoint was changed; training remains paused.

## Fresh confirmation after GPU concurrency tuning

Two alternating passes, each 96 matched continuations / 384 searched decisions per configuration, using the same checkpoint, seeds, chance streams, 128-simulation search budget and workload. Aggregate rates are total decisions divided by total measured seconds.

| Inference configuration | Game workers | Pass 1 moves/s | Pass 2 moves/s | Aggregate moves/s | Peak summed process RSS (GB) |
|---|---:|---:|---:|---:|---:|
| Local CPU, both models | 12 | 9.13 | 8.95 | 9.04 | 6.63 |
| Shared GPU CNN + local CPU MLP | 32 | 9.92 | 9.76 | 9.84 | 14.33 |

The gain appeared in both orders (about 9% each). This is a bounded throughput result, not a statistical guarantee for complete games or a long-duration thermal/power comparison. RSS includes game workers, the shared model service and native Java children. Shared pages can be counted repeatedly; this is not physical memory consumption or a complete accounting of Metal driver allocations.

## Fixed twelve-worker comparison

Three alternating longer-workload passes.

| Configuration | Aggregate moves/s | Per-pass moves/s |
|---|---:|---|
| local-cpu | 8.83 | 8.41, 9.08, 9.03 |
| gpu-cnn-local-mlp-0.5ms | 6.73 | 6.75, 6.68, 6.75 |

The same-count GPU hybrid is slower: enough game concurrency is necessary to supply useful batches. In the first twelve-worker confirmation, the CNN averaged 4.97 positions per forward; in the first 32-worker confirmation it averaged 13.73. Requests from independently progressing trees do not arrive as ready-made batches of 32 or 64.

## Screening results

The initial screen used 24 continuations (96 searched decisions) per configuration. These shorter rates include a larger end-of-case scheduling tail and should be compared within this table. The shared CPU control centralizes inference on one CPU thread; it measures the combination of centralization, queueing and batching, not isolated IPC cost.

| Twelve-worker configuration | Batch collection window | Moves/s |
|---|---:|---:|
| local-cpu | 0 ms | 7.64 |
| gpu-both-0.5ms | 0.5 ms | 2.16 |
| gpu-cnn-local-mlp-0.5ms | 0.5 ms | 5.81 |
| shared-cpu-0.5ms | 0.5 ms | 2.96 |
| gpu-both-2ms | 2 ms | 2.02 |
| gpu-cnn-local-mlp-2ms | 2 ms | 5.29 |
| gpu-both-0ms | 0 ms | 2.64 |

Higher-concurrency screening used 96 continuations per configuration. The first three rows use a 0.5 ms window and local CPU MLP; the last removes the explicit collection delay.

| Hybrid GPU game workers | Collection window | Moves/s |
|---:|---:|---:|
| 16 | 0.5 ms | 7.58 |
| 24 | 0.5 ms | 9.06 |
| 32 | 0.5 ms | 9.92 |
| 24 | 0 ms | 9.38 |

The best screened configuration was then confirmed against a fresh CPU baseline. The GPU selection was exploratory; the fresh confirmation workloads repeat the same 24 source continuations and are not independent held-out games. Thirty-two is the best worker count tested here, not a proven global optimum.

## Queue implementation

Each game runs in its own spawned process with independent game/RNG/search state and one PyTorch CPU thread. GPU workers send only permitted encoded features, a checkpoint key, worker ID and request sequence. One shared owner loads the pinned models and batches same-checkpoint requests. Clients normalize logits over their own legal actions and receive policy/value outputs. Local opponent MLP inference avoids sending its frequent small requests through the GPU service.

The service collects a bounded window, runs at most 64 rows per model forward, preserves request/row routing across chunks and propagates errors/timeouts. Each worker makes synchronous requests, which naturally bounds in-flight requests by worker count. Model startup and process initialization are timed separately and excluded from throughput. Normal timed collection includes queue wait/IPC, neural inference, CPU game/search operations, native opponent work, source-prefix replay, compact persistence and replay verification.

CPU game-state updates and tree work remain substantial, and queue/dispatch/transfer overhead limits the benefit of faster batched inference. The earlier fixed-batch MPS speedups do not translate directly into whole-search speedups. This implementation is an opt-in diagnostic; it has not been integrated into the training collector or human webapp.

## Correctness and reproducibility

All 1,512 timed diagnostic continuations passed replay checks: 6,048 searched decisions / 774,144 simulations across 21 cases. All 24 matched workloads had identical action/chance traces, sparse root-visit traces and final state/RNG checksums across CPU/GPU modes, worker counts, batching windows and repetitions. Checkpoint and pinned implementation hashes are unchanged.

Four focused unit tests passed for legal-action normalization, row routing, malformed/failed responses, nonfinite predictions and bounded timeouts. A live MPS audit ran 16 concurrent mixed-checkpoint clients with eight rows per request and a four-row forward cap, checking chunking, routing and CPU numerical parity over 128 rows. Maximum policy probability error was 1.67e-06; maximum value probability error was 2.98e-07.

Friendly robber uses visible victory points. No weights were trained or deployed. W4 and whole-game data splits are unchanged. Every diagnostic is explicitly marked as excluded from training and saves only gzip actions/chance events/sparse visits/seeds/checksums; no feature arrays or game-state snapshots are persisted. The 96-continuation cases repeat the same 24 source workloads four times with fixed seeds intentionally, so they are performance work, not 96 independent games.

## Reproduce and inspect

Runners: `.conda/bin/python -m agent_apis.neural.shared_gpu_benchmark` then `.conda/bin/python -m agent_apis.neural.gpu_queue_concurrency`. Live audit: `.conda/bin/python -m agent_apis.neural.shared_queue_parity`. Tests: `.conda/bin/python -m unittest tests.test_shared_inference -v`. Existing matching result files are reused; protocol/source hash mismatches are rejected.

Implementation: `agent_apis/neural/shared_inference.py`. Timed experiment records: `runs/catan-ai/shared-gpu-benchmark/`, including `protocol.json`, `summary.json`, `concurrency-protocol.json`, `concurrency-summary.json`, `audit.json`, `parity.json`, per-case `result.json` and compact journals. The original `game-worker-benchmark/protocol.json` pins checkpoint/source-game references.
