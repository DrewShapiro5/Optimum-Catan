# Game worker and GPU inference benchmark — November 18, 2026

Recommend **12 game workers, one PyTorch CPU thread per worker** for CPU self-play collection. The longer matched-workload test averaged 9.01 searched decisions/s at 12 workers, 9.04 at 14, and 8.97 at 16. Fourteen adds about 0.3% throughput and 1.1 GB summed RSS relative to twelve; sixteen adds no throughput. These runs support twelve as the practical choice, not a statistically proven optimum. No training controller configuration or playing model was changed; training remains paused.

## Original 1–8 worker sweep

24 matched continuations per case, four searched decisions each, 128 simulations per decision, two reverse-order passes. Rates aggregate decisions divided by total measured time across passes.

| Workers | CPU threads per worker | Searched decisions/s | Peak summed worker + child RSS (GB) |
|---:|---:|---:|---:|
| 1 | 1 | 1.27 | 0.67 |
| 1 | 2 | 1.24 | 0.68 |
| 1 | 6 | 1.14 | 0.66 |
| 2 | 1 | 2.28 | 1.19 |
| 2 | 2 | 2.15 | 1.12 |
| 4 | 1 | 3.87 | 2.17 |
| 4 | 2 | 3.41 | 2.12 |
| 6 | 1 | 5.31 | 3.05 |
| 6 | 2 | 4.61 | 3.00 |
| 8 | 1 | 6.09 | 3.89 |
| 8 | 2 | 5.29 | 3.89 |

One CPU thread per worker was faster at every worker count. The existing single-worker/six-thread baseline averaged 1.14 decisions/s; eight single-thread workers averaged 6.09, a 5.36× gain in this short-workload sweep.

## Longer high-count sweep

Each case repeats the same 24 workloads four times (96 continuations, 384 searched decisions), with two reverse-order passes. Repeated seeds are intentional fixed benchmark work, not independent games or training data. Longer queues reduce end-of-case idle time, so compare these rates within this table; do not combine them directly with the shorter sweep.

| Workers | CPU threads per worker | Searched decisions/s | Peak summed worker + child RSS (GB) |
|---:|---:|---:|---:|
| 8 | 1 | 7.32 | 4.38 |
| 10 | 1 | 8.60 | 5.82 |
| 12 | 1 | 9.01 | 6.51 |
| 14 | 1 | 9.04 | 7.57 |
| 16 | 1 | 8.97 | 8.44 |

Twelve workers provide about 23% more throughput than eight in the longer test. Individual-search latency increases as more games run concurrently; the goal here is total collection throughput. The machine has 14 CPU cores and 48 GiB RAM. RSS sums can double-count shared pages and are not total physical memory consumption.

## Apple GPU inference

Training is already configured for Apple GPU/MPS. Collection search currently evaluates on CPU. The XD-CNN supports MPS, and measured predictions agree with CPU to within 0.000002 absolute probability error.

| Positions per batch | CPU batch latency (ms) | MPS batch latency (ms) | MPS / CPU throughput |
|---:|---:|---:|---:|
| 1 | 2.29 | 3.37 | 0.68× |
| 8 | 8.76 | 4.30 | 2.04× |
| 32 | 63.88 | 9.90 | 6.45× |
| 64 | 121.18 | 16.77 | 7.22× |

These timings include public-observation encoding, tensor construction/transfer, model inference, legal-action normalization and result transfer. Search observations can cache encoded features; these are evaluator diagnostics, not measured multiworker GPU search speedups. Three warmups and three sets of ten timed calls per device/batch; median set reported.

The next GPU experiment should use a shared inference service with bounded batching across game workers, separating requests by checkpoint (current XD-CNN versus frozen opponent MLP). A short batch timeout is needed to limit worker stalls. Keep simulation budgets, root visits, RNG streams and compact replay checks unchanged. Game rules, state copying, chance generation and tree bookkeeping currently execute as CPU/Python logic. Moving these to GPU would require a separate substantial engine implementation. Simply moving each single-position request to MPS was slower here. No shared GPU service has been implemented or benchmarked, and no overall search speedup is claimed from the inference microbenchmark.

## Reproducibility and scope

All 1,488 diagnostic journals replayed successfully (5,952 searched decisions, 761,856 simulations). Every one of the 24 matched workloads reached the same final checksum across all worker/thread configurations and workload repetitions. The checkpoint checksum is unchanged. Each process owns its models, game state, RNG streams, search trees and native JSettlers bridge. Diagnostics save gzip actions/chance events/sparse visits/checksums, without game-state snapshots; they are explicitly excluded from training.

Workloads use the fresh visible-point-friendly-robber XD-CNN checkpoint, eight existing corrected completed-game boards, opening/25%/50% game prefixes, and a 50/25/25 current/historical/native-opponent mixture. The historical slot uses the current checkpoint for this initial-generation diagnostic. Each measured case includes prefix replay, search, native opponent work, compact persistence and replay verification. Process/model startup is measured separately and excluded. Model references and journal checksums are pinned in protocol/results.

This is a bounded throughput test, not complete-game collection, model strength, power efficiency or long-duration thermal certification. Native JVM startup occurs per short continuation, which differs from full games. W4 training, whole-game dataset splits and historical protocols/results were not modified.

Artifacts: `runs/catan-ai/game-worker-benchmark/{protocol,summary}.json`, `runs/catan-ai/high-worker-benchmark/{protocol,summary,gpu-inference,audit}.json`, and each case’s `result.json` and compressed journals. Runners: `agent_apis/neural/worker_benchmark.py` and `agent_apis/neural/worker_scaling_extension.py`.
