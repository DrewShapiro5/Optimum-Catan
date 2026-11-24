# Phase-one execution, November 4, 2026

The section-1 completion gate is satisfied: a written configuration contract,
resolved baseline JSON and machine-readable run-record schema exist. The
preflight command captures and validates actual provenance and device timings.
PyTorch 2.15.0 and JSON Schema validator 4.26.0 were installed into `.conda` and
pinned in the `experiment` optional dependency. No training has run.

The repository has no initial Git commit. Run records therefore use a null
revision plus individual and aggregate content hashes, including untracked
source files. A future commit will automatically supply revision metadata.

The seed-11 baseline configuration hash is
`4e7a801295e60771ab11897a869377e5398562a4b77087f4f59de9364b90f333`.
Each execution writes a separate ignored directory under `runs/catan-ai/`.
Use the full run record for exact source hashes, environment and raw timings.

Manifest totals and every listed compressed shard checksum were checked:

| Corpus | Games | Retained examples |
| --- | ---: | ---: |
| full-250 | 250 | 97,766 |
| downsampled | 1,017 | 64,503 |
| Total | 1,267 | 162,269 |

Both manifests agree on all 633 feature names and 332 action vocabulary entries.
This check establishes artifact identity and manifest consistency. The phase-2
ledger, replay, leakage, cross-split and demonstration-legality audits remain
required before model comparisons.

Hardware preflight detected an Apple M4 Pro, macOS 15.3, arm64, ten PyTorch CPU
threads, available CPU and MPS devices, and no CUDA device. Sandboxed detection
exposed CPU only; the hardware preflight was repeated outside the sandbox to
measure both supported devices.

Representative seed-11 measurements, 100 synchronized forwards after ten warmup
forwards, on a synthetic 633→256→256→256→333 GELU network:

| Device | Batch | Median ms | p95 ms | First forward ms |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 0.024 | 0.024 | 0.170 |
| CPU | 32 | 0.075 | 0.078 | 0.144 |
| MPS | 1 | 0.210 | 0.528 | 19.686 |
| MPS | 32 | 0.283 | 0.680 | 86.340 |

The benchmark uses zero input, inference mode and resident tensors; it excludes
encoding, search and host/device transfer. First forward excludes import and
model initialization and is not full agent cold start. These are provisional
compute measurements from one preflight, not statistical performance findings.
CPU is the promising starting device for this small workload. Evaluation-device
selection stays null until the real model/encoder workloads are measured.

Verification: four focused tests cover independent/repeatable RNG streams,
canonical configuration hashes, rejection of corrupt dataset shards and
required run-record provenance. The executable preflight checks the installed
PyTorch pin, hashes all corpus shards, benchmarks supported devices and
validates the resulting JSON against the schema. No learned provider or
JSettlers match result is claimed.

Next: execute phase 2's complete observation and resource-ledger audit. Its
feature schema must replace `pending-phase2-audit` in a new configuration
before training can begin. Freeze disjoint evaluation seed lists in phase 3.
