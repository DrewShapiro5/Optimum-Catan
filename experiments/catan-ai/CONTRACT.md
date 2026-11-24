# Catan AI experiment contract v1

Phase 2 has now resolved the initially pending model-input contract as
[schema 3](OBSERVATIONS.md); its configuration is `phase2.json` and its full
audit result is recorded in `PHASE2_RESULTS.md`. The phase-one configuration
remains unchanged as the historical preflight baseline.

Phase 1 implements section 1 of [the experiment plan](../../AI_EXPERIMENT_PLAN.md).
The machine-readable baseline is `phase1.json`; the record format is
`run-record.schema.json`. These files and the implementation are source inputs
and receive content hashes in each run. Change a configuration field to create
a new configuration identity; the seed is a separate run parameter. Repeats of
the same configuration and seed receive distinct timestamped run directories.

Use `.conda/bin/python` and install `.[dataset,experiment]`. PyTorch is the only
training framework; the experiment extra pins the version installed for this
phase. CPU inference remains supported. Benchmark available CPU/MPS/CUDA devices
before assigning an evaluation device. Phase-one synthetic measurements cannot
select a device for models whose encoders do not yet exist.

## Fixed information and rules

Two players, BLUE/RED, BASE board, 15 victory points, discard above nine,
standard robber, bank/port trades and no player trading. Evaluation swaps seats
on every board/seed pair. Other rules require a separate experiment and data.
Register the eventual learned provider under `agent_apis/agents.json`; preserve
the existing JSettlers provider. No placeholder learned provider is registered.

All architectures share one permitted observation path and the same 332 action
IDs, absolute coordinates and victim colors. Apply one global softmax after
masking illegal actions. Value means decision-actor win probability in [0,1];
convert existing outcomes with `(value_target + 1) / 2`. Demonstrations train the
policy; actual completed-game outcomes train value. Later search visits and
self-play outcomes are separately identified targets. Cutoffs are not outcomes.

Use the two schema-2 corpora once each. They have 633 source features including
exact tile numbers. The model feature schema is deliberately pending: phase 2
must add and audit the exact opponent-resource ledger, phase/rule information
and development-card constraints. No model training is authorized by a
successful preflight alone. No authoritative hidden hand, ordered deck, future
RNG, later action or terminal label may enter an observation. Catanatron remains
the dynamics engine; chance sampling must use independent futures.

## Configuration and random streams

Every experiment copies the baseline, names its stage and specifies model,
data view/composition, weighting, training, search and evaluation choices in
that JSON. Resolve all overrides into this configuration before hashing or
running. A checkpoint reference must include its SHA-256. Do not rely on CLI
defaults, environment variables or checkpoint filenames to imply changes.
The preflight CLI intentionally accepts only the preflight stage; subsequent
stages must implement their own acceptance gates before they can run.

`stream_seeds(seed)` derives stable, independent 64-bit seeds for board,
simulator, search, training and evaluation using SHA-256 with a versioned
namespace. `rng_streams(seed)` constructs isolated Python generators without
changing global state. Consumers must explicitly use the correct stream;
PyTorch/NumPy generators use the corresponding derived seed. Never reseed or
inspect an authoritative game's RNG during search. Screening uses seed 11;
confirmed configurations use 11, 23 and 47. A single seed is not confirmation.

## Run records

Run `.conda/bin/python -m agent_apis.experiment prepare` from the repository.
Optional `--config`, `--seed` and `--output` arguments identify another resolved
preflight configuration, seed or output directory. Default output is ignored
`runs/catan-ai/<experiment>-<config hash>-seed<seed>-<UTC timestamp>/run.json`.
Validate with `.conda/bin/python -m agent_apis.experiment validate PATH`.

Each record stores the resolved configuration and its canonical SHA-256,
seed/stream seeds, UTC creation time, status, source revision if available,
Git status, individual source hashes and an aggregate source hash, complete
installed package versions, Python executable/version, hardware, elapsed time,
dataset manifest and actual shard checksums, complete game/seed/split lists,
feature names, action vocabulary, teacher revision, checkpoint and metrics.
Source discovery includes untracked source files and respects Git ignores;
data, environments, generated outputs and images are excluded. With no initial
Git commit, revision is null and source hashes identify the actual inputs.

Schema validation rejects missing required fields; validation also recomputes
configuration identity and stream seeds. Preflight validates shard checksums
and manifest totals, not observation fidelity or legal labels. Phase 2 must
separately audit all actions, splits and observation reconstruction. A null
checkpoint is correct for preflight; trained results must record a hashed
checkpoint and metrics with units, denominators, protocol and failures. Preserve
raw timing samples. Later runners must capture failures and final elapsed time
and must not overwrite a completed record. Keep raw outputs ignored and commit
configuration files and analysis summaries.

## Evaluation boundary

The baseline records 100/500/2000 ms and 128/512/2048-simulation budgets;
500 ms is primary. Time the entire agent decision, record cold start separately,
return forced actions immediately, and avoid concurrent accelerator benchmarks.
JSettlers SMART is the primary opponent; errors must surface. Enforce p95 at
budget + max(20 ms, 5%); report every overrun. Cap at 1000 turns/10000 actions;
cutoffs receive 0.5 match score and are reported separately.

Phase 3 freezes tuning and final seed lists disjoint from corpus seeds. Keep
final confirmation reserved for phase 10. Use paired-seat bootstrap intervals;
aggregate confirmed models with a hierarchical bootstrap over training seed
and board pair. Claim superiority only with a lower 95% match-score bound above
50%, disclosed failures/cutoffs, and the legality and timing gates satisfied.
Training, model comparisons, full dataset audits, search and benchmark matches
belong to later phases and are not phase-one results.
