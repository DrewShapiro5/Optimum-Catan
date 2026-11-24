# Benchmark protocol v1

`benchmark-protocol-v1.json` freezes the experiment's evaluation contract.
`evaluation-seeds-v1.json` freezes 500 tuning and 500 reserved final board/seat
pairs. `phase3.json` is the resolved baseline configuration. No final game is
played in phase 3; the runner rejects final-seed requests. Phase 10 must add an
explicit locked-configuration evaluation path before final confirmation.

## Run a baseline

```sh
.conda/bin/python -m agent_apis.benchmark run --candidate RANDOM --tier smoke
.conda/bin/python -m agent_apis.benchmark run --candidate VALUE --tier smoke
.conda/bin/python -m agent_apis.benchmark run --candidate JSETTLERS --opponent RANDOM
```

Default opponent is installed JSettlers SMART. Smoke uses the first 20 tuning
pairs (40 games); screening uses 100 pairs; expanded-tuning uses all 500 pairs.
Optional `--budget-ms 100|500|2000`, `--seed`, `--config` and `--output` arguments
are resolved into the saved configuration before hashing. The seed argument
controls statistical resampling; board, simulator and policy RNG seeds come
from the frozen pair entries and remain shared across candidate comparisons.
Baseline policies have no search; time is an eligibility ceiling, not a delay.
Fixed simulations 128/512/2048 and time-limited search are supported by the
learned-provider interface once the later search implementation exists. The
CLI deliberately exposes only tuning schedules at this stage.

One invocation runs serially. An exclusive file lock rejects another harness
invocation even if it chooses a different output directory. Do not run other
jobs on the evaluation accelerator during timing. These phase-3 controls use
CPU. Later learned runs must record their selected device in configuration.

Each game starts with a fresh Java worker, to bound memory and make native
startup observable. Seated providers catch up from the permitted action history
on their next non-forced decision. Forced moves return immediately without
invoking a provider. Initialization and each seat's first non-forced decision
are reported separately as cold costs; raw logs retain them too.

## Board, seats and chance

Each pair has a board seed, independent simulator seed and independent agent
seed, derived from master 2026110403 with a versioned SHA-256 namespace. All
three seed types are unique across the tuning and final sets and disjoint from
both current corpora and the legacy source corpus. Frozen-list validation is
run before every benchmark, and file hashes are pinned in the configuration.
The generator is `frozen_seeds`; regeneration is tested to reproduce the
checked-in file exactly. Changing it is a new protocol, not an in-place update.

Game construction uses the board stream for map and starting color order. The
simulator then gets a separate RNG and separately shuffled initial development
deck. The ordered deck is never passed to a policy. The two games in a pair
assign candidate BLUE then RED, keeping the board/starting order fixed; thus
each candidate plays once from each starting seat. Board hashes must match.
Candidate and opponent policy RNGs are independent role-specific derivations.
Shared simulator seeds do not imply common trajectories after actions diverge.

JSettlers' upstream robot has internal randomness that this installed API does
not seed. Repeat runs therefore reproduce the protocol, boards and controlled
RNGs, not necessarily every native action. Fully controlled random-versus-random
traces are tested for exact repeated trajectory hashes.

## Opponents and provider identity

RANDOM samples uniformly over legal actions with its own RNG, including the
perspective-specific observation conversion in its measured decision.

VALUE reuses the framework's `base_fn` and `DEFAULT_WEIGHTS` in a one-step
afterstate player. The stock `ValueFunctionPlayer` copies the authoritative
future RNG; the benchmark variant instead seeds each afterstate independently.
It replaces the copied hidden development deck with neutral knight draws of
the same total size. The framework scorer uses only development-card holding
totals for this draw, so it cannot use the actual future draw type. Theft and
dice sample from the independent copied RNG. Opponent resource hands are exact
reconstructible information under the audited two-player contract. This is
explicitly an independent-future variant of the framework heuristic, not an
unchanged native policy or the later learned search. Its framework source hash,
weights and variant are recorded. Tests exercise invariance to changed
authoritative future RNG/deck in a midgame decision containing a roll and a
development-card option.

JSETTLERS uses its existing observation API unchanged. Before a run the harness
checks availability, the clean pinned upstream revision, provider source hash,
adapter/Gson JAR hashes, Java version and strategy metadata. Each native seat's
first non-forced call checks live synchronization reports `SOCRobotBrain` and
`SMART`; this check is included in cold latency. Errors propagate into the game
record and timing event, never to a replacement random policy.

Finite-bank trades rejected by the supplied legal mask are sent through
SMART's upstream `SOCRejectOffer` handler. The native brain can then retry or
end its turn under its own denial limit. Cumulative adapter translation
rejections are recorded per native seat and in the summary; an unrecovered
error still fails the game and run. This recovery was added after a smoke run
exposed an empty-bank trade. Old failed results remain preserved and are not
pooled with evaluations of the rebuilt adapter.

Frozen learned checkpoints are supported through a catalog-registered provider
with `from_checkpoint(path)` and `benchmark_choose(safe_observation, context,
rng)`. The latter returns a JSON action and receives only schema-3 permitted
observation data. The context provides a total decision deadline, an earlier
admission deadline reserving max(20 ms, 5% of budget), remaining milliseconds,
mode and simulation count. Encoding and preparation have already consumed part
of this deadline. Search must stop admitting work early enough to finish pending
inference; overruns are measured regardless of provider claims. Candidate and
opponent checkpoint path/SHA-256 references belong in their configuration specs
and are verified before use. No learned checkpoints exist yet; none are
invented or substituted in these baselines.

## Timing, outcomes and records

The clock starts before observation creation/serialization and includes every
provider operation, encoding, preparation, inference/search and selection.
It ends after action legality checking and vocabulary conversion. Authoritative
execution and opponent decisions are excluded. Raw timing events identify
pair, candidate color, actor/role, step, legal count, selected action ID,
forced/cold status, elapsed milliseconds, budget, overrun and any error.

The primary timing gate uses warm non-forced candidate p95 and requires it to
be at most budget + max(20 ms, 5% of budget). All-decision p50/p95/p99 and cold
first-decision p50/p95/p99 are separately reported. Every decision exceeding
the nominal budget is logged, including cold decisions. Candidate thinking
seconds and decisions/second are recorded; throughput includes forced actions
and therefore must not be interpreted as neural inference throughput.

Games stop at 1,000 turns or 10,000 actions and are unresolved cutoffs with 0.5
credit. Completed wins/losses, cutoffs, errors, illegal actions, completed-game
win rate, overall match score, cutoff-removed sensitivity, starting-seat results
and turn statistics are separate fields. An error is an invalid game with zero
candidate credit; it is never counted as a win. Errors receive no retries and
block promotion. Cutoffs above 1% also block promotion pending investigation.
Failure, timing and cutoff gates are explicit independent fields. A positive
interval alone cannot override a failed gate or authorize final-test tuning.

Outputs are ignored `runs/catan-ai/phase3-*/` directories containing resolved
`configuration.json`, complete `timings.jsonl`, `games.jsonl` and schema-validated
`run.json`. The record captures source/environment/provider identity, corpus
checksums/splits, exact evaluation pairs, configuration/seed, elapsed time,
checkpoint and log hashes. Game logs include the actual board hash, controlled
seeds, seat, outcome/error, turn/action totals and a trajectory hash. Interrupted
runs retain partial logs and configuration; they are not completed results.

## Statistical analysis

Report 95% percentile bootstrap intervals with 5,000 replicates. Resample whole
board/seat pairs, not games or moves independently. A pair's score is the mean
of its two games. `paired_difference` resamples the differences on identical
board/seat pairs, checking board hashes and simulator seeds. With three or more
distinct training seeds, `hierarchical_bootstrap` resamples training seeds and
shared board pairs and also reports each seed's mean. Neither helper converts
one training seed into statistical confirmation.

These helpers are importable from `agent_apis.benchmark`; pass the parsed game
records from `games.jsonl`. Compare records with the intended shared opponent,
protocol, seeds and budgets, and retain all failure/cutoff fields. All-identical
smoke outcomes can give a degenerate percentile interval; that is not proof of
the underlying true win probability. Phase-3 smoke results check the harness,
not a competitive-strength claim. Confirmation remains reserved for phase 10,
after configurations/checkpoints and controls are locked using tuning data.
