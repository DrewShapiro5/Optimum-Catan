# Phase-three results, November 4, 2026

The benchmark-harness completion gate is satisfied: paired baseline runs are
repeatable under the frozen protocol, every post-fix game completed legally,
provider errors surface, and complete timing logs and provider identities are
retained. No final evaluation games or learned-model selection were performed.

The [protocol and usage guide](BENCHMARK.md) describes the runner, statistical
analysis, deadline interface, scoring conventions and frozen-checkpoint hook.
[Machine-readable results](phase3-summary.json) retain metrics, provider hashes,
run references and the earlier failed run. [phase3.json](phase3.json) resolves
the baseline configuration. Both current corpora and the legacy source corpus
are excluded from all evaluation seeds.

The frozen seed file contains 500 tuning pairs and 500 final pairs. Smoke uses
20 tuning pairs with BLUE/RED swapped, screening 100 and expanded tuning 500.
Final games are rejected by this phase's runner and reserved for phase 10.

- Seed-file SHA-256: `a75e74b9ee5ad778db6e35041945be67cc62c9fab741e252be1aef32f2508d94`
- Protocol SHA-256: `0512be80936c4a021ad1091f5dd93ad6ee16dcc350287f1a7b61df6dd1ba788e`

## Post-fix baseline results

All runs below used the same rebuilt SMART binary, current benchmark/adapter
source, first 20 tuning pairs, two-player 15-point rules and 500 ms decision
ceiling. They ran serially on CPU. Each run contains 40 games.

| Candidate versus SMART | Wins–losses | Match score | Pair-bootstrap 95% interval | Warm non-forced p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 0–40 | 0% | 0–0% | 0.607 |
| Framework value variant, A | 28–12 | 70% | 52.5–85% | 5.031 |
| Same value variant, repeat B | 24–16 | 60% | 45–75% | 4.869 |

There were zero cutoffs, provider errors, illegal played actions or candidate
decision overruns in all 120 post-fix games. The 500 ms p95 ceiling was 525 ms;
all failure, cutoff and timing gates passed. Completed-game win rates equal
match scores, and removing cutoffs changes no result.

The value variant uses the framework's `base_fn` weights with independently
sampled afterstates and neutral hidden development draws. The unchanged
framework player was not used because it copies the authoritative future RNG.
This is a documented heuristic control, not a trained architecture.

| Candidate | Starting seat 0 | Starting seat 1 | Median/max turns | Thinking seconds | Decisions/sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0/20 | 0/20 | 103.5 / 173 | 0.516 | 10,500 |
| Value A | 13/20 | 15/20 | 105 / 139 | 7.001 | 961 |
| Value B | 11/20 | 13/20 | 111 / 159 | 7.231 | 983 |

Throughput includes immediate forced moves and is not inference throughput.
Candidate thinking excludes opponent time and authoritative execution.

| Candidate | All-decision p50/p95/p99 ms | Cold first-decision p50/p95/p99 ms |
| --- | --- | --- |
| Random | 0.004 / 0.508 / 0.609 | 0.327 / 0.505 / 3.663 |
| Value A | 0.645 / 4.260 / 6.348 | 5.143 / 5.946 / 6.308 |
| Value B | 0.665 / 4.009 / 6.181 | 5.159 / 5.534 / 5.876 |

Initialization times are also retained per seat/game. Applying 100 ms or 2,000 ms
ceilings to these same pure-policy timing samples gives zero overruns and
passes both p95 gates. These are latency checks, not additional matches or
search-budget comparisons. The policies do not use remaining time; actual
time/simulation-budget search comparisons belong to subsequent phases.

Repeat B minus A was −10 percentage points, with paired 95% interval
[−22.5, +2.5] points. Native SMART randomness is not seeded by the installed API,
so identical board/chance/policy seeds do not guarantee identical native
decisions. Deterministic random-versus-random trajectory replay is tested
separately. Repeated tuning pairs are not pooled as independent additional
boards, and these repeats are not separate training seeds.

These are smoke results, not final confirmation. A percentile bootstrap can
degenerate on all-identical outcomes; the random interval does not establish a
true zero win probability. One value run's lower bound exceeded 50%, while its
repeat's did not. No competitive superiority or learned-model result is claimed.

## Native bank-trade failure and recovery

The first value smoke run ended with 26 wins, 13 losses and one invalid game:
SMART requested four sheep for ore when the finite bank had no ore. The adapter
threw because it had no rejection path for an untranslatable bank trade. That
[failed run](../../runs/catan-ai/phase3-VALUE-6f12a1d375a9-20261104T213528912227Z/run.json)
remains preserved, along with the initial pre-fix random smoke run. Neither is
pooled with the post-fix table above.

The adapter now feeds an upstream `SOCRejectOffer` back to the existing native
brain. Its own nine-denial limit then governs native replanning/turn completion.
The upstream source revision and SMART strategy remain unchanged. No local
policy selects a replacement action. An unrecovered native error still fails
the game and produces a nonzero benchmark exit.

A perspective-redacted fixture reconstructed from the failing position is
checked into `tests/fixtures/jsettlers-empty-bank.json`. It masks hidden deck
composition. On three worker restarts, the native brain processed nine denied
trades and chose a legal END_TURN. The integration regression also checks this
path and idempotent repeated decisions. Post-fix runs record 0, 45 and 18 total
native translation rejections respectively, distinct from errors and illegal
played actions. Provider source/JAR hashes distinguish pre- and post-fix runs.

## Verification and artifacts

Twenty-seven tests passed across the benchmark, observation, provenance and
native API suites. These include paired/hierarchical bootstrap behavior,
seed exclusion, cutoff/error semantics, forced-move bypass, authoritative-RNG
independence, frozen-checkpoint hashing and safe inputs, final-seed/protocol
guards, real native bank-trade recovery, two-/four-player completion, HTTP/API
agreement and restart synchronization.

All three post-fix run records validated against the run-record schema. Their
raw game/timing file hashes were verified, board hashes match across candidates
and seats, and benchmark/adapter source hashes match the implementation. The
opponent JAR identities match across these three runs. Frozen seed/protocol file
hashes remain unchanged.

- [Random post-fix record](../../runs/catan-ai/phase3-RANDOM-2fd96178f5f5-20261104T214219642651Z/run.json)
- [Value A post-fix record](../../runs/catan-ai/phase3-VALUE-6f12a1d375a9-20261104T214023513237Z/run.json)
- [Value B post-fix record](../../runs/catan-ai/phase3-VALUE-6f12a1d375a9-20261104T214415684646Z/run.json)

Each record's directory contains complete `timings.jsonl`, `games.jsonl` and
resolved `configuration.json`. No frozen learned checkpoint exists yet; the
loader/budget-aware observation interface is implemented and fixture-tested,
with actual learned-opponent evaluations deferred until phase 5.

Phase 4 remains pending: stochastic search, hidden-development-card sampling,
outcome merging and its correctness acceptance tests.
