# Phase-four results, November 4, 2026

The section-4 correctness gate passed on the complete frozen 1,000-position
validation suite. The registered `STOCHASTIC_REFERENCE` provider now runs
root-sampled stochastic search through the existing Python and HTTP APIs.
It has a deterministic reference evaluator and frozen heuristic opponent;
no learned model has been trained and no playing-strength claim is made.

The [search contract](SEARCH.md) describes the information boundary, exact
history-consistent development-card belief, tree/history keys, probability
classes and reference components. [Machine-readable results](phase4-summary.json)
retain the stratified counts, diagnostics and full run reference.

The accepted [run record](../../runs/catan-ai/phase4-validation-e6f29e48626c-20261104T223507225065Z/run.json)
validated against the shared schema. It includes source/software/data hashes,
full split lists, configuration, test logs and the checksummed per-position
artifact. The full audit took 229.05 seconds.

Configuration SHA-256:
`e6f29e48626c5e76b91528686b908b62f51cd1a25b9b88ef8beaa95cace18d70`

Frozen-suite SHA-256:
`0460c371c0006d99d57ad5cdbae0bb14c1db1067378c0bfce9ce8a3d68e5e03f`

## Correctness gate

| Check | Result |
| --- | --- |
| Frozen validation positions, source identity and phase | 1,000 passed |
| Merged/unmerged selected actions legal | 2,000 passed |
| Weighted one-step value equivalence | Maximum error 6.66e-16; tolerance 1e-8 |
| Authoritative RNG/deck-order perturbation | Identical permitted observations in all 1,000 cases |
| Fixed-seed search invariance | Exact action/visit/value agreement for seeds 11, 23, 47 in tests |
| Hidden opponent card-type invariance | Identical root inputs and search statistics under fixed seeds |
| Root history sharing and private-draw visibility | Passed |
| Opponent policy private-information boundary | Passed |
| Chance probability mass and positive-outcome coverage | Passed |
| Dice/theft/development frequency tests | 100,000 draws each; all bounds passed |
| Implicit simulator RNG use | Guarded; explicit transition tests passed |
| Simulation action/depth cap | Zero capped simulations |
| Failures | Zero |

The three representative corpus-derived chance distributions contained 14
classes in total. Every observed frequency lay within the predeclared
simultaneous absolute bound 0.0071554, with familywise alpha 0.001. This is a
Hoeffding/union-bound check over the whole multinomial family, not a tolerance
chosen after observing counts. Separate handcrafted frequency tests also pass.

Handcrafted checks cover exact two-dice probabilities, 2:1 resource-frequency
theft, finite-supply development draws, newly bought card restrictions,
immediate wins, sequential own actions, discard sequences and free roads.
An immediate winning settlement also receives exact root utility and is selected
by search on a replayed pre-win position. Joint hidden-card tests verify that a
public play requires an old card and that later historical own draws correctly
change the posterior over earlier unknown opponent draws.

The development sampler uses exact dynamic-program continuation weights over
chronological purchase/play/age constraints. It checks own holdings and age
flags, public plays, remaining total, finite supply and absence of historical
terminal scores. It does not read hidden authoritative cards or weight past
strategic actions by an opponent likelihood model.

All evaluator/opponent callbacks receive roll labels explicitly replaced with
null; current state and normalized event effects carry the relevant information.
Non-acting root views omit the opponent's private legal actions. Opponent-only
private draws share root decision histories; own visible draws separate them.
Sampled worlds remain distinct below action edges, and root action evidence is
kept separate from chance-child evidence. Backups retain root utility throughout.

## Stratified outcome merging

The fixed suite uses only the existing game-level validation split. No training
labels, terminal outcomes or future actions enter the search observation.

| Stratum | Positions | Raw one-step outcomes | Merged one-step classes |
| --- | ---: | ---: | ---: |
| Setup | 200 | 200 | 200 |
| Middle | 300 | 4,802 | 1,361 |
| Endgame | 200 | 2,070 | 697 |
| Robber | 100 | 248 | 248 |
| Discard | 100 | 100 | 100 |
| Free road | 100 | 100 | 100 |
| Total | 1,000 | 7,520 | 2,706 |

For the fixed demonstrated action and same sampled world at each position,
merging removed 64.0% of elementary outcome branches while preserving weighted
one-step value. This compares identical transitions, independently of subsequent
search trajectory differences. Bank-shortage effects, phases, terminal results
and private knowledge remain distinguished.

## Search diagnostics and negative performance result

Both variants used 16 simulations on each of 841 non-forced positions. The other
159 roots returned immediately. Each variant completed 13,456 simulations.
All 1,000 positions also received an independent sampled-world one-step check,
including forced roots that skip belief preparation during ordinary selection.

| Metric | Unmerged | Merged |
| --- | ---: | ---: |
| Untraced simulations/second | 173.94 | 153.24 |
| Search elapsed seconds, including six traced probes | 80.90 | 91.59 |
| Raw outcomes across visited distributions | 144,766 | 144,634 |
| Outcome classes across visited distributions | 144,766 | 59,251 |
| Retained sampled chance children | 30,641 | 30,520 |
| Peak traced allocations, six-probe median, MB | 1.884 | 1.869 |
| Peak traced allocations, six-probe maximum, MB | 2.688 | 2.545 |
| Cumulative process peak RSS, MB | 183.337 | 183.337 |
| Maximum action depth | 14 | 14 |
| Maximum root-action depth | 4 | 4 |
| Maximum framework turn-counter delta | 2 | 2 |

Throughput excludes traced probes: 13,360 simulations per variant. Memory probes
use the first non-forced frozen position encountered in each of six strata;
RSS is recorded for every case but is cumulative process telemetry, not an
isolated per-variant tree size. MB means decimal megabytes. No memory advantage
is statistically established by six probes or by cumulative RSS.

Within the merged variant's own visited distributions, 144,634 elementary
outcomes became 59,251 classes, a 59.0% reduction. The search variants visit
slightly different trajectories, so their global raw counts are not assumed
identical. Their selected root actions differed on 94 positions. Grouping
changes the coupling of seeded finite-budget chance samples; this is compatible
with equal transition expectations and is not a playing-strength comparison.

The merged prototype was approximately 11.9% slower in untraced simulation
throughput despite fewer classes. Canonicalization and temporary successor
construction have overhead, and lazy storage already avoids retaining unsampled
children in both modes. No speedup is claimed and no throughput optimization is
promoted. The plan explicitly assumes no minimum speedup. Unmerged search stays
available as the diagnostic ablation; profiling/improvements belong to later
phases after this correctness baseline.

## Verification and scope

All 43 tests passed across search, benchmark, observation, provenance and native
agent APIs, including two-/four-player native completion, restart recovery and
the existing empty-bank trade regression. The new provider's Python and HTTP
paths were exercised, including unsupported-rule rejection and forced moves.
JSettlers integration and its pinned upstream source remain intact.

The accepted run's core search, belief, validation, catalog, test and
configuration hashes match the delivered implementation. Its `positions.jsonl`
and test-log hashes were verified. Earlier partial smoke/profiling attempts
remain in ignored run directories; interrupted attempts are marked in
`interruption.json` and are not accepted results. One was restarted to enforce
the callback normalization boundary; another separated all-case RSS telemetry
from per-stratum traced probes. Neither reduced the final 1,000-position scope,
simulation count or correctness gates.

These are correctness and diagnostic results, not JSettlers matches or a 500 ms
deployment qualification. Phase-three final evaluation seeds remain unused.
Phase 5 remains pending: train the common-head encoders, replace the reference
evaluator and heuristic opponent with the specified learned components, then
run the architecture comparisons under the frozen benchmark protocol.
