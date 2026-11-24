# Search optimization revision 1

The four-own-action cutoff is removed from default and neural search. The time
budget and 128-total-action simulation safety cap remain. Explicit shallow depth
parameters are retained only for historical controls and the frozen correctness
audit; they are not a four-actions-per-turn rule. Saved training configurations
and checkpoints are unchanged.

Dice production is cached by immutable map layout, settlement/city ownership and
robber position. Roads, resource exchanges and ordinary turns reuse that table.
The current finite bank is applied separately on every roll: different rolls
may become equivalent when resources run out, and shortage announcements remain
part of their identity. Seven remains a distinct outcome. Enumeration now
executes one representative successor per merged roll class, rather than all 36
ordered dice pairs. Raw 36-outcome enumeration remains the reference ablation.

Other changes:

- Decode the public root world once, then copy it and replace all sampled private
  holdings, age flags, purchase identities and deck composition. Tests compare
  these worlds with the original full decoder across different hidden draws.
- Reuse immutable sampled root worlds, chance children, private/public views,
  encoded features and canonical keys inside a decision. Mutation remains confined
  to newly copied successor states; private views keep their audited redaction.
- Cache immutable action-record serialization and increment development-history
  keys from the parent instead of repeatedly scanning old moves.
- Replace repeated linear searches of the 332-action vocabulary with a lookup.
- Estimate simulation cost before admitting another iteration near the deadline.
- Record simulation counts and achieved depths in future match timing events.

Multithreading uses independent root trees with independent named RNG streams.
Their root visits and visit-weighted values are aggregated by legal action ID.
A shared inference service batches requests to each frozen model. Worker trees
never share mutable game states or another player's private observation. Tests
cover legal results, visit accounting, separate-versus-batched predictions for
all four architectures, and exception propagation through inference queues.
Python tree work still encounters the GIL; batching and Torch kernels provide the
concurrency opportunity. More workers are therefore measured, not presumed faster.

On this M4 Pro, the serial control comparison uses the same 18 frozen validation
positions and 500 ms ceiling, with the old four-action cap held fixed for that
comparison. Median simulations increased from 38 to 65.5 (+72%). Neural forward
passes increased from 26.1% to 46.3% of elapsed time; prediction including encoding
increased from 29.9% to 50.4%. In the uncapped variant, median simulations were 59,
with sampled deepest paths reaching seven own actions and 14 total actions.
These are diagnostic samples, not measurements of playing-strength improvement.

A separate balanced 12-position comparison of uncapped search produced:

| Workers | Mean simulations | Median simulations | Maximum decision ms |
| --- | ---: | ---: | ---: |
| 1 | 63.67 | 61 | 486.7 |
| 2 | 72.42 | 70 | 463.5 |
| 4 | 85.75 | 90 | 493.0 |

Four workers improve mean throughput by approximately 35% over one on this test.
They are selected for CNN CPU search in [search-optimized-v1.json](search-optimized-v1.json).
Other models retain one worker until measured. Inference is now the largest
individual measured component, but raw forward passes are still below half the
serial time; state/search overhead remains a worthwhile optimization target.
The inference-service totals from parallel workers overlap and must not be
interpreted as disjoint percentages of wall time.

Raw measurements and scripts are retained under ignored `runs/catan-ai/`:
`search-opt-before-recheck.json`, `search-opt-depth4.json`,
`search-opt-unbounded.json`, `search-thread-benchmark.json`, and
`search-hotspots.txt`. The baseline modules are preserved in
`search-opt-baseline/`. CPU, float32, six Torch CPU threads, existing seed-11 CNN
and the same frozen MLP opponent were used throughout. Time-based measurements
are machine dependent; the sample sizes are not a latency certification.

Full-game evaluation remains interrupted. No checkpoints were retrained and no
new win-rate evaluation was launched during optimization. The optimized search
is a new revision; its results must not be pooled silently with older search
runs. The old pure-policy results remain historical evidence. Frozen final
seeds and the original benchmark protocol file remain unchanged.

Validation: all 81 repository tests passed, followed by targeted cache/batching/
parallel/provenance checks. The complete 1,000-position frozen search audit
passed with zero failures. Machine-readable measurements and source/artifact
hashes are in [search-optimization-results.json](search-optimization-results.json).
