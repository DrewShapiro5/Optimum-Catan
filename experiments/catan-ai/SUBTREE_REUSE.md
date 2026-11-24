# Search reuse across real decisions

User-requested prerequisite before further self-play training, November 13, 2026.
The implementation is available in `agent_apis/neural/subtree_reuse.py` and wired
into the checkpoint-backed `NEURAL_SEARCH` runtime. Enable explicitly with
`search.reuse_subtree = true`; false retains the previous fresh-tree baseline.
The existing default remains off because the position audit does not establish
a strength gain or consistent speed improvement. [Audit results](SUBTREE_REUSE_RESULTS.md).

Each independent search worker retains its own tree between decisions. On a new
request it verifies the public session ID, root seat, frozen evaluator/opponent
identity or caller revision, worker count and search settings. History must
strictly extend the previous request, begin with the action actually selected
by the aggregated workers, and reach an explored observation. Full permitted
snapshot, board, normalized history, legal actions and private knowledge must
match. Rewinds, different games, altered knowledge, unvisited outcomes and
oversized retained trees restart cleanly. Retention is capped at 4,096 decision nodes and 8,192 sampled Game states per worker.

The matched subtree is re-rooted and its siblings/ancestors are released. Root
world templates, sampled successor Game objects, chance classes, encoded views
and canonical keys are retained where compatible. The first new simulation can
reuse the actual stored root Game without decoding it again. Unseen feasible
private allocations still create a new template from the decoded public state.
Chance distributions remain indexed by full sampled-world keys; equal root
observations never pool different private states into one perfect-information
world. No authoritative Game or future RNG is passed into the reuse matcher.

Beliefs are rebuilt from the current permitted history at every new decision.
A conservative rule resets visits/returns whenever the opponent has purchased
any development card, or old search had a depth limit or capped simulation.
This discards old belief-weighted utility estimates while preserving compatible
Game objects, valid model priors and reusable transitions. With a known current
world, uncapped search can retain the accumulated root-utility statistics. Root
utility keeps the same player perspective. Retained visits are reported
separately from the new simulation budget.

Models remain frozen for a SearchSession. Callers replacing model contents in
place must change `model_revision` or call `clear()`/`runtime.reset_search()`.
Checkpoint-backed runtime revisions include configuration, checkpoint update
and frozen-opponent checksum, and replacing evaluator instances also resets.
Inference services are created anew per decision; retained trees receive the
new services and never submit to a closed old worker queue. A lock serializes
requests to the same session. Exceptions clear retained trees.

Game IDs are routing metadata carried on the decision context by the Python
benchmark and direct provider API. They are excluded from learned features. The direct API runtime cache is additionally scoped by game and seat, so alternating self-play seats do not evict or share each other's search session.
If no session ID is supplied, cross-decision reuse is disabled. Forced live
moves return immediately; the next non-forced request can match the full path
across forced moves and opponent actions. Chance purchases are normalized back
to the BUY action ID when matching recorded moves, while their visible card
identity remains part of the observation match.

Tests cover pruning/object reuse, exact retained-world reconstruction, hidden
allocations/age flags, posterior refresh, statistics reset, changed private
knowledge, session/model revision changes, actual aggregated worker actions,
new simulation accounting, and fresh root-decode avoidance. Existing stochastic
search, encoder, chance invariance, native API and benchmark regressions also
pass. No self-play training is part of this change.
