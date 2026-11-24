# Short evaluation schedule

The user requested a 10-game smoke test and 50-game total. This amendment uses
five board/seat pairs for smoke and at most 25 pairs for the complete run. The
10 smoke games are included in the 50; extension adds 40 games. It uses the
original frozen tuning seeds, 500 ms decision budget, optimized search, existing
seed11 checkpoints and the unchanged native SMART opponent. Final seeds remain
reserved. The original frozen protocol file is not rewritten.

All smoke tests run before any extension, in this order: H-GNN, Transformer,
XD-CNN, MLP. Existing pure-policy results are retained as historical comparison
references, using the common first at most 50 games (40 are available for the
old GNN/Transformer policy baselines). No new training or automatic additional
seeds follow. These are exploratory, single-seed results rather than final
confirmation.

The statistical futility policy is retained. Ten games alone do not establish
hopelessness: the corrected bound needs more pairs, so a zero-win smoke may
still extend. A run that satisfies futility later can stop before 50 games.
Provider errors stop the shorter run promptly and remain failures. Timing,
cutoff and illegal-action flags are preserved in the report.

The runner persists a checksum-verified checkpoint after every completed game.
If interrupted, it resumes that exact configuration's completed prefix, including
an odd completed seat, rather than replaying it. Uncommitted trailing log bytes
are retained separately. Model, provider and simulator compatibility are checked
before reuse. No games from older search revisions are pooled with this one.

Run with `.conda/bin/python -m agent_apis.neural.quick_evaluation`.
If games have finished and only reporting failed, use
`.conda/bin/python -m agent_apis.neural.quick_evaluation --report-only`.
This verifies the saved configurations, smoke prefixes and log checksums and
regenerates the report without launching games or training. It preserves the
original evaluation source hashes and records the reporter revision separately.
Live state (including PID): `runs/catan-ai/phase5-quick-evaluation.json`.
Live log: `runs/catan-ai/phase5-quick-evaluation.log`.
Finished results: `PHASE5_QUICK_RESULTS.json` and `PHASE5_QUICK_RESULTS.md`.
Resolved per-model configurations are in `phase5-quick-configs/`.

The shorter evaluation completed on November 7, 2026; see
[the results](PHASE5_QUICK_RESULTS.md). CNN won 12/50 games, Transformer 9/50,
GNN 1/40 and MLP 0/40. GNN and MLP stopped under the existing futility rule.
All search games had zero errors, cutoffs and illegal actions, and all four
models passed the warm latency gate. Final report generation initially rejected
the preserved Transformer policy baseline because its valid status was
`stopped-early`. The reporter now accepts futility-stopped historical references,
retains that status, and suppresses fixed-sample comparison intervals when either
side stopped early. Recovery replayed zero games.

Twenty-one targeted tests passed, including exact 10-to-50 prefix reuse, resumed
odd-seat accounting, schedule checksums, smoke-first ordering, and the existing
benchmark/futility checks, plus stopped-baseline handling and report-only recovery.
