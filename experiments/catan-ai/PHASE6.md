Current scope amendment (November 11, 2026): **XD-CNN only**. The user retired
Transformer and stopped its remaining tests. All XD-CNN phase-six confirmation
runs are complete; see [phase-six results](PHASE6_RESULTS.md). The original two-architecture
comparison remains incomplete. W4 is promising, but its paired hierarchical
improvement interval touches zero, so the conservative recipe remains D2/W0.
Future experiment work uses XD-CNN; do not restart the historical Transformer queue.

# Phase 6 corpus composition and weighting

Authorized November 7, 2026: proceed with Transformer and XD-CNN, validating
against random player and native JSettlers SMART. MLP and H-GNN leave the
candidate queue. The existing frozen seed-11 MLP remains the shared search
opponent model; this is separate from the actual validation opponents.

Phase 5's shortened single-seed screen supplies the shortlist. Its full
three-seed confirmation remains incomplete; phase 6 is an explicitly exploratory
continuation, with no claim of a statistically confirmed architecture winner.

The versioned protocol is [phase6.json](phase6.json). XD-CNN screens D0/W0,
D1/W0, D2/W0, D2/W1, D2/W2, D2/W3/V0, D2/W3/V1 and D2/W4 at seed 11.
Every run uses 20,000 optimizer updates, batch 256 and the existing augmentation,
optimizer, learning-rate schedule, audited schema and rule constants. All
recipes use the same complete, unweighted D2 validation set for checkpoint
selection and the same held-out D2 test metrics. Test results never select a
checkpoint or recipe.

D0 contains complete training decisions from full-250 (220 games / 85,773 rows).
D1 contains retained training decisions from both sources (1,118 games /
142,723 rows). D2 contains complete training decisions from both sources
(1,118 games / 431,470 rows). Original data and the existing derived cache stay
immutable. Validation and test games never enter any training distribution.
The W0 complete streams retain the original bounded game/decision shuffle;
retained D1 and weighted streams sample with replacement from their specified
probabilities. Counts and actual consumption are recorded rather than assuming
the same number of epochs or unique decisions.

W1 selects games uniformly, then decisions uniformly. W2 uses half uniform
sampling and half equal setup/middle/endgame pools. W3 uses the plan's early/late
policy curve and uniform (V0) or late-weighted (V1) value loss. Actor progress is
the zero-based decision ordinal divided by that actor's final ordinal, including
forced moves. It exists only in the sampler. Policy weights average one over
eligible training rows; V1 weights average one over all training rows.
W4 mixes half uniform with half family-balanced sampling. Water filling caps
the extra stream at 9x uniform, making the resulting mixture at most 5x uniform.
W2/W4 value weights are exactly `1 / (N * q)` and are applied once, without
sample-dependent re-normalization. Forced moves contribute only to value.
[phase6-sampling-audit.json](phase6-sampling-audit.json) records the exact
probability hashes, expected phase/family proportions and correction checks.

Each checkpoint plays both opponents in pure policy and optimized 500 ms search.
The prior 10-game smoke / 50-game total schedule is retained per opponent/mode;
smoke games are included in the total. All four opponent/mode smoke runs precede
extensions for that checkpoint. The unchanged futility rule can stop individual
runs; failures stop the queue for investigation. Cutoff/timing failures remain
reported and ineligible rather than silently changing the rules. Random results
are competence diagnostics; JSettlers search results determine recipe ranking.

Both baseline architectures are validated. Exact phase-5 D2/W0 training controls
are reused with checkpoint checksums; newly configured benchmark games are
measured independently. Then uniform D2/W0 plus the strongest feasible
non-uniform/coverage challenger are confirmed on both architectures with seeds
11, 23 and 47. Ties use the versioned recipe order. The conservative control
retains uniform unless the challenger's paired hierarchical 95% search
improvement interval is above zero on both architectures, with every required
run eligible. If no challenger is feasible, the queue stops for investigation.
No installed default changes automatically, and final evaluation seeds remain
reserved. The 50-game samples can produce an inconclusive result.

Run or resume serially with:

```sh
.conda/bin/python -m agent_apis.neural.phase6
```

Pinned MPS training needs accelerator access outside the macOS sandbox on this
machine; there is no silent CPU fallback. A single exclusive benchmark lock
prevents overlapping training/timing jobs. Training resumes optimizer, sampling,
augmentation and device RNG state. Validation resumes exact checksummed game
prefixes through the existing quick benchmark path. Changed sources,
checkpoints, provider identities or logs cause an explicit stop.

Live status: `runs/catan-ai/phase6-progress.json`.
Live log: `runs/catan-ai/phase6.log`.
Reviewable interim tables: [PHASE6_PROGRESS.md](PHASE6_PROGRESS.md).
The final `PHASE6_RESULTS.md` and `.json` are written only on actual completion.
