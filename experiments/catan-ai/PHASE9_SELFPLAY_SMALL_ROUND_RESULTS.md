# Small first self-play round results

Complete November 16, 2026. No background queue or automatic deployment.

Collected 100 completed games (27 reused at 512 simulations; 73 new at 128),
yielding 28,299 searched training examples. One collection cutoff was excluded.
Completed 1,000 fine-tuning updates in 208 seconds and all 40 tuning games.

| Evaluation | Wins | Losses | Cutoffs |
| --- | ---: | ---: | ---: |
| Parent vs JSettlers | 2 | 8 | 0 |
| Trained candidate vs JSettlers | 3 | 7 | 0 |
| Trained candidate vs parent | 4 | 5 | 1 |
| Candidate with historical simulated opponent vs JSettlers | 2 | 8 | 0 |

Candidate-minus-parent paired score difference against JSettlers: +10 percentage
points, 95% paired bootstrap interval [-20, +40] points. This small, single-seed
pilot does not establish improvement. All evaluation games had zero provider
errors and zero illegal actions. The head-to-head cutoff reached 1,000 turns.

Common original D2 validation Brier score worsened from 0.1390 to 0.1678; lower
is better. Candidate warm non-forced p95 latency against JSettlers was 568 ms,
above the 525 ms acceptance limit. Promotion eligibility is false. The existing
default checkpoint is unchanged, and final evaluation seeds remain unused.

Updated weights, optimizer/resume state, all game records and diagnostics remain
under `runs/catan-ai/selfplay-small-round/`. Machine-readable results are
`runs/catan-ai/selfplay-small-round/iteration-1/results.json`.
