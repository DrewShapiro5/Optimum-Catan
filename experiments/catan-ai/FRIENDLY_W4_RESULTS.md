# Friendly-robber W4 continuation results

Completed November 17, 2026. All background work finished. W4 remains the user-selected training default; no playing checkpoint was promoted.

Generated 250 complete native JSettlers SMART-versus-SMART friendly-robber games,
with 97,383 recorded decisions. Whole-game splits are 226 train, 12 validation,
and 12 test games. Only the training split supplied expert gradients; validation
remained unweighted and test remained unused. Per-game replay and checksums passed.

Continued the prior self-play checkpoint for 1,000 W4 updates in 81 seconds,
using 50% new expert training data and 50% the existing 100-game self-play data.
Inverse-probability value correction was applied once within each source.

| Evaluation condition | Wins | Losses | Cutoffs |
| --- | ---: | ---: | ---: |
| Parent vs JSettlers | 3 | 7 | 0 |
| W4 candidate vs JSettlers | 4 | 6 | 0 |
| W4 candidate vs parent | 5 | 5 | 0 |
| W4 candidate with historical simulated opponent vs JSettlers | 5 | 5 | 0 |

All 40 tuning games had zero errors and zero illegal actions. Candidate-minus-
parent paired score difference against JSettlers was +10 percentage points,
95% interval [-20, +40] points. Neither it nor the opponent-model ablation
establishes a strength gain from this small single-seed experiment.

On the friendly-robber validation split, policy NLL improved 0.871 to 0.762.
Value Brier score improved early (0.193 at step 0 to 0.145 at step 250), then
rose to 0.201 at step 1,000. Lower is better. This is overall probability error,
not an isolated calibration measurement. Only final-update playing strength was
measured; the early validation point is not evidence of a stronger checkpoint.
Candidate p95 warm non-forced latency vs JSettlers was 559 ms, above the 525 ms
gate. Final promotion eligibility is false. Default deployment is unchanged.

Artifacts: `runs/catan-ai/friendly-w4/`; final comparison:
`runs/catan-ai/friendly-w4/iteration-1/results.json`.
