# Phase 5 quick optimized-search evaluation

10-game smoke followed by at most 50 games total, including smoke. All four seed11 checkpoints use 500 ms optimized search. All smoke runs precede extensions. Existing policy references are reused, not replayed. No training seeds or final evaluation games are launched.

| Model | Search games | Wins | Losses | Cutoffs | Errors | Match score | Matched policy comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H-GNN | 40 | 1 | 39 | 0 | 0 | 0.025 | +0.000 over 40 games |
| TRANSFORMER | 50 | 9 | 41 | 0 | 0 | 0.180 | +0.175 over 40 games |
| XD-CNN | 50 | 12 | 38 | 0 | 0 | 0.240 | +0.000 over 50 games |
| MLP | 40 | 0 | 40 | 0 | 0 | 0.000 | +0.000 over 40 games |

These are exploratory single-seed tuning results, with broad uncertainty. Futility can end a run before 50 games. The existing statistical rule needs more than five pairs to establish futility; zero wins in a 10-game smoke alone is not proof of hopelessness. Full logs, paired bootstrap intervals and depth/simulation telemetry are linked from [PHASE5_QUICK_RESULTS.json](PHASE5_QUICK_RESULTS.json).
