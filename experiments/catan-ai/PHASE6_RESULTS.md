# Phase 6 results — XD-CNN-only scope

On November 11, 2026, the user retired Transformer and requested early stopping. All six XD-CNN model/recipe/seed combinations were already complete, including both modes and both opponents. The active Transformer evaluation was terminated; completed games/checkpoints and its partial logs remain preserved. The original two-architecture protocol is incomplete. Final evaluation seeds remain unused.

| Recipe | Seed | Mode | JSettlers wins / 50 | Cutoffs | JSettlers eligible |
| --- | ---: | --- | ---: | ---: | --- |
| D2-W0 | 11 | policy | 12 | 0 | True |
| D2-W0 | 23 | policy | 5 | 2 | False |
| D2-W0 | 47 | policy | 9 | 0 | True |
| D2-W0 | 11 | search | 11 | 0 | True |
| D2-W0 | 23 | search | 9 | 0 | True |
| D2-W0 | 47 | search | 10 | 0 | True |
| D2-W4 | 11 | policy | 11 | 0 | True |
| D2-W4 | 23 | policy | 17 | 0 | True |
| D2-W4 | 47 | policy | 14 | 0 | True |
| D2-W4 | 11 | search | 18 | 0 | True |
| D2-W4 | 23 | search | 20 | 0 | True |
| D2-W4 | 47 | search | 14 | 0 | True |

| Recipe | Mode | Wins / 150 | Match score | Hierarchical 95% CI | All opponent gates |
| --- | --- | ---: | ---: | --- | --- |
| D2-W0 | policy | 26 | 0.180 | [0.090, 0.280] | False |
| D2-W0 | search | 30 | 0.200 | [0.120, 0.280] | True |
| D2-W4 | policy | 42 | 0.280 | [0.180, 0.400] | True |
| D2-W4 | search | 52 | 0.347 | [0.227, 0.473] | True |

W4 minus W0 search match score: +14.7%, paired hierarchical 95% interval [+0.0%, +30.0%]. Selected recipe for subsequent XD-CNN work: **D2-W0**.

Intervals resample training seeds and shared board/seat pairs. Results are tuning comparisons after recipe screening; they do not establish superiority over JSettlers or supply reserved final-set confirmation. Search must separately demonstrate improvement over policy before selecting a deployed search default. No installed checkpoint default was changed by this report.
