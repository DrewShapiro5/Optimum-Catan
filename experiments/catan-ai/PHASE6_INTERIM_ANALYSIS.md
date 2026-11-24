# Phase 6 interim analysis

Snapshot: 2026-11-07T18:58:02.259789+00:00

The original serial queue remains active. No process was restarted and no completed games were replayed. Final evaluation seeds remain unused.

| Encoder | Training data | Mode | JSettlers wins/games | Cutoffs | Status |
| --- | --- | --- | ---: | ---: | --- |
| XD-CNN | D2/W0 | policy | 12/50 | 0 | completed |
| XD-CNN | D2/W0 | search | 11/50 | 0 | completed |
| TRANSFORMER | D2/W0 | policy | 1/40 | 0 | stopped-early |
| TRANSFORMER | D2/W0 | search | 9/50 | 0 | completed |

XD-CNN search minus policy is -2.0%, with paired board/seat 95% interval [-14.0%, +10.0%]. This does not demonstrate a search improvement.

Transformer search improves descriptively over its weak policy, but the policy was stopped under the futility rule. A fixed-sample improvement interval is inappropriate after that stopping decision. Neither model has demonstrated superiority over JSettlers.

Against random player, XD-CNN wins all 50 games in each mode, without cutoffs. Transformer policy has 41 wins, 3 losses and 6 cutoffs; search has 46 wins, 2 losses and 2 cutoffs. Every cutoff reached 1,000 turns. Both Transformer modes fail the predeclared 1% cutoff gate. These are behavioral completion concerns requiring investigation before promotion; they are not counted as wins. All completed baseline runs report zero errors and pass their timing gate.

Full-250-only XD-CNN (D0/W0) completed all 20,000 updates; the common validation criterion selects update 2,000. Its held-out policy NLL is 0.8597 versus 0.6602 for D2/W0, and Brier score is 0.1615 versus 0.1551. Lower is better for both. This preliminary imitation/calibration evidence favors broader D2 coverage; gameplay validation is still in progress and determines recipe ranking.

The remaining D1 and D2 weighting runs, challenger selection and three-seed checks remain pending. Continue the existing queue under its original protocol. No recipe or installed default has been changed based on this interim look.
