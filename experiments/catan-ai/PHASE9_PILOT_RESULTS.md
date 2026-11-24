# Phase 9 teacher-aggregation pilot results

Pilot complete. The full phase-nine improvement program remains incomplete. No model was installed and no larger queue is running.

| Item | Result |
| --- | --- |
| Learner / opponent | XD-CNN D2/W0 seed11 / JSettlers SMART |
| Collection | 20 completed games; 4 learner wins; no cutoffs |
| New training examples | 4,213 |
| Teacher queries | 3,416; 27 rejected/excluded (0.8%) |
| Fine-tuning | 1,000 updates, 50% original D2 / 50% new learner-state teacher labels |
| Selected checkpoint | Parent, update 0; weights verified identical |
| Ten-game tuning check | 2/10 wins; parent reference 3/10 |
| Errors / cutoffs | 0 / 0 |

| Common D2 validation metric | Parent | Final updated checkpoint |
| --- | ---: | ---: |
| Policy NLL | 0.6382 | 0.6759 |
| Value BCE | 0.4523 | 0.5097 |
| Combined selection loss | 1.0904 | 1.1856 |
| Brier | 0.1390 | 0.1580 |

The predeclared common-validation criterion selected the parent because none of the four fine-tuned validation checkpoints improved it. The selected weights are exactly the same as the parent, so the ten-game result is repeated-run variation, **not evidence of a learning gain or loss**. Playing strength of final.pt has not been measured.

The native teacher check, 16 regression tests, and interrupted-versus-uninterrupted MPS training integration passed; the integration ended with exactly equal final weights. Collection labels passed schema and legality checks; failed native reconstruction/translation queries were excluded. Only actual completed-game outcomes supervise value.

All data, updated checkpoints, validation history, and resume state are preserved under `runs/catan-ai/phase9-pilot/`. This small single-seed pilot does not establish that teacher aggregation cannot improve play. It shows that this 50/50, 1,000-update recipe did not improve the fixed supervised validation criterion. Final evaluation seeds remain unused.
