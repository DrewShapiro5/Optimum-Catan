# Phase 6 recovery analysis — November 8, 2026

The original queue (PID 90673) survived the reported power interruption. It is active again; the existing caffeinate helper is present. No duplicate process was launched. Runtime sources are unchanged. All four training checkpoints, 29 completed benchmark records and their artifact checksums were verified. The active batch has 41 checksum-committed games, including an odd seat; it can resume that exact prefix if the process later exits.

| XD-CNN recipe | JSettlers policy | JSettlers search | Random policy | Random search |
| --- | --- | --- | --- | --- |
| Complete D2, uniform | 12 wins / 50 | 11 wins / 50 | 50 wins / 50 | 50 wins / 50 |
| Full-250 only D0 | 0 wins / 40, futility stop | 0 wins / 40, futility stop | 33 wins, 7 losses, 10 cutoffs / 50 | 37 wins, 5 losses, 8 cutoffs / 50 |
| Retained D1 | 2 wins / 10, smoke only | 3 wins / 10, smoke only | 47 wins, 2 losses, 1 cutoff / 50 | In progress: 37 wins, 2 losses, 2 cutoffs / 41 |

D0 fails the cutoff gates against random and stops for futility against JSettlers. The observed gameplay and held-out imitation/calibration metrics favor the broader complete-D2 control over D0. D1 has completed 20,000 training updates, but its JSettlers smoke results are too small to rank the recipe. Its one random-policy cutoff already exceeds the 1% gate. All completed benchmark records have zero errors.

Transformer baseline results remain 1/40 policy wins (futility stopped) and 9/50 search wins against JSettlers. Against random, its 6/50 policy cutoffs and 2/50 search cutoffs fail promotion gates. No statistical architecture or recipe winner is established.

Continue the existing queue through the remaining D1 evaluation, then W1, W2, W3/V0, W3/V1 and W4 screening. Preserve cutoffs, timing overruns and partial logs. If a decision spans a machine suspension, its measured elapsed time can include that suspension; any resulting timing failure requires investigation rather than silently removing the measurement. No timing gate or decision log was changed during recovery.

Final evaluation seeds remain unused. Final recipe confirmation and selection remain pending.
