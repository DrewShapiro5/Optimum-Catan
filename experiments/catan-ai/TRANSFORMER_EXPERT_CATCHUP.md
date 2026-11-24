# Transformer expert catch-up — November 18, 2026

User requested simultaneous Transformer training on the CNN's new expert data
and catch-up on earlier cohorts. Continue historical seed-11 TRANSFORMER best
checkpoint (phase5, selected update 18000), preserving its historical W0 result.
All new optimization uses W4 and corrected visible-point friendly robber.

Runner: `python -m agent_apis.neural.transformer_catchup`.
Progress: `runs/catan-ai/transformer-expert-catchup-visible/progress.json`.
Log: `runs/catan-ai/transformer-expert-catchup-visible.log`.

Order: corrected fresh CNN expert manifest, then additional expert cohorts
001, 002, 003, etc. Only complete 2,000-game manifests are accepted, with
1,800/100/100 whole-game splits and no duplicate/cross-split journal paths.
Checksums are verified for all journal references. Train/validation features
are derived by replay with visible-point legality and actual chance checks;
test features are never loaded. At most 64 seeded positions/game enter RAM.

Use the unchanged pinned W4 trainer and MPS optimizer, 2,000 updates/cohort,
256-position batches, half uniform/half capped-family W4 sampling and value
correction once. Both independent trainer source halves draw from that cohort's
expert pool. Keep best validation checkpoint (including update zero), final
weights, optimizer and RNG recovery state. Each cohort has separate output.

This is continuation of historical Transformer weights, so comparison to the
fresh CNN is not a controlled architecture experiment. It uses the same expert
games and split membership, with the current expert protocol's sampling seed;
the original fresh corpus samples need not match the CNN pretraining samples.

The Transformer is a separate process sharing MPS with CNN training. No extra
game collectors or inference queue is created. The existing CNN controller and
pinned sources remain intact. This run adopts the CNN window deadline (November
18, 2026, 15:32 MST); it waits for new complete manifests until that deadline.
No automatic model deployment or window extension. Training checkpoint writes
and journal replay can finish slightly beyond the boundary before pausing.

Validation: ten relevant unit tests passed; historical checkpoint hash/schema
and architecture loaded successfully. New runner source is archived in its own
output folder. A future strength evaluation is needed before claiming improved
JSettlers performance.
