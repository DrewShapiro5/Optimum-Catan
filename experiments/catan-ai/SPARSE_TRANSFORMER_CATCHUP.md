# Sparse Transformer continuation — November 18, 2026

User authorized another three-hour XD-CNN expert-cohort window with simultaneous
sparse Transformer training and catch-up on previous cohorts. CNN resumes its
existing unchanged controller, with 12 single-thread native workers. The new
sparse model shares the MPS GPU for training; dense Transformer remains paused.

Runner: `.conda/bin/python -m agent_apis.neural.sparse_catchup`.
Progress: `runs/catan-ai/sparse-transformer-expert-catchup-visible/progress.json`.
Log: `runs/catan-ai/sparse-transformer-expert-catchup-visible.log`.

Starts from the selected typed-Exphormer ablation checkpoint (update 1750),
which has already seen cohort 8. Replays the original corrected 2,000-game
expert corpus, then numbered expert cohorts in order, including cohort 8.
This is continued training with repeated exposures, not a fresh architecture
benchmark or a claim that previously seen games are novel data.

Each complete cohort has 1,800 train, 100 validation and 100 test games. Preserve
whole-game splits; verify journal hashes; replay action/chance/final checksums
when deriving train/validation samples. Test features are never loaded. Derive
at most 64 seeded positions per game into RAM. Preserve existing compressed
journals and all historical runs.

Separate sparse trainer copies the pinned W4 update loop without modifying it:
2,000 AdamW updates/cohort, MPS optimizer, batch 256, W4 half uniform/half capped
family sampling, inverse-probability outcome correction exactly once. Add a
light 0.001 uniform-position squared-value-logit penalty, estimated using the
same single importance weight on the separate regularization term. Validate
every 250 updates including zero; continue earliest best combined validation
checkpoint. Preserve best/final weights and atomic optimizer/RNG recovery.

The architecture retains typed local/ownership/port links, expander overlay and
four global connector tokens. Experimental checkpoint metadata has its own
format/model marker and sparse-implementation checksum; standard model loader
will not silently interpret these weights as a dense Transformer.

Both processes adopt the CNN's new three-hour deadline, 23:48 MST November 18.
Pause safely at the boundary, preserving incomplete cohorts and optimizer state.
No automatic model deployment. Smoke verification performed a genuine MPS W4
update with the penalty and reloaded the resulting sparse checkpoint. Fourteen
relevant tests passed. Production source archive records the new runner.
