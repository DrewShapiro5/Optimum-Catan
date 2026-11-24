# Friendly-robber expert corpus and W4 self-play continuation

Authorized November 17, 2026: W4 is the default for all new training. Generate
new native JSettlers SMART-versus-SMART games using friendly robber first, then
continue training. No historical corpus, protocol, or result is overwritten.

Run/resume: `.conda/bin/python -m agent_apis.neural.friendly_w4`.
Progress: `runs/catan-ai/friendly-w4-progress.json`.
Log: `runs/catan-ai/friendly-w4.log`.
Protocol: [friendly-w4.json](friendly-w4.json).

Generate 250 completed two-player BASE/15-point/discard-nine games, matching
the earlier full-corpus size, with friendly robber explicitly enabled. Retain
every decision. Whole-game splits use 18 training, one validation and one test
slot per block of 20 completed games; assignment is independent of winners.
Fresh generation streams are checked against original corpus, tuning/final,
teacher pilot and previous self-play seeds. Cutoffs supply no outcome labels
and retry the same completion slot with a fresh seed; stop after 500 attempts.
Native API or legality failures stop for investigation without random fallback.

Store checksummed per-game schema-3 feature/legal-mask/native-action/outcome
shards under `data/jsettlers-friendly-250/`, with a manifest and separate
privileged initial/final states and action/chance replay logs. Every game's
actions and final state are independently replayed before commit; resource
inference is audited against engine hands. Model inputs are only the redacted
schema-3 features and legal masks. Outcomes are attached after completion from
each actor's perspective. An interrupted uncommitted game restarts; committed
games resume after checksum verification.

After all 250 games are committed, continue the last 1,000-update self-play
checkpoint (`selfplay-small-round/iteration-1/final.pt`) for 1,000 further MPS
updates. Batch 256 mixes 50% newly generated expert-training decisions and 50%
the existing 100 completed friendly-robber self-play games (28,299 examples).
Do not use old standard-robber expert decisions, validation, test, cutoffs or
evaluation games as training data. Preserve the existing self-play targets and
their 512/128-simulation provenance. No additional search collection is queued
for this bounded continuation. Validate every 250 updates on the new uniform,
friendly-robber expert validation split. Its test split remains unused.

W4 combines half uniform with half capped family-balanced sampling. It is
applied separately within the expert and self-play halves. For one-hot expert
targets it exactly matches the earlier W4 implementation: extra-stream cap
9x, resulting mixture cap 5x. For visit targets, family counts are fractional
target probability masses. Sample the joint row/action-family distribution,
using row marginal q_i and conditional target pi_ia*c_family/q_i, where
c_f = 0.5/N + 0.5*m_f/count_f. For hard targets the conditional target is
unchanged. This explicit extension avoids arbitrarily labeling a mixed visit
distribution by only its largest action. Apply value weights 1/(N*q_i) exactly
once, without sampled-batch weight normalization; policy follows the balanced
objective while value retains the uniform-per-source outcome objective.

Keep the prior supervised AdamW settings, D6 augmentation, 100-update warmup,
and restarted cosine schedule. Save complete optimizer-update boundaries every
100 updates, including optimizer, source/data digests, sampling, augmentation,
CPU and MPS RNGs. On an interrupted partial update, replay from the last atomic
committed checkpoint rather than saving inconsistent advanced RNG state.

Then run the same bounded four evaluation conditions as the prior small round,
10 games each, with friendly robber and noise-free 500-ms search. Compare to the
immediate parent checkpoint. Report paired uncertainty, timing, cutoffs, errors
and friendly-validation Brier change. Training-data weighting is now W4 by user
choice, independent of earlier conservative W0 selection. No automatic playing
checkpoint deployment, additional iteration or larger collection is authorized
by this bounded protocol. An exclusive benchmark lock serializes the workflow.
