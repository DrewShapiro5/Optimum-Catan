# Three-hour compact W4 iteration window

Authorized November 17, 2026: continue self-play iterations for three hours,
minimize generated-game storage, and schedule a timed research review to decide
whether continuing is worthwhile. W4 and friendly robber remain mandatory.

Run or resume a new authorized window:
`.conda/bin/python -m agent_apis.neural.compact_iterations --hours 3`.
Progress: `runs/catan-ai/compact-w4-progress.json`.
Log: `runs/catan-ai/compact-w4.log`.
Protocol: [compact-w4.json](compact-w4.json).

Each iteration collects eight completed games using the current/historical/
JSettlers 50/25/25 mixture and 128 new simulations per non-forced learned move.
Then continue from the current experimental checkpoint for 1,000 W4 updates,
mixing 50% the friendly JSettlers training split with 50% cumulative completed
self-play examples. This includes the preserved original 100 games and newly
completed compact journals. The same capped W4 joint-family sampling and
inverse-probability value correction apply. No original corpus is regenerated.

Generated collection and evaluation games store only gzip-compressed action IDs,
actual public/private chance event records, sparse root visit counts where
searched, seeds/rules, model/source references, result and checksums. There are
no per-move board snapshots, feature arrays, legal masks or full-tree telemetry
on disk. The initial game is recreated from isolated board/simulator seeds and
the pinned engine, and every action/chance result and final state/RNG checksum
is replay-verified. Privileged event identities are audit/replay data, never
fed directly to a model; actor-redacted features are derived before each
replayed action. Completed-game outcomes are attached after replay. Derived
training arrays stay in RAM. Original existing corpora/shards remain unchanged.
An immutable source.zip is saved once per run; model/optimizer checkpoints are
necessary persistent training state, separate from game journals.

Compressed partial journals are refreshed every ten actions. Interrupted games
and cutoffs have no value or policy training rows. An uncommitted game may
restart at its seed after a crash/review; no committed game is regenerated.
An orderly deadline interruption saves the exact replayable event prefix.
Abrupt-power-loss partial journals also replay their saved chance-event prefix.

For bounded feedback, after each iteration run five tuning board/seat pairs
for the parent and candidate against JSettlers in noise-free **pure policy**.
These fast comparisons help guide the review but do not establish a gain at
fixed search budget. Record paired uncertainty and friendly-robber unweighted
validation diagnostics. No playing checkpoint is deployed automatically.

The three-hour deadline is checked between game actions, before phases, and at
training guards. It pauses at the next safe boundary (normally seconds), keeps
the last atomic optimizer/RNG checkpoint, releases the benchmark lock, and
marks progress awaiting-review. It does not silently extend the run. The timed
review should inspect completed iterations, validation trajectories, paired
policy results, replay integrity, storage growth, throughput and cutoffs/errors,
then explain whether another bounded window is justified. A partial round can
resume from committed collection, training, and evaluation artifacts.

Keep the computer awake and Codex running for the scheduled local review.
