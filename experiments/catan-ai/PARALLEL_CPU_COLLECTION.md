# Twelve-worker CPU collection — November 18, 2026

Selected explicitly by the user after CPU/shared-GPU queue benchmarking: use
12 game workers, one PyTorch CPU thread per worker, local CPU model inference.
The 32-worker GPU experiment remains optional and is not the default.

Run/resume a bounded window:
`.conda/bin/python -m agent_apis.neural.parallel_cohorts --hours 3 --workers 12`.
Progress: `runs/catan-ai/fresh-xdcnn-visible-progress.json`.
Log: `runs/catan-ai/fresh-xdcnn-visible-parallel.log`.
Underlying rules, W4 trainer, current model lineage and cohort sizes are the
pinned fresh workflow, not the retired eight-game compact iterations.

The current corrected-rule lineage retains its 2,000-game expert corpus,
completed fresh XD-CNN pretraining, and all 13 committed self-play games.
Require 2,000 completed self-play games per update cohort. The optimizer retains
its existing Apple GPU configuration; game/search inference uses CPU. Sample at
most 64 positions per completed game into RAM, apply W4 and its value correction
once, preserve whole-game expert splits, select validation checkpoints and never
automatically deploy playing weights.

`parallel_cohorts.py` wraps the unchanged pinned serial engine/trainer. Each
worker owns its models, game, RNG streams, search trees and native Java bridge.
The original source digest and archive are preserved. A separate content-pinned
execution manifest/source archive under `fresh-xdcnn-visible/execution/` records
the new scheduler; new journals link this manifest. Committed old game bytes
are not rewritten.

A durable `round-NNN/parallel-plan.json` assigns completed-game slots, attempt
IDs, seeds, opponent kinds and historical checkpoint references before dispatch.
Opponent kind and learner seat are determined by the slot; a cutoff retries the
same slot with a fresh attempt seed. Consequently out-of-order completion and
cutoffs do not distort the 50/25/25 current/historical/JSettlers mixture. The
first 13 committed serial games are replay-verified and imported into this plan.
Resume verifies every committed planned journal and reuses pending assignments.
Unplanned or changed games/lineage fail validation.

Only the parent writes global progress and the plan. Workers write individual
active-game progress and gzip action/chance/sparse-visit journals. No game-state
snapshots or features are saved. Because the pinned engine restarts an unfinished
seed, its previous compressed prefix is retained under `partial-history/` before
restart; these prefixes and cutoffs never supply training outcomes. The shared
deadline or stop event pauses every active game at a safe action boundary and
retains replay-verified prefixes. Workers also stop if their parent exits.

The full run pauses after at most three hours; no automatic extension or model
deployment is created. Stop the parent with SIGTERM for a graceful pause.
Scheduler source changes are rejected during execution. Further changes require
an explicit execution-manifest/plan migration rather than rewriting old sources.

Validation before first launch: 12 scheduler/fresh/compact tests passed, including
out-of-order completions, same-family cutoff retries, stable resume seeds,
interrupted slots, exact cohort size, bounded attempts, replay and W4 correction.
A separate twelve-worker smoke run exercised interruption and restart twice with
four simulations per move; all prefixes replayed, original partials were
archived, all workers reported one CPU thread/CPU inference, and real training
progress stayed unchanged. Smoke artifacts are excluded from training under
`runs/catan-ai/parallel-collector-smoke/`. Production retains 128 simulations.
