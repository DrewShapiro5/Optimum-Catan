# Smaller first self-play round

The smaller round was started at the user’s request on November 14, 2026.
Its live progress and log are `runs/catan-ai/selfplay-small-round-progress.json`
and `runs/catan-ai/selfplay-small-round.log`.

The user paused the full queue on November 13, 2026. Its process was stopped
and the exclusive benchmark lock was verified released. The original progress
record is marked paused. All 27 completed games and 5,908 searched examples
remain unchanged. The unfinished 28th game's telemetry is preserved, but it
has no completed outcome and contributes no training labels.

The prepared pilot uses [phase9-selfplay-small-round.json](phase9-selfplay-small-round.json):

- One round with 100 completed games total: 27 existing plus 73 new.
- Existing games retain their 512-simulation provenance. New learned decisions
  use 128 simulations, with the same exploration, subtree reuse and rule checks.
- Friendly robber remains enabled. Continue the existing 50/25/25 completed-game
  current/historical/JSettlers mixture slots and deterministic seed sequence;
  committed games are not replayed. Restart the unfinished slot on its original
  seed, separately from its preserved incomplete telemetry.
- Train 1,000 updates from the original XD-CNN D2/W0 seed11 parent, sampling
  50% original expert decisions and 50% uniformly sampled searched decisions
  across the entire 100-game cohort. Use 100-update warmup, validation every
  250 updates and resumable checkpoint saves every 100 updates.
- Use five paired tuning boards (10 games) for each of the four existing
  evaluation conditions. This is exploratory pilot evidence. No automatic
  deployment, second iteration, or scale-up.

The immutable import manifest is
`runs/catan-ai/selfplay-small-round-seed-games.json`. Every source game JSON,
training shard, original source digest, parent checkpoint, and original search
budget is checked before importing. Original files are neither rewritten nor
relabeled. New artifacts have a separate directory and source/protocol identity.

Start/resume when requested:
`.conda/bin/python -m agent_apis.neural.selfplay_small_round`.
Check readiness without launching:
`.conda/bin/python -m agent_apis.neural.selfplay_small_round --preflight`.

Progress: `runs/catan-ai/selfplay-small-round-progress.json`.
Artifacts: `runs/catan-ai/selfplay-small-round/`.
The full queue stays paused; do not resume its original entry point for this
pilot. See [the original procedure](PHASE9_SELFPLAY.md) for unchanged search,
information-boundary, loss, and evaluation definitions.
