# Training preferences

Use W4 for all new policy/value training from November 17, 2026 onward:
half uniform and half capped action-family balancing, with inverse-probability
value-loss correction applied exactly once. Preserve historical protocols and
results with their original weighting. Friendly robber protects players with two or fewer VISIBLE victory points;
hidden VP cards must never remove protection. It is the default for new
games, training corpora and evaluations. Keep corpus splits at whole-game level.

For future combined model-search collectors, use the measured native CPU
workflow as the starting point: 16 total search workers, initially nine
model-model and seven learner-JSettlers, with 128 simulations per searched
decision and one inference thread per worker. Enforce 1,000 accepted completed
games of each kind before a 2,000-game update; adjust new dispatches as quotas
fill. The current Python/GPU hybrid did not improve sustained throughput.
Pursue CPU+GPU utilization by batching inference from native Rust search;
keep the existing GPU queue experimental. See
rust-catan-testing/hybrid-throughput-v2/RESULTS.md and
rust-catan-testing/hybrid-throughput-v2/FUTURE_COLLECTION.md.
These measurements use the current CNN; benchmark sparse-model workloads
separately. This is guidance for future combined schedulers, not an automatic
migration or training launch.

For the pinned fresh Python collector, retain 12 game workers with one PyTorch
CPU thread each and local CPU inference.
Keep the shared GPU inference queue experimental; do not use 32 workers by default.
The current workflow is
`python -m agent_apis.neural.parallel_cohorts --hours 3 --workers 12`.
The pinned fresh W4 trainer and its optimizer device remain unchanged.
The user retired eight-game iterations: require 2,000 completed games before
each update cohort. XD-CNN starts from entirely new random weights. First build
the new compact friendly expert corpus; then use large self-play cohorts.
Preserve every game's compressed actions/chance events and checksums; derive a
seeded sample of positions in RAM and train W4 minibatches. Select validation
checkpoints, preserve historical runs, and do not deploy models automatically.
See experiments/catan-ai/PARALLEL_CPU_COLLECTION.md, FRESH_XDCNN.md and the
latest experiment-plan status.

Current corrected-rule run: runs/catan-ai/fresh-xdcnn-visible/. Audit and reuse
only exactly replayable old journals under visible-point legality, preserving
whole-game splits, and replace invalid games before training. Human-versus-bot
browser views must stay on the human seat and never expose bot private cards.
See experiments/catan-ai/VISIBLE_ROBBER_PRIVACY_FIX.md.
