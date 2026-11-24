# Mixed native cohorts — November 19, 2026

Current user-authorized continuation replaces expert-only collection for future
cohorts. Run/resume: `.conda/bin/python -m agent_apis.neural.mixed_native_cohorts --hours 3`.
Live progress: `runs/catan-ai/mixed-native-visible/PROGRESS.md` and `progress.json`.
Log: `runs/catan-ai/mixed-native-visible.log`. Initial window ends 07:00 MST.

Each update cohort requires exactly 2,000 completed games: 1,000 native Java
JSettlers-vs-JSettlers demonstrations and 1,000 Rust-native neural self-play
games. Independently choose each actual seat's CNN or sparse-Transformer model
with probability one half. Thus expected matchups are 25% CNN-CNN, 25%
sparse-sparse, 50% cross-model; both cross seat orders occur. Fix matchups and
splits by completed-game slot, preserving them across cutoff retries. Fresh
attempt seeds exclude reserved evaluation seeds. At each cohort freeze the
latest validation-selected checkpoints of both models until collection ends.

The first mixed cohort reuses the first 1,000 completed slots from unfinished
expert cohort 12, with original 900 train / 50 validation / 50 test membership.
The remaining old expert games and paused runs remain intact. Subsequent expert
blocks use twelve single-thread local workers. Neural games use sixteen Rust
threads, three one-thread ONNX CPU sessions each (CNN, sparse Transformer,
frozen simulated-opponent MLP). No Python model/search/game call occurs inside
the Rust worker. Python prepares initial inputs, exports and audits models,
independently replays native journals and performs W4 training.

Native budget: 128 NEW simulations per non-forced decision (the native CLI's
`--simulations 128` setting), with the existing 128-action horizon, PUCT 1.5,
subtree reuse, root noise and action temperatures. This is a per-decision search
setting, not a shared allowance across multiple decisions in an entire game
turn. Actual opponents are randomly chosen neural models; simulated opponents
inside search remain the pinned frozen MLP. Visible-point friendly robber,
two players, 15 VP and discard limit nine remain in force.

Each source has whole-game 900/50/50 splits, yielding 1,800/100/100 overall.
Save every actual action/chance and sparse root visits in compressed native
journals, plus partial prefixes every 25 actions and live per-game progress.
Native replay checks actual RNG and final state. Before accepting for training,
Python independently verifies legal action IDs, chance/dice events, canonical
final state and state/RNG checksum, then preserves a compact compatible journal
referencing its immutable raw native journal and fixed model checkpoints.
Cutoffs and interrupted games never supply value outcomes and are replaced;
finished games are reused on resume. Original native benchmark directories are
unchanged; new full-game code lives in `rust-catan-testing/mixed-native/`.

Both models train on each accepted complete cohort: half expert-source and
half search-source samples per W4 minibatch, independently corrected exactly
once for value outcomes within each source. Derive at most 64 seeded positions
per game into RAM. CNN uses its pinned W4 trainer unchanged. Sparse model retains
typed Exphormer-style local/expander/global-connector attention and the 0.001
value-logit penalty. Train 2,000 AdamW updates/model, batch 256, existing MPS
optimizer. Train models sequentially to avoid competing optimizers. Validate on
the mixed 100-game validation split, including update zero; use earliest best
combined validation loss and preserve final and optimizer/RNG checkpoints.
Test features never enter training or selection. No automatic deployment.

Bounded three-hour windows stop at safe native action/Python update boundaries,
with prefixes and source archives preserved. SIGTERM to the Python parent sets
a native stop file and stops expert workers at safe boundaries. Restart clears
the stop marker while preserving prior attempts and archived partials.

Validation before launch: 15 relevant tests passed, including exact-mixture
training gate, split counts, matchup coverage, native visit-target replay and W4
sampling. Four complete Rust games (cross-model and sparse self-play) replayed
exactly under original Python rules. Both ONNX exports matched 270 PyTorch
observations each. Both models passed finite MPS backward checks on combined
expert/search targets without taking an optimizer update. Sparse/CNN self-play
mechanisms retain historical native benchmark parity evidence; this is a new
complete-game training workflow, so rates are measured from actual games rather
than inferred from continuation benchmarks.
