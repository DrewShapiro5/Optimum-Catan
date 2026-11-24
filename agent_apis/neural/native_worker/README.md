# Production Rust game worker

The worker runs Catan rules, observation encoding, development-card posterior sampling, chance enumeration, MCTS, subtree reuse, action sampling and ONNX CPU inference natively. Python prepares the initial board and verified model exports, then receives batches of actions to verify and persist. No Python call is made from inference or search. This package is independent of `rust-catan-testing/`; the experimental sources/results remain intact.

## Production cohort collection

```sh
python -m agent_apis.neural.parallel_cohorts --hours 3 --workers 12 --engine rust
```

This selects Rust for current/historical neural games. JSettlers continues through the existing Python/Java collector, preserving the opponent mix and completed-game slot/retry logic. `--engine python` retains the old backend and remains the default. The 12-worker default, one inference thread per game, original W4 trainer/MPS optimizer, validation selection, 2,000-completed-game update requirement, whole-game splits and model deployment policy are unchanged. Adding this backend does not launch a training run.

Execution-only scheduler migrations retain the original plan identity/assignments and append an execution-manifest revision. Old execution manifests must still match their checksums and the pinned engine/trainer source hashes. New games record their actual execution reference; historical journal bytes are not rewritten. Rule/model/cohort-size changes cannot be disguised as a backend migration.

## Standalone game collection

Install the `native` extra in the project environment and build with a Rust toolchain:

```sh
python -m pip install '.[native]'
agent_apis/neural/native_worker/build.sh
python -m agent_apis.neural.native_selfplay \
  --checkpoint runs/catan-ai/sparse-transformer-expert-catchup-visible/cohort-011/iteration-1/best.pt \
  --games 12 --workers 12 --simulations 128 \
  --output runs/catan-ai/native-collection
```

The checkpoint path is explicit. Both ordinary verified checkpoints (including XD-CNN/MLP) and typed Exphormer are supported. An optional `--opponent PATH` selects a different real-seat neural checkpoint and records learner targets only for the candidate seat; without it both seats use the candidate and both supply targets. The simulated opponent is the pinned frozen MLP, independently of real-seat opponent selection.

`--time-ms MS` is mutually exclusive with `--simulations N`; count defaults to 128. Time is a soft per-decision total-search budget, including root preparation. Ongoing work is not interruptible; tiny budgets may produce zero simulations and no training target for that decision. Cohort training uses its fixed simulation protocol. Full games stop at a winner or the existing 1,000-turn/10,000-action cutoffs. Diagnostic `--max-turns`/`--max-actions` can shorten a run, but cutoff/interrupted games cannot supply training outcomes.

Model exports are content-addressed and checksum-verified under `runs/catan-ai/native-assets/`. Architecture/schema/implementation hashes are checked, including sparse checkpoint markers; typed sparse attention is exported faithfully as dense masked attention. Export preserves the caller's CPU RNG and thread settings. Each native game process caches its models, with one intra-op and one inter-op thread per session. Export/build locks prevent concurrent cache corruption. Source and binary checksums trigger rebuilding when necessary.

The bundled ONNX Runtime 1.32.0 library and NumPy distribution archive are **macOS ARM64 artifacts**, validated on the current M4 Pro host. Their licenses/notices are retained in `runtime/` and `native/`. Other platforms require a matching ONNX Runtime (`ORT_DYLIB_PATH`) and a platform build of the standalone NumPy random distribution archive. The static archive contains distribution kernels; Rust supplies compatible RNG callbacks, with no embedded Python runtime.

## Python recording and training interface

```python
from agent_apis.neural.native_selfplay import load_journal
from agent_apis.neural.compact_iterations import MemoryData
import numpy as np

record = load_journal("runs/catan-ai/native-collection/game-00000.json.gz")
arrays = load_journal(
    "runs/catan-ai/native-collection/game-00000.json.gz",
    derive=True, limit=64, seed=11,
)
data = MemoryData([arrays], references=["game-00000"])
batch = data.sample(np.random.default_rng(11), 256)
```

The `.json.gz` format is the existing compact journal, with an additional `format="catan-compact-native-v1"` marker and `native` provenance. It stores:

- `pair`: board/simulator/agent seeds and stable `pair_id`;
- candidate, actual-opponent and frozen-search-opponent checkpoint references;
- `initial_sha256`, `final_sha256` (including simulator RNG/rules) and board hash;
- `events`: canonical action ID `a`, original Python chance-event record `event`, and optional sparse visit counts `v`;
- actual winner, completed/cutoff/interrupted status, turns, simulations and learner row counts;
- native protocol/source/binary/runtime/export hashes and a digest over events including visits.

Every incoming action is checked against Python's legal actions. Actual dice/card/steal results must match Python's natural seeded RNG. At completion, every native state field and MT19937 state must match Python, followed by the unchanged complete compact replayer. No raw state/feature tensors are saved in the journal. `fresh_cohorts.Fresh.load_samples` and the existing `derive`, `MemoryData` and W4 losses consume completed games without trainer changes. Position sampling remains seeded and in RAM; allocate **whole games** to train/validation/test before deriving positions. W4 sampling supplies inverse-probability value weights, which the original value loss applies exactly once.

Ten actions at a time are flushed to a gzip journal via atomic replacement. Live/aborted prefixes use `.partial.gz`; previous partials are archived by content hash before restarting a seed. Committed games cannot be overwritten. Only a native winner plus verified complete replay produces `status="completed"`. Corrupted chances, targets, checksums, incomplete games or incompatible search settings are rejected.

### Native wire protocol

The executable serves version-one newline-delimited JSON on stdin/stdout. One request contains an initial `Task`/board, seat seeds, initial MT state, two real-seat ONNX paths, a frozen opponent path, simulation-count **or** time budget, deadline, cancellation-file path and action/turn limits. Replies are `ready`, `events` (at most ten actual transitions, visits and search metrics), then `done` (status, native final state/RNG and counters), or `error`. The full final state exists only on this validation boundary; Python persists the compact journal. The client is serial per process; the collector uses one native client per game worker and handles cancellation at actual-action boundaries.

## Tests and measured smoke results

```sh
python -B -m unittest tests.test_native_selfplay tests.test_parallel_cohorts tests.test_fresh_cohorts tests.test_compact_iterations
CATAN_NATIVE_INTEGRATION=1 python -B -m unittest tests.test_native_selfplay
```

The integration tests collect two real 15-point native games, feed them directly through unchanged `Fresh.load_samples`, verify deterministic seeded reconstruction, preserve whole-game train/validation separation, run W4 sampling and a finite forward/backward/AdamW diagnostic step on an in-memory Exphormer copy, and evaluate the separate validation game. They also compare distinct historical seat models against the original Python PyTorch SearchSeat and check interruptions/overwrite rejection. No diagnostic checkpoint is saved, and these tests do not bypass the production cohort update gate.

Additional retained diagnostics are under `runs/catan-ai/native-integration-smoke/`: two complete eight-simulation games (587 and 520 actions), finite W4 training/validation evidence, and a **complete 128-simulation game with 912 actions, 617 searched decisions and 78,976 simulations**. All passed natural-chance/final-state/RNG replay and produced valid 681-feature/332-action training arrays. These smoke artifacts are excluded from production training collections. See `validation.json` there for final checksums and test results.
