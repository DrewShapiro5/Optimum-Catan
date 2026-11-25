# Catanotron GPU & Training CLI Tools

Reference for the CLI tools and scripts added for batched GPU self-play, head-to-head evaluation, and multi-cohort training. All shell wrappers (`*.sh` on macOS/Linux, `*.ps1` on Windows) are run from the repo root and forward any extra flags straight to the underlying Python module.

---

## 1. Multi-Cohort Training Pipeline (`./run_coreml_cohort.sh` / `.\run_cohort.ps1`)

**Module:** `agent_apis/neural/coreml_cohort_pipeline.py`  
**Purpose:** Collects replay-verified games (CNN vs. CNN via Rust CoreML `--pool`, plus JSettlers vs. CNN via Rust CoreML `--bridge-pool` + pooled Java JVMs), performs a stratified whole-game train/validation split (default 95/5 via `--validation-fraction`, no separate test split), loads all eligible training positions into RAM once per cohort and reslices `--positions-per-game` positions per game every `--resample-every` updates (keeping validation fixed), and trains both **XD-CNN** and **TYPED-EXPHORMER** on MPS with W4 weighting. Automatically chains each cohort's lowest-validation `best.pt` into the next cohort and is resumable per cohort and per game. Never auto-deploys checkpoints.

### Common Examples

```sh
# Run 3 sequential 2,000-game cohorts (default 1,000 CNN vs CNN + 1,000 JSettlers vs CNN)
./run_coreml_cohort.sh --n-cohorts 3

# Collect games only (skip MPS training)
./run_coreml_cohort.sh --n-cohorts 1 --skip-training

# Custom value-loss weight, game mix, and live per-game prints
./run_coreml_cohort.sh --n-cohorts 2 --value-loss-weight 0.25 --cnn-selfplay-games 1200 --jsettlers-cnn-games 800 --print-games
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--n-cohorts` | `1` | Number of sequential cohorts to run without restarting. |
| `--cnn-checkpoint` | `runs/catan-ai/selfplay-only-comparison/cnn/iteration-1/best.pt` | Starting CNN `.pt` checkpoint for Cohort 1. |
| `--transformer-checkpoint` | `runs/catan-ai/selfplay-only-comparison/sparse/iteration-1/best.pt` | Starting Transformer `.pt` checkpoint for Cohort 1. |
| `--output` | `runs/catan-ai/coreml-cohorts-2000` | Root directory for `cohort-001/`, `cohort-002/`, `multi_cohort_summary.json`, and `REPORT.md`. |
| `--cnn-selfplay-games` | `1000` | Completed CNN vs. CNN games per cohort. |
| `--jsettlers-cnn-games` | `1000` | Completed JSettlers vs. CNN games per cohort. |
| `--simulations` | `128` | MCTS simulation budget per searched decision. |
| `--setup-simulations` | Matches `--simulations` | MCTS simulation budget during the initial build phase (`is_initial_build_phase`). |
| `--setup-exploration-games` | `0` | Number of games per cohort played with `--setup-temperature` during initial placements (split proportionally across blocks). |
| `--setup-temperature` | `1.5` | Initial build phase sampling temperature for `--setup-exploration-games` (higher flattens the MCTS visit distribution during setup). |
| `--validation-fraction` | `0.05` | Fraction of games per block reserved for validation (remaining games go to train; no separate test split). Validation positions are derived once per cohort and never resampled. |
| `--positions-per-game` | `64` | Positions sliced per game per resampling round from the in-RAM training pool (always pins all initial build-phase placements). |
| `--resample-every` | `500` | Reslice `--positions-per-game` positions per game from the in-RAM pool every `N` training updates (`0` = slice once per cohort). |
| `--workers` | `128` | Concurrent Rust game workers for CNN vs. CNN (`--pool`). |
| `--jsettlers-workers` | Matches `--workers` | Concurrent Python/Rust workers for JSettlers vs. CNN (`--bridge-pool`). |
| `--jsettlers-jvms` | `12` | Parallel Java JSettlers `Bridge` JVMs (sharded consistently by game session). |
| `--gpu-backend` | Auto (`cuda` if available, else `coreml` on macOS, else `cpu-batch`) | Native Rust ONNX execution provider (`coreml`, `cuda`, or `cpu-batch`). CUDA uses FP16 ONNX by default unless `--fp32` is set. |
| `--train-device` | `auto` | PyTorch device for W4 training and validation (`auto`, `cuda`, `mps`, or `cpu`). |
| `--coreml-batch` / `--batch-size` | `64` | Static padded batch size for `coreml` (or `cuda` with `--fixed-batch`), otherwise max dynamic batch size for `cuda` / `cpu-batch`. |
| `--cuda-fixed-batch` / `--fixed-batch` | `false` | Use static padded batches (matching `--batch-size` / `--coreml-batch`) with a static ONNX graph on `cuda`. |
| `--gpu-streams` | `4` | Concurrent GPU inference owner threads. |
| `--mcts-lanes` | `1` | Concurrent virtual-loss MCTS leaf-selection lanes per game (`1` = pure sequential MCTS). |
| `--virtual-loss` | `1.0` | Virtual loss penalty applied to in-flight edges when `--mcts-lanes > 1`. |
| `--mcts-opponent` | `self-policy` | Opponent model inside MCTS simulations: `self-policy` (candidate's own policy, no-stop rollout), `adversarial-puct` (cached-prior no-stop MO-ISMCTS adversarial PUCT using candidate's own policy), or `frozen-mlp` (legacy seed-11 MLP). |
| `--adversarial-puct` | `false` | Shorthand for `--mcts-opponent adversarial-puct`. |
| `--vp-margin-alpha` | `0.03` | Weight on the `(2p-1)^2`-gated VP-margin utility blend in MCTS (`0.0` disables; max `0.5`). Zero effect at `p = 0.5`; gently breaks flat-Q ties near 0% and 100% win confidence. |
| `--wait-us` | `50` | Max batch-coalescing wait window in microseconds. |
| `--updates` | `2000` | W4 training updates per model per cohort. |
| `--value-loss-weight` | `0.25` | Training multiplier on value BCE loss (validation stays unweighted `policy_nll + value_bce`). |
| `--replay-cohorts` | `2` | Sliding training replay window in cohorts, including the current cohort (`1` = current only, `2` = current + 1 past). Validation always comes strictly from the current cohort. |
| `--seed` | `2026112001` | Base random seed (offset per cohort). |
| `--print-games` | `false` | Print a line to stdout on every completed game instead of every 50 games. |
| `--fp32` | `false` | Use FP32 instead of default FP16 for CUDA ONNX export. |
| `--skip-training` | `false` | Only collect and verify games; skip training. |

---

## 2. Head-to-Head Matchup Benchmark (`./run_matchup_benchmark.sh` / `.\run_matchup_benchmark.ps1`)

**Module:** `agent_apis/neural/matchup_benchmark.py`  
**Purpose:** Evaluates a model checkpoint against **JSettlers (SMART)** or another model checkpoint with paired BLUE/RED seats on matched boards, Wilson 95% confidence intervals, and live running win rates. Writes `summary.json` and `REPORT.md`.

### Common Examples

```sh
# CNN (128 sims, CoreML) vs JSettlers over 100 paired games
./run_matchup_benchmark.sh --candidate runs/catan-ai/coreml-cohorts-2000/cohort-001/cnn/iteration-1/best.pt --opponent jsettlers --games 100 --simulations 128 --backend coreml --workers 32 --coreml-batch 32 --gpu-streams 4 --mcts-lanes 4

# Policy-only (0 sims) CNN vs Transformer over 200 paired games
./run_matchup_benchmark.sh --candidate cnn --opponent transformer --policy-only --games 200 --workers 8

# Model vs Model (128 sims, CoreML)
./run_matchup_benchmark.sh --candidate path/to/new_cnn.pt --opponent path/to/old_cnn.pt --games 100 --simulations 128 --backend coreml --workers 64 --coreml-batch 64 --gpu-streams 4 --mcts-lanes 4
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--candidate` | `cnn` | Candidate `.pt` checkpoint path or preset (`cnn`, `transformer`, `latest-cnn`, `latest-transformer`). |
| `--opponent` | `jsettlers` | `jsettlers`, preset (`cnn`, `transformer`, `latest-cnn`, `latest-transformer`), or `.pt` checkpoint path. |
| `--games` | `20` | Total number of games to play. |
| `--workers` | `8` | Parallel game workers. |
| `--policy-only` | `false` | Pure greedy policy argmax (`0` simulations). Mutually exclusive with `--simulations`. |
| `--simulations` / `--candidate-simulations` | `64` | MCTS simulation budget per decision for the candidate. |
| `--opponent-simulations` | Same as `--simulations` | Separate simulation count for the opponent model (runs natively on Rust CPU/MPS/CoreML when both sides > 0; `0` = policy-only). |
| `--backend` | `cpu` | Inference backend: `cpu`, `mps`, `coreml`, `cuda` (in-process ORT CUDA EP), or `cuda-socket` (PyTorch CUDA TCP server). `cuda`, `cuda-socket`, and `mps` use FP16 by default unless `--fp32` is set. |
| `--max-batch` | `64` | Max batch rows for `mps` / `cuda-socket` server. |
| `--coreml-batch` / `--batch-size` | `32` | Static padded batch size for `coreml` (or `cuda` with `--fixed-batch`), otherwise max dynamic batch size for `cuda`. |
| `--cuda-fixed-batch` / `--fixed-batch` | `false` | Use static padded batches (matching `--batch-size` / `--coreml-batch`) with a static ONNX graph on `cuda`. |
| `--wait-us` | `100` | Batch-coalescing wait window in microseconds. |
| `--gpu-streams` | `2` | Concurrent GPU inference streams (`coreml` / `cuda`). |
| `--mcts-lanes` | `1` | Concurrent virtual-loss MCTS lanes per game. |
| `--virtual-loss` | `1.0` | Virtual loss penalty when `--mcts-lanes > 1`. |
| `--mcts-opponent` | `self-policy` | Candidate's opponent model inside MCTS: `self-policy` (default), `adversarial-puct` (cached-prior no-stop adversarial PUCT using own policy), or `frozen-mlp`. |
| `--opponent-mcts-opponent` | `self-policy` | Opponent model's MCTS opponent in model-vs-model matchups (`self-policy`, `adversarial-puct`, or `frozen-mlp`). |
| `--adversarial-puct` | `false` | Shorthand for `--mcts-opponent adversarial-puct`. |
| `--opponent-adversarial-puct` | `false` | Shorthand for `--opponent-mcts-opponent adversarial-puct`. |
| `--vp-margin-alpha` | `0.03` | Candidate weight on the `(2p-1)^2`-gated VP-margin utility blend in MCTS (`0.0` disables; max `0.5`). |
| `--opponent-vp-margin-alpha` | Same as `--vp-margin-alpha` | Separate VP-margin alpha for the opponent model in model-vs-model matchups. |
| `--python-search` | `false` | Force Python `SearchSession` instead of the native Rust worker. |
| `--seed` | `2026111955` | Base seed for paired boards and agents. |
| `--no-paired` | `false` | Disable paired BLUE/RED seat swapping on identical boards. |
| `--max-turns` | `1000` | Turn cutoff per game. |
| `--max-actions` | `10000` | Action cutoff per game. |
| `--fp32` | `false` | Use FP32 instead of default FP16 on CUDA/PyTorch GPU inference. |
| `--output` | `runs/catan-ai/matchup-benchmark/<timestamp>` | Directory for `summary.json` and `REPORT.md`. |

---

## 3. Throughput Benchmark (`./run_rust_gpu_benchmark.sh` / `.\run_rust_gpu_benchmark.ps1`)

**Module:** `agent_apis/neural/native_gpu_benchmark.py`  
**Purpose:** Benchmarks self-play throughput (`turns/s`, `decisions/s`, `sims/s`) across Rust CPU-only, PyTorch MPS Unix-socket batching, and in-process Rust CoreML EP batching, with MPS numerical parity and full replay verification. Outputs `summary.json`.

### Common Examples

```sh
# Benchmark CoreML pool for CNN and merge into existing summary.json
./run_rust_gpu_benchmark.sh --model cnn --skip-cpu --skip-mps --coreml-workers 128 --coreml-batch 64 --gpu-streams 4 --mcts-lanes 4 --seconds 120 --merge-existing

# Benchmark Transformer on CoreML with per-game completion logs
./run_rust_gpu_benchmark.sh --model transformer --skip-cpu --skip-mps --coreml-workers 32 --coreml-batch 32 --gpu-streams 2 --mcts-lanes 4 --print-games
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--model` | `cnn` | Preset model (`cnn` or `transformer`); sets default checkpoint and output dir. |
| `--checkpoint` | Preset latest `.pt` | Override checkpoint path. |
| `--output` | `runs/catan-ai/native-batched-gpu-benchmark[-transformer]` | Output directory for `summary.json` and temp journals. |
| `--seconds` | `120` | Timed measurement window per configuration. |
| `--simulations` | `128` | MCTS simulation budget per searched decision. |
| `--setup-simulations` | Matches `--simulations` | MCTS simulation budget during the initial build phase (`is_initial_build_phase`). |
| `--seed` | `2026111901` | Base seed for games. |
| `--cpu-workers` | `12` | Comma-separated worker counts for Rust CPU-only baseline. |
| `--mps-workers` | `12,16,24,48,64` | Comma-separated worker counts for PyTorch MPS Unix-socket server. |
| `--coreml-workers` | `16,32` | Comma-separated worker counts for in-process CoreML EP pool (macOS). |
| `--coreml-batch` | `32` | Static padded batch size for CoreML EP. |
| `--cuda-workers` | `32,64,128` | Comma-separated worker counts for in-process CUDA EP pool (Windows/Linux NVIDIA). |
| `--cuda-batch` | `64` | Max dynamic (or static when `--fixed-batch` is passed) batch size for CUDA EP pool. |
| `--cuda-fixed-batch` / `--fixed-batch` | `false` | Use static padded batches (matching `--cuda-batch`) with a static ONNX graph on CUDA EP. |
| `--fp32` | `false` | Use FP32 instead of default FP16 for GPU benchmarks (`cuda`, `cuda-socket`, `mps`). |
| `--cuda-socket-workers` | `32,64` | Comma-separated worker counts for PyTorch CUDA TCP socket server. |
| `--gpu-streams` | `1` | Concurrent GPU inference streams for CoreML/CUDA pool. |
| `--mcts-lanes` | `1` | Concurrent virtual-loss MCTS lanes per game. |
| `--virtual-loss` | `1.0` | Virtual loss penalty when `--mcts-lanes > 1`. |
| `--mcts-opponent` | `self-policy` | Opponent model inside MCTS simulations (`self-policy`, `adversarial-puct`, or `frozen-mlp`). |
| `--adversarial-puct` | `false` | Shorthand for `--mcts-opponent adversarial-puct`. |
| `--vp-margin-alpha` | `0.03` | Weight on the `(2p-1)^2`-gated VP-margin utility blend in MCTS (`0.0` disables; max `0.5`). |
| `--max-batch` | `64` | Max batch size for MPS / CUDA socket server. |
| `--wait-us` | `500` | Batch-coalescing wait window in microseconds. |
| `--skip-cpu` | `false` | Skip CPU-only baseline runs. |
| `--skip-mps` | Auto (`true` if no MPS) | Skip PyTorch MPS socket runs. |
| `--skip-coreml` | Auto (`true` off macOS) | Skip in-process CoreML EP pool runs. |
| `--skip-cuda` | Auto (`true` if no CUDA) | Skip in-process CUDA EP pool runs. |
| `--skip-cuda-socket` | Auto (`true` if no CUDA) | Skip PyTorch CUDA TCP socket runs. |
| `--merge-existing` | `false` | Merge new benchmark configs into an existing `summary.json`. |
| `--print-games` | `false` | Print a line whenever a game completes during a timed window. |

---

## 4. Native Rust Worker Binary (`catan-native-worker`)

**Binary:** `agent_apis/neural/native_worker/target/release/catan-native-worker`  
**Build script:** `agent_apis/neural/native_worker/build.sh`  
**Purpose:** Low-level JSON-lines engine invoked by the Python tools above. Supports four operating modes (mutually exclusive):
- **Default (no mode flag):** Single-game-per-process ONNX CPU worker.
- **`--gpu-socket <PATH>`:** Single-game worker sending `[f32; 681]` frames over a binary Unix-domain socket to `BinaryGpuBatchServer`.
- **`--pool`:** Multi-threaded self-play worker pool sharing an in-process batched ONNX/CoreML queue.
- **`--bridge-pool`:** Interactive multi-game pool (`init` / `choose` / `step` / `close`) used for Model vs. JSettlers on CoreML.

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--gpu-socket <PATH>` | *(none)* | Connect to a PyTorch MPS binary Unix-domain socket server. |
| `--worker-id <ID>` | `0` | Integer worker ID sent in `--gpu-socket` frames. |
| `--pool` | `false` | Run in multi-worker self-play pool mode. |
| `--bridge-pool` | `false` | Run in interactive multi-game bridge-pool mode (for JSettlers). |
| `--workers <N>` | `12` | Number of Rust worker threads (`--pool` / `--bridge-pool`). |
| `--gpu-backend <coreml\|cuda\|cpu-batch>` | `coreml` (macOS) / `cuda` (other) | In-process batch execution provider (`coreml`, `cuda`, or `cpu-batch`). |
| `--max-batch <B>` | `64` | Maximum rows per batched forward pass. |
| `--fixed-batch <B>` | `0` (dynamic) | Pad every batch to exact static size `B` (required for fast static CoreML graphs; enables static exhaustive cuDNN search on CUDA). |
| `--wait-us <US>` | `500` | Queue coalescing timeout in microseconds. |
| `--gpu-streams <S>` | `1` | Number of concurrent inference owner threads pulling from the batch queue. |

---

## 5. Windows + NVIDIA CUDA Setup & Examples

Install Rust (MSVC toolchain), Java (for JSettlers), CUDA-enabled PyTorch, and `onnxruntime-gpu` in your Windows Python environment. `native_assets.py` automatically discovers `onnxruntime.dll` from `onnxruntime-gpu` (or `ORT_DYLIB_PATH`) and prepends the ORT/PyTorch CUDA DLL directories to `PATH`.

CUDA paths use **FP16 by default** (pass `--fp32` to opt back in to FP32). Static padded batching can be enabled on `cuda` with `--fixed-batch` (`--cuda-fixed-batch`).

```powershell
# Multi-cohort training on Windows + RTX 3080 (FP16 CUDA EP games + PyTorch CUDA training)
.\run_cohort.ps1 --n-cohorts 3 --gpu-backend cuda --train-device cuda --workers 256 --batch-size 128 --gpu-streams 1 --wait-us 500 --mcts-lanes 1

# Multi-cohort training with static CUDA batch size 128
.\run_cohort.ps1 --n-cohorts 3 --gpu-backend cuda --train-device cuda --workers 256 --batch-size 128 --fixed-batch --gpu-streams 1 --wait-us 500 --mcts-lanes 1

# Head-to-head matchup benchmark (Model vs JSettlers or Model vs Model) on CUDA (FP16 default; add --fixed-batch or --fp32 if desired)
.\run_matchup_benchmark.ps1 --candidate latest-cnn --opponent jsettlers --games 100 --simulations 128 --backend cuda --workers 256 --batch-size 128 --gpu-streams 1 --wait-us 500 --mcts-lanes 1

# Throughput benchmark comparing FP16 dynamic vs static batching (add --fp32 for FP32 comparison)
.\run_rust_gpu_benchmark.ps1 --model cnn --skip-cpu --skip-cuda-socket --cuda-workers 256 --cuda-batch 128 --gpu-streams 1 --wait-us 500 --seconds 30 --mcts-lanes 1
.\run_rust_gpu_benchmark.ps1 --model cnn --skip-cpu --skip-cuda-socket --cuda-workers 256 --cuda-batch 128 --fixed-batch --gpu-streams 1 --wait-us 500 --seconds 30 --mcts-lanes 1
```
