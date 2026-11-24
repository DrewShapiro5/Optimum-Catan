$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

$python = if (Test-Path ".conda\python.exe") { ".conda\python.exe" } elseif (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

Write-Host "==> Building native Rust worker (release)..."
cargo build --release --manifest-path "agent_apis\neural\native_worker\Cargo.toml"

Write-Host "==> Running Rust CPU vs Batched GPU benchmark..."
& $python -B -m agent_apis.neural.native_gpu_benchmark @args
