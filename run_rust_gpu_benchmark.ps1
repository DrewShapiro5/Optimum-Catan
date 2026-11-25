$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "agent_apis\neural\win_run_helper.ps1") `
    -Module "agent_apis.neural.native_gpu_benchmark" `
    -Banner "==> Running Rust CPU vs Batched GPU benchmark..." `
    -ModuleArgs $args
exit $LASTEXITCODE
