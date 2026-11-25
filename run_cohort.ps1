$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "agent_apis\neural\win_run_helper.ps1") `
    -Module "agent_apis.neural.coreml_cohort_pipeline" `
    -Banner "==> Running cohort pipeline..." `
    -ModuleArgs $args
exit $LASTEXITCODE
