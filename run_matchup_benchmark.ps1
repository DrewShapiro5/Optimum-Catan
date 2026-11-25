$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "agent_apis\neural\win_run_helper.ps1") `
    -Module "agent_apis.neural.matchup_benchmark" `
    -ModuleArgs $args
exit $LASTEXITCODE
