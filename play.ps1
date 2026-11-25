$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = "1"

$python = if (Test-Path ".conda\python.exe") {
    (Resolve-Path ".conda\python.exe").Path
} elseif (Test-Path ".venv\Scripts\python.exe") {
    (Resolve-Path ".venv\Scripts\python.exe").Path
} elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
    (Join-Path $env:CONDA_PREFIX "python.exe")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    (Get-Command python).Source
} else {
    Write-Error "No Python environment found. Activate your Conda environment (or create .conda/.venv) first."
    exit 1
}

$pyDir = Split-Path $python -Parent
foreach ($dir in @($pyDir, (Join-Path $pyDir "Library\bin"))) {
    if ((Test-Path $dir) -and ($env:PATH -notlike "*$dir*")) {
        $env:PATH = "$dir;$env:PATH"
    }
}

$needsUiBuild = (-not (Test-Path "framework\ui\dist\index.html")) -or (
    Select-String -Path "framework\ui\dist\assets\*.js" -Pattern "localhost:5001" -Quiet -ErrorAction SilentlyContinue
)

if ($needsUiBuild) {
    $npm = if (Test-Path (Join-Path $pyDir "npm.cmd")) {
        Join-Path $pyDir "npm.cmd"
    } elseif (Test-Path (Join-Path $pyDir "Library\bin\npm.cmd")) {
        Join-Path $pyDir "Library\bin\npm.cmd"
    } elseif (Get-Command npm.cmd -ErrorAction SilentlyContinue) {
        (Get-Command npm.cmd).Source
    } elseif (Get-Command npm -ErrorAction SilentlyContinue) {
        (Get-Command npm).Source
    } else {
        Write-Error "Browser interface needs to be built for same-origin API calls, but npm was not found. Activate your Conda environment (with nodejs=24) or install Node.js 24, then re-run .\play.ps1."
        exit 1
    }

    Write-Host "==> Building browser interface (framework\ui\dist)..."
    Push-Location "framework\ui"
    $envLocal = ".env.production.local"
    $hadEnvLocal = Test-Path $envLocal
    try {
        if (-not (Test-Path "node_modules")) {
            & $npm ci --cache ..\..\.cache\npm
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        if (-not $hadEnvLocal) {
            [System.IO.File]::WriteAllText((Join-Path (Get-Location) $envLocal), "CTRON_API_URL=`n")
        }
        & $npm run build
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        if (-not $hadEnvLocal) {
            Remove-Item $envLocal -ErrorAction SilentlyContinue
        }
        Pop-Location
    }
}

& $python -B play_catan.py @args
exit $LASTEXITCODE
