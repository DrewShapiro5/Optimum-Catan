param(
    [Parameter(Mandatory = $true)]
    [string]$Module,
    [string]$Banner = "",
    [AllowEmptyCollection()]
    [string[]]$ModuleArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

$WinGuardCode = @'
using System;
using System.Runtime.InteropServices;

public static class WinRunGuard {
    public const int STD_INPUT_HANDLE = -10;
    public const uint ENABLE_QUICK_EDIT_MODE = 0x0040;
    public const uint ENABLE_EXTENDED_FLAGS = 0x0080;
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const int ProcessPowerThrottling = 4;
    public const uint PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1;
    public const uint PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1;

    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_POWER_THROTTLING_STATE {
        public uint Version;
        public uint ControlMask;
        public uint StateMask;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr GetStdHandle(int nStdHandle);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetProcessInformation(
        IntPtr hProcess,
        int ProcessInformationClass,
        ref PROCESS_POWER_THROTTLING_STATE ProcessInformation,
        uint ProcessInformationSize
    );
}
'@

if (-not ("WinRunGuard" -as [type])) {
    Add-Type -TypeDefinition $WinGuardCode -ErrorAction SilentlyContinue | Out-Null
}

$stdinHandle = [IntPtr]::Zero
[uint32]$origConsoleMode = 0
$hasConsole = $false
$proc = [System.Diagnostics.Process]::GetCurrentProcess()
$origPriority = $proc.PriorityClass

try {
    if ("WinRunGuard" -as [type]) {
        $stdinHandle = [WinRunGuard]::GetStdHandle([WinRunGuard]::STD_INPUT_HANDLE)
        if ($stdinHandle -ne [IntPtr]::Zero -and [WinRunGuard]::GetConsoleMode($stdinHandle, [ref]$origConsoleMode)) {
            $hasConsole = $true
            $newMode = ($origConsoleMode -band (-bnot [WinRunGuard]::ENABLE_QUICK_EDIT_MODE)) -bor [WinRunGuard]::ENABLE_EXTENDED_FLAGS
            [void][WinRunGuard]::SetConsoleMode($stdinHandle, $newMode)
        }
        [void][WinRunGuard]::SetThreadExecutionState([WinRunGuard]::ES_CONTINUOUS -bor [WinRunGuard]::ES_SYSTEM_REQUIRED)
        $state = New-Object WinRunGuard+PROCESS_POWER_THROTTLING_STATE
        $state.Version = [WinRunGuard]::PROCESS_POWER_THROTTLING_CURRENT_VERSION
        $state.ControlMask = [WinRunGuard]::PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        $state.StateMask = 0
        [void][WinRunGuard]::SetProcessInformation(
            $proc.Handle,
            [WinRunGuard]::ProcessPowerThrottling,
            [ref]$state,
            [uint32][System.Runtime.InteropServices.Marshal]::SizeOf($state)
        )
    }
    try {
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal
    } catch {}

    $python = if (Test-Path ".conda\python.exe") { ".conda\python.exe" } elseif (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

    Write-Host "==> Building native Rust worker (release)..."
    cargo build --release --manifest-path "agent_apis\neural\native_worker\Cargo.toml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($Banner) {
        Write-Host $Banner
    }
    & $python -B -m $Module @ModuleArgs
    exit $LASTEXITCODE
} finally {
    if ($hasConsole) {
        [void][WinRunGuard]::SetConsoleMode($stdinHandle, $origConsoleMode)
    }
    if ("WinRunGuard" -as [type]) {
        [void][WinRunGuard]::SetThreadExecutionState([WinRunGuard]::ES_CONTINUOUS)
    }
    try {
        $proc.PriorityClass = $origPriority
    } catch {}
}
