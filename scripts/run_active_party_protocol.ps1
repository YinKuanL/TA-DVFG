param(
    [string]$Device = "cuda",
    [string]$Seeds = "42,43,44,45,46",
    [int]$Epochs = 300,
    [switch]$HardOnly,
    [switch]$DryRun,
    [switch]$Force
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }

$ArgsList = @(
    (Join-Path $Root "experiments\run_experiments.py"),
    "active_party_protocol",
    "--device", $Device,
    "--seeds", $Seeds,
    "--epochs", $Epochs,
    "--strict-cache",
    "--continue-on-error",
    "--no-plots"
)
if ($HardOnly) { $ArgsList += @("--match", "_hard_") }
if ($DryRun) { $ArgsList += "--dry-run" }
if ($Force) { $ArgsList += "--force" }

& $Python @ArgsList
exit $LASTEXITCODE
