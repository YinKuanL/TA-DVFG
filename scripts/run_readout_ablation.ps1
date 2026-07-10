param(
    [string]$Device = "auto",
    [string]$Seeds = "42,43,44,45,46",
    [int]$Epochs = 300,
    [switch]$DryRun,
    [switch]$Force
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }

$ArgsList = @(
    (Join-Path $Root "experiments\run_experiments.py"),
    "readout_ablation",
    "active_party_evaluator",
    "--device", $Device,
    "--seeds", $Seeds,
    "--epochs", $Epochs,
    "--continue-on-error"
)
if ($DryRun) { $ArgsList += "--dry-run" }
if ($Force) { $ArgsList += "--force" }

& $Python @ArgsList
exit $LASTEXITCODE
