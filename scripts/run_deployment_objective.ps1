param(
  [string]$Device = "cuda",
  [string[]]$Seeds = @("42", "43", "44", "45", "46"),
  [int]$Epochs = 300,
  [switch]$DryRun,
  [switch]$Force,
  [switch]$StrictCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SeedText = ($Seeds -join ",")

$RunArgs = @(
  (Join-Path $RepoRoot "experiments\run_experiments.py"),
  "deployment_objective",
  "--results-root", (Join-Path $RepoRoot "results\deployment_objective_runs"),
  "--device", $Device,
  "--epochs", "$Epochs",
  "--seeds", $SeedText,
  "--no-plots"
)

if ($StrictCache) { $RunArgs += "--strict-cache" }
if ($Force) { $RunArgs += "--force" }
if ($DryRun) { $RunArgs += "--dry-run" }

& $Python @RunArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $DryRun) {
  & $Python (Join-Path $RepoRoot "experiments\deployment_objective_analysis.py") `
    --input-root (Join-Path $RepoRoot "results\deployment_objective_runs\deployment_objective") `
    --output-dir (Join-Path $RepoRoot "results\deployment_objective")
}
