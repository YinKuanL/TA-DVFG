param(
    [string]$Device = "cuda",
    [string[]]$Seeds = @("42", "43", "44", "45", "46"),
    [int]$Epochs = 300,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$StrictCache
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
$Runner = Join-Path $Root "experiments\run_experiments.py"
$Analyzer = Join-Path $Root "experiments\strong_setting_analysis.py"
$RunsRoot = Join-Path $Root "results\strong_setting_runs"
$OutputRoot = Join-Path $Root "results\strong_setting"
$SeedText = ($Seeds -join ",")

$ArgsList = @(
    $Runner,
    "strong_setting",
    "--results-root", $RunsRoot,
    "--device", $Device,
    "--seeds", $SeedText,
    "--epochs", $Epochs,
    "--continue-on-error",
    "--no-plots"
)
if ($StrictCache) { $ArgsList += "--strict-cache" }
if ($DryRun) { $ArgsList += "--dry-run" }
if ($Force) { $ArgsList += "--force" }

& $Python @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $DryRun) {
    & $Python $Analyzer `
        --input-root (Join-Path $RunsRoot "strong_setting") `
        --output-dir $OutputRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
