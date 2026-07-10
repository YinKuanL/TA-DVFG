param(
    [ValidateSet("cached", "training", "all")]
    [string]$Stage = "all",
    [string]$Device = "cuda",
    [string]$Seeds = "42,43,44,45,46",
    [int]$Epochs = 300,
    [switch]$DryRun,
    [switch]$Force
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
$Runner = Join-Path $Root "experiments\run_experiments.py"

function Invoke-ExperimentBatch {
    param(
        [string[]]$Suites,
        [switch]$StrictCache
    )
    $ArgsList = @(
        $Runner
    ) + $Suites + @(
        "--device", $Device,
        "--seeds", $Seeds,
        "--epochs", $Epochs,
        "--continue-on-error",
        "--no-plots"
    )
    if ($StrictCache) { $ArgsList += "--strict-cache" }
    if ($DryRun) { $ArgsList += "--dry-run" }
    if ($Force) { $ArgsList += "--force" }
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Stage -in @("cached", "all")) {
    Invoke-ExperimentBatch -StrictCache -Suites @(
        "split_validation",
        "decentralized_readout",
        "active_party_identity",
        "validation_label_budget",
        "topology_frequency"
    )
}

if ($Stage -in @("training", "all")) {
    Invoke-ExperimentBatch -Suites @(
        "strict_label_training",
        "party_scalability"
    )
}
