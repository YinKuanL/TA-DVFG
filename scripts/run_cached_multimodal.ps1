param(
    [string]$Device = "cuda",
    [string]$Seeds = "42,43,44,45,46",
    [int]$Epochs = 300,
    [ValidateSet("core", "ablation", "all")]
    [string]$Mode = "all",
    [switch]$NoPlot,
    [switch]$Force
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }

$Suites = @()
if ($Mode -eq "core" -or $Mode -eq "all") { $Suites += "multimodal_core" }
if ($Mode -eq "ablation" -or $Mode -eq "all") { $Suites += "modality_ablation" }

$ArgsList = @(
    (Join-Path $Root "experiments\run_experiments.py")
) + $Suites + @(
    "--device", $Device,
    "--seeds", $Seeds,
    "--epochs", $Epochs,
    "--continue-on-error"
)
if ($NoPlot) { $ArgsList += "--no-plots" }
if ($Force) { $ArgsList += "--force" }

& $Python @ArgsList
exit $LASTEXITCODE
