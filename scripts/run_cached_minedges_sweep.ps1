param(
    [string]$Device = "cuda",
    [string]$Seeds = "42,43,44,45,46",
    [int]$Epochs = 300,
    [ValidateSet("all", "build-cache", "sweep")]
    [string]$Stage = "all",
    [switch]$Force
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }

$ArgsList = @(
    (Join-Path $Root "experiments\run_cached_minedges_sweep.py"),
    "--stage", $Stage,
    "--device", $Device,
    "--seeds", $Seeds,
    "--epochs", $Epochs
)
if ($Force) { $ArgsList += "--force" }

& $Python @ArgsList
exit $LASTEXITCODE
