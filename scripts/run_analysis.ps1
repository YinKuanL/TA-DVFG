param(
    [string]$ResultsRoot = "results"
)

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
$ResolvedResults = Join-Path $Root $ResultsRoot

& $Python (Join-Path $Root "experiments\aggregate_results.py") --results-root $ResolvedResults
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "experiments\statistical_tests.py") `
    --input (Join-Path $ResolvedResults "combined\raw_results.csv") `
    --output (Join-Path $ResolvedResults "combined\paired_significance.csv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "experiments\make_ablation_plots.py") `
    --input (Join-Path $ResolvedResults "combined\summary_results.csv") `
    --output-dir (Join-Path $ResolvedResults "combined\figures")
exit $LASTEXITCODE
