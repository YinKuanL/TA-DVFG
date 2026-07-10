$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path

if (-not (Test-Path (Join-Path $RepoRoot "AGENTS.md"))) {
    throw "Run this script from the repository root after copying the workspace files."
}

$RequiredDirs = @(
    "docs",
    "codex_tasks",
    ".codex"
)

foreach ($dir in $RequiredDirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $dir) | Out-Null
}

Write-Host "Research workspace initialized at:"
Write-Host "  $RepoRoot"
Write-Host ""
Write-Host "Control files:"
Get-ChildItem (Join-Path $RepoRoot "docs") -File | ForEach-Object {
    Write-Host "  docs/$($_.Name)"
}
Write-Host ""
Write-Host "No paper text, experiments, historical results, or caches were modified."
