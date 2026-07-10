param(
    [Parameter(Mandatory=$true)]
    [string]$Name
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Location).Path
$Template = Join-Path $RepoRoot "codex_tasks\TASK_TEMPLATE.md"

if (-not (Test-Path $Template)) {
    throw "Task template not found. Run from the repository root."
}

$SafeName = ($Name.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
$Stamp = Get-Date -Format "yyyyMMdd"
$Target = Join-Path $RepoRoot "codex_tasks\$Stamp-$SafeName.md"

if (Test-Path $Target) {
    throw "Task file already exists: $Target"
}

Copy-Item $Template $Target

$CurrentTask = Join-Path $RepoRoot "docs\CURRENT_TASK.md"
@"
# Current Task

## Goal

See ``codex_tasks/$Stamp-$SafeName.md``.

## Approved scope

Planning only until the task file is completed and approved.

## Files allowed to change

- ``codex_tasks/$Stamp-$SafeName.md``
- ``docs/CURRENT_TASK.md``

## Files explicitly protected

All paper, experiment, result, cache, and historical output files.

## Approved commands

Read-only inspection only.

## Expensive work approved?

No.

## Stop condition

Stop after the plan and wait for approval.
"@ | Set-Content -Path $CurrentTask -Encoding UTF8

Write-Host "Created task:"
Write-Host "  $Target"
Write-Host ""
Write-Host "Updated:"
Write-Host "  $CurrentTask"
