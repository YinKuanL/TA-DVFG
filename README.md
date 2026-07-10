# TA-DVFG Research Repository

This repository contains the research code, experiment artifacts, and paper
material for the TA-DVFG project.

## Canonical Paper Source

The Git-tracked canonical paper source now lives under `paper/source/`.

- `GitHub` is the canonical version history.
- `Overleaf` is the compilation and visual review mirror.
- `Codex` is the synchronization agent between the two.

The canonical paper roots are:

- `paper/source/main.tex`
- `paper/source/supplementary.tex`
- `paper/source/ReproducibilityChecklist.tex`

The source directory also contains the required bibliography, AAAI style files,
and only the figures/assets referenced by those compile roots.

`CameraReady2027.tex` is not part of the canonical paper source because the
current paper uses `main.tex` and `supplementary.tex` as the real entry points.

## Synchronization Workflow

Normal workflow:

1. Pull the latest `origin/main`.
2. Compare `paper/source/` against the configured Overleaf project.
3. Edit `paper/source/` locally.
4. Validate the local LaTeX structure.
5. Synchronize the matching source to Overleaf.
6. Commit and push GitHub.

Detailed synchronization rules are documented in
`docs/PAPER_SYNC_PROTOCOL.md`.

## Repository Scope

This repository also includes experiment code, cached outputs, and supporting
analysis material. Those artifacts are not implicitly regenerated as part of a
paper-edit task.
