# Paper Repository Instructions

## Canonical paper source

The canonical Git-tracked paper source is `paper/source/`.

Compile roots:

- `paper/source/main.tex`
- `paper/source/supplementary.tex`
- `paper/source/ReproducibilityChecklist.tex`

## Before paper edits

Codex must:

1. `git fetch origin`
2. `git checkout main`
3. `git pull --ff-only origin main`
4. compare Git paper source with Overleaf

## After approved paper edits

Codex must:

1. validate local LaTeX structure;
2. push only the changed paper source files to Overleaf;
3. verify the Overleaf content matches local source;
4. commit the same source to Git;
5. push the current Git branch.

## No silent divergence

Never leave `GitHub version != Overleaf version` without explicitly reporting
it.

## No automatic conflict overwrite

If both sides changed, stop before writing.
