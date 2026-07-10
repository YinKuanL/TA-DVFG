# Paper Synchronization Protocol

## Roles

- `GitHub` is the canonical version history.
- `Overleaf` is the compilation and visual review mirror.
- `paper/source/` is the canonical local paper directory.

## Before Every Paper Task

1. `git fetch origin`
2. `git checkout main`
3. `git pull --ff-only origin main`
4. inspect `git status`
5. compare `paper/source/` with the latest Overleaf source

## Conflict Rule

If GitHub and Overleaf both changed since the last sync:

`STOP`

Do not automatically overwrite either side.

Report:

- files changed on GitHub
- files changed on Overleaf
- conflicting files

## Normal GitHub-First Task

1. pull GitHub
2. edit `paper/source/` locally
3. validate
4. push matching source to Overleaf
5. verify Overleaf source
6. commit Git
7. push GitHub

## Overleaf-First Manual Edit

1. detect Overleaf change
2. import the exact Overleaf change into `paper/source/`
3. commit the synchronization
4. push GitHub
5. begin the requested paper task
