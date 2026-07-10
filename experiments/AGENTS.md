# Experiment Instructions

These rules apply to experimental code and outputs.

## Approval gate

Do not run a new experiment until the purpose, exact configuration, outputs, and stop condition are explicit.

A request to inspect code is not permission to run training.
A request to fix a plot is not permission to rerun experiments.
A request to update the paper is not permission to regenerate results.

## Protected data

By default, never modify or overwrite:

- raw result CSVs;
- per-seed results;
- historical summary CSVs;
- archived experiment folders;
- cached datasets;
- checkpoints;
- submission figures tied to prior results.

Create new versioned output paths for approved reruns.

## Reproducibility

Before an approved run, capture:

- git commit;
- command;
- config;
- dataset/version;
- seeds;
- device;
- output directory.

Afterward, report failures and missing seeds honestly.

## Analysis

- Never select a baseline using test performance unless explicitly marked oracle/post-hoc.
- Do not compare incompatible datasets, protocols, budgets, party counts, or information access as if directly matched.
- Distinguish deployment methods from stronger-information references.
- Keep active-party and passive-party label access assumptions explicit.
- Preserve per-seed evidence whenever available.

## Cheap checks first

Prefer:

```text
static inspection
→ config validation
→ smoke test
→ single small run
→ approved full run
```

Stop at the first stage that answers the question.
