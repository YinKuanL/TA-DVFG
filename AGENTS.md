# Research Repository Instructions

This repository contains research code, experimental evidence, and paper material. Correctness, traceability, and scope control matter more than speed.

## Operating mode

Default sequence:

```text
UNDERSTAND → PLAN → APPROVE → EDIT → VERIFY → REPORT → STOP
```

For a substantial task, do not begin by editing.

## Before any substantial change

1. Read `docs/CURRENT_TASK.md`.
2. Read the relevant control documents in `docs/`.
3. Inventory the relevant files instead of assuming paths.
4. Explain the current architecture in simple terms.
5. Identify the minimum files needed for the task.
6. Propose a scoped plan.
7. List expensive commands or experiments separately.
8. Stop before execution unless the task already includes explicit approval to execute.

## Scope control

- Modify only the approved files.
- Do not rewrite unrelated sections.
- Do not run unrelated experiments.
- Do not add "helpful" baselines, ablations, datasets, seeds, or plots without approval.
- Do not perform repository-wide cleanup during a paper task.
- Do not rename files casually; paper references and scripts may depend on them.

## Evidence integrity

Treat the following as protected by default:

- historical CSV files;
- raw outputs;
- experiment logs;
- checkpoints;
- cached datasets;
- archived plots;
- submitted PDFs;
- prior camera-ready or submission snapshots.

Do not modify, delete, regenerate, or overwrite protected artifacts unless explicitly named in the approved task.

Never:
- fabricate results;
- average across incompatible settings;
- silently drop seeds;
- select methods using test performance unless explicitly labeled as oracle/post-hoc;
- strengthen a claim beyond its evidence.

## Experiments

Before running an experiment, state:

```text
Purpose:
Question answered:
Exact command:
Expected runtime class: small / medium / large
Expected outputs:
Files that may change:
Stop condition:
```

Run only approved experiments.

Prefer:
1. static inspection;
2. unit or smoke checks;
3. one-seed / tiny-run validation;
4. targeted experiment;
5. full benchmark only when explicitly approved.

## Paper changes

Paper changes must preserve:
- factual consistency with experiments;
- claim-to-evidence traceability;
- citation correctness;
- conference page limits and formatting constraints;
- consistency between main paper and supplement.

Do not claim a compile succeeded unless it actually succeeded.

## Required completion report

```text
TASK COMPLETED
Files changed:
Files intentionally not changed:
What changed:
Why:
Commands run:
Validation:
Evidence updated:
Remaining risks:
Suggested next step:
```

Then stop.
