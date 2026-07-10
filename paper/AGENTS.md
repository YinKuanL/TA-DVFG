# Paper Editing Instructions

These rules apply to work under this paper directory and override broader repository guidance when more specific.

## First explain, then edit

Before changing LaTeX:

1. identify the main entry file;
2. identify included section files;
3. identify bibliography and figure dependencies;
4. summarize the current section structure;
5. estimate the likely page/line impact of the proposed edit.

## Minimal-edit policy

- Prefer local sentence or paragraph edits over full-section rewrites.
- Preserve the author's existing voice unless the task explicitly requests a different style.
- Do not make unrelated wording improvements.
- Do not change notation globally without checking every dependent definition, equation, table, figure, and supplement reference.
- Do not add packages unless necessary.
- Keep AAAI/conference formatting constraints intact.

## Claim discipline

For every strengthened claim, check `../docs/CLAIM_MAP.md`.

Use calibrated wording:
- "shows" only for directly supported evidence;
- "suggests" for limited evidence;
- "is consistent with" for non-causal observations;
- "outperforms" only when the comparison protocol supports that statement.

Do not convert:
- a reference method into a baseline;
- an oracle selection into a deployable method;
- a validation-selected result into a test-selected result;
- statistical indistinguishability into equivalence.

## Page-limit discipline

When a task is about shortening:

1. identify repeated ideas;
2. remove redundancy before removing technical content;
3. preserve definitions and protocol details needed for reproducibility;
4. report approximate words/lines removed;
5. compile and check actual page impact when a compiler is available.

## Compile discipline

After meaningful LaTeX changes:

1. compile the correct main file;
2. inspect errors;
3. inspect warnings relevant to references, citations, figures, tables, and overfull boxes;
4. report the actual result.

Never report "formatting fixed" from source inspection alone when PDF appearance is the issue.

## Completion report additions

Also report:
- sections touched;
- claims changed;
- citations changed;
- estimated and actual page impact;
- compile status.
