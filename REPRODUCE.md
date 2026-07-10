# Reproduction Notes

## Canonical Compile Roots

The canonical Git-tracked paper sources are:

- `paper/source/main.tex`
- `paper/source/supplementary.tex`
- `paper/source/ReproducibilityChecklist.tex`

These same sources are mirrored to the configured Overleaf project for
compilation and visual review.

## Required Local Paper Files

The paper source directory contains:

- the three compile roots listed above;
- `paper/source/references.bib`;
- `paper/source/aaai2027.sty`;
- `paper/source/aaai2027.bst`;
- the figures referenced by the main paper and supplement.

## Validation Scope

Static validation for a paper-source sync should confirm:

1. each compile root exists;
2. referenced bibliography, style, and figure dependencies are present;
3. no wording or numeric scientific content changed during synchronization.

Do not claim a successful PDF compile unless a LaTeX compiler was actually run.
