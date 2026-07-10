# Global Top-k Wording Audit

Scope: `main_aaai27_alignment_submission_ready.tex`,
`supplementary_aaai27_alignment_submission_ready.tex`, and `appendix.tex`.

## Finding

The current paper is mostly safe, but the abstract and conclusion can still be
read as claiming that TA-DVFG's primary contribution is higher active-readout
accuracy. The evidence is more nuanced:

- Global Top-k Reliability is often the strongest centralized prediction
  selection reference.
- TA-DVFG's main capability difference is not centralized active-readout
  dominance; it learns an explicit sparse peer graph and supports sparse P2P
  deployment plus party-local post-consensus readout.
- The strongest empirical wins for TA-DVFG are against Full Mesh, fixed matched
  sparse controls, pairwise/complementarity topology heuristics, and
  reliability-selected latent fusion on ACM/DBLP Hard.

## Lines To Patch

- `main_aaai27_alignment_submission_ready.tex:85-93`: abstract result sentence
  mentions Full Mesh and matched sparse controls, but should add that Global
  Top-k is a centralized prediction-selection reference with a different
  capability profile.
- `main_aaai27_alignment_submission_ready.tex:654-660`: discussion already
  separates MovieLens/HGB roles, but can explicitly say that Global Top-k is
  not the target to dominate on a single active readout.
- `main_aaai27_alignment_submission_ready.tex:831-834`: conclusion says learned
  sparse graphs can outperform baselines; clarify which baselines and preserve
  the Global Top-k caveat.

## Recommended Framing

Use this sentence once, preferably in the discussion:

> TA-DVFG is not designed to dominate centralized Global Top-k on a single
> active readout; its purpose is to recover comparable task-level utility while
> learning explicit sparse peer paths that support deployment-specific active
> and local inference.

Do not overuse "diagnostic" as a shield. State the capability difference once,
then report Global Top-k plainly.
