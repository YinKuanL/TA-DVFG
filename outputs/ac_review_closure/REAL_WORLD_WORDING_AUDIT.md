# Real-World Evidence Wording Audit

Scope: `main_aaai27_alignment_submission_ready.tex` and
`supplementary_aaai27_alignment_submission_ready.tex`.

## Finding

The current wording is mostly reviewer-safe. It already states that MovieLens
15-party Main/Hard are controlled stress constructions rather than natural
15-organization federations. The remaining risk is the phrase
"real-world predictors" or "real-world MovieLens evaluation" being read as
"naturally occurring multi-organization federation."

## Safe Terms

Use:

- real-data heterogeneous predictor benchmark
- real-world recommendation task with constructed vertical party interfaces
- controlled 15-party weak-predictor stress setting

Avoid:

- real-world federation
- real-world multi-organization federation
- real-world heterogeneous federation

## Lines To Patch

- `main_aaai27_alignment_submission_ready.tex:631`: heading
  "Real-world MovieLens evaluation" should become "MovieLens real-data
  heterogeneous predictor evaluation".
- `main_aaai27_alignment_submission_ready.tex:480`: phrase
  "heterogeneous real-world predictors" should become "real-data heterogeneous
  predictor interfaces".
- `supplementary_aaai27_alignment_submission_ready.tex:74`: same wording
  adjustment.

## Required Limitation

Keep the limitation explicit:

> MovieLens-5 uses real recommendation data and explicitly heterogeneous
> predictors, but it is not a naturally occurring five-organization federation.
