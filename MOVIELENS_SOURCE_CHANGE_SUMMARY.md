# MovieLens Source-Change Summary

Date: 2026-07-05

## Code And Output Changes

- Re-ran MovieLens five-party suite for seeds 42--46 using current `experiments/movielens/run_movielens_tadvfg.py`.
- Re-ran MovieLens 15-party Main and Hard suites for seeds 42--46.
- Regenerated extended MovieLens statistics, Pareto inputs, paired statistics, figures, 15-party aggregation, and packaged outputs.
- Archived pre-rerun outputs under `outputs/movielens_archive_pre_leakage_safe_20260705/`.
- Copied leakage-safe outputs back to canonical paths under `outputs/movielens`, `outputs/movielens_main15`, `outputs/movielens_hard15`, and `outputs/movielens_final_package`.

## Paper/Supplement Changes

- Updated MovieLens macros and 15-party table values in `main_aaai27_alignment_submission_ready.tex`.
- Updated MovieLens summary table values in `Collaboration_without_Representation_Alignment_supplement_fixed.tex`.
- Corrected the supplement readout equation: the active readout set is all non-isolated parties when the topology has edges, not active-party union non-isolated parties.
- Clarified that the reliability weight epsilon `1e-3` is distinct from the MovieLens wrapper's final-reliability filtering floor, which is `0.0`.

## Remaining Noncanonical Stale Files

The repository contains many historical TeX drafts with old MovieLens values. They were not all edited in this closure pass. Treat `main_aaai27_alignment_submission_ready.tex` and `Collaboration_without_Representation_Alignment_supplement_fixed.tex` as the updated sources from this pass.
