# AC Review Closure Status

Scope: targeted audits only. No new full benchmark suite was launched.

## Phase Status

- Phase 0 inventory: complete.
- Phase 1 Best Matched selection audit: complete; blocking issue found for grouped paper row.
- Phase 2A disjoint-validation audit: existing ACM/DBLP Hard split-validation summaries found and audited.
- Phase 2B candidate-level validation-to-test replay: not run in this pass.
- Phase 3 Global Top-k wording audit: complete; patch snippet generated.
- Phase 4 real-world wording audit: complete; patch snippet generated.
- Phase 5 optional exact-search sanity check: not run.
- Phase 6 scalability/privacy wording: patch snippet generated.
- Final paper integration: not started; do not patch main paper until the blocking Best Matched label is resolved.

## Highest-Priority Decision

The paper grouped row named `Best Matched Sparse` is generated in `experiments/final_analysis.py` by selecting the fixed matched family with the largest test mean per dataset/setting. This is a post-hoc test envelope, not a deployable validation-selected baseline. It should be relabeled `Oracle Best Matched` or replaced in the main table by a predeclared matched control.

## Generated Patch Snippets

- `paper_global_topk_patch.tex`
- `paper_real_world_wording_patch.tex`
- `paper_scalability_privacy_patch.tex`

These are suggestions only; `main_aaai27_alignment_submission_ready.tex` and
`supplementary_aaai27_alignment_submission_ready.tex` were not modified.
