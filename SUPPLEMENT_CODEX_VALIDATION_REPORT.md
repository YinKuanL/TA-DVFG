# TA-DVFG Supplement Codex Validation Report

Date: 2026-07-05

## Overall Status

FAIL / BLOCKED for final submission-grade approval.

Reason: the repository does not contain `aaai2027.sty`, no TeX Live/MacTeX tools are installed on PATH, and bundled Tectonic fails with Windows `Access is denied (os error 5)`. Therefore I could not produce or visually inspect a final compiled PDF. I made only minimal source corrections that are directly supported by implementation evidence and copied missing already-existing figure assets into the source-expected `figures/` directory.

## Deliverables Produced

- Corrected source: `Collaboration_without_Representation_Alignment_supplement_fixed.tex`
- Backup of original candidate: `Collaboration_without_Representation_Alignment_supplement_fixed.tex.bak_codex_verify`
- Validation report: `SUPPLEMENT_CODEX_VALIDATION_REPORT.md`
- Final PDF: not produced; blocked by missing/blocked LaTeX toolchain.

## Step 1: Source And Compilation Audit

Status: FAIL / BLOCKED.

Inspected:

- `Collaboration_without_Representation_Alignment_supplement_fixed.tex`
- repository root
- `figures/`
- `overleaf_tadvfg_aaai2026/figures/`
- bundled LaTeX helper at `C:\Users\Yin Kuan\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py`

Findings:

- Candidate source uses `\usepackage[submission]{aaai2027}`.
- Recursive search found no `aaai2027.sty`.
- PATH contains no `pdflatex`, `latexmk`, `kpsewhich`, `xelatex`, or `lualatex`.
- Bundled Tectonic 0.16.9 was detected, but compilation failed with `Access is denied (os error 5)`.
- Several referenced figures were missing from root `figures/` but existed in `overleaf_tadvfg_aaai2026/figures/`; these were copied into `figures/`.
- Source-level checks found no literal `Section ;`, no literal `??`, no duplicate labels, and no unresolved `\ref{...}` targets by textual label scan.

Compilation warning summary:

- No final `.log` was produced.
- Tectonic attempt failed before LaTeX diagnostics.
- TeX Live attempt skipped because no TeX Live/MacTeX installation was detected.

Final page count:

- Unavailable because no final PDF was produced.

## Step 2: Algorithm-To-Code Verification

Status: PARTIAL PASS, with source corrections made.

Inspected:

- `main experiment/ta_dvfg_hgb_reliability.py`
- relevant result configs under `results/`

Verified implementation facts:

- Candidate edge set for graph-validation TA-DVFG is the complete undirected party graph: `[(i, j) for i in range(n) for j in range(i + 1, n)]`.
- Headline audited configuration appears in run manifests/configs: `m=5`, `tau=0.001`, `B=15`, `d_max=2`, `K=1`, `self_weight=0.85`.
- Consensus neighbor weights use `max(reliability, 1e-3)` and normalize over neighbors.
- Active readout weights use `max(reliability, 1e-3)^1.0 * (degree + 1)^0.5`.
- Constants are exposed as defaults/fixed config values: reliability floor `1e-3`, reliability power `1.0`, topology-degree readout power `0.5`.
- Graph-validation candidate list is sorted by descending pair utility and descending party indices. If validation scores tie, higher utility wins; exact utility ties retain the first candidate in that deterministic order.

Changes made:

- Lines 156--158: replaced generic `(r+\epsilon)^\gamma` consensus weights with implemented `max(r,10^{-3})` weights.
- Lines 174--176: replaced generic active-readout weights with implemented reliability and degree weighting.
- Lines 236--239: replaced lexicographic tie-breaking sentence with exact implemented deterministic ordering.
- Line 314: updated proof wording from generic epsilon/gamma to positive reliability floor.
- Lines 1005--1007: replaced generic epsilon/gamma prose with literal values `10^{-3}`, `1.0`, and `0.5`.

## Step 3: MovieLens Correctness Audit

Status: PASS for checked numeric/code items.

Inspected:

- `models/movielens_parties.py`
- `experiments/movielens/run_movielens_tadvfg.py`
- `outputs/movielens_final_package/raw_results/movielens_tadvfg_summary.csv`
- `outputs/movielens_final_package/raw_results/movielens_tadvfg_per_seed.csv`
- `outputs/movielens_final_package/raw_results/movielens_15party_summary.csv`

Verified:

- Five-party seeds are `42,43,44,45,46`.
- Task metric is ROC-AUC.
- MovieLens runner exports aligned two-class probabilities with shape `[party, target, 2]`.
- Binary scores are implemented as two-class softmax probabilities; using column `[:,1]` for ROC-AUC is mathematically consistent with the source statement `(1-p,p)`.
- Five-party numbers match packaged outputs:
  - Best Single: `0.742081...` AUC, `0` communication.
  - Adaptive Pairwise: `0.750768...` AUC, `5,201,092` communication.
  - Full Mesh: `0.749516...` AUC, `10,002,100` communication.
  - TA-DVFG: `0.751205...` AUC, `3,200,672` communication.
- 15-party Main/Hard values and link counts are traceable to `movielens_15party_summary.csv`.
- Source wording remains narrow: it does not claim universal AUC superiority.

## Step 4: Matched-Baseline Audit

Status: PARTIAL PASS.

Inspected:

- `Collaboration_without_Representation_Alignment_supplement_fixed.tex`
- `outputs/ac_review_closure/BEST_MATCHED_SELECTION_REPORT.md`
- `outputs/ac_review_closure/best_matched_selection_audit.csv`
- core cached result CSVs under `results/core_cached_v2/`
- legacy result CSVs under `main experiment/`

Findings:

- The supplement source contains no primary `Best Matched Sparse` row.
- The only `best matched` occurrence in the supplement is an explanatory caution that a post-hoc best matched row would be oracle/test-selected.
- Existing audit files confirm that `experiments/final_analysis.py` previously generated `Best Matched Sparse` as a post-hoc test-mean envelope over fixed matched controls. That row is not reintroduced in this supplement.
- Individual matched controls (`Ring Matched`, `Random Matched`, `Expander Matched`) remain backed by fixed per-run outputs.

Ring-vs-Expander investigation:

- For full 15-edge degree-2 controls, `fixed_ring` and `expander` often produce the same 15-cycle topology, e.g. selected edges `[[0,1],[0,14],[1,2],...,[13,14]]`.
- This explains identical rows in some settings as intentional constructor equivalence under `N=15`, degree-2, 15-edge cycle-like configuration, not a logging bug.
- Matched sparse variants are not always identical and remain separately reported.

## Step 5: Communication-Accounting Audit

Status: PARTIAL PASS.

Inspected:

- `main experiment/ta_dvfg_hgb_reliability.py`
- MovieLens packaged raw summaries
- supplement Table S5 source

Verified:

- HGB communication accounting separates peer exchange, readout, and topology-control traffic in output columns.
- MovieLens five-party totals match packaged output values.
- Source distinguishes sparse P2P consensus from active readout collection.
- Source states TA-DVFG does not collect hidden states and that party-local readout requires no global collection.
- Table S5 wording says HGB and MovieLens communication totals should not be directly compared and that hidden-width/prediction-payload columns use different natural units.

Not fully verified:

- Every Table S5 number was not recomputed from first principles in this pass.
- PDF readability of Table S5 could not be inspected because compilation failed.

## Step 6: Numerical And Statistical Consistency Audit

Status: PARTIAL PASS.

Verified by direct source/output inspection:

- MovieLens five-party values match packaged CSVs.
- K-sensitivity ACM/DBLP Hard summary exists and matches source values to rounding.
- Alignment source statements are traceable to `figures/alignment_acmhard5_*` and alignment suite text.

Arithmetic checked in source:

- ACM nested scaling gap stated as `12.59` points.
- DBLP nested scaling gap stated as `5.16` points.
- ACM TA-DVFG drop stated as `2.41` points.
- ACM Full Mesh drop stated as `15.00` points.
- DBLP TA-DVFG drop stated as `0.76` points.
- DBLP Full Mesh drop stated as `5.92` points.
- Alignment comparison `77.10 - 85.60 = -8.50` points is arithmetically correct.
- Five-pair exact two-sided Wilcoxon minimum `0.0625` is correct for five nonzero paired signs.

Not fully verified:

- Every mean, standard deviation, CI, p-value, win/tie/loss count, and bold/underline designation against raw packaged outputs. This remains incomplete because the final PDF and full auxiliary source-generated layout could not be produced.

## Step 7: Final Layout Audit

Status: FAIL / BLOCKED.

Reason:

- No final PDF was produced.
- Therefore page-by-page visual inspection, page count, float order, table clipping, overfull boxes, black boxes, and figure-only page checks could not be completed.

## Files Changed

- `Collaboration_without_Representation_Alignment_supplement_fixed.tex`
- `figures/label_budget_topology_frequency_tight.png`
- `figures/minedges_sensitivity_tight.png`
- `figures/consensus_ablation_p11.png`
- `figures/consensus_ablation_p12.png`
- `figures/consensus_ablation_p21.png`
- `figures/consensus_ablation_p22.png`
- `figures/edge_budget_acm_edges.pdf`
- `figures/edge_budget_dblp_edges.pdf`
- `SUPPLEMENT_CODEX_VALIDATION_REPORT.md`

## Unresolved Issues

1. `aaai2027.sty` is missing from the repository.
2. No system TeX Live/MacTeX toolchain is available.
3. Bundled Tectonic fails with `Access is denied (os error 5)`.
4. No final compiled PDF exists from the corrected source.
5. Final page count and visual layout checks are unavailable.

## Required Next Action

Install or provide the real AAAI 2027 LaTeX template, especially `aaai2027.sty`, and make a working LaTeX compiler available. Then rerun clean compilation twice and perform the PDF visual audit before declaring the supplement submission-ready.

